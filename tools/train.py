"""Reproducible transfer-learning experiment on verified paired reference clips."""
from pathlib import Path
import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pingpong.model import load_model
from pingpong.detector import MEAN,STD,heatmap_point


class Windows(Dataset):
    def __init__(self,folder,prefix,limit):
        self.arrays=[];self.labels=[];self.windows=[]
        for video in sorted(folder.glob(prefix+'*.mp4')):
            cache=video.with_suffix('.npy');labels=video.with_suffix('.labels.npy')
            if not cache.exists() or not labels.exists():continue
            images=np.load(cache,mmap_mode='r');xy=np.load(labels)
            if len(images)!=len(xy):raise ValueError('Unaligned data: '+video.name)
            clip=len(self.arrays);self.arrays.append(images);self.labels.append(xy)
            self.windows.extend((clip,i) for i in range(0,len(xy)-2,3))
        if len(self.windows)>limit:
            indices=np.linspace(0,len(self.windows)-1,limit,dtype=int)
            self.windows=[self.windows[i] for i in indices]
        if not self.windows:raise ValueError('No verified paired clips. Run tools/prepare_reference.py first.')

    def __len__(self):return len(self.windows)

    def __getitem__(self,index):
        clip,start=self.windows[index]
        # Cached frames are BGR; use identical RGB normalization at inference and training.
        images=self.arrays[clip][start:start+3,:,:,::-1].astype(np.float32)/255
        images=((images-MEAN)/STD).transpose(0,3,1,2).reshape(9,288,512).copy()
        return torch.from_numpy(images),torch.from_numpy(self.labels[clip][start:start+3].copy())


def targets(labels,grid_x,grid_y):
    x=labels[:,:,0,None,None];y=labels[:,:,1,None,None]
    valid=(labels[:,:,2,None,None]>0).float()
    return torch.exp(-((grid_x-x)**2+(grid_y-y)**2)/(2*2.5**2))*valid


def evaluate(model,loader,device):
    tp=fp=fn=tn=0;errors=[];losses=[]
    yy,xx=torch.meshgrid(torch.arange(288,device=device),torch.arange(512,device=device),indexing='ij')
    model.eval()
    with torch.inference_mode():
        for images,labels in loader:
            images=images.to(device);labels=labels.to(device)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                logits=model(images)[0];target=targets(labels,xx,yy)
                loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(20.,device=device))
            losses.append(float(loss))
            pred=logits.float().cpu().numpy();truth=labels.cpu().numpy()
            for batch in range(len(pred)):
                for j in range(3):
                    x,y,_=heatmap_point(pred[batch,j],.5);visible=truth[batch,j,2]>0
                    if visible and x is not None:
                        error=float(np.hypot(x-truth[batch,j,0],y-truth[batch,j,1]));errors.append(error)
                        if error<=6:tp+=1
                        else:fp+=1;fn+=1
                    elif visible:fn+=1
                    elif x is not None:fp+=1
                    else:tn+=1
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn)
    return dict(loss=float(np.mean(losses)),precision=p,recall=r,f1=2*p*r/max(1e-9,p+r),
                tp=tp,fp=fp,fn=fn,tn=tn,mean_error_512px=float(np.mean(errors)) if errors else None)


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--train-windows',type=int,default=360);parser.add_argument('--val-windows',type=int,default=90)
    parser.add_argument('--batch',type=int,default=2);parser.add_argument('--lr',type=float,default=1e-5)
    args=parser.parse_args();seed=42
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d-%H%M%S')
    run=root/'runs'/('reference-transfer-'+stamp);run.mkdir(parents=True)
    folder=root/'datasets/reference';train=Windows(folder,'00_',args.train_windows);val=Windows(folder,'01_',args.val_windows)
    g=torch.Generator().manual_seed(seed)
    loader=DataLoader(train,batch_size=args.batch,shuffle=True,num_workers=0,generator=g)
    validation=DataLoader(val,batch_size=args.batch,num_workers=0)
    model=load_model(root,root/'models/blurball_pretrained.pt',device)
    # Adapt the final multi-resolution stage and heatmap heads; preserve earlier representations.
    for name,p in model.named_parameters():p.requires_grad=name.startswith(('stage4.','final_layers.'))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler(device,enabled=device=='cuda')
    config=dict(**vars(args),seed=seed,device=device,torch=torch.__version__,cuda=torch.version.cuda,
        model='BlurBall pretrained; fine-tune stage4 and final_layers',train_prefix='00_',val_prefix='01_',
        training_windows=len(train),validation_windows=len(val),input_size=[512,288],augmentation='none',AMP=device=='cuda',
        cudnn_deterministic=True,deterministic_algorithms_enforced=False,workers=0,
        upstream_commit='2f0f5496f7ba4b5b1a36790749935121b2ce972d',
        caveat='Upstream pretraining may contain both reference groups. This validates transfer training, not independent generalization.',
        supplied_data='Excluded: 327/358 supplied video/CSV pairs have inconsistent frame counts.')
    (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    started=time.perf_counter();baseline=evaluate(model,validation,device)
    history=[dict(epoch=0,validation=baseline)];best=baseline['f1'];best_epoch=0
    torch.save({'model_state_dict':model.state_dict(),'epoch':0},run/'best.pt')
    print('baseline',baseline,flush=True)
    yy,xx=torch.meshgrid(torch.arange(288,device=device),torch.arange(512,device=device),indexing='ij')
    for epoch in range(1,args.epochs+1):
        # Frozen batchnorm statistics stay fixed even when the trainable layers update.
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.BatchNorm2d):module.eval()
        losses=[]
        for batch,(images,labels) in enumerate(loader):
            images=images.to(device);labels=labels.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                logits=model(images)[0];target=targets(labels,xx,yy)
                loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(20.,device=device))
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise RuntimeError('Non-finite gradient')
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
            if batch==0:print('sanity',epoch,'loss',float(loss.detach()),'grad_norm',float(norm),flush=True)
        metrics=evaluate(model,validation,device)
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=metrics,lr=optimizer.param_groups[0]['lr'])
        history.append(row)
        state={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
               'scaler_state_dict':scaler.state_dict(),'epoch':epoch}
        torch.save(state,run/'last.pt')
        if metrics['f1']>best or (metrics['f1']==best and metrics['loss']<history[best_epoch]['validation']['loss']):
            best=metrics['f1'];best_epoch=epoch;torch.save(state,run/'best.pt')
        (run/'metrics.json').write_text(json.dumps(dict(baseline=baseline,history=history,best_epoch=best_epoch,
                elapsed_seconds=time.perf_counter()-started),indent=2),encoding='utf-8')
        print('epoch',epoch,row,flush=True)
    best_model=load_model(root,run/'best.pt',device);final=evaluate(best_model,validation,device)
    result=dict(run=str(run),best_epoch=best_epoch,baseline=baseline,final=final,
                elapsed_seconds=time.perf_counter()-started,
                max_allocated_vram_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                independent_test=False)
    (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (root/'reports/training_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot([r['epoch'] for r in history],[r['validation']['f1'] for r in history],marker='o')
    ax.set(xlabel='Epoch',ylabel='Reference validation F1',title='Transfer experiment (not independent test)');ax.grid(alpha=.2)
    fig.savefig(run/'curve.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
