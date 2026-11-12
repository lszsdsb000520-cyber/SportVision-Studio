"""Source-grouped transfer learning; the official test subset is evaluated only at the end."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,copy,hashlib,json,random,sys,time
import cv2,numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from pingpong.model import load_model
from pingpong.detector import preprocess,heatmap_point
from train import targets


class ExtendedWindows(Dataset):
    def __init__(self,folder,manifest,split,augment=False):
        self.folder=folder;self.records=[r for r in manifest['windows'] if r['split']==split]
        self.augment=augment;self.images={};self.labels={}
        for record in self.records:
            for name in record['images']:
                if name not in self.images:
                    image=cv2.imread(str(folder/name))
                    if image is None:raise ValueError('Bad image '+name)
                    self.images[name]=image
            for name in record['labels']:
                if name not in self.labels:self.labels[name]=json.loads((folder/name).read_text())

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        record=self.records[index];frames=[self.images[n] for n in record['images']]
        if self.augment:
            alpha=random.uniform(.9,1.1);beta=random.uniform(-8,8)
            frames=[np.clip(f.astype(np.float32)*alpha+beta,0,255).astype(np.uint8) for f in frames]
        labels=np.array([[self.labels[n]['x'],self.labels[n]['y'],self.labels[n]['visible']] for n in record['labels']],np.float32)
        return torch.from_numpy(preprocess(frames)[0]),torch.from_numpy(labels)


def evaluate(model,loader,thresholds=(.5,)):
    counts={t:dict(tp=0,fp=0,fn=0,tn=0,errors=[]) for t in thresholds};losses=[]
    yy,xx=torch.meshgrid(torch.arange(288,device='cuda'),torch.arange(512,device='cuda'),indexing='ij')
    model.eval()
    with torch.inference_mode():
        for images,labels in loader:
            images=images.cuda();labels=labels.cuda()
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(images)[0]
                loss=F.binary_cross_entropy_with_logits(logits,targets(labels,xx,yy),pos_weight=torch.tensor(20.,device='cuda'))
            losses.append(float(loss));pred=logits.float().cpu().numpy();truth=labels.cpu().numpy()
            for b in range(len(pred)):
                for j in range(3):
                    for threshold in thresholds:
                        r=counts[threshold];x,y,_=heatmap_point(pred[b,j],threshold)
                        visible=truth[b,j,2]>0
                        if visible and x is not None:
                            error=float(np.hypot(x-truth[b,j,0],y-truth[b,j,1]));r['errors'].append(error)
                            if error<=6:r['tp']+=1
                            else:r['fp']+=1;r['fn']+=1
                        elif visible:r['fn']+=1
                        elif x is not None:r['fp']+=1
                        else:r['tn']+=1
    results={}
    for t,r in counts.items():
        errors=r.pop('errors');p=r['tp']/max(1,r['tp']+r['fp']);recall=r['tp']/max(1,r['tp']+r['fn'])
        results[str(t)]=dict(**r,precision=p,recall=recall,f1=2*p*recall/max(1e-9,p+recall),
            mean_error_512px=float(np.mean(errors)) if errors else None,loss=float(np.mean(losses)))
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--batch',type=int,default=4);parser.add_argument('--lr',type=float,default=1e-5)
    args=parser.parse_args();seed=42
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available();assert torch.version.cuda>='12.8'
    manifest_path=ROOT/'datasets/extended/manifest.json';manifest=json.loads(manifest_path.read_text())
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d-%H%M%S')
    run=ROOT/'runs'/('extended-transfer-'+stamp);run.mkdir(parents=True)
    config=dict(**vars(args),seed=seed,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),AMP=True,
        input_size=[512,288],normalization='RGB ImageNet, three consecutive frames',
        data_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),split_rule=manifest['split_rule'],
        optimizer='AdamW, weight_decay=1e-4',model='BlurBall; train stage4 and final_layers only',
        augmentation='consistent brightness/contrast per 3-frame window',workers=0,
        cudnn_deterministic=True,deterministic_algorithms_enforced=False,
        selection='validation F1 at fixed 0.5; loss tie-break; test never used for selection',
        matching_radius_512px=6,license=manifest['license'],
        pretrained_sha256=hashlib.sha256((ROOT/'models/blurball_pretrained.pt').read_bytes()).hexdigest())
    config['training_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('RUN',run,flush=True)
    generator=torch.Generator().manual_seed(seed)
    train=ExtendedWindows(manifest_path.parent,manifest,'train',True)
    val=ExtendedWindows(manifest_path.parent,manifest,'val')
    loader=DataLoader(train,batch_size=args.batch,shuffle=True,num_workers=0,generator=generator)
    validation=DataLoader(val,batch_size=args.batch,num_workers=0)
    model=load_model(ROOT,ROOT/'models/blurball_pretrained.pt','cuda')
    for name,p in model.named_parameters():p.requires_grad=name.startswith(('stage4.','final_layers.'))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda');yy,xx=torch.meshgrid(torch.arange(288,device='cuda'),torch.arange(512,device='cuda'),indexing='ij')
    started=time.time();baseline=evaluate(model,validation,(.3,.4,.5,.6))
    history=[dict(epoch=0,validation=baseline['0.5'])];best=baseline['0.5']['f1'];best_loss=baseline['0.5']['loss'];best_epoch=0
    torch.save(dict(model_state_dict=model.state_dict(),epoch=0),run/'best.pt')
    print('BASELINE',baseline,flush=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.BatchNorm2d):module.eval()
        losses=[]
        for batch,(images,labels) in enumerate(loader):
            images=images.cuda();labels=labels.cuda();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(images)[0];loss=F.binary_cross_entropy_with_logits(logits,targets(labels,xx,yy),pos_weight=torch.tensor(20.,device='cuda'))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
            if batch==0:print('SANITY',epoch,float(loss),float(norm),flush=True)
        metrics=evaluate(model,validation)['0.5'];row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=metrics,lr=args.lr)
        history.append(row)
        state=dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),scaler_state_dict=scaler.state_dict(),epoch=epoch)
        torch.save(state,run/'last.pt')
        if (metrics['f1'],-metrics['loss'])>(best,-best_loss):
            best=metrics['f1'];best_loss=metrics['loss'];best_epoch=epoch;torch.save(state,run/'best.pt')
        (run/'metrics.json').write_text(json.dumps(dict(baseline=baseline,history=history,best_epoch=best_epoch),indent=2),encoding='utf-8')
        print('EPOCH',row,flush=True)
    selected=load_model(ROOT,run/'best.pt','cuda');final_val=evaluate(selected,validation,(.3,.4,.5,.6))
    # Lock the selected model and threshold before opening official test samples.
    threshold=float(max(final_val,key=lambda t:(final_val[t]['f1'],-abs(float(t)-.5))))
    selection=dict(best_epoch=best_epoch,threshold=threshold,validation=final_val)
    (run/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    test=ExtendedWindows(manifest_path.parent,manifest,'test');testing=DataLoader(test,batch_size=args.batch,num_workers=0)
    final_test=evaluate(selected,testing,(threshold,))[str(threshold)]
    base_model=load_model(ROOT,ROOT/'models/blurball_pretrained.pt','cuda')
    base_threshold=float(max(baseline,key=lambda t:(baseline[t]['f1'],-abs(float(t)-.5))))
    base_test=evaluate(base_model,testing,tuple(sorted({.5,base_threshold})))
    result=dict(run=str(run),best_epoch=best_epoch,selected_threshold=threshold,baseline_validation=baseline,
        selected_validation=final_val,baseline_test=base_test,baseline_calibrated_threshold=base_threshold,
        selected_test=final_test,train_windows=len(train),val_windows=len(val),test_windows=len(test),
        elapsed_seconds=time.time()-started,peak_vram_bytes=torch.cuda.max_memory_allocated(),
        caveats=manifest['caveats'],metric='ball localization F1 at 6px on 512x288; not bounce/end-event accuracy')
    (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'reports/extended/training_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].plot([h['epoch'] for h in history],[h['validation']['f1'] for h in history],marker='o');axes[0].set(xlabel='Epoch',ylabel='Validation F1')
    axes[1].plot([h['epoch'] for h in history[1:]],[h['train_loss'] for h in history[1:]],marker='o');axes[1].set(xlabel='Epoch',ylabel='Training loss')
    fig.tight_layout();fig.savefig(run/'curve.png',dpi=160);plt.close(fig)
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':main()
