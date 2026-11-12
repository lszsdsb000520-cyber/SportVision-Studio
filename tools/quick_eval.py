from pathlib import Path
import argparse
import csv
import json
import sys
import time
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pingpong.model import load_model
from pingpong.detector import preprocess, heatmap_point


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,default=root/'models/blurball_pretrained.pt')
    p.add_argument('--clips',type=int,default=8)
    p.add_argument('--windows',type=int,default=6)
    p.add_argument('--output',type=Path,default=root/'reports/baseline.json')
    args=p.parse_args()
    audit=json.loads((root/'reports/data_audit/audit.json').read_text(encoding='utf-8'))
    if audit['errors']:
        raise RuntimeError('Supplied video/CSV alignment has not passed audit. Do not report these diagnostic matches as accuracy.')
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_model(root,args.weights,device).eval()
    files=sorted((root/'video/labeled').glob('val_*.mp4'))
    files=[files[i] for i in np.linspace(0,len(files)-1,args.clips,dtype=int)]
    errors=[];tp=fp=fn=tn=0; records=[]; total_time=0
    for path in files:
        with path.with_suffix('.csv').open() as f: labels=list(csv.DictReader(f))
        cap=cv2.VideoCapture(str(path))
        starts=np.linspace(0,max(0,len(labels)-3),args.windows,dtype=int)
        for start in starts:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(start))
            frames=[]
            for _ in range(3):
                ok,frame=cap.read()
                if ok:frames.append(frame)
            if len(frames)!=3:continue
            x=torch.from_numpy(preprocess(frames)).to(device)
            if device=='cuda':torch.cuda.synchronize()
            t=time.perf_counter()
            with torch.inference_mode():pred=model(x)[0][0].float().cpu().numpy()
            total_time+=time.perf_counter()-t
            for j,hm in enumerate(pred):
                px,py,conf=heatmap_point(hm,0.5)
                row=labels[start+j];visible=row['Visibility']=='1'
                error=None
                if visible and px is not None:
                    error=float(np.hypot(px-float(row['X']),py-float(row['Y'])))
                    errors.append(error)
                    if error<=6:tp+=1
                    else:fp+=1;fn+=1
                elif visible:fn+=1
                elif px is not None:fp+=1
                else:tn+=1
                records.append(dict(file=path.name,frame=int(start+j),gt_visible=visible,
                                    pred_x=px,pred_y=py,confidence=conf,error_512px=error))
            if start==starts[len(starts)//2]:
                show=frames[1].copy(); row=labels[start+1]
                if row['Visibility']=='1':
                    gt=(round(float(row['X'])*show.shape[1]/512),round(float(row['Y'])*show.shape[0]/288))
                    cv2.circle(show,gt,10,(0,255,0),2)
                px,py,_=heatmap_point(pred[1],0.5)
                if px is not None:cv2.drawMarker(show,(round(px*show.shape[1]/512),round(py*show.shape[0]/288)),(0,0,255),cv2.MARKER_CROSS,22,2)
                out=root/'reports/model_checks';out.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(out/(path.stem+'.jpg')),cv2.resize(show,(960,540)))
        cap.release()
        print('evaluated',path.name,flush=True)
    precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
    result=dict(weights=str(args.weights),device=device,evaluated_frames=len(records),clips=len(files),
                threshold=0.5,matching_radius_512px=6,tp=tp,fp=fp,fn=fn,tn=tn,
                precision=precision,recall=recall,f1=2*precision*recall/max(1e-9,precision+recall),
                mean_localization_error_512px=float(np.mean(errors)) if errors else None,
                inference_fps=len(records)/max(total_time,1e-9),records=records,
                caveat='Sampled validation frames only; pretrained training overlap is unknown; not a contest accuracy claim.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
