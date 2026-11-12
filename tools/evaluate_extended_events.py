"""Diagnostic bounce scoring on short annotated windows, not full-match accuracy."""
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from train_extended import ExtendedWindows
from pingpong.model import load_model
from pingpong.detector import heatmap_point
from pingpong.analysis import TrackPoint
from pingpong.events import analyze_events,recover_tracks


def score(weights,threshold,manifest):
    dataset=ExtendedWindows(ROOT/'datasets/extended',manifest,'test');model=load_model(ROOT,weights,'cuda').eval()
    bursts={}
    with torch.inference_mode():
        for i,record in enumerate(dataset.records):
            images,_=dataset[i]
            with torch.autocast('cuda',dtype=torch.float16):logits=model(images[None].cuda())[0][0].float().cpu().numpy()
            key=(record['source'],record['frames'][0]//1)
            # Consecutive disjoint triplets are merged, stopping at sampling gaps.
            group=bursts.setdefault(record['source'],[])
            for frame,hm in zip(record['frames'],logits):
                x,y,confidence=heatmap_point(hm,threshold)
                cx,cy,_=heatmap_point(hm,min(.25,threshold*.5))
                group.append((frame,x,y,confidence,cx,cy))
    tp=fp=fn=0;details=[]
    source_labels=ROOT/'video/table_tennis_data-main/table_tennis_data-main/data/raw/game_data/test'
    for source,raw in bursts.items():
        labels=json.loads((source_labels/(source+'.json')).read_text());segments=[]
        for p in sorted(raw):
            if not segments or p[0]>segments[-1][-1][0]+1:segments.append([])
            segments[-1].append(p)
        for segment in segments:
            first,last=segment[0][0],segment[-1][0]
            points=[TrackPoint(i,i/120,x,y,c,source='model',candidate_x=cx,candidate_y=cy) for i,(_,x,y,c,cx,cy) in enumerate(segment)]
            recover_tracks(points,120,(512,288));events,_=analyze_events(points,120,(),(512,288))
            pred=[first+e['frame'] for e in events if e['kind']=='bounce' and 3<=e['frame']<len(points)-3]
            truth=[int(k) for k,v in labels.items() if v=='bounce' and first+3<=int(k)<=last-3]
            unmatched=set(truth);matched=0
            for frame in pred:
                close=[f for f in unmatched if abs(f-frame)<=2]
                if close:unmatched.remove(min(close,key=lambda f:abs(f-frame)));matched+=1
            tp+=matched;fp+=len(pred)-matched;fn+=len(unmatched)
            details.append(dict(source=source,start=first,end=last,predicted=pred,truth=truth))
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn)
    return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(1e-9,p+r),windows=details)


def main():
    torch.set_num_threads(4)
    manifest=json.loads((ROOT/'datasets/extended/manifest.json').read_text())
    result=json.loads((ROOT/'reports/extended/training_result.json').read_text())
    report=dict(baseline=score(ROOT/'models/blurball_pretrained.pt',.5,manifest),
        trained=score(Path(result['run'])/'best.pt',result['selected_threshold'],manifest),
        tolerance_frames=2,caveat='Short event-centred sparse-label windows, with 3-frame boundary exclusion; not full-match detection precision or recall.')
    (ROOT/'reports/extended/bounce_diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({k:{n:v for n,v in r.items() if n!='windows'} for k,r in report.items() if isinstance(r,dict)})


if __name__=='__main__':main()
