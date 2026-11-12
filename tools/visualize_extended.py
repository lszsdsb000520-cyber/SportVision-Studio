"""Show fixed test cases and both improvements and regressions without changing selection."""
from pathlib import Path
import json,sys
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from train_extended import ExtendedWindows
from pingpong.model import load_model
from pingpong.detector import heatmap_point


def main():
    torch.set_num_threads(4);manifest=json.loads((ROOT/'datasets/extended/manifest.json').read_text())
    result=json.loads((ROOT/'reports/extended/training_result.json').read_text())
    dataset=ExtendedWindows(ROOT/'datasets/extended',manifest,'test')
    old=load_model(ROOT,ROOT/'models/blurball_pretrained.pt','cuda').eval()
    new=load_model(ROOT,Path(result['run'])/'best.pt','cuda').eval()
    cases=[]
    with torch.inference_mode():
        for i in range(0,len(dataset),2):
            images,labels=dataset[i]
            with torch.autocast('cuda',dtype=torch.float16):
                a=old(images[None].cuda())[0][0,1].float().cpu().numpy()
                b=new(images[None].cuda())[0][0,1].float().cpu().numpy()
            p=heatmap_point(a,.5);q=heatmap_point(b,result['selected_threshold']);gt=labels[1].numpy()
            def correct(point):return point[0] is not None and gt[2]>0 and np.hypot(point[0]-gt[0],point[1]-gt[1])<=6
            cases.append((i,p,q,gt,correct(p),correct(q)))
    fixed=cases[::max(1,len(cases)//3)][:3]
    improved=[r for r in cases if not r[4] and r[5]][:3]
    regressed=[r for r in cases if r[4] and not r[5]][:3]
    chosen=fixed+improved+regressed;panels=[]
    for i,p,q,gt,ok_old,ok_new in chosen:
        record=dataset.records[i];image=cv2.imread(str(ROOT/'datasets/extended'/record['images'][1]))
        if gt[2]:cv2.circle(image,(round(gt[0]),round(gt[1])),7,(255,255,255),1)
        if p[0] is not None:cv2.drawMarker(image,(round(p[0]),round(p[1])),(0,160,255),cv2.MARKER_CROSS,12,1)
        if q[0] is not None:cv2.drawMarker(image,(round(q[0]),round(q[1])),(80,255,80),cv2.MARKER_DIAMOND,12,1)
        text=f"{record['source']} f{record['frames'][1]} old={int(ok_old)} new={int(ok_new)}"
        cv2.putText(image,text,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
        panels.append(image)
    while len(panels)%2:panels.append(np.zeros_like(panels[0]))
    collage=cv2.vconcat([cv2.hconcat(panels[i:i+2]) for i in range(0,len(panels),2)])
    cv2.imwrite(str(ROOT/'reports/extended/comparison.jpg'),collage)
    print('white=GT; orange=old; green=new; fixed cases plus changes in either direction')


if __name__=='__main__':main()
