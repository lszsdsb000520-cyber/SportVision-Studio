"""Download small, paired upstream samples to validate the training pipeline."""
from pathlib import Path
import csv
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parent))
from inspect_reference import request, ROOT


def main():
    work=ROOT/'work/reference'
    dest=ROOT/'datasets/reference';dest.mkdir(parents=True,exist_ok=True)
    tasks=[]
    with zipfile.ZipFile(work/'all_csv_midpoint_annotations.zip') as z:
        for group in ('00','01'):
            listing=json.loads((work/(group+'_listing.json')).read_text(encoding='utf-8'))
            videos=sorted(Path(url).name for url,size in listing if url.endswith('.mp4'))[:12]
            for video in videos:
                name=f'{group}_{Path(video).stem}'
                labelname=f'all_csv_annotations/{group}_csv_{Path(video).stem}.csv'
                if labelname not in z.namelist():continue
                data=z.read(labelname).decode().replace('\r\r\n','\n')
                (dest/(name+'.csv')).write_text(data,encoding='utf-8')
                tasks.append((f'{group}/rallies_videos/{video}',dest/(name+'.mp4')))
    def fetch(item):
        source,target=item
        if not target.exists():
            data=request(source);target.write_bytes(data)
        print('reference',target.name,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(fetch,tasks))
    report=[]
    for _,video in tasks:
        with video.with_suffix('.csv').open(encoding='utf-8') as f:labels=list(csv.DictReader(f))
        cap=cv2.VideoCapture(str(video));meta=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4))
        frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(cv2.resize(frame,(512,288)))
        cap.release()
        record=dict(file=video.name,decoded=len(frames),metadata=meta,labels=len(labels),width=w,height=h)
        report.append(record)
        if len(frames)==len(labels):
            import numpy as np
            np.save(video.with_suffix('.npy'),np.array(frames,dtype=np.uint8))
            xy=np.array([[float(r['X'])*512/w,float(r['Y'])*288/h,int(r['Visibility'])] for r in labels],np.float32)
            np.save(video.with_suffix('.labels.npy'),xy)
    (dest/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
