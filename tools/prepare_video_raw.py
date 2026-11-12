"""Keep author-published demos separate from local original-weight reproductions."""
from pathlib import Path
from functools import reduce
from math import gcd
import hashlib,json,shutil,sys
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pingpong.detector import Detector


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_gif(source,target):
    with Image.open(source) as image:
        durations=[]
        for i in range(image.n_frames):image.seek(i);durations.append(max(10,image.info.get('duration',40)))
        quantum=reduce(gcd,durations);fps=1000/quantum
        width=image.width//2*2;height=image.height//2*2
        writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height));count=0
        if not writer.isOpened():raise IOError('Cannot create author demo conversion')
        try:
            for i,duration in enumerate(durations):
                image.seek(i);frame=cv2.cvtColor(np.array(image.convert('RGB')),cv2.COLOR_RGB2BGR)[:height,:width]
                for _ in range(duration//quantum):writer.write(frame);count+=1
        finally:writer.release()
    return dict(frames=count,fps=fps,duration_seconds=sum(durations)/1000)


def reproduce(source,target,detector):
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);width=int(cap.get(3));height=int(cap.get(4))
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    predictions=[];previous=None;index=0
    try:
        if not cap.isOpened() or not writer.isOpened():raise IOError('Cannot open baseline replay')
        while True:
            frames=[]
            for _ in range(3):
                ok,frame=cap.read()
                if ok:frames.append(frame)
            if not frames:break
            points=detector.predict(frames+[frames[-1]]*(3-len(frames)),.5)
            for frame,(x,y,confidence) in zip(frames,points):
                x=x*width/512 if x is not None else None;y=y*height/288 if y is not None else None
                predictions.append(dict(frame=index,x=x,y=y,confidence=confidence))
                if x is not None:
                    point=(round(x),round(y));cv2.circle(frame,point,9,(0,255,255),2)
                    if previous is not None:cv2.line(frame,previous,point,(220,220,30),2)
                    previous=point
                else:previous=None
                cv2.putText(frame,'Original BlurBall weights | LOCAL REPLAY | no static filter',(20,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                cv2.putText(frame,f'Frame {index} | threshold 0.50 | step 3',(20,62),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                writer.write(frame);index+=1
    finally:cap.release();writer.release()
    target.with_suffix('.predictions.json').write_text(json.dumps(predictions),encoding='utf-8')
    return dict(frames=index,fps=fps,threshold=.5,step=3,provider=detector.provider)


def main():
    folder=ROOT/'video_raw';published=folder/'authors_published';inputs=folder/'unannotated_inputs';replays=folder/'original_weights_reproduced'
    for directory in (published,inputs,replays):directory.mkdir(parents=True,exist_ok=True)
    entries=[]
    sources=[('BlurBall',ROOT/'external/blurball/demo.gif','https://github.com/cogsys-tuebingen/blurball/blob/main/demo.gif'),
             ('TTNet',ROOT/'TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch-master/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch-master/docs/demo.gif',
              'https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/blob/master/docs/demo.gif')]
    for name,source,url in sources:
        target=published/(name+'_author_demo.gif');shutil.copy2(source,target)
        video=target.with_suffix('.mp4');metadata=convert_gif(target,video)
        entries.append(dict(category='author_published_demo',file=str(video.relative_to(folder)),original=str(target.relative_to(folder)),source_url=url,
            original_sha256=digest(target),conversion='GIF to MP4; no model rerun or added annotation',**metadata))
        print('author demo',name,metadata,flush=True)
    detector=Detector(ROOT/'models/detector.onnx',acceleration='cpu')
    for name,source in [('00_001',ROOT/'datasets/reference/00_001.mp4'),('01_001',ROOT/'datasets/reference/01_001.mp4'),('V009',ROOT/'video/labeled/V009.mp4')]:
        target=inputs/(name+'.mp4');shutil.copy2(source,target)
        assert digest(target)==digest(source)
        entries.append(dict(category='unannotated_input',file=str(target.relative_to(folder)),source=str(source),sha256=digest(source),
            source_url='https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7' if name!='V009' else None,
            note='Unmodified MP4 copy; no label overlay. V009 is the user-provided local case, not claimed as an author-distributed example.'))
        replay=replays/(name+'_original_weights.mp4');metadata=reproduce(target,replay,detector)
        entries.append(dict(category='local_original_weight_reproduction',file=str(replay.relative_to(folder)),input=str(target.relative_to(folder)),
            checkpoint_sha256=digest(ROOT/'models/blurball_pretrained.pt'),onnx_sha256=digest(ROOT/'models/detector.onnx'),
            pipeline='Local RGB/heatmap wrapper; no fine-tuning, no recovery, no static filter, no bounce algorithm; not an author-published output',**metadata))
        print('reproduced',name,metadata,flush=True)
    (folder/'manifest.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')


if __name__=='__main__':main()
