"""Sample source-disjoint, explicitly annotated triplets via HTTP video seeking."""
from pathlib import Path
import argparse,json,time,hashlib
from concurrent.futures import ThreadPoolExecutor
import av,cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'video/table_tennis_data-main/table_tennis_data-main'
DEST=ROOT/'datasets/extended'


def select_runs(labels,limit):
    frames=sorted(map(int,labels));runs=[];run=[]
    for frame in frames:
        if run and frame!=run[-1]+1:
            if len(run)>=12:runs.append(run)
            run=[]
        run.append(frame)
    if len(run)>=12:runs.append(run)
    # Fixed uniform coverage, decided without looking at model predictions.
    chosen=np.linspace(0,len(runs)-1,min(limit,len(runs)),dtype=int)
    return [runs[i][:12] for i in chosen]


def prepare(item):
    name,split,count=item;upstream='train' if name.startswith('game') else 'test'
    label_name=name.replace('game_','train_')
    label_path=SOURCE/'data/raw/ball_data'/upstream/(label_name+'.json')
    labels=json.loads(label_path.read_text());bursts=select_runs(labels,count)
    image_folder=DEST/'images'/split;label_folder=DEST/'labels'/split
    image_folder.mkdir(parents=True,exist_ok=True);label_folder.mkdir(parents=True,exist_ok=True)
    url='https://lab.osai.ai/datasets/openttgames/data/'+name+'.mp4'
    container=None;windows=[];decoded=0;started=time.time()
    try:
        for n,frames in enumerate(bursts):
            files=[image_folder/f'{name}_{i:07d}.png' for i in frames]
            if not all(p.exists() for p in files):
                for attempt in range(3):
                    try:
                        if container is None:
                            container=av.open(url,timeout=(25,25));stream=container.streams.video[0]
                            stream.thread_type='AUTO';stream.codec_context.thread_count=2
                            assert float(stream.average_rate)==120 and stream.start_time==0
                        container.seek(int(frames[0]/120/stream.time_base),stream=stream)
                        found={}
                        for frame in container.decode(stream):
                            index=round(float(frame.pts*stream.time_base)*120)
                            if index>frames[-1]:break
                            if index in frames:
                                assert (frame.width,frame.height)==(1920,1080)
                                found[index]=cv2.resize(frame.to_ndarray(format='bgr24'),(512,288))
                        if len(found)!=len(frames):raise RuntimeError(f'Frame alignment failed: {name} {frames[0]}')
                        for index,image in found.items():
                            if not cv2.imwrite(str(image_folder/f'{name}_{index:07d}.png'),image):raise IOError('Image cache failed')
                        decoded+=len(found);break
                    except Exception:
                        if container is not None:container.close();container=None
                        if attempt==2:raise
                print('cached',name,n+1,'/',len(bursts),'elapsed',round(time.time()-started),flush=True)
            for index in frames:
                point=labels[str(index)];visible=point['x']>=0 and point['y']>=0
                record=dict(frame=index,x=point['x']*512/1920 if visible else 0.,
                            y=point['y']*288/1080 if visible else 0.,visible=int(visible),
                            source_file=str(label_path.relative_to(ROOT)))
                (label_folder/f'{name}_{index:07d}.json').write_text(json.dumps(record),encoding='utf-8')
            for offset in range(0,10,3):
                indices=frames[offset:offset+3]
                windows.append(dict(source=name,split=split,frames=indices,
                    images=[str((image_folder/f'{name}_{i:07d}.png').relative_to(DEST)) for i in indices],
                    labels=[str((label_folder/f'{name}_{i:07d}.json').relative_to(DEST)) for i in indices]))
    finally:
        if container is not None:container.close()
    return dict(source=name,split=split,windows=windows,bursts=len(bursts),decoded_frames=decoded,
        elapsed_seconds=time.time()-started,label_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),url=url)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--train-bursts',type=int,default=40)
    parser.add_argument('--val-bursts',type=int,default=30);parser.add_argument('--test-bursts',type=int,default=12)
    args=parser.parse_args();DEST.mkdir(parents=True,exist_ok=True)
    items=[(f'game_{i}','train',args.train_bursts) for i in (1,2,3,5)]
    items+=[('game_4','val',args.val_bursts)]
    items += [(f'test_{i}','test',args.test_bursts) for i in (3,5,7)]
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(prepare,items))
    windows=[w for r in records for w in r.pop('windows')]
    manifest=dict(version=1,seed=42,sampling='uniform uninterrupted explicit-label runs, 12 frames per burst, disjoint triplets',
        dimensions=[512,288],fps=120,sources=records,windows=windows,
        split_rule='train=game1,2,3,5; val=game4; test=official test3,5,7, held out from fitting and selection',
        caveats=['Only explicit ball labels are supervised; missing keys are unknown, never negative.',
                 'Original ball labels are semi-automatic; published BlurBall pretraining membership is not auditable.',
                 'Fixed subset, not the full dataset. Source-disjoint local evaluation.'],
        license='CC BY-NC-SA 4.0',source_annotations=str(SOURCE))
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('COMPLETE', {s:sum(w['split']==s for w in windows) for s in ('train','val','test')},flush=True)


if __name__=='__main__':main()
