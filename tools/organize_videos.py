"""Create duration-based copies and clearly identified ~30s test compilations."""
from pathlib import Path
import csv
import json
import shutil
import cv2

ROOT=Path(__file__).resolve().parents[1]
VIDEO=ROOT/'video'
DEST=VIDEO/'by_duration'


def inspect(path):
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise ValueError('Cannot open '+str(path))
        fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps<=0 or frames<=0:raise ValueError('Invalid duration '+str(path))
        return dict(fps=fps,frames=frames,width=int(cap.get(3)),height=int(cap.get(4)),seconds=frames/fps)
    finally:cap.release()


def group(seconds):
    if seconds<5:return '01_under_5s'
    if seconds<10:return '02_5_to_10s'
    if seconds<20:return '03_10_to_20s'
    if seconds<40:return '04_20_to_40s'
    return '05_40s_and_above'


def compile_test(camera):
    prefix='val_match1' if camera=='oblique' else 'val_match2'
    sources=[VIDEO/'labeled'/f'{prefix}_{i:03d}_{camera}.mp4' for i in (14,6,0)]
    info=inspect(sources[0]);fps=info['fps'];size=(info['width'],info['height'])
    folder=DEST/'00_30s_tests';folder.mkdir(exist_ok=True)
    target=folder/f'test_30s_{camera}_compilation.mp4'
    if target.exists():raise FileExistsError('Will not overwrite '+str(target))
    temporary=target.with_name(target.stem+'.partial.mp4')
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    if not writer.isOpened():raise ValueError('Cannot create '+str(target))
    total=0;segments=[]
    try:
        for part,source in enumerate(sources,1):
            source_info=inspect(source)
            if abs(source_info['fps']-fps)>.01 or (source_info['width'],source_info['height'])!=size:
                raise ValueError('Mismatched source format')
            cap=cv2.VideoCapture(str(source));start=total;count=0
            try:
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    # Mark cuts clearly in a small top-left badge, away from the ball/table.
                    cv2.rectangle(frame,(12,12),(440,87),(12,23,38),-1)
                    cv2.putText(frame,f'COMBINED TEST / PART {part} OF 3',(24,42),
                                cv2.FONT_HERSHEY_SIMPLEX,.65,(120,240,230),2,cv2.LINE_AA)
                    cv2.putText(frame,'Separate rallies; cuts at part changes',(24,70),
                                cv2.FONT_HERSHEY_SIMPLEX,.53,(225,225,225),1,cv2.LINE_AA)
                    writer.write(frame);total+=1;count+=1
            finally:cap.release()
            if count!=source_info['frames']:raise ValueError('Incomplete source decode')
            segments.append(dict(part=part,source=str(source.relative_to(VIDEO)),
                start_frame=start,end_frame_exclusive=total,start_seconds=start/fps,
                end_seconds=total/fps,source_frames=count))
    finally:writer.release()
    check=cv2.VideoCapture(str(temporary));decoded=0
    while True:
        ok,_=check.read()
        if not ok:break
        decoded+=1
    check.release()
    if decoded!=total:raise ValueError('Output decode mismatch')
    temporary.replace(target)
    report=dict(file=str(target.relative_to(VIDEO)),camera=camera,fps=fps,frames=total,
        decoded_frames=decoded,seconds=total/fps,segments=segments,
        warning='Compilation of three independent rallies, not a continuous recording. No ground truth is supplied. Review cuts separately when judging tracks or bounce events.',
        audio=False)
    target.with_suffix('.segments.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    if DEST.exists():raise FileExistsError('Destination already exists; no files changed: '+str(DEST))
    files=sorted(p for p in VIDEO.rglob('*') if p.is_file() and p.suffix.lower() in
                 ('.mp4','.webm','.mov','.avi','.mkv','.ogv'))
    DEST.mkdir()
    for folder in ('01_under_5s','02_5_to_10s','03_10_to_20s','04_20_to_40s','05_40s_and_above'):
        (DEST/folder).mkdir()
    entries=[]
    for source in files:
        metadata=inspect(source);bucket=group(metadata['seconds']);target=DEST/bucket/source.name
        if target.exists():raise FileExistsError('Duplicate filename '+source.name)
        shutil.copy2(source,target)
        label=source.with_suffix('.csv')
        if label.exists():shutil.copy2(label,target.with_suffix('.csv'))
        entries.append(dict(file=source.name,group=bucket,seconds=round(metadata['seconds'],4),
            fps=metadata['fps'],frames=metadata['frames'],width=metadata['width'],height=metadata['height'],
            original=str(source.relative_to(VIDEO)),copy=str(target.relative_to(VIDEO)),
            label_copied=label.exists()))
    with (DEST/'duration_index.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader()
        writer.writerows(sorted(entries,key=lambda r:r['seconds'],reverse=True))
    tests=[compile_test('oblique'),compile_test('side')]
    from collections import Counter
    report=dict(original_videos=len(files),groups=dict(Counter(r['group'] for r in entries)),
                test_videos=tests,originals_moved=False)
    (DEST/'organization.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
