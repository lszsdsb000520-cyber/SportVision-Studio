"""Rename video copies and sidecars with an explicit, reversible path map."""
from pathlib import Path
import csv
import json
import shutil
import hashlib
import cv2

ROOT=Path(__file__).resolve().parents[1]
VIDEO=ROOT/'video'


def main():
    if (VIDEO/'name_map.json').exists():raise FileExistsError('A rename map already exists; no files changed.')
    source=sorted((VIDEO/'labeled').glob('*.mp4'))
    assignments={p.name:f'V{i:03d}' for i,p in enumerate(source,1)}
    assignments.update({'test_30s_oblique_compilation.mp4':'T01','test_30s_side_compilation.mp4':'T02'})
    catalog=[];plan=[]
    for video in sorted(VIDEO.rglob('*.mp4')):
        ident=assignments[video.name];new=video.with_name(ident+video.suffix.lower())
        if new.exists():raise FileExistsError(str(new))
        plan.append(dict(old=str(video),new=str(new),kind='video'))
        for suffix in ('.csv','.segments.json'):
            sidecar=video.with_suffix(suffix)
            if sidecar.exists():plan.append(dict(old=str(sidecar),new=str(new.with_suffix(suffix)),kind='sidecar'))
        analysis=video.parent/'v2_analysis'/(video.stem+'.analysis.json')
        if analysis.exists():plan.append(dict(old=str(analysis),new=str(analysis.with_name(ident+'.analysis.json')),kind='project'))
        if video.parent.name=='labeled' or ident.startswith('T'):
            cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
            catalog.append(dict(id=ident,file=str(new.relative_to(VIDEO)),original_name=video.name,
                camera='oblique' if 'oblique' in video.name else 'side',
                split=video.stem.split('_')[0],frames=frames,fps=fps,seconds=frames/fps,
                sha256=hashlib.sha256(video.read_bytes()).hexdigest()))
    backup=ROOT/'work/rename-backup';backup.mkdir(parents=True,exist_ok=False)
    for f in VIDEO.rglob('*'):
        if f.is_file() and f.suffix.lower() in ('.json','.csv','.md'):
            target=backup/f.relative_to(VIDEO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    mapping=dict(format_version=1,status='in_progress',renames=plan)
    (VIDEO/'name_map.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    for item in plan:
        old,new=Path(item['old']),Path(item['new'])
        if not old.resolve().is_relative_to(VIDEO.resolve()) or not new.resolve().is_relative_to(VIDEO.resolve()):raise ValueError('Unsafe path')
        try:old.rename(new)
        except PermissionError:
            shutil.copy2(old,new);item['locked_alias']=True
    absolute={p['old']:p['new'] for p in plan}
    relative={str(Path(p['old']).relative_to(VIDEO)):str(Path(p['new']).relative_to(VIDEO)) for p in plan}
    basenames={old:ident+'.mp4' for old,ident in assignments.items()}
    def rewrite(value):
        if isinstance(value,dict):return {k:rewrite(v) for k,v in value.items()}
        if isinstance(value,list):return [rewrite(v) for v in value]
        if isinstance(value,str):
            if value in absolute:return absolute[value]
            if value in relative:return relative[value]
            if value in basenames:return basenames[value]
        return value
    for file in VIDEO.rglob('*.json'):
        if file.name=='name_map.json':continue
        data=json.loads(file.read_text(encoding='utf-8'));file.write_text(json.dumps(rewrite(data),indent=2),encoding='utf-8')
    index=VIDEO/'by_duration/duration_index.csv'
    with index.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        row['original_name']=row['file']
        for key in ('file','original','copy'):row[key]=rewrite(row[key])
    with index.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+['original_name']);writer.writeheader();writer.writerows(rows)
    mapping['status']='complete';mapping['locked_aliases']=sum(p.get('locked_alias',False) for p in plan)
    (VIDEO/'name_map.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    (VIDEO/'catalog.json').write_text(json.dumps(dict(videos=catalog),indent=2),encoding='utf-8')
    print(json.dumps(dict(video_paths=sum(p['kind']=='video' for p in plan),unique_ids=len(catalog),
        sidecars=sum(p['kind']=='sidecar' for p in plan),locked_aliases=mapping['locked_aliases'],
        tests=[p for p in catalog if p['id'].startswith('T')]),indent=2),flush=True)


if __name__=='__main__':main()
