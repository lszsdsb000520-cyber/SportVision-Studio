"""Human-reviewed, revisioned data corpus. Model output is not automatically ground truth."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import copy
import csv
import hashlib
import json
import shutil
import uuid
import cv2
from .analysis import validate_project
from .events import event_kind
from .media import workspace_root,resolve_video,media_details
from .review_state import ensure_numbers
from .cancellation import checkpoint
from .video_io import VideoReader


def timestamp():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='microseconds')


def digest(path,cancelled=None):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            checkpoint(cancelled);h.update(chunk)
    return h.hexdigest()


def approval_root():return workspace_root()/'video_success'


def invalidate_review(project):
    if project.get('review',{}).get('status')=='approved':
        project['review']['status']='modified_after_approval'


def log_edit(project,action,before=None,after=None):
    invalidate_review(project)
    project.setdefault('edit_history',[]).append(dict(at=timestamp(),action=action,before=copy.deepcopy(before),after=copy.deepcopy(after)))


def save_approved(project,folder=None,events_reviewed=False,ball_positions_reviewed=False,cancelled=None):
    if not events_reviewed:raise ValueError('请先人工复核整段视频的事件，并勾选审核声明。')
    tracks=validate_project(project);source=resolve_video(project['video'])
    if not source.exists():raise FileNotFoundError('原视频不存在，不能审核入库。')
    if len(project.get('corners',[]))!=4:raise ValueError('请先核对球台边界，再审核入库。')
    cap=VideoReader(source,cancelled)
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    info=project.get('decode_info',{})
    compatible_count=count==len(tracks) or (info.get('reported_frames')==count and info.get('decoded_frames')==len(tracks))
    if not compatible_count or (w,h)!=(project['width'],project['height']) or abs(fps-project['fps'])>.05:
        raise ValueError('项目与视频帧数、尺寸或帧率不匹配，不能审核入库。')
    keys=set()
    for e in project['events']:
        if event_kind(e) not in ('bounce','out','net','stroke'):raise ValueError('事件类型无效。')
        if not (0<=e['x']<w and 0<=e['y']<h):raise ValueError('事件坐标超出视频范围。')
        key=(e['frame'],event_kind(e))
        if key in keys:raise ValueError('同一帧存在重复的同类型事件，请先删除重复项。')
        keys.add(key)
    folder=Path(folder or approval_root());folder.mkdir(parents=True,exist_ok=True)
    source_hash=digest(source,cancelled);origin=project.get('data_origin',{})
    if project.get('video_sha256') and source_hash!=project['video_sha256']:
        raise ValueError('原视频内容已变化，请重新分析并审核。')
    details=media_details(source,sha256=source_hash)
    if origin.get('video_sha256')==source_hash:ident=origin['media_id']
    else:origin={};ident=details['id']
    if not ident.replace('_','').replace('-','').isalnum():ident='clip_'+digest(source)[:10]
    revision='review_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    final=folder/ident/revision;final.parent.mkdir(exist_ok=True)
    stage=final.parent/('.pending_'+uuid.uuid4().hex);stage.mkdir()
    result=copy.deepcopy(project)
    ensure_numbers(result)
    imported_name=Path(project.get('imported_video_name',source.name)).name
    video_name=Path(imported_name).stem+'_success'+source.suffix
    result['imported_video_name']=imported_name
    try:
        with source.open('rb') as src,(stage/video_name).open('wb') as dst:
            while chunk:=src.read(1024*1024):
                if cancelled and cancelled():raise InterruptedError('已取消审核入库。')
                dst.write(chunk)
        video_hash=digest(stage/video_name,cancelled)
        if video_hash!=source_hash:raise ValueError('复制期间原视频发生变化，请重新审核。')
        for e in result['events']:
            e.update(kind=event_kind(e),confirmed=True,type=event_kind(e),human_reviewed=True)
        result['source_video']=str(source);result['video']=video_name
        group=origin.get('group_id','sha256:'+video_hash)
        original=details.get('original_name','') if details.get('sha256')==video_hash else ''
        if not origin and original.startswith(('train_match','val_match')):
            import re
            match=re.search(r'(train|val)_match(\d+)',original)
            group=f'tt3d_{match[1]}_session{(int(match[2])+1)//2}'
        elif not origin and original.startswith('test_30s_'):group='tt3d_val_session1'
        result['data_origin']=dict(media_id=ident,video_sha256=video_hash,group_id=group,original_name=origin.get('original_name',original))
        result['review']=dict(status='approved',at=timestamp(),events_complete=True,
            ball_positions_complete=bool(ball_positions_reviewed),video_sha256=video_hash,
            statement='Human explicitly reviewed all events; ball scope is separate.')
        (stage/'project.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        with (stage/'events.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['EventNumber','Frame','Time','Kind','X','Y','TableX','TableY','Source','HumanReviewed'])
            writer.writeheader()
            for e in result['events']:
                xy=e.get('table_xy') or ['','']
                writer.writerow(dict(EventNumber=e['event_number'],Frame=e['frame'],Time=e['time'],Kind=event_kind(e),X=e['x'],Y=e['y'],
                    TableX=xy[0],TableY=xy[1],Source=e['source'],HumanReviewed=True))
        reviewed_frames=0
        with (stage/'ball_labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['Frame','Visibility','X','Y','Status','Source'])
            writer.writeheader()
            for p in tracks:
                reviewed=ball_positions_reviewed or p.source=='manual'
                reviewed_frames+=int(reviewed)
                writer.writerow(dict(Frame=p.frame,Visibility=int(p.x is not None) if reviewed else '',
                    X=p.x if reviewed and p.x is not None else '',Y=p.y if reviewed and p.y is not None else '',
                    Status='reviewed' if reviewed else 'unreviewed',Source=p.source))
        manifest=dict(format_version=1,status='approved',media_id=ident,revision=revision,
            decode_info=project.get('decode_info'),
            reviewed_at=result['review']['at'],
            video_sha256=video_hash,project_sha256=digest(stage/'project.json'),frames=count,width=w,height=h,fps=fps,
            coordinate_space='original_video_pixels',events=len(result['events']),reviewed_ball_frames=reviewed_frames,
            events_reviewed=True,all_ball_positions_reviewed=bool(ball_positions_reviewed),group_id=group,
            video_file=video_name,files=[video_name,'project.json','events.csv','ball_labels.csv'])
        (stage/'approved.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        if cancelled and cancelled():raise InterruptedError('已取消审核入库。')
        stage.rename(final)
    except Exception:
        if stage.resolve().is_relative_to(folder.resolve()):shutil.rmtree(stage)
        raise
    return dict(directory=str(final),manifest=manifest)


def prepare_training_index(folder=None,destination=None,cancelled=None):
    checkpoint(cancelled)
    folder=Path(folder or approval_root());destination=Path(destination or workspace_root()/'datasets/reviewed')
    destination.mkdir(parents=True,exist_ok=True)
    records=[];seen=set()
    # Latest approved revision per video fingerprint, without duplicating older labels.
    manifests=[(f,json.loads(f.read_text(encoding='utf-8'))) for f in folder.rglob('approved.json')]
    for file,manifest in sorted(manifests,key=lambda item:item[1].get('reviewed_at',item[1].get('revision','')),reverse=True):
        checkpoint(cancelled)
        base=file.parent
        if manifest.get('status')!='approved' or manifest['video_sha256'] in seen:continue
        if digest(base/manifest.get('video_file','video.mp4'),cancelled)!=manifest['video_sha256'] or digest(base/'project.json',cancelled)!=manifest['project_sha256']:
            raise ValueError('审核数据已被修改，请重新审核后再生成训练清单。')
        seen.add(manifest['video_sha256']);records.append(dict(**manifest,directory=str(base)))
    if not records:raise ValueError('video_success 中还没有人工审核通过的数据。')
    groups=sorted({r['group_id'] for r in records})
    val_groups={groups[-1]} if len(groups)>=2 else set()
    for record in records:record['split']='val' if record['group_id'] in val_groups else 'train'
    result=dict(format_version=1,created_at=timestamp(),clips=records,
        grouping='source_session_or_video_sha256; revisions and paired source views stay together',
        has_independent_validation=bool(val_groups),
        group_provenance='inferred source session names or content hashes; source independence requires separate verification',
        warning='Only reviewed rows are training labels; unreviewed rows are unknown, never negative labels.')
    checkpoint(cancelled)
    (destination/'training_index.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return dict(file=str(destination/'training_index.json'),clips=len(records),groups=len(groups),
                reviewed_ball_frames=sum(r['reviewed_ball_frames'] for r in records),independent_validation=bool(val_groups))
