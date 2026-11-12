from pathlib import Path
import csv
import json
from dataclasses import asdict
import cv2
import numpy as np
import pytest
from pingpong.analysis import TrackPoint
from pingpong.library import save_approved,prepare_training_index,digest


def project_fixture(tmp_path):
    video=tmp_path/'sample.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),25,(160,96))
    for i in range(8):writer.write(np.full((96,160,3),40+i*5,np.uint8))
    writer.release()
    points=[asdict(TrackPoint(i,i/25,80.,48.,.9)) for i in range(8)]
    points[2].update(source='manual',x=82.,y=49.)
    points[3].update(source='manual',x=None,y=None)
    return dict(format_version=1,video=str(video),fps=25,width=160,height=96,frame_count=8,
        corners=[[20,10],[140,10],[140,85],[20,85]],tracks=points,
        events=[dict(frame=2,time=.08,x=82.,y=49.,source='manual',kind='bounce',type='bounce',confirmed=True,table_xy=[.8,1.4])])


def test_only_explicitly_approved_data_enters_library(tmp_path):
    project=project_fixture(tmp_path);library=tmp_path/'video_success'
    with pytest.raises(ValueError):save_approved(project,library)
    assert not library.exists()
    accepted=save_approved(project,library,events_reviewed=True)
    folder=Path(accepted['directory'])
    assert accepted['manifest']['video_file']=='sample_success.mp4'
    assert digest(folder/'sample_success.mp4')==digest(project['video'])
    assert json.loads((folder/'project.json').read_text())['video']=='sample_success.mp4'
    with (folder/'ball_labels.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert rows[0]['Visibility']=='' and rows[0]['Status']=='unreviewed'
    assert rows[2]['Visibility']=='1' and rows[3]['Visibility']=='0'
    assert accepted['manifest']['reviewed_ball_frames']==2
    index=prepare_training_index(library,tmp_path/'training')
    assert index['clips']==1 and index['independent_validation'] is False


def test_revisions_are_immutable_and_training_uses_latest_revision(tmp_path):
    project=project_fixture(tmp_path);library=tmp_path/'video_success'
    first=save_approved(project,library,events_reviewed=True)
    project['events'][0]['x']=85.
    second=save_approved(project,library,events_reviewed=True,ball_positions_reviewed=True)
    assert first['directory']!=second['directory']
    assert json.loads((Path(first['directory'])/'project.json').read_text())['events'][0]['x']==82.
    result=prepare_training_index(library,tmp_path/'training')
    assert result['clips']==1 and result['reviewed_ball_frames']==8
    (Path(second['directory'])/'project.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError):prepare_training_index(library,tmp_path/'training')


def test_cancelled_approval_does_not_create_an_approved_revision(tmp_path):
    project=project_fixture(tmp_path);library=tmp_path/'video_success'
    with pytest.raises(InterruptedError):save_approved(project,library,events_reviewed=True,cancelled=lambda:True)
    assert not list(library.rglob('approved.json'))


def test_changed_source_cannot_reuse_previous_analysis_labels(tmp_path):
    project=project_fixture(tmp_path);project['video_sha256']=digest(project['video'])
    path=Path(project['video']);path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError):save_approved(project,tmp_path/'video_success',events_reviewed=True)


def test_reapproving_a_saved_project_preserves_its_source_group(tmp_path):
    project=project_fixture(tmp_path)
    project['data_origin']=dict(media_id='EXAMPLE',video_sha256=digest(project['video']),group_id='session_A')
    first=save_approved(project,tmp_path/'library',events_reviewed=True)
    folder=Path(first['directory']);saved=json.loads((folder/'project.json').read_text())
    saved['video']=str(folder/saved['video'])
    second=save_approved(saved,tmp_path/'library',events_reviewed=True)
    assert second['manifest']['media_id']=='EXAMPLE' and second['manifest']['group_id']=='session_A'
    assert second['manifest']['video_file']=='sample_success.mp4'
