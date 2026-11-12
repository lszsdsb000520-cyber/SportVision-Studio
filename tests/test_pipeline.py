from pathlib import Path
import cv2
import numpy as np
import pytest
import pingpong.detector as module
from pingpong.analysis import save_project,validate_project


def make_video(path,frames):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(160,96))
    assert writer.isOpened()
    for i in range(frames):writer.write(np.full((96,160,3),40+i*10,np.uint8))
    writer.release()


class FakeDetector:
    provider='test';backend_note=''
    def __init__(self,*args,**kwargs):pass
    def predict(self,frames,threshold):return [(256.,144.,.9)]*3


@pytest.mark.parametrize('frames',[1,2,3,4,7])
def test_every_decoded_frame_has_exactly_one_prediction(tmp_path,monkeypatch,frames):
    path=tmp_path/'source.mp4';make_video(path,frames)
    monkeypatch.setattr(module,'Detector',FakeDetector)
    project=module.analyze_video(path,'unused')
    assert len(project['tracks'])==frames
    assert [p['frame'] for p in project['tracks']]==list(range(frames))
    assert project['tracks'][-1]['time']==(frames-1)/25
    assert project['tracks'][0]['x']==80 and project['tracks'][0]['y']==48
    validate_project(project)


def test_cancel_does_not_return_partial_analysis(tmp_path,monkeypatch):
    path=tmp_path/'source.mp4';make_video(path,7)
    monkeypatch.setattr(module,'Detector',FakeDetector)
    assert module.analyze_video(path,'unused',cancelled=lambda:True) is None


def test_export_preserves_source_and_cleans_cancelled_output(tmp_path,monkeypatch):
    path=tmp_path/'source.mp4';make_video(path,7);before=path.read_bytes()
    monkeypatch.setattr(module,'Detector',FakeDetector)
    project=module.analyze_video(path,'unused')
    with pytest.raises(ValueError):module.export_video(path,project)
    output=tmp_path/'tracked.mp4'
    assert module.export_video(output,project,cancelled=lambda:True) is False
    assert not output.exists() and not (tmp_path/'tracked.partial.mp4').exists()
    assert module.export_video(output,project)
    cap=cv2.VideoCapture(str(output));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==7;cap.release()
    assert path.read_bytes()==before


def test_heatmap_empty_and_weighted_center():
    assert module.heatmap_point(np.full((8,8),-20.),.5)[:2]==(None,None)
    logits=np.full((8,8),-20.);logits[3,4]=10
    assert module.heatmap_point(logits,.5)[:2]==(4.,3.)


def test_compilation_cuts_split_model_windows_and_preserve_frame_count(tmp_path,monkeypatch):
    import json
    path=tmp_path/'source.mp4';make_video(path,8)
    path.with_suffix('.segments.json').write_text(json.dumps(dict(frames=8,segments=[dict(start_frame=0),dict(start_frame=4)])),encoding='utf-8')
    seen=[]
    class Spy(FakeDetector):
        def predict(self,frames,threshold):
            seen.append([float(f.mean()) for f in frames]);return super().predict(frames,threshold)
    monkeypatch.setattr(module,'Detector',Spy)
    project=module.analyze_video(path,'unused')
    assert len(project['tracks'])==8 and project['break_frames']==[4]
    assert len(seen)==4 and len(set(seen[1]))==1 and len(set(seen[-1]))==1
