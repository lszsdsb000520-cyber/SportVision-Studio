import copy
import numpy as np
import pytest
from pingpong.annotations import mark_invisible_range,AnnotationHistory
from pingpong.detector import render_frame


def fixture():
    return dict(fps=25,corners=[[10,10],[190,10],[190,90],[10,90]],
        tracks=[dict(frame=i,time=i/25,x=40.+i*5,y=40.,confidence=.9,source='model') for i in range(8)],
        events=[dict(frame=3,x=55.,y=40.,kind='bounce',type='bounce',confirmed=True,source='manual')],
        review=dict(status='approved'))


def test_invisible_range_is_inclusive_and_keeps_events_by_default():
    project=fixture();original=copy.deepcopy(project)
    assert mark_invisible_range(project,2,5)==4
    assert all(p['x'] is None and p['source']=='manual' for p in project['tracks'][2:6])
    assert project['tracks'][1]==original['tracks'][1] and project['tracks'][6]==original['tracks'][6]
    assert project['events']==original['events']
    assert project['review']['status']=='modified_after_approval'
    before=copy.deepcopy(project)
    with pytest.raises(ValueError):mark_invisible_range(project,6,2)
    assert project==before


def test_range_deletion_and_undo_restore_labels_and_keep_audit():
    project=fixture();history=AnnotationHistory();history.remember(project)
    mark_invisible_range(project,2,5,True)
    assert not project['events'] and project['rejected_events']
    restored=history.undo(project)
    assert restored['events'] and restored['tracks'][3]['x']==55.
    assert restored['review']['status']=='modified_after_approval'
    assert [e['action'] for e in restored['edit_history']]==['ball_invisible_range','undo_annotation']
    assert history.undo(restored) is None


def test_render_layers_are_independent_and_pure_view_is_unchanged():
    project=fixture();frame=np.zeros((100,200,3),np.uint8)
    off=dict(trail=False,ball=False,bounces=False,terminal=False,boundary=False)
    project['events'].append(dict(frame=3,x=150.,y=65.,kind='net',confirmed=False))
    assert np.array_equal(render_frame(frame,project,3,options=off),frame)
    for name in off:
        image=render_frame(frame,project,3,options={**off,name:True})
        assert np.any(image),name
    assert not np.any(frame)
