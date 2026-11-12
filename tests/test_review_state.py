import numpy as np
from pingpong.review_state import ensure_numbers,reuse_numbers,matches_filter
from pingpong.detector import render_frame


def test_numbers_survive_deletion_and_recalculation_without_reuse():
    project=dict(events=[dict(frame=10,kind='bounce'),dict(frame=20,kind='bounce')])
    ensure_numbers(project);assert [e['event_number'] for e in project['events']]==[1,2]
    project['events'].pop(0);project['events'].append(dict(frame=30,kind='net'));ensure_numbers(project)
    assert [e['event_number'] for e in project['events']]==[2,3]
    candidates=[dict(frame=21,kind='bounce'),dict(frame=30,kind='out')]
    reuse_numbers(project['events'],candidates)
    assert candidates[0]['event_number']==2 and 'event_number' not in candidates[1]


def test_filters_are_explicit_and_use_exact_ids_or_frames():
    event=dict(frame=100,event_number=7,kind='bounce',confirmed=False,table_xy=[.5,.5])
    assert matches_filter(event,'pending','#007')
    assert matches_filter(event,'pending','#7')
    assert matches_filter(event,'far','100')
    assert not matches_filter(event,'near') and not matches_filter(event,'confirmed')
    assert not matches_filter(event,'all',frame=300,fps=30)


def test_hidden_event_overlays_are_absent_but_labels_unchanged():
    frame=np.zeros((100,200,3),np.uint8)
    event=dict(frame=5,x=60.,y=50.,kind='bounce',confirmed=False,event_number=2)
    project=dict(fps=30,events=[event],tracks=[],corners=[])
    assert np.any(render_frame(frame,project,5))
    assert not np.any(render_frame(frame,project,5,options={'hidden_event_numbers':[2]}))
    assert not np.any(render_frame(frame,project,5,options={'visible_event_numbers':[]}))
    assert project['events']==[event]
