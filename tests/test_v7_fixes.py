import threading
import numpy as np
import pytest
from pingpong.analysis import TrackPoint
from pingpong.events import analyze_events
from pingpong.motion import suppress_spikes
from pingpong.label_layout import label_slots
from pingpong.cancellation import TaskCancelled


def test_racket_reversal_is_not_a_table_bounce():
    corners=[[100,100],[500,100],[500,400],[100,400]]
    stroke=[TrackPoint(i,i/60,250+4*abs(i-15),230-5*abs(i-15),.9) for i in range(31)]
    events,_=analyze_events(stroke,60,corners,(640,480))
    assert any(e['kind']=='stroke' and e['table_xy'] is None for e in events)
    assert not any(e['kind']=='bounce' for e in events)
    for i,p in enumerate(stroke):p.x=200+i*4
    events,_=analyze_events(stroke,60,corners,(640,480))
    assert any(e['kind']=='bounce' for e in events)


def test_short_spurious_excursion_is_removed_without_creating_a_ball():
    track=[TrackPoint(i,i/60,100+i*10,300.,.8) for i in range(20)]
    track[8].x=1000;track[9].x=1000
    assert len(suppress_spikes(track,60,(1920,1080)))==2
    assert track[8].x is None and track[9].x is None
    track[8].x=1000;track[8].y=300.;track[8].source='manual'
    suppress_spikes(track,60,(1920,1080));assert track[8].x==1000


def test_dense_labels_have_a_slot_for_every_id():
    slots=label_slots([(i,100.,100.) for i in range(100)],220,1070)
    assert len(slots)==100
    boxes=list(slots.values())
    for i,(x,y,w,h) in enumerate(boxes):
        assert x>=0 and y>=0 and x+w<=220 and y+h<=1070
        for a,b,c,d in boxes[i+1:]:assert not(x<a+c and a<x+w and y<b+d and b<y+h)


def test_postprocessing_can_be_cancelled():
    points=[TrackPoint(i,i/60,100+i,200.,.9) for i in range(2000)]
    calls=0
    def cancelled():
        nonlocal calls
        calls+=1;return calls>20
    with pytest.raises(TaskCancelled):analyze_events(points,60,cancelled=cancelled)
