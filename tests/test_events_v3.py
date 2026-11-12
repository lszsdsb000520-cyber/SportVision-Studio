from pingpong.analysis import TrackPoint
from pingpong.events import analyze_events,recover_tracks,terminal_candidates,rally_hints
from pingpong.events import _select_rally_endings

CORNERS=[[100,100],[500,100],[500,400],[100,400]]


def test_gapped_v_can_be_recovered_without_equal_count_fabrication():
    points=[TrackPoint(i,i/60,250+i*2,230-5*abs(i-15),.9) for i in range(31)]
    for i in range(13,17):points[i].x=points[i].y=None
    events,_=analyze_events(points,60,CORNERS,(640,480))
    assert any(e['kind']=='bounce' and abs(e['frame']-15)<=1 and not e['confirmed'] for e in events)
    # Counts are a review cue. The hint generator must not append a balancing event.
    uneven=[dict(frame=i*30,time=i/2,kind='bounce',table_xy=[.5,.5]) for i in range(3)]
    before=len(uneven);assert rally_hints(uneven,60);assert len(uneven)==before


def test_recovery_requires_a_real_peak_and_both_reliable_neighbours():
    points=[TrackPoint(i,i/60,100+i*10,200.,.9) for i in range(7)]
    points[3]=TrackPoint(3,.05,None,None,.3,candidate_x=130,candidate_y=201)
    assert recover_tracks(points,60,(640,480))==1 and points[3].source=='temporal_recovery'
    points[3]=TrackPoint(3,.05,None,None,.3,candidate_x=500,candidate_y=400)
    assert recover_tracks(points,60,(640,480))==0 and points[3].x is None
    points[3]=TrackPoint(3,.05,None,None,.3,candidate_x=130,candidate_y=201)
    assert recover_tracks(points,60,(640,480),break_frames=[3])==0


def sequence(ys,total=60):
    return [TrackPoint(i,i/50,300. if i<len(ys) else None,float(ys[i]) if i<len(ys) else None,.9) for i in range(total)]


def test_crossing_net_is_not_the_same_as_being_blocked():
    passing=sequence([350-10*i for i in range(28)])
    assert not any(e['kind']=='net' for e in terminal_candidates(passing,50,CORNERS,(640,480),[],[]))
    blocked=sequence([350,342,334,326,318,310,302,294,286,278,270,260,257,263,273,285,300,315,330])
    events=terminal_candidates(blocked,50,CORNERS,(640,480),[],[])
    assert any(e['kind']=='net' and not e['confirmed'] for e in events)


def test_out_needs_observed_departure_not_just_a_missing_ball():
    lost=sequence([340+i*2 for i in range(8)])
    assert not any(e['kind']=='out' for e in terminal_candidates(lost,50,CORNERS,(640,480),[],[]))
    leaving=sequence([320+i*10 for i in range(15)])
    assert any(e['kind']=='out' for e in terminal_candidates(leaving,50,CORNERS,(640,480),[],[]))


def test_terminal_motion_in_middle_is_rejected_when_rally_continues():
    blocked=sequence([350,342,334,326,318,310,302,294,286,278,270,260,257,263,273,285,300,315,330])
    assert not terminal_candidates(blocked,50,CORNERS,(640,480),[],[dict(frame=25,table_xy=[.5,.5])])
    # Even without a detected bounce, continued tracking beyond the terminal
    # observation window contradicts a rally ending at this point.
    candidates=[dict(frame=12,kind='net')]
    assert not _select_rally_endings(candidates,list(range(90)),50,[],[])


def test_net_then_out_is_one_terminal_event_and_next_rally_is_independent():
    candidates=[dict(frame=12,kind='net'),dict(frame=19,kind='out'),dict(frame=72,kind='out')]
    selected=_select_rally_endings(candidates,list(range(25))+list(range(65,80)),50,[],[])
    assert [(e['kind'],e['frame']) for e in selected]==[('net',12),('out',72)]
    assert all(e['terminal'] for e in selected)
    # A cut also separates rallies even with continuous detections.
    selected=_select_rally_endings([dict(frame=12,kind='out'),dict(frame=32,kind='net')],list(range(40)),50,[20],[])
    assert len(selected)==2
