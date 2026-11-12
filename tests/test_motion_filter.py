from pingpong.analysis import TrackPoint
from pingpong.motion import suppress_stationary


def points(total=30):return [TrackPoint(i,i/60,100.,200.,.8,motion_score=1.) for i in range(total)]


def test_sustained_static_artifact_is_rejected_but_manual_label_is_protected():
    track=points();track[10].source='manual'
    rejected=suppress_stationary(track,60,(1920,1080))
    assert rejected and track[10].x==100 and track[10].source=='manual'
    assert track[15].x is None and track[15].candidate_x is None


def test_turning_point_and_real_motion_are_not_stationary_artifacts():
    track=points(9)
    assert not suppress_stationary(track,60,(1920,1080))
    track=points()
    for p in track:p.x+=p.frame*7
    assert not suppress_stationary(track,60,(1920,1080))
    track=points()
    for p in track:p.motion_score=35.
    assert not suppress_stationary(track,60,(1920,1080))


def test_clip_boundaries_do_not_accumulate_static_evidence():
    track=points(16)
    assert not suppress_stationary(track,60,(1920,1080),break_frames=[8])
