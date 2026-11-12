from pingpong.analysis import TrackPoint
from pingpong.live import LiveEvents


CORNERS = [[180,180],[460,180],[550,360],[90,360]]


def monitor(heights):
    tracker=LiveEvents(50,(640,480),CORNERS)
    for index in range(100):
        y = heights[index] if index<len(heights) else None
        tracker.history.append(TrackPoint(index,index/50,300. if y is not None else None,y,.9))
        tracker.scan(index/50,None)
    return tracker.counts


def test_live_terminal_events_need_observed_motion_and_do_not_repeat():
    blocked=[350,342,334,326,318,310,302,294,286,278,270,260,257,263,273,285,300,315,330]
    assert monitor(blocked)['net']==1
    assert monitor([320+i*10 for i in range(15)])['out']==1
    assert monitor([350-10*i for i in range(28)])['net']==0
    assert monitor([340+i*2 for i in range(8)])['out']==0
