"""Evidence-based recovery, terminal-event candidates and rally review hints."""
import math
import cv2
import numpy as np
from .analysis import propose_bounces,homography,map_point
from .cancellation import checkpoint

ALGORITHM='contact_classification_v7'


def recover_tracks(points,fps,image_size,break_frames=(),cancelled=None):
    """Recover only real low-score model peaks supported by reliable neighbours."""
    reliable=[i for i,p in enumerate(points) if p.x is not None and p.source in ('model','manual')]
    maximum=max(3,min(8,round(fps*.12)));recovered=0
    for left,right in zip(reliable,reliable[1:]):
        checkpoint(cancelled)
        if right-left<=1 or right-left>maximum or not same_segment(left,right,break_frames):continue
        a=np.array([points[left].x,points[left].y]);b=np.array([points[right].x,points[right].y])
        if np.linalg.norm(b-a)/(right-left)>image_size[0]*.055:continue
        for i in range(left+1,right):
            p=points[i]
            if p.x is not None or p.candidate_x is None or p.candidate_y is None:continue
            if p.confidence<.2:continue
            candidate=np.array([p.candidate_x,p.candidate_y]);expected=a+(b-a)*(i-left)/(right-left)
            if np.linalg.norm(candidate-expected)>image_size[0]*.018+(right-left)*2:continue
            extra=np.linalg.norm(candidate-a)+np.linalg.norm(b-candidate)-np.linalg.norm(b-a)
            if extra>image_size[1]*.035:continue
            p.x,p.y=map(float,candidate);p.source='temporal_recovery';recovered+=1
    return recovered


def event_kind(event):
    return event.get('kind',event.get('type','bounce').replace('_candidate',''))


def table_half(event):
    xy=event.get('table_xy')
    return None if not xy else ('far' if xy[1]<1.37 else 'near')


def same_segment(a,b,cuts):return not any(min(a,b)<c<=max(a,b) for c in cuts)


def _net_geometry(corners):
    inverse=np.linalg.inv(homography(corners))
    net=cv2.perspectiveTransform(np.array([[[0,1.37],[1.525,1.37]]],np.float32),inverse)[0]
    tangent=net[1]-net[0];axis=np.array([-tangent[1],tangent[0]])
    axis=axis/max(1e-6,np.linalg.norm(axis))
    if np.dot(np.mean(np.array(corners)[2:],axis=0)-net.mean(axis=0),axis)<0:axis=-axis
    return net,axis


def _rally_spans(available,fps,break_frames):
    """Cuts and sustained observation gaps separate provisional rally spans."""
    spans=[]
    for frame in available:
        if not spans or frame-spans[-1][1]>max(3,round(fps*.7)) or not same_segment(spans[-1][1],frame,break_frames):
            spans.append([frame,frame])
        else:
            spans[-1][1]=frame
    return spans


def _select_rally_endings(candidates,available,fps,break_frames,bounces):
    """A terminal event ends a rally; an ongoing rally cannot contain one."""
    selected=[]
    for start,end in _rally_spans(available,fps,break_frames):
        last_bounce=max((e['frame'] for e in bounces if start<=e['frame']<=end),default=start-1)
        # A following table contact contradicts termination. Also require the
        # candidate to lie in the observed tail, rather than in a continuing rally.
        tail=[e for e in candidates if start<=e['frame']<=end
              and e['frame']>last_bounce and end-e['frame']<=fps*.6]
        if not tail:continue
        # A ball stopped by the net may subsequently fall outside the table.
        # That is one net ending, not a second out ending.
        net=[e for e in tail if event_kind(e)=='net']
        event=min(net or tail,key=lambda e:e['frame'])
        event.update(terminal=True,rally_start_frame=start,rally_end_frame=end)
        selected.append(event)
    return selected


def terminal_candidates(points,fps,corners,image_size,break_frames,bounces,cancelled=None):
    """Classify only the end of each rally as suspected out or blocked by net."""
    if len(corners)!=4 or len(points)<8:return []
    poly=np.asarray(corners,np.float32);net,axis=_net_geometry(corners)
    plane=homography(corners)
    height=image_size[1];width=image_size[0];radius=max(5,round(fps*.18))
    available=[i for i,p in enumerate(points) if p.x is not None]
    result=[]
    for pos,i in enumerate(available):
        checkpoint(cancelled)
        if i<3:continue
        before=[j for j in available[max(0,pos-radius):pos] if i-j<=radius and same_segment(i,j,break_frames)]
        after=[j for j in available[pos+1:pos+radius+1] if j-i<=radius and same_segment(i,j,break_frames)]
        if len(before)<3:continue
        p=points[i];xy=np.array([p.x,p.y]);past=points[before[0]]
        velocity=(xy-np.array([past.x,past.y]))/max(1,i-before[0])
        speed=float(np.linalg.norm(velocity))
        inside=cv2.pointPolygonTest(poly,(float(p.x),float(p.y)),True)
        # A net crossing by itself is not a blocked ball. Require approach AND slowdown/
        # reversal with a downward fall on the same side, without a later opposite bounce.
        net_distance=abs(float(np.dot(xy-net.mean(axis=0),axis)))
        projection=float(np.dot(xy-net[0],net[1]-net[0])/max(1,np.dot(net[1]-net[0],net[1]-net[0])))
        if len(after)>=3 and net_distance<height*.045 and -.15<projection<1.15 and speed>width*.0015:
            next_p=points[after[-1]];future=np.array([next_p.x,next_p.y]);v2=(future-xy)/max(1,after[-1]-i)
            approach=abs(np.dot(np.array([past.x,past.y])-net.mean(axis=0),axis))>net_distance+height*.008
            direction_before=float(np.dot(velocity,axis));direction_after=float(np.dot(v2,axis))
            stopped=np.linalg.norm(v2)<speed*.5 or direction_before*direction_after<0
            fell=future[1]>p.y+height*.012
            remained=float(np.dot(future-net.mean(axis=0),axis))*float(np.dot(xy-net.mean(axis=0),axis))>=0
            other_bounce=any(i<e['frame']<i+fps*.6 for e in bounces)
            if approach and stopped and fell and remained and not other_bounce:
                result.append(dict(frame=i,time=p.time,x=p.x,y=p.y,table_xy=None,kind='net',
                    type='net_candidate',confirmed=False,source='terminal_motion',
                    evidence=['approach_net','slowdown_or_reversal','fall_same_side'],reason='suspected_blocked_by_net'))
        # For out, require a continuous tracked departure from the table neighbourhood,
        # persistent downward/outward motion, and a trajectory ending or approaching the
        # image edge. Losing detection alone is deliberately insufficient.
        near_before=any(cv2.pointPolygonTest(poly,(float(points[j].x),float(points[j].y)),True)>=-height*.04 for j in before)
        below=p.y>np.max(poly[:,1])+height*.025
        surface=cv2.perspectiveTransform(np.array([[[p.x,p.y]]],np.float32),plane)[0,0]
        plane_exit=surface[0]<-.08 or surface[0]>1.605 or surface[1]<-.10 or surface[1]>2.84
        destination='far' if surface[1]<1.37 else 'near'
        already_landed=any(0<i-e['frame']<fps*.8 and table_half(e)==destination for e in bounces)
        edge=p.x<width*.035 or p.x>width*.965 or p.y>height*.95
        outward=velocity[1]>height*.002 or edge
        tail=[j for j in available[pos+1:pos+radius*3+1] if j-i<fps*.3 and same_segment(i,j,break_frames)]
        persistent=len(after)>=2 and all(cv2.pointPolygonTest(poly,(float(points[j].x),float(points[j].y)),True)<0 for j in after[:3])
        near_cut=any(0<c-i<radius for c in break_frames)
        ended=(not tail or tail[-1]-i<radius*.7) and i<len(points)-max(3,round(fps*.1))
        if inside<-height*.025 and near_before and (below or (plane_exit and not already_landed)) and outward and persistent and (ended or edge) and not near_cut:
            result.append(dict(frame=i,time=p.time,x=p.x,y=p.y,table_xy=None,kind='out',
                type='out_candidate',confirmed=False,source='terminal_motion',
                evidence=['tracked_table_departure','outside_surface','persistent_outward_motion'],reason='suspected_out'))
    return _select_rally_endings(result,available,fps,break_frames,bounces)


def rally_hints(events,fps,break_frames=()):
    bounces=sorted([e for e in events if event_kind(e)=='bounce' and table_half(e)],key=lambda e:e['frame'])
    hints=[]
    for a,b in zip(bounces,bounces[1:]):
        if not same_segment(a['frame'],b['frame'],break_frames):continue
        if not fps*.18<b['frame']-a['frame']<fps*2.2:continue
        if any(a['frame']<e['frame']<b['frame'] and event_kind(e) in ('net','out') for e in events):continue
        if table_half(a)==table_half(b):
            hints.append(dict(start=a['frame'],end=b['frame'],kind='possible_missing_bounce',
                expected_half='near' if table_half(a)=='far' else 'far',
                reason='same_half_consecutive; may also be a double bounce or an incorrect candidate'))
    return hints


def analyze_events(points,fps,corners=(),image_size=(1920,1080),break_frames=(),cancelled=None):
    checkpoint(cancelled)
    strong=propose_bounces(points,fps,corners,image_size,break_frames,cancelled=cancelled)
    recovery=propose_bounces(points,fps,corners,image_size,break_frames,sensitivity='recall',cancelled=cancelled)
    events=[]
    for event in strong:
        event.update(kind='bounce',review_priority='normal');events.append(event)
    for event in recovery:
        if any(abs(event['frame']-e['frame'])<max(3,round(fps*.1)) for e in events):continue
        event.update(kind='bounce',review_priority='high',source='recovery_v_fit');events.append(event)
    for event in events:
        if event.get('contact_kind')=='stroke':
            event.update(kind='stroke',type='stroke_candidate',table_xy=None,reason='horizontal_reversal_suspected_racket_contact',review_priority='high')
    events.extend(terminal_candidates(points,fps,corners,image_size,break_frames,[e for e in events if event_kind(e)=='bounce'],cancelled=cancelled))
    events.sort(key=lambda e:e['frame'])
    hints=rally_hints(events,fps,break_frames)
    for e in events:
        checkpoint(cancelled)
        e['half']=table_half(e) if event_kind(e)=='bounce' else None
        if event_kind(e)=='bounce' and any(points[i].source=='temporal_recovery' for i in range(max(0,e['frame']-3),min(len(points),e['frame']+4))):
            e.update(review_priority='high',recovered_ball_evidence=True)
        if e.get('recovery')=='relaxed_v_fit' and any(h['start']<e['frame']<h['end'] for h in rally_hints(strong,fps,break_frames)):
            e['reason']='V_turn_supported_and_alternation_gap'
    return events,hints
