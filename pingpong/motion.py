"""Reject persistent, image-static background detections without a per-frame speed cutoff."""
from collections import deque
import cv2,numpy as np
from .cancellation import checkpoint


class LocalMotion:
    def __init__(self):self.history=deque(maxlen=7)

    def append(self,frame):
        small = frame if frame.shape[:2] == (288,512) else cv2.resize(frame,(512,288))
        self.history.append(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY))

    def score(self,x,y):
        if x is None or len(self.history)<4:return None
        x=int(np.clip(round(x),2,509));y=int(np.clip(round(y),2,285))
        now=self.history[-1][y-2:y+3,x-2:x+3].astype(np.float32)
        return max(float(np.percentile(np.abs(now-old[y-2:y+3,x-2:x+3].astype(np.float32)),85))
                   for old in list(self.history)[:-3:3])


def suppress_stationary(points,fps,image_size,break_frames=(),cancelled=None):
    radius=max(3.,image_size[0]*.0035);window=max(12,round(fps*.55))
    minimum=max(6,round(fps*.12));span=max(5,round(fps*.18));rejected={}
    observations=[]
    for i,p in enumerate(points):
        if p.source=='manual':continue
        x=p.x if p.x is not None else p.candidate_x;y=p.y if p.y is not None else p.candidate_y
        if x is None or p.motion_score is None or p.motion_score>7 or p.confidence<.2:continue
        observations.append((i,x,y))
    for pos,(i,x,y) in enumerate(observations):
        checkpoint(cancelled)
        close=[(j,a,b) for j,a,b in observations[max(0,pos-window):pos+window+1]
            if abs(i-j)<=window and np.hypot(a-x,b-y)<=radius
            and not any(min(i,j)<cut<=max(i,j) for cut in break_frames)]
        if len(close)<minimum or close[-1][0]-close[0][0]<span:continue
        if len(close)/max(1,close[-1][0]-close[0][0]+1)<.4:continue
        for j,a,b in close:
            p=points[j]
            rejected[j]=dict(frame=p.frame,x=a,y=b,confidence=p.confidence,reason='persistent_stationary_image_feature')
    for i in rejected:
        points[i].x=points[i].y=None;points[i].candidate_x=points[i].candidate_y=None
        points[i].source='stationary_rejected'
    return list(sorted(rejected.values(),key=lambda r:r['frame']))


def suppress_spikes(points,fps,image_size,break_frames=(),cancelled=None):
    """Drop low-confidence one/two-frame excursions from an otherwise coherent track."""
    valid=[i for i,p in enumerate(points) if p.x is not None and p.y is not None];rejected={};margin=image_size[0]*.045
    for pos in range(1,len(valid)-1):
        checkpoint(cancelled)
        for count in (1,2):
            if pos+count>=len(valid):continue
            left=valid[pos-1];right=valid[pos+count];indices=valid[pos:pos+count]
            if right-left>max(5,round(fps*.1)) or any(left<c<=right for c in break_frames):continue
            if any(points[i].source=='manual' or points[i].confidence>=.85 for i in indices):continue
            a=np.array([points[left].x,points[left].y]);b=np.array([points[right].x,points[right].y])
            if np.linalg.norm(b-a)/(right-left)>image_size[0]*.025:continue
            if not all(np.linalg.norm(np.array([points[i].x,points[i].y])-(a+(b-a)*(i-left)/(right-left)))>margin for i in indices):continue
            for i in indices:rejected[i]=dict(frame=points[i].frame,x=points[i].x,y=points[i].y,confidence=points[i].confidence,reason='isolated_incoherent_excursion')
    for i in rejected:
        points[i].x=points[i].y=points[i].candidate_x=points[i].candidate_y=None;points[i].source='inconsistent_rejected'
    return sorted(rejected.values(),key=lambda r:r['frame'])
