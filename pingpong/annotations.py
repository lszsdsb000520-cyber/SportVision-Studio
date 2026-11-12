"""Reversible human annotation operations in original frame coordinates."""
import copy
from .library import log_edit,invalidate_review


def mark_invisible_range(project,start,end,remove_events=False):
    if not 0<=start<=end<len(project['tracks']):
        raise ValueError('起始帧必须小于或等于结束帧，并位于视频范围内。')
    before=copy.deepcopy(project['tracks'][start:end+1])
    for point in project['tracks'][start:end+1]:
        point.update(x=None,y=None,source='manual',confidence=1.,candidate_x=None,candidate_y=None)
    removed=[]
    if remove_events:
        removed=[e for e in project['events'] if start<=e['frame']<=end]
        project.setdefault('rejected_events',[]).extend(copy.deepcopy(removed))
        project['events']=[e for e in project['events'] if not start<=e['frame']<=end]
    log_edit(project,'ball_invisible_range',before,dict(start=start,end=end,removed_events=removed))
    return end-start+1


class AnnotationHistory:
    def __init__(self,limit=10):
        self.limit=limit;self.items=[]

    def remember(self,project):
        self.items.append(copy.deepcopy(project))
        self.items=self.items[-self.limit:]

    def undo(self,current):
        if not self.items:return None
        restored=self.items.pop()
        restored['edit_history']=copy.deepcopy(current.get('edit_history',[]))
        log_edit(restored,'undo_annotation',None,dict(previous_algorithm=restored.get('event_algorithm')))
        # Undo never certifies a clip that has changed since approval.
        invalidate_review(restored)
        return restored
