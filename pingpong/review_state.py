"""Stable event identifiers and display-only review filters."""
from .events import event_kind,table_half


def ensure_numbers(project):
    events=project.get('events',[]);used=set()
    next_number=max([project.get('next_event_number',1)]+[e.get('event_number',0)+1 for e in events if isinstance(e.get('event_number'),int)])
    for event in events:
        number=event.get('event_number')
        if not isinstance(number,int) or number<1 or number in used:
            number=next_number;next_number+=1;event['event_number']=number
        used.add(number)
    project['next_event_number']=next_number


def hidden_numbers(project):
    return set(project.get('view_settings',{}).get('hidden_event_numbers',[]))


def matches_filter(event,mode,search='',frame=None,fps=30):
    kind=event_kind(event)
    if mode=='pending' and event.get('confirmed'):return False
    if mode=='confirmed' and not event.get('confirmed'):return False
    if mode in ('bounce','out','net','stroke') and kind!=mode:return False
    if mode in ('near','far') and (kind!='bounce' or table_half(event)!=mode):return False
    if search:
        query=search.strip().lstrip('#')
        if query.isdigit():query=str(int(query))
        if query not in (str(event.get('event_number','')),str(event['frame'])):return False
    if frame is not None and abs(event['frame']-frame)>fps*2:return False
    return True


def reuse_numbers(previous,candidates):
    available=[e for e in previous if e.get('event_number')]
    for event in candidates:
        matched=[e for e in available if event_kind(e)==event_kind(event) and abs(e['frame']-event['frame'])<=2]
        if matched:
            old=min(matched,key=lambda e:abs(e['frame']-event['frame']))
            event['event_number']=old['event_number'];available.remove(old)
