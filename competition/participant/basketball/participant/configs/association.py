"""Associate players using velocity, jersey appearance, and scene continuity."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class State:
    tid: int
    box: np.ndarray
    appearance: np.ndarray
    velocity: np.ndarray
    frame: int
    shot: int
    hits: int = 1


def iou(a,b):
    wh = np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter = wh.prod(2)
    return inter / np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]
                             +(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter,1e-6)


class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def step(self, fid, shot, boxes, colors, confidence):
        self.tracks = [t for t in self.tracks if fid-t.frame <= 150]
        order = np.argsort(-confidence)
        boxes,colors,confidence = boxes[order],colors[order],confidence[order]
        matches = {}
        assigned = set()
        if self.tracks and len(boxes):
            ts = self.tracks
            gap = np.array([fid-t.frame for t in ts])
            same = np.array([shot == t.shot for t in ts])
            old = np.stack([t.box for t in ts])
            v = np.stack([t.velocity for t in ts])
            moved = old + v*np.minimum(gap,6)[:,None]
            overlap = iou(moved,boxes)
            bc = (boxes[:,:2]+boxes[:,2:])/2
            tc = (moved[:,:2]+moved[:,2:])/2
            scale = np.maximum(30,(moved[:,3]-moved[:,1]))[:,None]
            dist = np.linalg.norm(tc[:,None]-bc[None],axis=2)/scale
            delta = np.abs(np.stack([t.appearance for t in ts])[:,None]-colors)
            color = delta.mean(axis=2)
            known = np.array([t.hits >= 3 for t in ts])[:,None]
            immediate = (same & (gap <= 4))[:,None]
            near = (same & (gap <= 30))[:,None]
            short = immediate & ((overlap > .12)|(dist < .58)) & ((color < .22)|(overlap > .55))
            bridge = near & ~immediate & (dist < .45+gap[:,None]*.023) & (color < .12) & known
            cost = np.where(immediate,.60*(1-overlap)+.32*dist+1.6*color,
                            .66+.35*dist+2.4*color)
            valid = short|bridge
            rr,cc = linear_sum_assignment(np.where(valid,cost,1e5))
            for i,j in zip(rr,cc):
                if valid[i,j]:
                    matches[j] = ts[i]
                    assigned.add(i)
            # Across a cut, jersey color alone is ambiguous for teammates.
            # Reuse an ID only if the match is distinctive in both directions.
            left = [j for j in range(len(boxes)) if j not in matches and confidence[j] >= .45]
            stale = [i for i,t in enumerate(ts) if i not in assigned and t.hits >= 12
                     and (not same[i] or gap[i] > 30)]
            if left and stale:
                z = color[np.ix_(stale,left)]
                for a,b in zip(*linear_sum_assignment(z)):
                    i,j = stale[a],left[b]
                    other_t = np.delete(z[:,b],a)
                    other_d = np.delete(z[a,:],b)
                    margin = min(other_t.min(initial=1),other_d.min(initial=1))-z[a,b]
                    if z[a,b] < .045 and margin > .017:
                        matches[j] = ts[i]
        out = []
        for j,(box,col,conf) in enumerate(zip(boxes,colors,confidence)):
            if j in matches:
                t = matches[j]
                gap = fid-t.frame
                previous, previous_box = t.frame,t.box.copy()
                if t.shot == shot and gap <= 4:
                    t.velocity = .55*t.velocity + .45*(box-t.box)/gap
                else:
                    t.velocity.fill(0)
                if conf >= .44:
                    t.appearance = .88*t.appearance+.12*col
                t.box,t.frame,t.shot,t.hits = box,fid,shot,t.hits+1
                out.append((t, previous, previous_box, float(conf)))
            elif conf >= .40:
                t = State(self.next_id,box,col,np.zeros(4,np.float32),fid,shot)
                self.next_id += 1
                self.tracks.append(t)
                out.append((t, None, None, float(conf)))
        return out
