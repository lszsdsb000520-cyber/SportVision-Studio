"""Trajectory events and table geometry; no ground truth is read here."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import math
import cv2
import numpy as np
from scipy.signal import find_peaks
from .cancellation import checkpoint


@dataclass
class TrackPoint:
    frame: int
    time: float
    x: float | None
    y: float | None
    confidence: float
    source: str = 'model'
    candidate_x: float | None = None
    candidate_y: float | None = None
    motion_score: float | None = None


def homography(corners):
    pts = np.asarray(corners, np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError('请依次选择球台的四个角点。')
    contour = pts.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 100:
        raise ValueError('四个角点必须沿球台边缘依次选择，且不能交叉或共线。')
    target = np.array([[0, 0], [1.525, 0], [1.525, 2.74], [0, 2.74]], np.float32)
    return cv2.getPerspectiveTransform(pts, target)


def map_point(x, y, corners):
    if len(corners) != 4:
        return None
    if cv2.pointPolygonTest(np.asarray(corners, np.float32), (float(x), float(y)), False) < 0:
        return None
    point = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), homography(corners))[0, 0]
    return [round(float(point[0]), 4), round(float(point[1]), 4)]


def _turning_point(indices, xs, ys, weights, center, radius, scale, relaxed=False,max_gap=3,cancelled=None):
    """Fit a continuous two-branch trajectory, rather than a smoothed peak."""
    selected=np.abs(indices-center)<=radius
    t=indices[selected].astype(float);x=xs[selected];y=ys[selected];w=weights[selected]
    minimum=2 if relaxed else 3
    if len(t)<(5 if relaxed else 7) or np.sum(t<center)<minimum or np.sum(t>center)<minimum:return None
    if np.max(np.diff(t))>max_gap:return None
    root_w=np.sqrt(w)
    smooth_matrix=np.column_stack([np.ones(len(t)),t-center,(t-center)**2])
    smooth_coef=np.linalg.lstsq(smooth_matrix*root_w[:,None],y*root_w,rcond=None)[0]
    smooth_error=float(np.average((smooth_matrix@smooth_coef-y)**2,weights=w))
    best=None
    span=max(1.2,(max_gap-1)/2) if relaxed else 1.2
    for tau in np.linspace(center-span,center+span,31 if relaxed else 25):
        checkpoint(cancelled)
        if np.sum(t<tau)<minimum or np.sum(t>tau)<minimum:continue
        left=np.minimum(t-tau,0);right=np.maximum(t-tau,0)
        matrix=np.column_stack([np.ones(len(t)),left,right,left**2,right**2])
        coef=np.linalg.lstsq(matrix*root_w[:,None],y*root_w,rcond=None)[0]
        cost=float(np.average((matrix@coef-y)**2,weights=w))
        # In image coordinates, down-then-up is positive-then-negative velocity.
        slope=.3 if relaxed else .65
        if coef[1]<slope*scale or coef[2]>-slope*scale:continue
        if coef[1]-coef[2]<(1.2 if relaxed else 2.)*scale:continue
        if best is None or cost<best[0]:best=(cost,tau,coef,matrix)
    if best is None:return None
    cost,tau,coef,matrix=best
    # A smooth airborne arc can also have a lowest pixel; it is not a sharp bounce.
    if smooth_error-cost<(.3 if relaxed else .7)*scale**2 or cost>smooth_error*(.88 if relaxed else .65):return None
    if np.sqrt(cost)>max((4.5 if relaxed else 2.5)*scale,(.2 if relaxed else .12)*np.ptp(y)):return None
    cx=np.linalg.lstsq(matrix*root_w[:,None],x*root_w,rcond=None)[0]
    return dict(fit_frame=float(tau),x=float(cx[0]),y=float(coef[0]),
                fit_rmse_px=float(np.sqrt(cost)),incoming_vy=float(coef[1]),outgoing_vy=float(coef[2]),
                incoming_vx=float(cx[1]),outgoing_vx=float(cx[2]))


def propose_bounces(points, fps, corners=(), image_size=(1920,1080), break_frames=(),sensitivity='strict',cancelled=None):
    """Detect sharp V turns using adjacent motion; all proposals remain unconfirmed."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('视频帧率无效。')
    scale=image_size[1]/1080
    relaxed=sensitivity=='recall'
    radius=max(5,min(14,round(fps*.15))) if relaxed else max(4,min(9,round(fps*.1)))
    max_gap=max(3,min(6,round(fps*.09))) if relaxed else 3
    valid=[i for i,p in enumerate(points) if p.x is not None and p.y is not None
           and np.isfinite([p.x,p.y]).all()]
    segments=[];current=[];cuts=set(break_frames)
    for i in valid:
        checkpoint(cancelled)
        if current:
            prev=current[-1]
            jump=np.hypot(points[i].x-points[prev].x,points[i].y-points[prev].y)
            if i-prev>max_gap or jump>image_size[0]*.16 or any(prev<c<=i for c in cuts):
                segments.append(current);current=[]
        current.append(i)
    if current:segments.append(current)
    events=[]
    for segment in segments:
        if len(segment)<(5 if relaxed else 7):continue
        indices=np.array(segment);xs=np.array([points[i].x for i in segment],float)
        ys=np.array([points[i].y for i in segment],float)
        weights=np.clip([points[i].confidence for i in segment],.15,1.)
        dense=np.arange(indices[0],indices[-1]+1)
        interp=np.interp(dense,indices,ys)
        peaks,_=find_peaks(interp,prominence=(1.2 if relaxed else 2.)*scale,distance=max(3,round(fps*.1)))
        for peak in peaks:
            center=int(dense[peak])
            fit=_turning_point(indices,xs,ys,weights,center,radius,scale,relaxed=relaxed,max_gap=max_gap,cancelled=cancelled)
            if fit is None:continue
            frame=int(round(fit['fit_frame']))
            x,y=fit['x'],fit['y']
            observed=points[frame].x is not None
            if observed:
                # Place the visible marker on the observed bottom of the V, not an
                # extrapolated sub-frame point below the displayed trajectory.
                nearby=[i for i in segment if abs(i-fit['fit_frame'])<=1.2]
                frame=max(nearby,key=lambda i:points[i].y)
                x,y=float(points[frame].x),float(points[frame].y)
            mapped=map_point(x,y,corners) if len(corners)==4 else None
            vx1,vx2=fit['incoming_vx'],fit['outgoing_vx']
            reversal=vx1*vx2<0 and min(abs(vx1),abs(vx2))>max(2*scale,.2*max(abs(vx1),abs(vx2)))
            edge_uncertain=False
            if len(corners)==4 and mapped is None:
                distance=cv2.pointPolygonTest(np.asarray(corners,np.float32),(float(x),float(y)),True)
                if not relaxed or distance < -image_size[1]*.015:continue
                xy=cv2.perspectiveTransform(np.array([[[x,y]]],np.float32),homography(corners))[0,0]
                mapped=[float(xy[0]),float(xy[1])];edge_uncertain=True
            events.append(dict(frame=points[frame].frame,time=points[frame].time,
                x=x,y=y,table_xy=mapped,confirmed=False,source='v_turn_fit',type='bounce_candidate',
                fit_frame=fit['fit_frame'],fit_rmse_px=fit['fit_rmse_px'],
                incoming_vx=vx1,outgoing_vx=vx2,contact_kind='stroke' if reversal else 'bounce',
                vertex_observed=observed,edge_uncertain=edge_uncertain,recovery='relaxed_v_fit' if relaxed else 'strict_v_fit'))
    # Suppress duplicate local candidates, preferring the better supported fit.
    selected=[]
    for event in sorted(events,key=lambda e:e['fit_rmse_px']):
        if not any(abs(event['frame']-e['frame'])<max(3,round(fps*.1)) for e in selected):selected.append(event)
    return sorted(selected,key=lambda e:e['frame'])


def validate_project(project):
    if project.get('format_version') != 1:
        raise ValueError('不支持的项目文件版本。')
    fps = float(project['fps'])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('项目中的帧率无效。')
    tracks = [TrackPoint(**p) for p in project['tracks']]
    for i,p in enumerate(tracks):
        if p.frame != i or not math.isfinite(p.time):
            raise ValueError('项目轨迹的帧号或时间无效。')
        if (p.x is None) != (p.y is None):
            raise ValueError('项目轨迹的坐标缺失不一致。')
        if p.x is not None and not all(math.isfinite(v) for v in (p.x,p.y)):
            raise ValueError('项目轨迹中存在无效坐标。')
    if project.get('corners'):
        homography(project['corners'])
    for event in project.get('events',[]):
        if not 0 <= int(event['frame']) < len(tracks):
            raise ValueError('项目中的事件帧号越界。')
    return tracks


def save_project(path, project):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(project, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def export_csv(folder, project):
    from .review_state import ensure_numbers
    ensure_numbers(project)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for filename, fields, rows in [
        ('tracks.csv', ['frame','time','x','y','confidence','source'], project['tracks']),
        ('events.csv', ['event_number','frame','time','x','y','table_x_m','table_y_m','confirmed','source','type'],
         [dict(**{k:v for k,v in e.items() if k!='table_xy'},
               table_x_m=(e.get('table_xy') or ['', ''])[0],
               table_y_m=(e.get('table_xy') or ['', ''])[1]) for e in project['events']])]:
        with (folder/filename).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    save_project(folder/'project.json', project)
