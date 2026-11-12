"""Fixed-camera tracking from a Windows video device or paced video frames."""
from collections import deque
from dataclasses import asdict, replace
from math import ceil
import time

import cv2
import numpy as np

from .analysis import homography
from .boundary import TableBoundaryDetector
from .camera import CameraReader
from .cancellation import checkpoint
from .detector import Detector, preprocess, stream_detections
from .events import analyze_events, recover_tracks, event_kind, _net_geometry
from .motion import suppress_stationary, suppress_spikes
from .video_io import VideoReader


def grazing_contacts(points, fps, corners, height):
    """Detect a sharp deflection across the net band with evidence on both sides."""
    net, normal = _net_geometry(corners)
    tangent = net[1]-net[0]
    center = net.mean(axis=0)
    available = [p for p in points if p.x is not None]
    result = []
    for index in range(2,len(available)-3):
        before, a, b, after = available[index-2], available[index], available[index+1], available[index+3]
        if b.frame-a.frame>max(2,fps*.045) or after.time-before.time>.22:
            continue
        xy, nxt = np.array([a.x,a.y]), np.array([b.x,b.y])
        d1, d2 = np.dot(xy-center,normal), np.dot(nxt-center,normal)
        if d1*d2>=0:
            continue
        crossing = xy+(nxt-xy)*(d1/(d1-d2))
        projection = np.dot(crossing-net[0],tangent)/max(1,np.dot(tangent,tangent))
        net_base = net[0]+np.clip(projection,0,1)*tangent
        if not 0<=projection<=1 or not -height*.045<=crossing[1]-net_base[1]<=height*.008:
            continue
        incoming = (xy-[before.x,before.y])/max(1,a.frame-before.frame)
        outgoing = ([after.x,after.y]-nxt)/max(1,after.frame-b.frame)
        speed = np.linalg.norm(incoming)
        after_speed = np.linalg.norm(outgoing)
        if speed<height*.004 or not .15*speed<after_speed<.68*speed:
            continue
        if np.dot(incoming,outgoing)>speed*after_speed*.9:
            continue
        result.append(dict(frame=a.frame,time=a.time,x=float(crossing[0]),y=float(crossing[1]),table_xy=None,
            kind='net',type='net_candidate',confirmed=False,terminal=False,source='net_band_deflection',
            evidence=['net_band_crossing','speed_loss','direction_change','observed_on_both_sides']))
    return result


class LiveEvents:
    def __init__(self, fps, size, corners):
        self.fps, self.size, self.corners = fps, size, corners
        self.history = deque(maxlen=round(fps*2.6))
        self.recent = deque(maxlen=160)
        self.counts = dict(bounce=0, net=0, out=0)
        self.next_scan = 0.
        self.next_number = 1
        self.added = []
        self.terminal_rally_end = -10.

    def scan(self, now, cancelled):
        self.added = []
        if len(self.history)<8 or now<self.next_scan:
            return
        self.next_scan = now+.25
        start = self.history[0].frame
        points = [replace(p,frame=i) for i,p in enumerate(self.history)]
        recover_tracks(points,self.fps,self.size,cancelled=cancelled)
        suppress_stationary(points,self.fps,self.size,cancelled=cancelled)
        suppress_spikes(points,self.fps,self.size,cancelled=cancelled)
        candidates, _ = analyze_events(points,self.fps,self.corners,self.size,cancelled=cancelled)
        candidates.extend(grazing_contacts(points,self.fps,self.corners,self.size[1]))
        for event in sorted(candidates,key=lambda item:item['frame']):
            kind = event_kind(event)
            if kind not in self.counts:
                continue
            terminal = event.get('terminal',False)
            delay = .7 if terminal else .22
            age = now-event['time']
            if age<delay or age>1.3:
                continue
            if terminal and event['time']<=self.terminal_rally_end+.55:
                continue
            if any(abs(event['time']-old['time'])<.14 and kind==event_kind(old) for old in self.recent):
                continue
            event['frame'] += start
            if 'fit_frame' in event:event['fit_frame'] += start
            for key in ('rally_start_frame','rally_end_frame'):
                if key in event:event[key] += start
            event['event_number'] = self.next_number
            self.next_number += 1
            self.counts[kind] += 1
            self.recent.append(event)
            self.added.append(event)
            if terminal:self.terminal_rally_end = event['time']


def run_live(source, model, threshold, acceleration, cancelled, publish, corners=(), on_camera=None):
    is_camera = isinstance(source,int)
    reader = CameraReader(source,cancelled) if is_camera else VideoReader(source,cancelled)
    try:
        if is_camera and on_camera:
            on_camera(reader)
        ok, first = reader.read()
        source_fps = reader.fps
        if not ok:raise ValueError('输入没有可读取的画面。')
        size = (first.shape[1],first.shape[0])
        publish(dict(frame=first,status='正在确认固定机位球台…',corners=[]))
        if len(corners)!=4:
            table = TableBoundaryDetector(acceleration=acceleration)
            if is_camera:
                corners = table.detect(first)['corners']
            else:
                try:initial = table.detect(first)
                except ValueError:initial = None
                if initial is not None and initial['mask_quad_iou']>=.8:
                    corners = initial['corners']
                else:
                    calibration=table.detect_clip(first,str(source),fps=source_fps,cancelled=cancelled,fixed_camera=True)
                    if calibration['review_required']:
                        raise ValueError('球台边界缺少一致的多帧证据，请手动标台。')
                    corners=calibration['corners']
        homography(corners)
        checkpoint(cancelled)
        if is_camera:
            publish(dict(frame=first,status='球台已标定，正在准备检测模型…',corners=corners))
        detector = Detector(model,acceleration=acceleration)
        tensor = preprocess([first]*3)
        detector.run_tensor(tensor)
        warmed=time.perf_counter()
        for _ in range(5):
            checkpoint(cancelled)
            detector.run_tensor(tensor)
        capacity=15/max(.001,time.perf_counter()-warmed)
        stride=max(1,ceil(source_fps/min(60,max(3,capacity*.65))))
        fps=source_fps/stride
        tracker = LiveEvents(fps,size,corners)
        started = time.perf_counter()
        if is_camera:
            reader.started = started
            reader.sample_step = stride
            feed = reader
        else:
            reader.release()
            reader = VideoReader(source,cancelled,parallel=True,model_input=True,sample_step=stride)
            feed = reader
            started = time.perf_counter()

        def pace(timestamp):
            while True:
                checkpoint(cancelled)
                remaining = started+timestamp-time.perf_counter()
                if remaining<=0:return
                time.sleep(min(.01,remaining))

        last_publish = 0.
        frames = 0
        lags = deque(maxlen=2400)
        with stream_detections(feed,detector,size,fps,threshold,cancelled=cancelled,pace=None if is_camera else pace,
                               capacity=1 if is_camera else 3) as detections:
            for frame,batch,reset,_ in detections:
                if reset:tracker.history.clear()
                tracker.history.extend(batch)
                frames += len(batch)
                timestamp = batch[-1].time
                tracker.scan(timestamp,cancelled)
                now = time.perf_counter()
                lags.append(max(0.,now-started-timestamp))
                if tracker.added or now-last_publish>=1/30:
                    last_publish=now
                    publish(dict(frame=frame,size=size,tracks=[asdict(p) for p in list(tracker.history)[-25:]],
                        corners=corners,fps=fps,time=timestamp,events=list(tracker.recent),counts=dict(tracker.counts),
                        status='摄像头检测中' if is_camera else '视频模拟中',frames=frames,lag=lags[-1],
                        sample_step=stride,captured_at=started+timestamp))
        return dict(frames=frames,fps=fps,source_fps=source_fps,sample_step=stride,counts=tracker.counts,events=list(tracker.recent),
            elapsed_seconds=time.perf_counter()-started,lag_p95_seconds=float(np.percentile(lags,95)) if lags else 0.)
    finally:
        reader.release()
