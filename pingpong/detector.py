"""Portable ONNX inference. Never imports labels or training annotations."""
from contextlib import contextmanager
from queue import Queue, Full, Empty
from threading import Thread, Event
from dataclasses import asdict
from pathlib import Path
import time
import json
import hashlib
import cv2
import numpy as np
from .analysis import TrackPoint, propose_bounces
from .events import analyze_events,recover_tracks,event_kind,ALGORITHM
from .media import resolve_video
from .motion import LocalMotion,suppress_stationary,suppress_spikes
from .cancellation import checkpoint,cancellable_result,cancellable_false
from .video_io import VideoReader,VideoWriter
from .gpu import preferred_adapter
from .fast_input import normalizer

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
NORMALIZE_BGR = np.ascontiguousarray(((np.arange(256, dtype=np.float32).reshape(256,1,1)/255-MEAN)/STD)[:,:,::-1])
CHANNEL_LUTS = tuple(np.ascontiguousarray(NORMALIZE_BGR[:,:,c]) for c in range(3))


def preprocess(frames):
    tensor = np.empty((1,len(frames)*3,288,512),dtype=np.float32)
    normalize = normalizer()
    for index,frame in enumerate(frames):
        small = frame if frame.shape[:2] == (288,512) else cv2.resize(frame, (512,288))
        if normalize is not None and small.dtype==np.uint8 and small.strides[1:]==(3,1) and small.strides[0]>0:
            normalize(small.ctypes.data,small.strides[0],*(lut.ctypes.data for lut in CHANNEL_LUTS),tensor[0,index*3].ctypes.data)
        else:
            for channel,plane in enumerate(cv2.split(small)):
                cv2.LUT(plane,CHANNEL_LUTS[channel],dst=tensor[0,index*3+2-channel])
    return tensor


def heatmap_point(logits, threshold, probabilities=False):
    hm = np.clip(logits,0,1) if probabilities else 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    peak = float(hm.max())
    if peak < threshold:
        return None, None, peak
    peak_y, peak_x = np.unravel_index(hm.argmax(), hm.shape)
    mask = (hm >= threshold).astype(np.uint8)
    _, _, _, (left, top, width, height) = cv2.floodFill(mask, None, (int(peak_x), int(peak_y)), 2, flags=8)
    ys, xs = np.where(mask[top:top+height,left:left+width] == 2)
    ys += top
    xs += left
    weights = hm[ys,xs]
    return float(np.average(xs, weights=weights)), float(np.average(ys, weights=weights)), peak


class Detector:
    def __init__(self, model_path, threads=4, acceleration='auto'):
        import onnxruntime as ort
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError('未找到检测模型，请在界面中选择 detector.onnx。')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        providers=['CPUExecutionProvider']
        self.backend_note=''
        if acceleration != 'cpu' and 'DmlExecutionProvider' in ort.get_available_providers():
            options.enable_mem_pattern=False
            options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            device_id=int(acceleration) if acceleration in ('0','1') else preferred_adapter()
            providers=[('DmlExecutionProvider',{'device_id':device_id}),'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(str(model_path), options, providers=providers)
        except Exception as exc:
            if len(providers)==1:
                raise
            self.backend_note=f'显卡初始化失败，已使用 CPU：{exc}'
            self.session=ort.InferenceSession(str(model_path),options,providers=['CPUExecutionProvider'])
        self.provider=self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.probability_output=self.session.get_modelmeta().custom_metadata_map.get('heatmap_activation')=='sigmoid'

    def predict(self, frames, threshold=0.5):
        return self.decode_heatmaps(self.run_tensor(preprocess(frames)),threshold)

    def run_tensor(self, tensor):
        return self.session.run(None, {self.input_name: tensor})[0][0]

    def decode_heatmaps(self, result, threshold):
        maps = np.clip(result,0,1) if self.probability_output else 1/(1+np.exp(-np.clip(result,-40,40)))
        predictions=[]
        self.last_candidates=[]
        for hm in maps:
            peak=float(hm.max())
            if peak>=threshold:
                predictions.append(heatmap_point(hm,threshold,True))
                self.last_candidates.append(None)
            else:
                predictions.append((None,None,peak))
                self.last_candidates.append(heatmap_point(hm,min(.25,threshold*.5),True) if peak>=min(.25,threshold*.5) else None)
        return predictions


def points_from_predictions(detector, predictions, frames, times, start, size, motion):
    width,height = size
    points=[]
    for j,(x,y,confidence) in enumerate(predictions[:len(frames)]):
        candidate = getattr(detector,'last_candidates',[None]*3)[j]
        cx,cy = candidate[:2] if candidate is not None else (None,None)
        motion.append(frames[j])
        score = motion.score(x if x is not None else cx,y if y is not None else cy)
        points.append(TrackPoint(start+j,times[j],None if x is None else x*width/512,None if y is None else y*height/288,confidence,
            candidate_x=None if cx is None else cx*width/512,candidate_y=None if cy is None else cy*height/288,motion_score=score))
    return points


@contextmanager
def prefetch(items, cancelled, capacity=3):
    queue = Queue(maxsize=capacity)
    stopped = Event()

    def send(item):
        while not stopped.is_set():
            try:
                queue.put(item, timeout=.05)
                return
            except Full:
                continue

    def produce():
        try:
            for item in items:
                if stopped.is_set():return
                send(item)
            send(None)
        except Exception as error:
            send(error)

    def consume():
        while True:
            checkpoint(cancelled)
            try:
                item = queue.get(timeout=.05)
            except Empty:
                continue
            if isinstance(item, Exception):
                raise item
            if item is None:
                return
            yield item

    worker = Thread(target=produce, name='analysis-decoder', daemon=True)
    worker.start()
    try:
        yield consume()
    finally:
        stopped.set()
        worker.join()


def input_batches(reader, fps, cuts, cancelled, pace):
    frames, times = [], []
    index, start = 0, 0
    latest = previous_time = None
    reset = False
    while True:
        checkpoint(cancelled)
        ok, raw = reader.read()
        timestamp = reader.last_timestamp
        cut = index in cuts or (previous_time is not None and timestamp-previous_time>max(.2,3/fps))
        if frames and (not ok or cut):
            if pace:pace(times[-1])
            yield latest, frames, times, start, reset
            frames, times = [], []
        if not ok:return
        if not frames:start,reset=index,cut
        frames.append(raw if raw.shape[:2]==(288,512) else cv2.resize(raw,(512,288)))
        times.append(timestamp)
        index += 1
        latest, previous_time = raw, timestamp
        if len(frames)==3:
            if pace:pace(timestamp)
            yield latest, frames, times, start, reset
            frames, times = [], []


@contextmanager
def stream_detections(reader, detector, size, fps, threshold, cuts=(), cancelled=None, pace=None, capacity=3):
    """Overlap video preparation, GPU inference and CPU trajectory extraction."""
    def prepare():
        for batch in input_batches(reader,fps,set(cuts),cancelled,pace):
            frames=batch[1]
            padded=frames+[frames[-1]]*(3-len(frames))
            yield batch, preprocess(padded) if hasattr(detector,'run_tensor') else padded

    with prefetch(prepare(),cancelled,capacity) as prepared:
        def infer():
            for batch,tensor in prepared:
                checkpoint(cancelled)
                started=time.perf_counter()
                if hasattr(detector,'run_tensor'):
                    result=detector.run_tensor(tensor)
                else:
                    result=detector.predict(tensor,threshold)
                yield batch,result,time.perf_counter()-started

        with prefetch(infer(),cancelled,capacity) as inferred:
            def extract():
                motion=LocalMotion()
                for (raw,frames,times,start,reset),result,elapsed in inferred:
                    if reset:motion=LocalMotion()
                    if hasattr(detector,'decode_heatmaps'):
                        result=detector.decode_heatmaps(result,threshold)
                    points=points_from_predictions(detector,result,frames,times,start,size,motion)
                    yield raw,points,reset,elapsed
            yield extract()


@cancellable_result
def analyze_video(video, model_path, threshold=0.5, corners=(), progress=None, cancelled=None, acceleration='auto',motion_filter=True):
    started=time.perf_counter()
    checkpoint(cancelled)
    video=resolve_video(video)
    fingerprint=hashlib.sha256()
    with Path(video).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            checkpoint(cancelled);fingerprint.update(chunk)
    video_sha256=fingerprint.hexdigest();initial_stat=Path(video).stat()
    cap = VideoReader(video,cancelled,parallel=True,model_input=True)
    if not cap.isOpened():
        raise ValueError('无法打开视频，请检查路径及视频编码。')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or min(width,height) <= 0:
        cap.release()
        raise ValueError('视频尺寸或帧率无效。')
    break_frames=[]
    segment_file=Path(video).with_suffix('.segments.json')
    if segment_file.exists():
        try:
            metadata=json.loads(segment_file.read_text(encoding='utf-8'))
            if abs(metadata['frames']-total)>2:raise ValueError('片段清单与视频帧数不匹配。')
            break_frames=sorted(set(int(p['start_frame']) for p in metadata['segments'] if int(p['start_frame'])>0))
            if any(not 0<b<total for b in break_frames):raise ValueError('片段切换帧号越界。')
        except Exception:
            cap.release();raise
    points=[]
    inference_seconds=0.
    try:
        detector=Detector(model_path,acceleration=acceleration);checkpoint(cancelled)
        with stream_detections(cap,detector,(width,height),fps,threshold,break_frames,cancelled) as detections:
            for raw,batch,reset,elapsed in detections:
                if reset and batch[0].frame not in break_frames:break_frames.append(batch[0].frame)
                points.extend(batch)
                inference_seconds+=elapsed
                if progress:progress(len(points),max(total,len(points)),raw,points[-1])
        decode_info=cap.info()
    finally:cap.release()
    if not points:raise ValueError('视频中没有可解码的帧。')
    break_frames=sorted(cut for cut in set(break_frames) if 0<cut<len(points))
    final_stat=Path(video).stat()
    if (final_stat.st_size,final_stat.st_mtime_ns)!=(initial_stat.st_size,initial_stat.st_mtime_ns):
        raise ValueError('分析过程中原视频发生变化，请重新分析。')
    recovered=recover_tracks(points,fps,(width,height),break_frames,cancelled=cancelled)
    checkpoint(cancelled)
    rejected=suppress_stationary(points,fps,(width,height),break_frames,cancelled=cancelled) if motion_filter else []
    if motion_filter:rejected+=suppress_spikes(points,fps,(width,height),break_frames,cancelled=cancelled)
    events,hints=analyze_events(points,fps,corners,(width,height),break_frames,cancelled=cancelled)
    checkpoint(cancelled)
    elapsed = time.perf_counter()-started
    return dict(format_version=1, video=str(Path(video).resolve()), fps=fps, width=width,
                video_sha256=video_sha256,
                height=height, frame_count=len(points), threshold=threshold, corners=list(corners),
                decode_info=decode_info,duration_seconds=decode_info['duration_seconds'],
                model=Path(model_path).name, detector='BlurBall / HRNet, three-frame inference',
                provenance='Model inference only; annotations are never used by the detector.',
                tracks=[asdict(p) for p in points],events=events,review_hints=hints,recovered_ball_frames=recovered,
                break_frames=break_frames,
                event_algorithm=ALGORITHM,
                motion_filter_enabled=motion_filter,motion_rejections=rejected,
                performance=dict(elapsed_seconds=elapsed, end_to_end_fps=len(points)/elapsed,
                                 inference_fps=len(points)/max(inference_seconds,1e-9),
                                 provider=detector.provider, backend_note=detector.backend_note),
                limitations=['触台为二维轨迹规则候选，未经人工确认。',
                             '未标定球台时不输出桌面坐标；空中轨迹不映射到桌面。'])


def render_frame(frame, project, index, trail=24, options=None):
    image = frame.copy()
    options=options or {}
    tracks = project.get('tracks', [])
    previous = None
    for p in (tracks[max(0,index-trail):index+1] if options.get('trail',True) else []):
        if p['frame'] in project.get('break_frames',[]):previous=None
        if p['x'] is None:
            previous = None
            continue
        xy = (round(p['x']), round(p['y']))
        if previous is not None and np.linalg.norm(np.array(xy)-previous) < image.shape[1]*0.2:
            cv2.line(image, tuple(previous), xy, (220, 220, 30), 3, cv2.LINE_AA)
        previous = np.array(xy)
    if options.get('ball',True) and index < len(tracks) and tracks[index]['x'] is not None:
        p = tracks[index]
        cv2.circle(image, (round(p['x']),round(p['y'])), 8, (0,255,255), 2, cv2.LINE_AA)
    for e in project.get('events', []):
        if e.get('event_number') in options.get('hidden_event_numbers',[]):continue
        allowed=options.get('visible_event_numbers')
        if allowed is not None and e.get('event_number') not in allowed:continue
        if not options.get('bounces' if event_kind(e)=='bounce' else ('strokes' if event_kind(e)=='stroke' else 'terminal'),True):continue
        if any(min(e['frame'],index)<cut<=max(e['frame'],index) for cut in project.get('break_frames',[])):continue
        if abs(e['frame']-index) <= max(1, round(project['fps']*0.15)):
            kind=event_kind(e)
            color=(100,230,60) if e['confirmed'] else {'bounce':(0,160,255),'out':(70,90,255),'net':(220,110,230),'stroke':(255,180,80)}.get(kind,(0,160,255))
            marker={'bounce':cv2.MARKER_DIAMOND,'out':cv2.MARKER_TILTED_CROSS,'net':cv2.MARKER_SQUARE,'stroke':cv2.MARKER_STAR}.get(kind,cv2.MARKER_DIAMOND)
            cv2.drawMarker(image,(round(e['x']),round(e['y'])),color,marker,22,2)
            if options.get('selected_event_number') is not None and e.get('event_number')==options.get('selected_event_number'):
                cv2.circle(image,(round(e['x']),round(e['y'])),17,(255,255,255),2)
            if e.get('event_number') and options.get('event_numbers',True):
                cv2.putText(image,f"#{e['event_number']:03d}",(round(e['x'])+14,round(e['y'])-10),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
    corners = project.get('corners', [])
    if options.get('boundary',True) and len(corners)==4:
        cv2.polylines(image,[np.array(corners,np.int32)],True,(150,210,50),2,cv2.LINE_AA)
    return image


@cancellable_false
def export_video(path, project, progress=None, cancelled=None, render_options=None):
    checkpoint(cancelled)
    path = Path(path)
    if path.resolve() == Path(project['video']).resolve():
        raise ValueError('输出路径不能覆盖原始视频。')
    partial = path.with_name(path.stem+'.partial.mp4')
    if partial.exists():
        raise FileExistsError('临时输出文件已存在，请更换导出文件名。')
    cap = VideoReader(project['video'],cancelled)
    writer = VideoWriter(partial,project['fps'],(project['width'],project['height']))
    completed = False
    try:
        if not cap.isOpened() or not writer.isOpened():
            raise ValueError('无法读取输入视频或创建输出视频。')
        for i in range(project['frame_count']):
            if cancelled and cancelled():
                return False
            ok,frame = cap.read()
            if not ok:
                raise ValueError(f'导出时视频在第 {i} 帧中断。')
            image = render_frame(frame,project,i,options=render_options)
            if render_options is None or any(render_options.get(key,True) for key in ('trail','ball','bounces','terminal','boundary')):
                cv2.putText(image,f'Frame {i}',(24,40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            writer.write(image,project['tracks'][i]['time'])
            if progress and i%15==0:
                progress(i+1,project['frame_count'],image,None)
        checkpoint(cancelled);completed = True
    finally:
        cap.release()
        writer.release()
        if not completed and partial.exists():
            partial.unlink()
    partial.replace(path)
    return True
