"""FFmpeg/PyAV reader with OpenCV fallback and decoded-frame timestamps."""
from fractions import Fraction
import math
import av,cv2,numpy as np
from av.video.reformatter import VideoReformatter
from .cancellation import checkpoint,TaskCancelled
from .fast_input import ModelInputSampler


class VideoReader:
    def __init__(self,path,cancelled=None,parallel=False,model_input=False,sample_step=1):
        self.cancelled=cancelled;checkpoint(cancelled)
        self.path=str(path);self.container=None;self.cap=None;self.backend='PyAV/FFmpeg'
        self.pos=0;self.last_timestamp=0.;self.last_duration=0.;self.times=[];self.exact_times=None
        self.errors=0;self.warnings=[];self.eof=False;self.pending=None;self._fallback_used=False;self.last_rotation=0
        self._recover_enabled=True
        self.reformatter=VideoReformatter() if parallel else None
        self.model_input=model_input
        self.sample_step=max(1,int(sample_step))
        self.input_sampler=ModelInputSampler() if model_input else None
        self.native_size=None
        try:
            self.container=av.open(self.path);self.stream=self.container.streams.video[0]
            self.stream.codec_context.thread_count=4 if parallel else 2
            if parallel:self.stream.thread_type='AUTO'
            rates=[self.stream.average_rate,self.stream.base_rate,self.stream.guessed_rate]
            self.fps=next((float(r) for r in rates if r and math.isfinite(float(r)) and float(r)>0),30.)
            self.origin=float((self.stream.start_time or 0)*self.stream.time_base)
            self.duration=float(self.stream.duration*self.stream.time_base) if self.stream.duration else (self.container.duration or 0)/av.time_base
            self.reported=int(self.stream.frames or round(self.duration*self.fps))
            self.decoder=self._frames();raw,t,d=next(self.decoder)
            self.width,self.height=self.native_size;self.pending=(raw,t,d)
        except TaskCancelled:
            self.release()
            raise
        except Exception:
            self.release()
            self._open_cv();ok,raw=self.cap.read()
            if not ok:self.cap.release();raise ValueError('无法解码视频：文件可能损坏，或当前解码器不支持此编码。')
            self.width,self.height=raw.shape[1],raw.shape[0];self.pending=(raw,0.,1/self.fps)
        self.reported=max(1,self.reported)

    def _open_cv(self):
        self.cap=cv2.VideoCapture(self.path)
        if not self.cap.isOpened():raise ValueError('备用解码器也无法打开视频。')
        self.backend='OpenCV/FFmpeg fallback';self.fps=float(self.cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps<=0:self.fps=30.
        self.reported=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT));self.duration=max(0,self.reported/self.fps)

    def _frames(self):
        generated=self.pos
        for packet in self.container.demux(self.stream):
            checkpoint(self.cancelled)
            try:frames=packet.decode()
            except av.error.FFmpegError:
                self.errors+=1;continue
            for frame in frames:
                time=float(frame.pts*self.stream.time_base)-self.origin if frame.pts is not None else generated/self.fps
                duration=float(frame.duration*self.stream.time_base) if frame.duration else 1/self.fps
                rotation=int(frame.rotation or 0)
                if rotation or any(side.type.name=='DISPLAYMATRIX' for side in frame.side_data):self.last_rotation=rotation
                rotation=self.last_rotation%360
                selected=generated%self.sample_step==0
                generated+=1
                if not selected:continue
                size=(frame.height,frame.width) if rotation in (90,270) else (frame.width,frame.height)
                if self.native_size is None:self.native_size=size
                raw=self.input_sampler.resize(frame,rotation) if self.input_sampler and size==self.native_size else None
                if raw is None:
                    raw=self._convert(frame,rotation)
                    if size!=self.native_size and self.model_input:raw=cv2.resize(raw,self.native_size)
                    if self.model_input:raw=cv2.resize(raw,(512,288))
                yield raw,max(0.,time),duration

    def _convert(self, frame, rotation):
        if self.reformatter is None:
            raw=frame.to_ndarray(format='bgr24')
        else:
            bgr=self.reformatter.reformat(frame,format='bgr24',threads=2)
            plane=bgr.planes[0]
            rows=np.frombuffer(plane,dtype=np.uint8).reshape(bgr.height,plane.line_size)
            raw=rows[:,:bgr.width*3].reshape(bgr.height,bgr.width,3)
        if rotation==90:return cv2.rotate(raw,cv2.ROTATE_90_COUNTERCLOCKWISE)
        if rotation==180:return cv2.rotate(raw,cv2.ROTATE_180)
        if rotation==270:return cv2.rotate(raw,cv2.ROTATE_90_CLOCKWISE)
        return raw

    def read(self):
        checkpoint(self.cancelled)
        try:
            if self.pending is not None:raw,t,d=self.pending;self.pending=None
            elif self.container:raw,t,d=next(self.decoder)
            else:
                for _ in range(self.sample_step-1):
                    if not self.cap.grab():break
                ok,raw=self.cap.read()
                if not ok:self.eof=True;return False,None
                t=self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if t<self.last_timestamp or (self.pos and t==0):t=self.pos/self.fps
                d=1/self.fps
        except (StopIteration,av.error.FFmpegError) as error:
            if isinstance(error,av.error.FFmpegError):self.errors+=1
            self.eof=True
            if self.container and self._recover_enabled and not self._fallback_used and self.pos+2<self.reported and self.last_timestamp+1/self.fps<self.duration-.1:
                self._fallback_used=True
                alternate=cv2.VideoCapture(self.path)
                if alternate.isOpened():
                    alternate.set(cv2.CAP_PROP_POS_MSEC,(self.last_timestamp+1/self.fps)*1000)
                    ok,raw=alternate.read()
                    timestamp=alternate.get(cv2.CAP_PROP_POS_MSEC)/1000
                    if ok and timestamp>self.last_timestamp:
                        self.container.close();self.container=None;self.cap=alternate
                        self.backend='PyAV then OpenCV recovery';self.eof=False;self.pending=(raw,timestamp,1/self.fps)
                        return self.read()
                alternate.release()
            return False,None
        output_size=(512,288) if self.model_input else (self.width,self.height)
        if raw.shape[1::-1]!=output_size:
            raw=cv2.resize(raw,output_size)
            if not self.model_input and 'resolution_normalized' not in self.warnings:self.warnings.append('resolution_normalized')
        self.last_timestamp=t;self.last_duration=d
        if self.pos==len(self.times):self.times.append(t)
        self.pos+=1;self.eof=False;return True,raw

    def get(self,key):
        return {cv2.CAP_PROP_FPS:self.fps,cv2.CAP_PROP_FRAME_COUNT:len(self.exact_times) if self.exact_times is not None else self.reported,
                cv2.CAP_PROP_FRAME_WIDTH:self.width,cv2.CAP_PROP_FRAME_HEIGHT:self.height,
                cv2.CAP_PROP_POS_FRAMES:self.pos,cv2.CAP_PROP_POS_MSEC:self.last_timestamp*1000}.get(key,0)

    def set_timeline(self,times):self.exact_times=list(times)

    def set(self,key,value):
        if key!=cv2.CAP_PROP_POS_FRAMES:return False
        self._recover_enabled=False
        index=max(0,int(value));target=self.exact_times[index] if self.exact_times is not None and index<len(self.exact_times) else index/self.fps
        if self.container:
            self.container.seek(int((target+self.origin)/self.stream.time_base),stream=self.stream,backward=True)
            self.decoder=self._frames();self.pending=None
            for raw,t,d in self.decoder:
                if t>=target-1e-6:self.pending=(raw,t,d);self.pos=index;self.eof=False;return True
            return False
        self.pending=None;success=self.cap.set(key,index);self.pos=index;return success

    def isOpened(self):return self.container is not None or (self.cap is not None and self.cap.isOpened())

    def info(self):
        actual=len(self.times);end=self.last_timestamp+self.last_duration
        partial=bool(self.duration and end<self.duration-max(.1,3/self.fps))
        return dict(backend=self.backend,reported_frames=self.reported,decoded_frames=actual,
            packet_errors=self.errors,complete=not partial and not self.errors,
            status='partial' if partial else ('recovered' if self.errors else ('metadata_adjusted' if actual!=self.reported else 'complete')),
            duration_seconds=end,container_duration_seconds=self.duration,warnings=self.warnings)

    def release(self):
        if self.container:self.container.close();self.container=None
        if self.cap:self.cap.release();self.cap=None


class VideoWriter:
    """H.264 output preserving source presentation times, including VFR."""
    def __init__(self,path,fps,size):
        self.container=av.open(str(path),'w',options={'movflags':'+faststart'})
        self.stream=self.container.add_stream('libx264',rate=Fraction(float(fps)).limit_denominator(100000),options={'preset':'veryfast','crf':'22'})
        self.size=tuple((int(n)+1)//2*2 for n in size)
        self.stream.width,self.stream.height=self.size;self.stream.pix_fmt='yuv420p'
        self.time_base=Fraction(1,60000);self.stream.codec_context.time_base=self.time_base;self.stream.time_base=self.time_base
        self.stream.codec_context.max_b_frames=0;self.stream.codec_context.thread_count=2
        self.fps=fps;self.index=0;self.last_pts=-1;self.closed=False

    def isOpened(self):return not self.closed

    def write(self,image,timestamp=None):
        if image.shape[1]!=self.size[0] or image.shape[0]!=self.size[1]:
            image=cv2.copyMakeBorder(image,0,self.size[1]-image.shape[0],0,self.size[0]-image.shape[1],cv2.BORDER_CONSTANT,value=0)
        frame=av.VideoFrame.from_ndarray(np.ascontiguousarray(image),format='bgr24')
        pts=round((self.index/self.fps if timestamp is None else timestamp)/float(self.time_base))
        frame.pts=max(self.last_pts+1,pts);frame.time_base=self.time_base;self.last_pts=frame.pts
        for packet in self.stream.encode(frame):self.container.mux(packet)
        self.index+=1

    def release(self):
        if not self.closed:
            try:
                for packet in self.stream.encode():self.container.mux(packet)
            finally:self.container.close();self.closed=True
