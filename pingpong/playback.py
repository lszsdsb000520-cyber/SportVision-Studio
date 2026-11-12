"""Bounded background playback; analysis and frame-accurate seeking stay independent."""
from bisect import bisect_left
from collections import deque
import threading
import av,cv2
from .video_io import VideoReader


class PlaybackDecoder:
    def __init__(self,path,start_frame,fps,times=None,speed=1.,capacity=8,size=None):
        self.path=path;self.start_frame=start_frame;self.fps=fps;self.times=times
        self.start_time=times[start_frame] if times else start_frame/fps
        self.step=max(1e-6,speed/30.);self.capacity=capacity;self.size=size
        self.queue=deque();self.condition=threading.Condition();self.stopped=threading.Event()
        self.ready=threading.Event();self.done=False;self.error=None
        self.decoded_frames=0;self.converted_frames=0;self.max_queue=0
        self.thread=threading.Thread(target=self._run,daemon=True,name='video-playback');self.thread.start()

    def _put(self,index,timestamp,image):
        with self.condition:
            while len(self.queue)>=self.capacity and not self.stopped.is_set():self.condition.wait(.05)
            if self.stopped.is_set():return
            self.queue.append((index,timestamp,image));self.converted_frames+=1
            self.max_queue=max(self.max_queue,len(self.queue));self.ready.set()

    def pop_due(self,timestamp):
        latest=None
        with self.condition:
            while self.queue and self.queue[0][1]<=timestamp+1e-6:latest=self.queue.popleft()
            self.condition.notify_all()
        return latest

    def empty(self):
        with self.condition:return not self.queue

    def stop(self):
        self.stopped.set()
        with self.condition:self.condition.notify_all()

    def _native(self):
        container=av.open(str(self.path))
        try:
            stream=container.streams.video[0];stream.thread_type='AUTO';stream.codec_context.thread_count=4
            origin=float((stream.start_time or 0)*stream.time_base)
            container.seek(int((self.start_time+origin)/stream.time_base),stream=stream,backward=True)
            next_time=self.start_time+self.step;last=None;emitted=self.start_frame;rotation=0
            for frame in container.decode(stream):
                if self.stopped.is_set():return
                self.decoded_frames+=1
                timestamp=float(frame.pts*stream.time_base)-origin if frame.pts is not None else self.decoded_frames/self.fps
                index=bisect_left(self.times,timestamp-1e-6) if self.times else round(timestamp*self.fps)
                if index<=self.start_frame:continue
                if self.times and index>=len(self.times):break
                value=int(frame.rotation or 0)
                if value or any(side.type.name=='DISPLAYMATRIX' for side in frame.side_data):rotation=value
                last=(frame,index,timestamp,rotation)
                if timestamp+1e-6<next_time:continue
                self._put(index,timestamp,self._image(frame,rotation));emitted=index
                next_time=timestamp+self.step
            if last and last[1]>emitted and not self.stopped.is_set():
                frame,index,timestamp,rotation=last;self._put(index,timestamp,self._image(frame,rotation))
        finally:container.close()

    def _image(self,frame,rotation):
        image=frame.to_ndarray(format='bgr24');rotation%=360
        if rotation==90:image=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation==180:image=cv2.rotate(image,cv2.ROTATE_180)
        elif rotation==270:image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
        if self.size and image.shape[1::-1]!=self.size:image=cv2.resize(image,self.size)
        return image

    def _run(self):
        try:
            try:self._native()
            except (av.error.FFmpegError,ValueError,IndexError):
                if self.converted_frames:raise
                reader=VideoReader(self.path,self.stopped.is_set)
                try:
                    if self.times:reader.set_timeline(self.times)
                    if not reader.set(cv2.CAP_PROP_POS_FRAMES,self.start_frame+1):return
                    next_time=self.start_time+self.step
                    while not self.stopped.is_set():
                        ok,image=reader.read()
                        if not ok:break
                        self.decoded_frames+=1;timestamp=reader.last_timestamp
                        if timestamp+1e-6>=next_time:
                            self._put(int(reader.get(cv2.CAP_PROP_POS_FRAMES))-1,timestamp,image);next_time=timestamp+self.step
                finally:reader.release()
        except InterruptedError:pass
        except Exception as error:self.error=str(error)
        finally:self.done=True;self.ready.set()
