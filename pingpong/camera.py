"""Keep Windows capture isolated from inference and recover blocked drivers."""
import multiprocessing as mp
import time

import cv2
import numpy as np

from .cancellation import checkpoint


MAX_BYTES = 3840 * 2160 * 3
BACKENDS = (cv2.CAP_MSMF, cv2.CAP_DSHOW)


def publish_frame(pixels, meta, lock, frame, fps):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or frame.nbytes > len(pixels):
        return False
    if lock.acquire(timeout=.1):
        try:
            np.frombuffer(pixels, np.uint8, count=frame.nbytes)[:] = frame.ravel()
            meta[:] = (meta[0] + 1, frame.shape[0], frame.shape[1], time.perf_counter(), fps)
        finally:
            lock.release()
    return True


def capture_device(index, backend, stopped, pixels, meta, lock):
    """All native device operations happen in this process and on one thread."""
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(index, backend)
    try:
        if not capture.isOpened():
            return
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 <= fps <= 240 else 30.
        misses = 0
        while not stopped.value:
            ok, frame = capture.read()
            if not ok or frame is None:
                misses += 1
                if misses >= 3:
                    return
                time.sleep(.05)
                continue
            misses = 0
            if not publish_frame(pixels, meta, lock, frame, fps):
                return
    finally:
        capture.release()


class CameraReader:
    def __init__(self, index, cancelled=None, *, capture_target=capture_device, startup_timeout=6., stall_timeout=2.):
        self.index = index
        self.cancelled = cancelled
        self.capture_target = capture_target
        self.startup_timeout = startup_timeout
        self.stall_timeout = stall_timeout
        self.context = mp.get_context('spawn')
        self.process = None
        self.channel = None
        self.status = '正在打开摄像头…'
        self.fps = 30.
        self.started = time.perf_counter()
        self.last_timestamp = 0.
        self.sample_step = 1
        self.frame_shape = None
        self.last_key = None
        self.generation = 0
        self.failures = 0
        self.last_arrival = None
        self._open()

    def _open(self):
        ctx = self.context
        self.generation += 1
        backend = BACKENDS[(self.generation - 1) % len(BACKENDS)]
        stopped = ctx.RawValue('b', 0)
        pixels = ctx.RawArray('B', MAX_BYTES)
        meta = ctx.RawArray('d', 5)
        lock = ctx.Lock()
        process = ctx.Process(target=self.capture_target,
                              args=(self.index, backend, stopped, pixels, meta, lock),
                              daemon=True)
        process.start()
        self.process = process
        self.stopped = stopped
        self.channel = (self.generation, pixels, meta, lock)
        self.opened = time.perf_counter()
        self.last_arrival = None

    def preview(self, previous=None, step=1):
        """Read only a fresh frame; never block on a driver or wait on the UI thread."""
        channel = self.channel
        if channel is None:
            return None
        generation, pixels, meta, lock = channel
        if not lock.acquire(False):
            return None
        try:
            sequence, height, width, captured, fps = meta[:]
            key = generation, int(sequence)
            if not sequence or (previous and generation == previous[0] and key[1] < previous[1] + step):
                return None
            shape = int(height), int(width), 3
            frame = np.frombuffer(pixels, np.uint8, count=shape[0]*shape[1]*3).copy().reshape(shape)
            return key, frame, captured, fps
        finally:
            lock.release()

    def read(self):
        while True:
            checkpoint(self.cancelled)
            sample = self.preview(self.last_key, self.sample_step)
            now = time.perf_counter()
            if sample is not None and now - sample[2] < self.stall_timeout:
                self.last_key, frame, captured, self.fps = sample
                if self.frame_shape is None:
                    self.frame_shape = frame.shape
                elif frame.shape != self.frame_shape:
                    self.status = '摄像头分辨率已变化，请重新开始监测并标台。'
                    raise ValueError(self.status)
                if captured - self.opened >= 5.:
                    self.failures = 0
                self.last_arrival = captured
                self.last_timestamp = max(0., captured - self.started)
                self.status = '摄像头采集中'
                return True, frame
            deadline = (self.opened + self.startup_timeout if self.last_arrival is None
                        else self.last_arrival + self.stall_timeout)
            if not self.process.is_alive() or now >= deadline:
                self.failures += 1
                self.status = '摄像头取帧中断，正在切换驱动重连…'
                self.release()
                if self.failures >= 3:
                    self.status = '摄像头取帧超时。请检查设备连接及是否被其他程序占用，再重新开始。'
                    raise ValueError(self.status)
                checkpoint(self.cancelled)
                self._open()
            time.sleep(.01)

    def release(self):
        self.channel = None
        process, self.process = self.process, None
        if process is None:
            return
        self.stopped.value = 1
        process.join(.25)
        if process.is_alive():
            process.terminate()
            process.join(1.)
        if not process.is_alive():
            process.close()
