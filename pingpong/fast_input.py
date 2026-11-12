"""Optional exact-pixel decoder fast path for the tested 1080p YUV420 format."""
import ctypes
from functools import lru_cache
import sys

import av
import cv2
import numpy as np
from av.video.reformatter import VideoReformatter

from .paths import asset_path


@lru_cache(maxsize=1)
def native_library():
    if sys.platform != 'win32':
        return None
    try:
        return ctypes.CDLL(str(asset_path('assets/fast_input.dll')))
    except OSError:
        return None


@lru_cache(maxsize=1)
def sampler():
    function = getattr(native_library(),'sample_yuv420',None)
    if function is None:return None
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    function.argtypes = [pointer]*3+[integer]*3+[pointer]*5+[integer]*2+[pointer]
    function.restype = None
    return function


@lru_cache(maxsize=1)
def normalizer():
    function = getattr(native_library(),'normalize_bgr',None)
    if function is None:return None
    function.argtypes = [ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*4
    function.restype = None
    return function


@lru_cache(maxsize=2)
def color_table(colorspace, color_range):
    """Convert every possible 8-bit YUV color with FFmpeg's native converter."""
    frame = av.VideoFrame(4096,4096,'yuv420p')
    frame.colorspace = colorspace
    frame.color_range = color_range
    # Four consecutive luma levels share a chroma cell.
    luma = (np.arange(4096)//2%64)*4 + np.arange(4096)%2
    y, u, v = [np.frombuffer(p,np.uint8).reshape(p.height,p.line_size)[:,:p.width] for p in frame.planes]
    y[::2] = luma
    y[1::2] = luma+2
    chroma = np.arange(2048*2048,dtype=np.int32).reshape(2048,2048)//64
    u[:], v[:] = chroma//256, chroma%256
    bgr = VideoReformatter().reformat(frame,format='bgr24',threads=2).to_ndarray()
    table = np.empty(bgr.size+1,np.uint8)
    table[:-1].reshape(2048,2048,2,2,3)[:] = bgr.reshape(2048,2,2048,2,3).transpose(0,2,1,3,4)
    table[-1] = 0
    return table


def axis_coordinates(source, destination):
    positions = (np.arange(destination)+.5)*source/destination-.5
    whole = np.floor(positions).astype(np.int32)
    return whole, np.rint((positions-whole)*2048).astype(np.int32)


class ModelInputSampler:
    def __init__(self):
        self.function = sampler()
        self.enabled = self.function is not None
        self.axes = None
        self.verified = 0
        self.color = None

    def resize(self, frame, rotation):
        if not self.enabled or rotation or frame.format.name!='yuv420p' or (frame.width,frame.height)!=(1920,1080):
            return None
        if self.axes is None:
            xs, wx = axis_coordinates(1920,512)
            ys, wy = axis_coordinates(1080,288)
            self.axes = xs,ys,wx,wy
        color = frame.colorspace,frame.color_range
        if self.color!=color:
            self.verified=0
            self.color=color
        try:
            table = color_table(*color)
        except (ValueError, av.error.FFmpegError, MemoryError):
            self.enabled = False
            return None
        output = np.empty((288,512,3),np.uint8)
        planes = frame.planes
        self.function(*(plane.buffer_ptr for plane in planes),*(plane.line_size for plane in planes),
                      table.ctypes.data,*(axis.ctypes.data for axis in self.axes),512,288,output.ctypes.data)
        if self.verified<3:
            full = cv2.resize(frame.to_ndarray(format='bgr24'),(512,288))
            if not np.array_equal(output,full):
                self.enabled = False
                return full
            self.verified += 1
        return output
