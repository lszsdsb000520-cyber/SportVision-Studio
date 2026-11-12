import time
import cv2,numpy as np
from pingpong.playback import PlaybackDecoder
from pingpong.video_io import VideoReader,VideoWriter


def clip(path,count=80,fps=120):
    writer=VideoWriter(path,fps,(160,96))
    for i in range(count):writer.write(np.full((96,160,3),i,np.uint8))
    writer.release();return [i/fps for i in range(count)]


def test_buffer_is_bounded_and_stop_wakes_waiting_decoder(tmp_path):
    path=tmp_path/'clip.mp4';times=clip(path)
    worker=PlaybackDecoder(path,0,120,times,speed=.25,capacity=3)
    assert worker.ready.wait(2)
    time.sleep(.05);worker.stop();worker.thread.join(2)
    assert not worker.thread.is_alive() and worker.max_queue<=3 and not worker.error


def test_playback_seeks_by_timestamps_and_keeps_original_pixels(tmp_path):
    path=tmp_path/'clip.mp4';times=clip(path)
    worker=PlaybackDecoder(path,37,120,times,speed=1.,capacity=80)
    worker.thread.join(3);assert worker.done and not worker.error
    item=worker.pop_due(times[50]);assert item is not None
    index,timestamp,image=item
    assert 37<index<=50 and timestamp==times[index]
    reader=VideoReader(path);reader.set_timeline(times);assert reader.set(cv2.CAP_PROP_POS_FRAMES,index)
    assert np.array_equal(image,reader.read()[1]);reader.release()
    last=worker.pop_due(100);assert last[0]==79 and worker.empty()


def test_variable_rate_slow_motion_and_tail_are_not_truncated(tmp_path):
    path=tmp_path/'vfr.mp4';times=[0.,.01,.05,.08,.15,.18]
    writer=VideoWriter(path,25,(160,96))
    for i,t in enumerate(times):writer.write(np.full((96,160,3),i*20,np.uint8),t)
    writer.release();worker=PlaybackDecoder(path,0,25,times,speed=.25)
    worker.thread.join(3);assert worker.done and not worker.error
    assert worker.pop_due(.01)[0]==1
    assert worker.pop_due(.05)[0]==2
    assert worker.pop_due(1)[0]==5
