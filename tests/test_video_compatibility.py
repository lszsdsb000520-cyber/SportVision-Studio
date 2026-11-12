from pathlib import Path
from fractions import Fraction
import av,cv2,numpy as np,pytest
from pingpong.video_io import VideoReader,VideoWriter
import pingpong.detector as pipeline


@pytest.mark.parametrize('codec,extension,pixfmt',[('libx264','.mp4','yuv420p'),('libx265','.mp4','yuv420p'),('ffv1','.mkv','yuv420p'),('mjpeg','.avi','yuvj420p'),('libvpx-vp9','.webm','yuv420p')])
def test_common_encodings_are_decoded(codec,extension,pixfmt,tmp_path):
    if codec not in av.codecs_available:pytest.skip('Encoder unavailable')
    options={'x265-params':'pools=1:frame-threads=1:log-level=error'} if codec=='libx265' else {}
    path=tmp_path/('clip'+extension);container=av.open(str(path),'w');stream=container.add_stream(codec,rate=25,options=options)
    stream.width=128;stream.height=96;stream.pix_fmt=pixfmt;stream.codec_context.thread_count=1
    for i in range(5):
        frame=av.VideoFrame.from_ndarray(np.full((96,128,3),40+i*10,np.uint8),format='bgr24')
        for packet in stream.encode(frame):container.mux(packet)
    for packet in stream.encode():container.mux(packet)
    container.close();reader=VideoReader(path);frames=[]
    while True:
        ok,image=reader.read()
        if not ok:break
        frames.append(image)
    assert len(frames)==5 and frames[0].shape==(96,128,3)
    reader.release()


def test_variable_timestamps_and_seek_are_preserved(tmp_path):
    path=tmp_path/'vfr.mp4';writer=VideoWriter(path,25,(127,95));times=[0.,.04,.10,.18,.20]
    for i,t in enumerate(times):writer.write(np.full((95,127,3),i*30,np.uint8),t)
    writer.release();reader=VideoReader(path);decoded=[]
    while True:
        ok,frame=reader.read()
        if not ok:break
        decoded.append(reader.last_timestamp)
    assert np.allclose(decoded,times,atol=1e-4)
    reader.set_timeline(times);assert reader.set(cv2.CAP_PROP_POS_FRAMES,3)
    assert reader.read()[0] and abs(reader.last_timestamp-.18)<1e-4
    reader.release()


def test_cancel_at_final_partial_batch_discards_analysis(tmp_path,monkeypatch):
    path=tmp_path/'last.mp4';writer=VideoWriter(path,25,(160,96))
    for i in range(7):writer.write(np.full((96,160,3),i*10,np.uint8))
    writer.release();cancelled=False
    class Fake:
        provider='test';backend_note=''
        def __init__(self,*args,**kwargs):pass
        def predict(self,frames,threshold):return [(256.,144.,.9)]*3
    monkeypatch.setattr(pipeline,'Detector',Fake)
    def progress(done,total,*args):
        nonlocal cancelled
        if done==7:cancelled=True
    assert pipeline.analyze_video(path,'unused',progress=progress,cancelled=lambda:cancelled) is None


def test_display_rotation_is_applied(tmp_path):
    path=tmp_path/'rotated.mp4';container=av.open(str(path),'w');stream=container.add_stream('libx264',rate=25)
    stream.width=128;stream.height=96;stream.pix_fmt='yuv420p';stream.set_display_rotation(90)
    raw=np.zeros((96,128,3),np.uint8);raw[:24,:24]=(0,0,255)
    for packet in stream.encode(av.VideoFrame.from_ndarray(raw,format='bgr24')):container.mux(packet)
    for packet in stream.encode():container.mux(packet)
    container.close();reader=VideoReader(path);ok,image=reader.read();reader.release()
    assert ok and image.shape==(128,96,3) and image[-10,10,2]>150
