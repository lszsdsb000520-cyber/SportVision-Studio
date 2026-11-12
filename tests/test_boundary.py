import cv2
import numpy as np
import pytest
from pingpong.boundary import corners_from_mask,order_corners,refine_border_lines,color_supported_surface,TableBoundaryDetector


def test_table_mask_produces_four_surface_corners():
    mask=np.zeros((320,640),np.float32)
    cv2.fillConvexPoly(mask,np.array([[190,140],[430,150],[500,240],[130,235]],np.int32),.99)
    result=corners_from_mask(mask,(1080,1920,3))
    corners=np.array(result['corners'])
    assert corners.shape==(4,2) and result['mask_quad_iou']>.9
    assert corners[:2,1].mean()<corners[2:,1].mean()


def test_no_table_does_not_return_a_fake_rectangle():
    with pytest.raises(ValueError):corners_from_mask(np.zeros((320,640),np.float32),(1080,1920,3))


def test_surface_edge_refinement_stays_near_mask_and_handles_hough_shape():
    frame=np.zeros((480,640,3),np.uint8)
    points=np.array([[200,200],[440,200],[520,360],[120,360]],np.int32)
    cv2.fillConvexPoly(frame,points,(80,110,70));cv2.polylines(frame,[points],True,(240,240,240),3)
    result=np.array(refine_border_lines(frame,order_corners(points)))
    assert result.shape==(4,2)
    assert np.max(np.linalg.norm(result-order_corners(points),axis=1))<20


def test_missing_interior_and_split_net_recover_from_local_image_evidence():
    frame=np.zeros((320,640,3),np.uint8)
    quad=np.array([[190,180],[450,180],[510,245],[130,245]],np.int32)
    cv2.fillConvexPoly(frame,quad,(130,15,5))
    surface=np.zeros((320,640),np.float32);cv2.fillConvexPoly(surface,quad,.99)
    # Large internal gaps reproduce incomplete model masks without deleting pixels.
    surface[182:226,205:430]=0
    surface[:,315:325]=0
    with pytest.raises(ValueError):corners_from_mask(surface,frame.shape)
    result=color_supported_surface(frame,surface)
    assert result['mask_quad_iou']>.9
    assert np.max(np.linalg.norm(np.array(result['corners'])-order_corners(quad),axis=1))<4


def test_color_recovery_requires_model_support():
    frame=np.full((320,640,3),(130,15,5),np.uint8)
    with pytest.raises(ValueError):color_supported_surface(frame,np.zeros((320,640),np.float32))


def test_nearby_frames_do_not_consume_caller_frame_and_need_consensus(monkeypatch):
    import pingpong.video_io as video_io
    released=[]
    class Reader:
        def __init__(self,*args):pass
        def set(self,*args):return True
        def read(self):return True,np.full((320,640,3),1,np.uint8)
        def release(self):released.append(True)
    monkeypatch.setattr(video_io,'VideoReader',Reader)
    class Detector:
        detect_clip=TableBoundaryDetector.detect_clip
        def detect(self,frame):
            if not frame.any():raise ValueError('Incomplete first-frame mask')
            return dict(corners=[[100,100],[500,100],[550,250],[50,250]])
    frame=np.zeros((320,640,3),np.uint8)
    result=Detector().detect_clip(frame,'unused',0,120)
    assert len(result['verified_frames'])==3 and not frame.any() and released==[True]


def test_valid_current_boundary_is_not_replaced_by_different_later_view(monkeypatch):
    import pingpong.video_io as video_io
    class Reader:
        def __init__(self,*args):pass
        def set(self,*args):return True
        def read(self):return True,np.ones((320,640,3),np.uint8)
        def release(self):pass
    monkeypatch.setattr(video_io,'VideoReader',Reader)
    current=[[100,100],[500,100],[550,250],[50,250]]
    class Detector:
        detect_clip=TableBoundaryDetector.detect_clip
        def detect(self,frame):return dict(corners=(np.array(current)+([80,0] if frame.any() else [0,0])).tolist())
    result=Detector().detect_clip(np.zeros((320,640,3),np.uint8),'unused',0,120)
    assert result['corners']==current and result['verified_frames']==[0] and result['review_required']
