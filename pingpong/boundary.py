"""Table segmentation and ordered surface corners, independent of ball detection."""
from pathlib import Path
import cv2
import numpy as np
from .analysis import homography
from .paths import app_root,asset_path
from .cancellation import checkpoint
from .gpu import preferred_adapter


def table_model_path():
    local=app_root()/'models/table.onnx'
    return local if local.exists() else asset_path('models/table.onnx')


def order_corners(quad):
    """Choose opposite short end-lines, then order far-left to near-left."""
    quad=np.asarray(quad,np.float32).reshape(4,2)
    center=quad.mean(axis=0)
    quad=quad[np.argsort(np.arctan2(quad[:,1]-center[1],quad[:,0]-center[0]))]
    lengths=np.linalg.norm(np.roll(quad,-1,axis=0)-quad,axis=1)
    pair=0 if lengths[0]+lengths[2]<=lengths[1]+lengths[3] else 1
    ends=[(pair,(pair+1)%4),((pair+2)%4,(pair+3)%4)]
    far=min(ends,key=lambda e:(float(quad[list(e),1].mean()),float(quad[list(e),0].mean())))
    first,second=sorted(far,key=lambda i:(float(quad[i,0]),float(quad[i,1])))
    step=1 if (first+1)%4==second else -1
    ordered=np.array([quad[(first+i*step)%4] for i in range(4)],np.float32)
    homography(ordered)
    return ordered


def corners_from_mask(probability,image_shape):
    """Fit a convex quadrilateral; reject incomplete or implausible masks."""
    h,w=probability.shape
    mask=(probability>.8).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=[c for c in contours if cv2.contourArea(c)>w*h*.004]
    if not contours:raise ValueError('没有找到完整球台，请换一帧重试，或手动选择四个角点。')
    contour=max(contours,key=cv2.contourArea)
    hull=cv2.convexHull(contour);perimeter=cv2.arcLength(hull,True)
    quad=None
    for ratio in (.004,.006,.008,.01,.015,.02,.03,.04,.06,.08):
        approx=cv2.approxPolyDP(hull,perimeter*ratio,True)
        if len(approx)==4:
            quad=approx.reshape(4,2).astype(np.float32);break
    if quad is None:raise ValueError('球台轮廓被遮挡或边界不完整，请换一帧或手动标定。')
    area=abs(cv2.contourArea(quad))
    if area<w*h*.004 or area>w*h*.65:raise ValueError('自动边界形状不可靠，请手动标定。')
    poly=np.zeros_like(mask);cv2.fillConvexPoly(poly,quad.astype(np.int32),255)
    overlap=np.count_nonzero((poly>0)&(mask>0))/max(1,np.count_nonzero((poly>0)|(mask>0)))
    if overlap<.68:raise ValueError('自动边界与台面不一致，请换一帧重试。')
    ih,iw=image_shape[:2]
    corners=order_corners(quad*np.array([iw/w,ih/h],np.float32)).tolist()
    return dict(corners=corners,mask_quad_iou=float(overlap),method='TT3D table segmentation + convex surface fit')


def color_supported_surface(frame,probability):
    """Recover missing table interior only near high-confidence model evidence."""
    h,w=probability.shape
    hsv=cv2.cvtColor(cv2.resize(frame,(w,h)),cv2.COLOR_BGR2HSV)
    seed=(probability>.8).astype(np.uint8)
    contours,_=cv2.findContours(seed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=[c for c in contours if cv2.contourArea(c)>w*h*.004]
    if not contours:raise ValueError('没有足够的球台证据，保留原标定，请手动选择四角。')
    # Both halves may be separated by the net; retain significant seed components.
    support=np.zeros_like(seed)
    cv2.drawContours(support,contours,-1,1,cv2.FILLED)
    hull=cv2.convexHull(np.concatenate(contours));region=np.zeros_like(seed)
    cv2.fillConvexPoly(region,hull,1)
    region=cv2.dilate(region,np.ones((25,25),np.uint8))
    samples=hsv[(support>0)&(hsv[:,:,1]>60)]
    if len(samples)<200:raise ValueError('台面颜色证据不足，请手动标定。')
    histogram=np.bincount(samples[:,0],minlength=180)
    hue=int(np.argmax(sum(np.roll(histogram,i) for i in range(-4,5))))
    distance=np.abs(samples[:,0].astype(float)-hue)
    samples=samples[np.minimum(distance,180-distance)<=12]
    low=np.percentile(samples,5,axis=0);high=np.percentile(samples,95,axis=0)
    distance=np.abs(hsv[:,:,0].astype(float)-hue);distance=np.minimum(distance,180-distance)
    mask=((distance<=12)&(hsv[:,:,1]>=max(40,low[1]-40))&
          (hsv[:,:,2]>=max(20,low[2]-45))&(hsv[:,:,2]<=min(255,high[2]+60))&(region>0)).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    surface=np.zeros_like(seed)
    for label in range(1,count):
        if stats[label,cv2.CC_STAT_AREA]>w*h*.002 and np.count_nonzero((labels==label)&(support>0))>100:
            surface[labels==label]=1
    contours,_=cv2.findContours(surface,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:raise ValueError('未找到与模型一致的台面区域，请手动标定。')
    hull=cv2.convexHull(np.concatenate(contours));perimeter=cv2.arcLength(hull,True)
    quad=None
    for ratio in (.004,.006,.008,.01,.015,.02,.03,.04):
        candidate=cv2.approxPolyDP(hull,perimeter*ratio,True)
        if len(candidate)==4:quad=candidate.reshape(4,2);break
    if quad is None:raise ValueError('台面遮挡过多，无法可靠拟合四角，请手动标定。')
    poly=np.zeros_like(seed);cv2.fillConvexPoly(poly,quad,1)
    area=np.count_nonzero(poly);model_area=np.count_nonzero(support)
    overlap=np.count_nonzero((poly>0)&(surface>0))/max(1,np.count_nonzero((poly>0)|(surface>0)))
    coverage=np.count_nonzero((poly>0)&(support>0))/max(1,model_area)
    if not w*h*.004<area<w*h*.65 or area>model_area*3.5 or overlap<.8 or coverage<.65:
        raise ValueError('颜色区域与球台证据不一致，保留原标定，请手动选择四角。')
    ih,iw=frame.shape[:2]
    corners=order_corners(quad*np.array([iw/w,ih/h],np.float32)).tolist()
    return dict(corners=corners,mask_quad_iou=float(overlap),model_seed_coverage=float(coverage),
                method='TT3D seeds + local color-supported surface + border lines')


class TableBoundaryDetector:
    """Reusable single-frame detector for future camera pipelines."""
    def __init__(self,model_path=None,acceleration='auto'):
        import onnxruntime as ort
        path=Path(model_path or table_model_path())
        if not path.exists():raise FileNotFoundError('未找到球台模型 table.onnx，请使用完整软件包。')
        opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
        providers=['CPUExecutionProvider']
        if acceleration!='cpu' and 'DmlExecutionProvider' in ort.get_available_providers():
            opts.enable_mem_pattern=False;opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            device=int(acceleration) if acceleration in ('0','1') else preferred_adapter()
            providers=[('DmlExecutionProvider',{'device_id':device}),'CPUExecutionProvider']
        self.session=ort.InferenceSession(str(path),opts,providers=providers)

    def detect(self,frame):
        rgb=cv2.cvtColor(cv2.resize(frame,(640,320)),cv2.COLOR_BGR2RGB)
        tensor=rgb.transpose(2,0,1).astype(np.float32)[None]
        logits=self.session.run(None,{'image':tensor})[0][0,0]
        probability=1/(1+np.exp(-np.clip(logits,-40,40)))
        try:result=corners_from_mask(probability,frame.shape)
        except ValueError as original:
            try:result=color_supported_surface(frame,probability)
            except ValueError as recovery:raise ValueError(str(original)+' '+str(recovery)) from recovery
        result['corners']=refine_border_lines(frame,result['corners'])
        return result

    def detect_clip(self,frame,video=None,frame_index=0,fps=30.,cancelled=None,fixed_camera=False):
        """Check nearby frames through a private reader, never consuming preview frames."""
        from .video_io import VideoReader
        checkpoint(cancelled);results=[];errors=[]
        def attempt(image,index):
            checkpoint(cancelled)
            try:
                result=self.detect(image);result['sample_frame']=index;results.append(result)
            except ValueError as error:errors.append(str(error))
        attempt(frame,frame_index)
        limit=max(12.,np.hypot(*frame.shape[:2])*.012)
        reference=cv2.resize(frame,(80,45),interpolation=cv2.INTER_AREA).astype(np.int16)
        if video:
            reader=VideoReader(video,cancelled,parallel=True) if fixed_camera else VideoReader(video,cancelled)
            try:
                offsets=(.25,.5,1.,1.5) if fixed_camera else (.25,.5,1.)
                for offset in offsets:
                    checkpoint(cancelled);index=frame_index+max(1,round(fps*offset))
                    if not reader.set(cv2.CAP_PROP_POS_FRAMES,index):continue
                    ok,image=reader.read()
                    if ok and image.shape==frame.shape:
                        thumbnail=cv2.resize(image,(80,45),interpolation=cv2.INTER_AREA).astype(np.int16)
                        if not fixed_camera or np.mean(np.abs(thumbnail-reference))<14:
                            attempt(image,index)
                    if len(results)>=3 and (not fixed_camera or results[0].get('mask_quad_iou',0)>=.8):break
            finally:reader.release()
        checkpoint(cancelled)
        if not results:raise ValueError('当前帧及附近帧未能可靠识别球台，视频帧未删除。可继续手动标定。 '+(errors[0] if errors else ''))
        groups=[[other for other in results if np.max(np.linalg.norm(np.array(item['corners'])-np.array(other['corners']),axis=1))<=limit] for item in results]
        group=groups[0] if results[0]['sample_frame']==frame_index else max(groups,key=len)
        if fixed_camera and len(group)<2:
            stable=max(groups,key=len)
            if len(stable)>=2:group=stable
        if results[0]['sample_frame']!=frame_index and len(group)<2:
            raise ValueError('附近帧的球台边界不稳定，保留当前画面与原标定，请手动选择四角。')
        # Preserve a real detected boundary rather than averaging across camera motion.
        selected=next((item for item in group if item['sample_frame']==frame_index),
                      max(group,key=lambda item:item.get('mask_quad_iou',0)))
        return {**selected,'verified_frames':[item['sample_frame'] for item in group],
                'attempted_frame_count':len(results)+len(errors),'review_required':len(group)<2}


def refine_border_lines(frame,corners):
    """Use nearby straight image edges to correct inward segmentation boundaries."""
    ih,iw=frame.shape[:2];scale=min(1.,960/iw)
    image=cv2.resize(frame,None,fx=scale,fy=scale)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,45,120)
    lines=cv2.HoughLinesP(edges,1,np.pi/720,25,minLineLength=22,maxLineGap=12)
    if lines is None:return corners
    quad=np.array(corners,float)*scale;fitted=[]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    for a,b in zip(quad,np.roll(quad,-1,axis=0)):
        direction=b-a;length=np.linalg.norm(direction);unit=direction/length
        candidates=[]
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            start=np.array([x1,y1],float);end=np.array([x2,y2],float);vector=end-start
            size=np.linalg.norm(vector)
            if size<length*.16:continue
            angle=abs(float(np.dot(vector/size,unit)))
            if angle<.985:continue
            mid=(start+end)/2;project=np.dot(mid-a,unit)/length
            offset=mid-a;distance=abs(float(unit[0]*offset[1]-unit[1]*offset[0]))
            if not -.08<=project<=1.08 or distance>15:continue
            samples=np.linspace(start,end,24).round().astype(int)
            samples[:,0]=np.clip(samples[:,0],0,image.shape[1]-1);samples[:,1]=np.clip(samples[:,1],0,image.shape[0]-1)
            pixels=hsv[samples[:,1],samples[:,0]]
            white=np.mean((pixels[:,2]>125)&(pixels[:,1]<110))
            score=size*(.65+.65*white)*angle/(1+distance*.025)
            candidates.append((score,start,end))
        if not candidates:return corners
        _,start,end=max(candidates,key=lambda c:c[0]);fitted.append((start,end-start))
    refined=[]
    for i in range(4):
        p,u=fitted[(i-1)%4];q,v=fitted[i]
        matrix=np.column_stack([u,-v])
        if abs(np.linalg.det(matrix))<1e-4:return corners
        tau=np.linalg.solve(matrix,q-p)[0];refined.append(p+tau*u)
    refined=np.array(refined)/scale
    if np.max(np.linalg.norm(refined-np.array(corners),axis=1))>iw*.03:return corners
    if np.any(refined<0) or np.any(refined[:,0]>=iw) or np.any(refined[:,1]>=ih):return corners
    try:homography(refined)
    except ValueError:return corners
    return refined.tolist()
