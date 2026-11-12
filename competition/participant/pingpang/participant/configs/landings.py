"""Table-aware bounce candidates from a sparse, time-aligned ball trajectory."""
import cv2
import numpy as np


def surface_from_logits(logits, width, height):
    mask = (logits > 1.38).astype(np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    cs,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cs]
    if not areas or max(areas) < mask.size*.003:
        return None
    keep = [c for c,a in zip(cs,areas) if a >= .15*max(areas)]
    hull = cv2.convexHull(np.concatenate(keep)).astype(np.float32)
    if cv2.contourArea(hull) > mask.size*.42 or cv2.contourArea(hull) > sum(areas)*2.5:
        return None
    return hull * np.array([width/mask.shape[1],height/mask.shape[0]],np.float32)


def choose_surface(surfaces):
    if len(surfaces) < 2:
        return None
    bounds = np.asarray([cv2.boundingRect(q) for q in surfaces],np.float32)
    middle = np.median(bounds,axis=0)
    error = np.abs(bounds-middle).sum(axis=1)/max(50,middle[2]+middle[3])
    best = int(error.argmin())
    if np.count_nonzero(error < .18) < 2:
        return None
    return surfaces[best]


def propose(data, surface=None, *, min_gap=7, conf=.67, margin=46, score_limit=.6):
    if len(data) < 3:
        return []
    p = np.asarray(data,dtype=np.float64)
    if len(p) >= 4:
        # Remove isolated detector jumps that immediately return to the old path.
        dt = p[2:,0]-p[:-2,0]
        w = (p[1:-1,0]-p[:-2,0])/dt
        interp = p[:-2,1:3]+w[:,None]*(p[2:,1:3]-p[:-2,1:3])
        spikes = (dt<=4)&(np.linalg.norm(p[1:-1,1:3]-interp,axis=1)>105)&(
            np.linalg.norm(p[2:,1:3]-p[:-2,1:3],axis=1)<64)
        p = p[np.r_[True,~spikes,True]]
    candidates = []
    for i in range(1,len(p)-1):
        t,x,y,c = p[i]
        g1,g2 = t-p[i-1,0],p[i+1,0]-t
        if c<conf or g1>3 or g2>3:
            continue
        fall,rise = y-p[i-1,2], y-p[i+1,2]
        distance = (cv2.pointPolygonTest(surface,(x,y),True) if surface is not None else 0)
        if distance < -margin:
            continue
        left,right = p[max(0,i-2):i],p[i+1:i+3]
        vin = (p[i,1:3]-left[-1,1:3])/g1
        vout = (right[0,1:3]-p[i,1:3])/g2
        if max(np.linalg.norm(vin),np.linalg.norm(vout)) < 3:
            continue
        if np.linalg.norm(vin) > 240 or np.linalg.norm(vout)>240:
            continue
        if not (fall>=1 and rise>=.6):
            # A high or oblique camera can project both legs of a physical
            # bounce in the same y direction. Look for an abrupt velocity
            # break at the table, with horizontal motion remaining consistent.
            if surface is None or distance < -12 or len(left)<2 or len(right)<2:
                continue
            if t-left[0,0]>4 or right[-1,0]-t>4:
                continue
            a = (left[-1,1:3]-left[0,1:3])/(left[-1,0]-left[0,0])
            b = (right[-1,1:3]-right[0,1:3])/(right[-1,0]-right[0,0])
            change = a[1]-b[1]
            if change < 15 or vin[1]-vout[1]<max(6,.29*change):
                continue
            if a[0]*b[0]<0 and min(abs(a[0]),abs(b[0]))>6:
                continue
            if abs(a[0]-b[0]) > max(17,.56*max(abs(a[0]),abs(b[0]))):
                continue
            if abs(vin[0]-vout[0])>max(20,.63*max(abs(vin[0]),abs(vout[0]))):
                continue
            expected_a = left[-1,2]+a[1]*g1
            expected_b = right[0,2]-b[1]*g2
            if abs(expected_a-expected_b)>max(20,.77*change):
                continue
            if max(expected_a-y,y-expected_b)>max(21,.8*change):
                continue
            score = 1. + min(2,change*.036)+.5*c-.015*max(0,-distance)
            if score>=score_limit:
                candidates.append((float(score),int(t),float(x),float(y)))
            continue
        support = 0.
        if len(left)==2 and t-left[0,0]<=4:
            support += np.clip((y-left[0,2])*.04,-1.2,1.)
        if len(right)==2 and right[-1,0]-t<=4:
            support += np.clip((y-right[-1,2])*.04,-1.2,1.)
        score = (min(fall/g1,rise/g2)*.20 + min(3,(fall/g1+rise/g2)*.07)
                 + .8*c + support - .015*max(0,-distance))
        # Favor the actual impact over a nearby weak apex. A single soft
        # extremum can precede the sharp break when the camera is elevated.
        if surface is not None and distance >= 0 and len(left)==2 and len(right)==2:
            a = (left[-1,2]-left[0,2])/(left[-1,0]-left[0,0])
            b = (right[-1,2]-right[0,2])/(right[-1,0]-right[0,0])
            if g1==g2==1 and max(t-left[0,0],right[-1,0]-t)<=3:
                score += .045*np.clip(a-b, -18, 40)
        if vin[0]*vout[0] < 0 and min(abs(vin[0]),abs(vout[0])) > 11:
            score -= 1.8
        if score >= score_limit:
            candidates.append((float(score),int(t),float(x),float(y)))
    if surface is not None:
        # Recover a single missing contact frame only when two independently
        # fitted legs meet near that frame and inside the tabletop.
        for a,b,c,d in zip(p,p[1:],p[2:],p[3:]):
            gap = int(c[0]-b[0])
            if gap!=2 or b[0]-a[0]>2 or d[0]-c[0]>2 or min(b[3],c[3])<.77:
                continue
            vin = (b[1:3]-a[1:3])/(b[0]-a[0])
            vout = (d[1:3]-c[1:3])/(d[0]-c[0])
            change = vin[1]-vout[1]
            if change<22 or (vin[0]*vout[0]<0 and min(abs(vin[0]),abs(vout[0]))>7):
                continue
            if abs(vin[0]-vout[0])>max(18,.6*max(abs(vin[0]),abs(vout[0]))):
                continue
            offset = (c[2]-b[2]-gap*vout[1])/change
            if not .25<=offset<=1.75:
                continue
            left = b[1:3]+vin*offset
            right = c[1:3]+vout*(offset-gap)
            x,y = (left+right)/2
            if abs(left[0]-right[0])<=32 and cv2.pointPolygonTest(surface,(x,y),True)>=-13:
                candidates.append((1.7+min(1,change*.02),int(b[0])+1,float(x),float(y)))
    accepted = []
    for score,t,x,y in sorted(candidates,reverse=True):
        if all(abs(t-old[0]) >= min_gap for old in accepted):
            accepted.append((t,x,y))
    return sorted(accepted)
