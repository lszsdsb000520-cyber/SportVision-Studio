"""Court geometry and player appearance features."""
import cv2
import numpy as np


def geometry(dets, floor, width, height):
    box = dets[:, :4] / np.array([width,height,width,height], np.float32)
    x1,y1,x2,y2 = box.T
    w,h = x2-x1,y2-y1
    x = (x1+x2)/2
    fh,fw = floor.shape
    ix = np.clip((x.astype(np.float64)*fw).astype(int),0,fw-1)
    iy = np.clip((y2.astype(np.float64)*fh).astype(int),0,fh-1)
    sums = cv2.integral(floor.astype(np.uint8))
    foot = []
    for a,b,c in ((1,3,2),(0,5,3),(4,5,5),(1,2,8)):
        l,r = np.maximum(0,ix-c),np.minimum(fw,ix+c+1)
        t,d = np.maximum(0,iy-a),np.minimum(fh,iy+b)
        foot.append((sums[d,r]-sums[d,l]-sums[t,r]+sums[t,l])/((d-t)*(r-l)))
    return np.column_stack((dets[:,4], x, abs(x-.5), y1, y2, w, h, w/np.maximum(h,.001),
                            h/np.maximum(y2,.05), np.column_stack(foot)))


def appearances(rgb, boxes):
    patches = np.zeros((len(boxes),12,9,3),np.uint8)
    valid = np.zeros(len(boxes),dtype=bool)
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        bw,bh = x2-x1,y2-y1
        a,b,c,d = map(int,(x1+.23*bw,y1+.19*bh,x2-.23*bw,y1+.57*bh))
        if a < c and b < d:
            patches[i] = cv2.resize(rgb[b:d,a:c],(9,12),interpolation=cv2.INTER_AREA)
            valid[i] = True
    if not len(boxes):
        return np.empty((0,9),np.float32)
    lab = cv2.cvtColor(patches.reshape(-1,9,3),cv2.COLOR_RGB2LAB).reshape(-1,108,3)
    # Quantiles at positions 26.75, 53.5 and 80.25 of the 108 pixels.
    lab = np.partition(lab,(26,27,53,54,80,81),axis=1)
    p = lab[:,[26,53,80]].astype(np.float32)
    q = lab[:,[27,54,81]].astype(np.float32)
    result = ((p+(q-p)*np.array([.75,.5,.25],np.float32)[None,:,None])/255).reshape(-1,9)
    result[~valid] = 0
    return result


def appearance(rgb, box):
    return appearances(rgb,np.asarray([box]))[0]
