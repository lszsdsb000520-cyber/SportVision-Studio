"""Same detector inputs with reusable planar letterbox storage."""
import cv2
import numpy as np

from ..player_detector import PlayerDetector, PAD_COLOR


class FastDetector(PlayerDetector):
    def _letterbox(self, rgb):
        h,w = rgb.shape[:2]
        key = h,w
        if getattr(self,"_shape",None) != key:
            scale = min(self.in_w/w,self.in_h/h)
            nw,nh = round(w*scale),round(h*scale)
            ox,oy = (self.in_w-nw)//2,(self.in_h-nh)//2
            self._shape = key
            self._params = (scale,ox,oy)
            self._size = (nw,nh)
            self._lut = np.arange(256,dtype=np.float32)/255
            self._tensor = np.empty((1,3,self.in_h,self.in_w),np.float32)
            self._tensor[:] = self._lut[list(PAD_COLOR)][None,:,None,None]
            self._inner = self._tensor[0,:,oy:oy+nh,ox:ox+nw]
        resized = cv2.resize(rgb,self._size,interpolation=cv2.INTER_LINEAR)
        for k,plane in enumerate(cv2.split(resized)):
            cv2.LUT(plane,self._lut,dst=self._inner[k])
        return self._tensor,self._params
