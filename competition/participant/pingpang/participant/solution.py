"""Competition entry: ball heatmaps, table geometry, and rebound detection."""
from pathlib import Path

import cv2
import numpy as np

from core.types import Landing
from .base_solution import BaseSolution
from .configs.landings import choose_surface, propose, surface_from_logits
from .configs.runtime import model_runner

WEIGHTS = Path(__file__).resolve().parent/"weights"
MEAN = np.array([.485,.456,.406],np.float32)
STD = np.array([.229,.224,.225],np.float32)
LUT = np.ascontiguousarray(((np.arange(256,dtype=np.float32)[:,None]/255-MEAN)/STD).T)


class Solution(BaseSolution):
    def prepare(self, context):
        cv2.setNumThreads(1)
        self.ball, backend = model_runner(WEIGHTS/"ball.onnx",context,"input",(1,9,288,512))
        self.table, table_backend = model_runner(WEIGHTS/"table.onnx",context,
                                                  "image",(1,3,320,640),use_trt=False)
        self.tensor = np.empty((1,9,288,512),np.float32)
        self.tensor[:] = np.tile(LUT[:,0],3)[None,:,None,None]
        self.ball(self.tensor)
        self.table(np.zeros((1,3,320,640),np.float32))
        context.info.update(backend=backend,table_backend=table_backend,
                            windows="centered for frontal or unknown; grouped for 45/90")

    def reset(self, video_meta):
        self.tensor[:] = np.tile(LUT[:,0],3)[None,:,None,None]
        self.trajectory = []
        self.surfaces = []
        self.previous_frame = None
        self.group = []
        self.grouped = str(video_meta.angle).strip().startswith(("45","90"))
        n = video_meta.n_frames
        self.samples = {min(n-1,j) for j in (0,n//5,n//2,3*n//4)}

    def process_frame(self, rgb, frame_meta):
        fid = frame_meta["frame_id"]
        h,w = rgb.shape[:2]
        if fid in self.samples:
            sample = cv2.resize(rgb,(640,320)).transpose(2,0,1)
            result = self.table(np.ascontiguousarray(sample,dtype=np.float32)[None])[0,0]
            surface = surface_from_logits(result,w,h)
            if surface is not None:
                self.surfaces.append(surface)
        offset = len(self.group)*3 if self.grouped else 6
        if not self.grouped:
            self.tensor[0,:6] = self.tensor[0,3:].copy()
        small = cv2.resize(rgb,(512,288))
        for i,channel in enumerate(cv2.split(small)):
            cv2.LUT(channel,LUT[i],dst=self.tensor[0,offset+i])
        if self.grouped:
            self.group.append(fid)
            if len(self.group)==3:
                self._flush_group(w,h)
            return []
        heat = self.ball(self.tensor)[0,1]
        if self.previous_frame is not None:
            self._observe(heat,self.previous_frame,w,h)
        self.previous_frame = fid
        return []

    def _flush_group(self,w,h):
        n = len(self.group)
        if not n:
            return
        for i in range(n,3):
            self.tensor[0,3*i:3*i+3] = self.tensor[0,3*n-3:3*n]
        heat = self.ball(self.tensor)[0]
        for channel,fid in enumerate(self.group):
            self._observe(heat[channel],fid,w,h)
        self.group.clear()

    def _observe(self, heat, frame, w, h):
        _,confidence,_,(x,y) = cv2.minMaxLoc(heat)
        if confidence < .55:
            return
        x0,y0 = max(0,x-4),max(0,y-4)
        patch = heat[y0:y+5,x0:x+5]
        cutoff = max(.55,confidence*.78)
        yy,xx = np.nonzero(patch>=cutoff)
        if not len(xx):
            return
        weights = patch[yy,xx]
        px = (np.average(xx,weights=weights)+x0)*w/512
        py = (np.average(yy,weights=weights)+y0)*h/288
        self.trajectory.append((frame,float(px),float(py),float(confidence)))

    def finish_video(self, video_meta):
        if self.grouped:
            self._flush_group(video_meta.width,video_meta.height)
        if self.previous_frame is not None:
            self.tensor[0,:6] = self.tensor[0,3:].copy()
            self._observe(self.ball(self.tensor)[0,1],self.previous_frame,
                          video_meta.width,video_meta.height)
        surface = choose_surface(self.surfaces)
        return [Landing(t,x,y) for t,x,y in propose(self.trajectory,surface)
                if 0 <= t < video_meta.n_frames and 0 <= x < video_meta.width and 0 <= y < video_meta.height]
