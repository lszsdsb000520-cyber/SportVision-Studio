"""Competition entry: broadcast scene filtering and player identity tracking."""
from pathlib import Path
import os

import cv2
import numpy as np
from scipy.special import expit

from core.types import Track
from .base_solution import BaseSolution
from .player_detector import weights_path
from .configs.association import Tracker
from .configs.fast_detector import FastDetector
from .configs.player_basis import basis
from .configs.player_features import appearances, geometry
from .configs.runtime import model_runner
from .configs.scene_features import scene_features
from .configs.scorebug import on_air

CONFIG = Path(__file__).resolve().parent/"configs"


def read_classifier(path):
    with np.load(path) as data:
        mean,scale,p = data["mean"],data["scale"],data["coefficients"]
    return lambda x: expit(((x-mean)/scale)@p[:-1]+p[-1])


class Solution(BaseSolution):
    def prepare(self, context):
        cv2.setNumThreads(1)
        infer, backend = model_runner(weights_path(),context,"images",(1,3,960,960))
        self.det = FastDetector(infer,classes=[0],layout="e2e",conf_th=.18,top_k=25)
        self.det.detect(np.zeros((720,1280,3),np.uint8))
        self.scene_model = read_classifier(CONFIG/"scene.npz")
        with np.load(CONFIG/"player_filter_v2.npz") as data:
            mean,scale,p = data["mean"],data["scale"],data["coefficients"]
            self.player_threshold = float(data["threshold"])
        self.player_model = lambda x: expit(basis(x,mean,scale)@p[:-1]+p[-1])
        self.stride = 2 if os.getenv("BB_STRIDE") == "2" else 1
        context.info.update(backend=backend,association="motion+appearance",
                            scene_filter="official-training-data",player_filter="quadratic-calibration")

    def reset(self, video_meta):
        self.tracker = Tracker()
        self.prev = None
        self.shot = 0
        self.game = True
        self.scene_ema = None
        self.records = {}
        self.overlay_samples = []

    def process_frame(self, rgb, frame_meta):
        fid = frame_meta["frame_id"]
        h,w = rgb.shape[:2]
        features,floor,gray = scene_features(rgb)
        cut = self.prev is not None and np.abs(gray.astype(np.float32)-self.prev).mean()>35
        self.prev = gray
        p = float(self.scene_model(features))
        if fid%5==0:
            thumb = cv2.resize(rgb,(192,108),interpolation=cv2.INTER_AREA)
            self.overlay_samples.append((fid,p,cv2.cvtColor(thumb,cv2.COLOR_RGB2GRAY)))
        if cut:
            self.shot += 1
            self.scene_ema = None
        self.scene_ema = p if self.scene_ema is None else .66*self.scene_ema+.34*p
        game = self.scene_ema >= (.34 if self.game else .42)
        if not game:
            if self.game:
                self.shot += 1
            self.game = False
            return []
        restart = not self.game
        self.game = True
        if fid%self.stride and not cut and not restart:
            return []
        dets = self.det.detect(rgb)
        if not len(dets):
            return []
        prob = self.player_model(geometry(dets,floor,w,h))
        det_score = dets[:,4]
        keep = (prob >= self.player_threshold)&(det_score >= .20)
        dets,prob = dets[keep],prob[keep]
        quality = np.asarray(.64*prob+.36*dets[:,4],np.float32)
        order = np.argsort(-quality)[:12]
        dets,quality = dets[order],quality[order]
        colors = appearances(rgb,dets[:,:4])
        for track,previous,old,quality_score in self.tracker.step(fid,self.shot,dets[:,:4],colors,quality):
            key = (fid,track.tid)
            self.records[key] = (track.box.copy(),quality_score,track.shot)
            old_row = self.records.get((previous,track.tid))
            if old_row and old_row[2] == self.shot and fid-previous <= 3 and track.hits >= 3:
                for missing in range(previous+1,fid):
                    ratio = (missing-previous)/(fid-previous)
                    self.records[missing,track.tid] = (old*(1-ratio)+track.box*ratio,old_row[1]*.93,self.shot)
        return []

    def finish_video(self, video_meta):
        from collections import Counter, defaultdict
        unavailable = on_air(self.overlay_samples) or []
        hits = Counter(tid for _,tid in self.records)
        rows = defaultdict(list)
        for (fid,tid),(box,quality,_) in self.records.items():
            if hits[tid] >= 3 and not any(a<=fid<=b for a,b in unavailable):
                rows[fid].append((quality,tid,box))
        return [Track(frame_id=fid,track_id=tid,bbox=tuple(map(float,box)))
                for fid,items in sorted(rows.items())
                for _,tid,box in sorted(items,key=lambda x:(-x[0],x[1]))[:11]]
