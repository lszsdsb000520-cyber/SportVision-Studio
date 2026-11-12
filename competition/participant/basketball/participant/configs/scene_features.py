"""Low-cost broadcast camera features shared by training and inference."""
import cv2
import numpy as np


def floor_mask(hsv):
    h, s, v = cv2.split(hsv)
    return ((h >= 6) & (h <= 35) & (s >= 22) & (s <= 170) & (v >= 105))


def scene_features(rgb):
    small = cv2.resize(rgb, (96, 54), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    mask = floor_mask(hsv)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    edge = np.abs(cv2.Laplacian(gray, cv2.CV_32F)) > 24
    grid = []
    for top in (0, 9, 18, 27, 36, 45):
        for left in (0, 24, 48, 72):
            region = np.s_[top:top+9, left:left+24]
            patch = hsv[region]
            grid.extend((mask[region].mean(), edge[region].mean(),
                         patch[..., 1].mean()/255, patch[..., 2].mean()/255))
    return np.asarray(grid, dtype=np.float32), mask, gray
