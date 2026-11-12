"""Allow court-player calibration to model position/scale/color interactions."""
import numpy as np


def basis(x,mean,scale):
    x = np.clip((x-mean)/scale,-5,5)
    i,j = np.triu_indices(x.shape[1])
    return np.column_stack((x,x[:,i]*x[:,j]))
