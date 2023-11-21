from scipy.signal import normalize 
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from copy import deepcopy

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def cheops_trend_func(x, df, residuals):
    
    # unpack data frame with cheops data
    bg = df['bg'] # background flux
    contam = df['contam'] # contamination in the aperature 
    smear = df['smear'] # smear correction
    phi = df['phi'] # roll angel 
    dx = df['dx'] # x point spread function centroid coordinates
    dy = df['dy'] # y point spread function centroid coordinates
    x = x
    
    trend_array = np.transpose(np.array([normalize(x)**2,\
                                         normalize(bg),\
                                         normalize(contam),\
                                         normalize(smear),\
                                         normalize(dx),\
                                         normalize(dy),\
                                         normalize(dx)**2,\
                                         normalize(dy)**2 ,\
                                         normalize(dx)*normalize(dy),\
                                         np.sin(normalize(phi)),\
                                         np.cos(normalize(phi)),\
                                         np.sin(2*normalize(phi)),\
                                         np.cos(2*normalize(phi)),\
                                         np.sin(3*normalize(phi)),\
                                         np.cos(3*normalize(phi))]))
    
    A = np.c_[normalize(x)-np.nanmin(normalize(x)), np.ones(len(x))]
    Aquad = trend_array
    A = np.c_[Aquad, A]
    c1, _, _, _ = lstsq(A, residuals)
    trend_model = np.sum(c1*A, axis=1)
    return trend_model


