import copy
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pickle
import pywt
import torch

from matplotlib.patches import Ellipse
from numpy.random import binomial, uniform
from scipy.signal import fftconvolve
from scipy.stats import gennorm

def save_obj(obj, name, data_dir):
    with open(data_dir+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name, data_dir):
    with open(data_dir+name+'.pkl', 'rb') as f:
        return pickle.load(f)
        
def _block_diag(T):
    """
    Function to build a block diagonal matrix from a tensor
    with 3 dimensions. Each block, corresponds to elements
    given at dim 1 and 2
    """

    arrs = [np.atleast_2d(a) for a in [T[i,:,:] for i in range(T.shape[0])]]

    shapes = np.array([a.shape for a in arrs])
    out_dtype = np.find_common_type([arr.dtype for arr in arrs], [])
    out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc

    V=out

    return V

def _discretewv(N, wv, J):
    """
    Function to retrieve the discrete wavelets.
    
    INPUT
    =====
    T: int, lenght of the time series to be generated
    N: int, number of basis vectors to obtain
    wv: str, one of the discrete wavelet families 
    J: int, maximum decomposition level
    
    OUTPUT
    ======
    basis: list of numpy array, decimated wavelet coefficients.
    """
    
    ##############################
    #Step 1
    #Generate an empty array of 
    #length 2**J.
    ##############################
    
    T = 2**J
    v = np.zeros(T)
    
    ##############################
    #Step 2
    #Decompose the empty array
    #by using the DWT of interest
    ##############################
    
    vd = pywt.wavedec(v, wavelet=wv, level=J, axis=0)
    
    ##############################
    #Step 3
    #For each level, retrieve the 
    #nonzero coefficients
    ##############################
    
    basis = torch.zeros(J,T,1)
    
    for i in range(1,len(vd)):
        mod_vd = list(copy.deepcopy(vd))
        mod_vd[-i][0] = 1.
        
        rec = pywt.waverec(mod_vd, wv, axis=0).reshape(-1,1)
        
        basis[i-1]+=torch.tensor(rec)
        
    return basis


