# !/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Global image imthresh/segmentation function using extended Otsu's method.
This multi-level thresholding method is extended to handle RGB
images, via the Karhunen-Loeve transform performed on each of the
R,G,B channels. Segmentation is carried out on the image
component that contains most of the information energy.

IDX,sep = imthresh(I,N) segments the image I into N labels by means of Otsu's
N-level (multi) thresholding method.

The function returns an array IDX containing the cluster indices (from 1 to N)
of each point. The value (sep) of the separability criterion within the range
[0 1]. Zero is obtained only with data having less than N values, whereas one
(optimal value) is obtained only with N-valued arrays. Zero values are
assigned to non-finite (NaN or Inf) pixels.

If only I is provided then N =2.

The number of labels N cannot be less than 2, or more than the image
levels (N<255).

Otsu N (1979), A Threshold Selection Method from Gray-Level Histograms,
IEEE Trans. Syst. Man Cybern. 9: 62-66.
'''

import numpy as np
import cv2 as cv
import sys
import warnings
from numpy import linalg
from scipy import optimize

__author__ = "G. Chliveros and K. Papakonstantinou"
__copyright__ = "Copyright 2019, WatchOver Project"
__credits__ = ["", "", "", ""] # people who reported bug fixes
__license__ = "LGPL"
__version__ = "1.0.1"
__maintainer__ = "TBA"
__email__ = "TBA"
__status__ = "Prototype" # "Prototype", "Development", or "Production"


def imthresh(I, n):
    '''
    
    
    Parameters
    ----------
    I : int multidimensional array
        Input frame.
    n : int
        Number of labels.
        
    Returns
    -------
    TYPE
        DESCRIPTION.
        
    '''
   # Evaluate number of input arguments
   # ATTENTION! This evaluation must be the first thing the function encounters!
    nargin = len(locals())
    
    def sig_func(k):
        '''
        Function to be minimized if n>=4
        
        Parameters
        ----------
        k : TYPE
            DESCRIPTION.
        
        Returns
        -------
        y : TYPE
            DESCRIPTION.
        
        '''
        
        muT = np.sum(np.reshape(np.arange(1, nbins+1), (1, nbins)) * P)
        sigma2T = np.sum((np.reshape(np.arange(1, nbins+1), (1, nbins)) ** 2) * P)
        k = np.round(k * (nbins - 1) + 1)
        k = np.sort(k, axis=0)
        
        if np.logical_or(k.all() < 1, k.all() > nbins):
            y = 1
            
            return
        
        k = np.insert(k, 0, 0)
        k = np.append(k, nbins)
        k = np.int32(k)
        sigma2B = 0
        
        for j in range(n):
            wj = np.sum(P[k[j]+1:k[j+1]], axis=0)
            if wj == 0:
                y = 1
                
                return
            muj = np.sum((np.arange(k[j]+1, k[j+1]) * P[k[j]+1:k[j+1]]) / wj)
            
            sigma2B = sigma2B + wj * np.power((muj - muT), 2)
        y = 1 -sigma2B / sigma2T # Within the range [0 1]
        
        return (y)
    
    # Convert frame datatype from BGR to RGB colour format
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)
    
    # Get frame dimmensions
    height, width, col = I.shape
    
    isRGB = np.bool(col == 3)
    
    # Check if input frame is an RGB image
    assert col >= 2, 'The input must be a 2-D array or an RGB image.'
    
    # Checking n (number of classes)
    if nargin == 1:
        n = 2
        
    elif n == 1:
        #IDX = np.full((height, width, col), np.nan)
        #sep = 0
        n = 2
        warnings.warn('n is too low - n value has been changed to 2...')
        
    elif np.logical_or(n != np.abs(np.round(n)), n == 0):
        sys.exit('n must be a strictly positive integer!')
    
    # TODO: In future release: (n>255); n= 255; via recursion
    elif n > 18:
        n = 18
        warnings.warn('n is too high - n value has been changed to 18...')
        
    # Convert datatype to float32
    I = np.int32(I)
    
    # Perform a KTL if isRGB, and keep the component of highest energy
    if isRGB:
        # Reshape input frame
        I = np.reshape(I, (height * width, 3))
        # Evaluate eigen values - Numpy requires transposed arrays!
        V, D = linalg.eig(np.cov(I.T))
        # Evaluate index where max eigen value is present
        c = np.argmax(V)
        # Reshape D variable - One of Python's quirks
        D = np.reshape(D[:, c], (3, 1))
        # Component with the highest energy
        I = np.reshape((I @ D), (height, width))
        
    # Convert to 256 levels (0 to 255)
    I = I - np.min(I)
    I = np.round(I / np.max(I) * 255)
    
    # Probability distribution
    unI = np.sort(np.unique(I))
    nbins = np.minimum(len(unI), 256)
    
    if nbins == n:
        IDX = np.ones((height, width), dtype=np.float32)
        for i in range(n):
            IDX[I==unI[i]] = i
        sep = 1
        
        return
    
    elif nbins < n:
        IDX = np.full((height, width), np.nan)
        sep = 0
        
        return
    
    elif nbins < 256:
        hist, pixval = np.histogram(I.ravel(), bins=unI)
        
    else:
        hist, pixval = np.histogram(I.ravel(), bins=256)
        
    # Regularised histogram
    P = hist / np.sum(hist)
    
    P = np.reshape(P, (len(P), 1))
    
    # Reduce counter reference for unI var, for the GC to catch the var.
    unI = None
    
    # Zeroth and first-order cumulative moments
    w = np.cumsum(P, axis=0)
    
    
    # Maximal sigmaB^2 and Segmented image
    if n == 2:
        mu = np.cumsum(np.reshape(np.arange(0, nbins, dtype=np.uint16),
                         (nbins, 1)) *P, axis=0)
        #mu = np.uint16(mu)
        
        sigma2B = (mu[-1] * w[1:-1] - mu[1:-1]) ** 2 / (w[1:-1]
                                              / (1- w[1:-1]))
        
        # Max values
        maxsig = np.max(sigma2B, axis=0)
        
        # Max value indicies
        k = np.argmax(sigma2B, axis=0)
        
        # Segmented image
        IDX = np.ones_like(I)
        
        IDX[I > pixval[k+1]] = 2
        
        # Separability criterion
        sep = maxsig / np.sum((np.arange(0, nbins) - mu[-1]) ** 2 * P, axis=0)
        
    elif n == 3:
        mu = np.cumsum((np.arange(0, nbins, dtype=np.uint16) * P))
        #w0 = w
        w0 = [w]
        P=[P]
        w2 = np.cumsum(np.fliplr(P), axis=0)
        w2 = np.fliplr(w2)
        
        w0, w2 = np.mgrid[np.min(w0):np.max(w0):((np.max(w0)-np.min(w0))/256), np.min(w2):np.max(w2):((np.max(w2)-np.min(w2))/256)]
        mu0 = mu / w
        
        mu2 = np.fliplr(np.cumsum(np.fliplr(np.arange(0, nbins) * P)) /
                        np.cumsum(np.fliplr(P), axis=0))
        mu0, mu2 = np.mgrid[np.min(mu0):np.max(mu0):((np.max(mu0)-np.min(mu0)) / 256), np.min(mu2):np.max(mu2):(np.max(mu2)-np.min(mu2) / 256)]
        w1 = 1 - w0 - w2
        w1[w1 <= 0] = np.nan
        sigma2B = w0 * (mu0 - mu[-1])**2 + w2 * (mu2 - mu[-1])**2 + (w0 *
                  (mu0 - mu[-1])) + w2 * (mu2 - mu[-1])** 2/w1
        sigma2B[np.isnan(sigma2B)] = 0 # zeroing if k1 >= k2
        maxsig = np.max(sigma2B)
        k = np.argmax(sigma2B)
        #k2, k1 = np.unravel_index(k, [nbins, nbins])
        k2, k1 = np.divmod(k,nbins)
        k2 = k2+1
        k1= k1+1
        
        # Segmented image
        
        IDX = np.full((np.shape(I)), 3, dtype=np.int32)
        IDX[I <= pixval[k1]] = 1
        IDX[np.logical_and(I > pixval[k1], I <= pixval[k2])] = 2
        
        # Seperability criterion
        # sep = maxsig / np.sum(np.arange(0, nbins) - mu[-1]**2 * P)
        # BUG: datatype issue to calculate sep
        sep = -1
        
    else:
        mu = np.cumsum((np.arange(0, nbins, dtype=np.uint16) * P))
        k0 = np.linspace(0, 1, n+1)
        k0 = k0[1:n]
        k, y,_, _, _ = optimize.fmin(sig_func, k0, xtol=1, full_output=True)
        k = np.int32(np.round((k * (nbins-1)+1)))
        
        # Segmented image
        IDX = np.full(np.shape(I), n)
        IDX[I <= pixval[k[0]]] = 1
        
        for i in range(n-2):
            IDX[np.logical_and(I > pixval[k[i]], I <= pixval[k[i+1]])] = i+1
            
        # Seperability criterion
        sep = 1 - y
        
    IDX[~np.isfinite(I)] = 0
    out = np.array((IDX, sep))
    
    return (out)

# Example implementation...
#=============================================================================#
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img=cv.imread('hull6.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # just for correct colour format at imshow
    plt.imshow(img)
    plt.show()
    # BUG: optimizer does not work for values 14 & >18.
    n=2 # define n labels for segments
    idx, sep = imthresh(img,n)
    result = np.where(idx == 2) # find label 2
    coords=list(zip(result[0], result[1]))
    I2 = np.zeros(np.shape(img))
    for i in range(np.shape(coords)[0]):
        temp = coords[i]
        I2[temp[0],temp[1],:] = img[temp[0],temp[1],:]
    plt.imshow(np.int16(I2)) # typecasting to int16 just for proper colour
    plt.show()
