from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np
import scipy.misc

from .misc import *

def load_image(img_path):
    # H x W x C => C x H x W
    im = np.transpose(scipy.misc.imread(img_path), (2, 0, 1)) 
    im = to_torch(im).float()/255  # Normalize to [0, 1]
    return im

def upsample(x, scale_factor=1):   
    assert x.dim() == 4, 'Input should be a 4-dim tensor'
    model = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    output = model(torch.autograd.Variable(x))
    output = output.data
    return output

def downsample(x, scale_factor=1):    
    model = nn.AvgPool2d(scale_factor, stride=scale_factor)
    output = model(torch.autograd.Variable(x))
    return output.data

def resize(img, owidth, oheight):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    img = scipy.misc.imresize(
            img,
            (oheight, owidth)
        )
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    return img
