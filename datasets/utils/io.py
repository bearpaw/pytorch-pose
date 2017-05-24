import torch
import numpy as np
import scipy.misc
from misc import *

def load_image(img_path):
    # H x W x C => C x H x W
    im = np.transpose(scipy.misc.imread(img_path), (2, 0, 1)) 
    im = to_torch(im).float()/255  # Normalize to [0, 1]
    return im