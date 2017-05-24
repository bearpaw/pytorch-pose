import torch
import numpy as np
import scipy.misc as misc

__all__ = [
    'load_image',
]

def load_image(img_path):
    im = np.transpose(misc.imread(img_path), (2, 0, 1)) # H x W x C => C x H x W
    im = torch.from_numpy(im).float()/255               # normalize to [0, 1]
    return im
