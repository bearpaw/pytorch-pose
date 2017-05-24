import  torch
import  torch.nn as nn
import  torchvision.transforms as transforms
from    torch.autograd import Variable
import  scipy.ndimage as ndi
import  scipy.misc as misc
import  math
import  numpy as np
import  pdb
import  matplotlib.pyplot as plt


class Transformer(object):
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu


    def upsample(self, x, scale_factor=1):   
        assert(x.dim() == 4, 'Input should be a 4-dim tensor')
        model = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        output = model(Variable(x))
        output = output.data
        return output

    def downsample(self, x, scale_factor=1):    
        model = nn.AvgPool2d(scale_factor, stride=scale_factor)
        output = model(Variable(x))
        return output.data
