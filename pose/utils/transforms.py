from __future__ import absolute_import
import torch
import torch.nn as nn
from   torch.autograd import Variable
import torchvision.transforms as transforms
import os
import numpy as np
import scipy.ndimage as ndi
import scipy.misc as misc
import matplotlib.pyplot as plt
import torch.utils.data as data
import random
import math

from .misc import *
from .imutils import *
from PIL import Image

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, x.size(1), x.size(2))
    return (x - mean.view(3, 1, 1).expand_as(x)) / std.view(3, 1, 1).expand_as(x)


def shufflelr(x, width, dataset='mpii'):
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    else:
        print('Not supported dataset: ' + dataset)

    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = x[pair[0], :].clone()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x


def fliplr(x):
    x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    return x.astype(float)

def rotate(img, angle):
    return to_torch(ndi.rotate(to_numpy(img), angle, axes=(2, 1), reshape=False, mode='nearest'))

def get_transform(center, scale, rot, res):
    h = 200 * scale
    t = torch.eye(3)

    # Scaling
    t[0][0] = res*1.0 / h
    t[1][1] = res*1.0 / h

    # Translation
    t[0][2] = res*1.0 * (-center[0] / h + .5)
    t[1][2] = res*1.0 * (-center[1] / h + .5)


    # Rotation
    if rot != 0:
        rot = -rot
        r = torch.eye(3)
        ang = rot * math.pi / 180
        s = math.sin(ang)
        c = math.cos(ang)
        r[0][0] = c
        r[0][1] = -s
        r[1][0] = s
        r[1][1] = c
        # Need to make sure rotation is around center
        t_ = torch.eye(3)
        t_[0][2] = -res/2
        t_[1][2] = -res/2
        t_inv = torch.eye(3)
        t_inv[0][2] = res/2
        t_inv[1][2] = res/2
        t = t_inv.mm(r.mm(t_.mm(t)))

    return t

def transform(pt, center, scale, rot, res, invert=False):
    pt_ = torch.ones(3, 1)
    pt_[0],pt_[1] = pt[0]-1,pt[1]-1

    t = get_transform(center, scale, rot, res)
    if invert:
        t = torch.inverse(t)

    new_point = t.mm(pt_)[0:2]

    return new_point.int().squeeze()+1

def crop(img, center, scale, rot, res):
    ndim = img.dim()
    if ndim == 2:
        img.unsqueeze_(0)
    ht,wd = img.size(1), img.size(2)
    tmpImg, newImg = img, np.zeros((img.size(0), res, res))


    # Modify crop approach depending on whether we zoom in/out
    # This is for efficiency in extreme scaling cases
    scaleFactor = (200.0 * scale) / res

    if scaleFactor < 2: 
        scaleFactor = 1
    else:
        newH = int(math.floor(ht/ scaleFactor))
        newW = int(math.floor(wd/ scaleFactor))

        if max(newH, newW) < 2:
           # Zoomed out so much that the image is now a single pixel or less
           if ndim == 2:
                newImg = newImg.squeeze()
           return newImg
        else:
           tmpImg = resize(img, newW, newH)
           ht,wd = tmpImg.size(1), tmpImg.size(2)
    # Calculate upper left and bottom right coordinates defining crop region
    c,s = center/scaleFactor, scale/scaleFactor
    ul = transform([1,1], c, s, 0, res, invert=True)
    br = transform([res+1,res+1], c, s, 0, res, invert=True)

    if scaleFactor >= 2: 
        br = ul + res
    # If the image is to be rotated, pad the cropped area
    pad = int(math.ceil((torch.norm((ul-br).float())-(br[0]-ul[0]))/2))

    if rot != 0:
        ul.add_(-pad)
        br.add_(pad)

    # Define the range of pixels to take from the old image
    old_ = torch.IntTensor([max(1, ul[1]), min(br[1], ht+1) - 1,
                       max(1, ul[0]), min(br[0], wd+1) - 1])
    # And where to put them in the new image
    new_ = torch.IntTensor([max(1, -ul[1] + 2), min(br[1], ht+1) - ul[1],
                       max(1, -ul[0] + 2), min(br[0], wd+1) - ul[0]])

    # Initialize new image and copy pixels over
    newImg = torch.zeros(img.size(0), br[1] - ul[1], br[0] - ul[0])

    try:
        newImg[:, new_[0]-1:new_[1], new_[2]-1:new_[3]].copy_(tmpImg[:, old_[0]-1:old_[1], old_[2]-1:old_[3]])
    except:
        print('Error occurred during cropping')
        raise

    # Rotate the image and remove padded area
    if rot != 0:
        newImg = rotate(newImg, rot)
        newImg = newImg[:, pad:newImg.size(1)-pad, pad:newImg.size(2)-pad].clone()


    if scaleFactor < 2:
        newImg = resize(newImg,res,res)
    if ndim == 2: 
        newImg = newImg.squeeze()

    return newImg     