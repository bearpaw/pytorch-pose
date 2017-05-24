import json
import torch
import torch.nn as nn
from    torch.autograd import Variable
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import scipy.misc as misc
import matplotlib.pyplot as plt
import torch.utils.data as data
import random
import math

import pprint

from utils.io import *
from utils.misc import *
from utils.transforms import *

import scipy.misc as misc




class Mpii(data.Dataset):

    def __init__(self, jsonfile, img_folder, inp_res=256, out_res=64, train=True, gsize=2,
        scale_factor=0.25, rot_factor=30):
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.gsize = gsize
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor

        # create train/val split
        with open(jsonfile) as anno_file:   
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid.append(idx)
            else:
                self.train.append(idx)

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        pts = torch.Tensor(a['joint_self'])
        c = torch.Tensor(a['objpos'])
        s = a['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img_path) # CxHxW

        r = 0
        if self.is_train == True:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf,1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf,2*rf)[0] if random.random() <= 0.9 else 0

            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp(0, 1)

            # Flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='mpii')
                c[0] = img.size(2) - c[0]

        # Prepare image and groundtruth map
        inp = crop(img, c, s, r, self.inp_res)

        # Generate ground truth
        tpts = pts.clone()
        for i in range(nparts):
            if tpts[i, 2] > 0:
                tpts[i, 0:2] = transform(tpts[i, 0:2], c, s, r, self.out_res)

        target = torch.zeros(nparts, self.out_res, self.out_res)
        for i in range(nparts):
            if tpts[i, 2] > 0:
                vinp = downsample(inp, 4)
                draw_gaussian(target[i], tpts[i], 2)
                # print('Max %f min %f' % (target[i].max(), target[i].min()))
                # vinp  = (vinp*0.5 + target[i].expand(3, 64, 64)*0.5)
                # imshow(vinp)
                # plt.show()

        # show_joints(inp, tpts*4)
        # plt.show()

        return inp, target

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)


# mpii = Mpii('../data/mpii/mpii_annotations.json', '../data/mpii/images', train=False)
# mpii.__getitem__(10)