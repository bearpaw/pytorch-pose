from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *


class Mscoco(data.Dataset):
    def __init__(self, is_train=True, **kwargs):
        self.is_train   = is_train # training set or test set
        self.inp_res    = kwargs['inp_res']
        self.out_res    = kwargs['out_res']
        self.sigma      = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']
        self.year       = kwargs['year']
        self.jsonfile   = kwargs['anno_path']
        img_folder = kwargs['image_path'] # root image folders

        if is_train:
            self.img_folder = '{}/train{}'.format(img_folder, self.year)    # root image folders
        else:
            self.img_folder = '{}/val{}'.format(img_folder, self.year)    # root image folders

        # create train/val split
        with open(self.jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid.append(idx)
            else:
                self.train.append(idx)
        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        meanstd_file = './data/coco/mean.pth.tar'
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            print('==> compute mean')
            mean = torch.zeros(3)
            std = torch.zeros(3)
            cnt = 0
            for index in self.train:
                cnt += 1
                print( '{} | {}'.format(cnt, len(self.train)))
                a = self.anno[index]
                img_path = os.path.join(self.img_folder, a['img_paths'])
                img = load_image(img_path) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        pts = torch.Tensor(a['joint_self'])
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        # c = torch.Tensor(a['objpos']) - 1
        c = torch.Tensor(a['objpos'])
        s = a['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img_path)  # CxHxW

        r = 0
        if self.is_train:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

            # Flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='mpii')
                c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        tpts = pts.clone()
        target = torch.zeros(nparts, self.out_res, self.out_res)
        target_weight = tpts[:, 2].clone().view(nparts, 1)
        for i in range(nparts):
            if tpts[i, 2] > 0: # COCO visible: 0-no label, 1-label + invisible, 2-label + visible
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r))
                target[i], vis = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)
                target_weight[i, 0] = vis

        # Meta info
        meta = {'index' : index, 'center' : c,
                'scale' : s, 'pts' : pts,
                'tpts' : tpts, 'img_path' : img_path,
                'target_weight': target_weight}

        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)

def coco(**kwargs):
    return Mscoco(**kwargs)

coco.njoints = 17  # ugly but works
