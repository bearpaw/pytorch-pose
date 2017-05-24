import json
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils.data as data
import random

import pprint
import utils

class Mpii(data.Dataset):

    def __init__(self, jsonfile, img_folder, inp_res=256, out_res=64, train=True, gsize=2,
        scale_factor=0.25, rot_factor=30):
        self.img_folder = img_folder # root image folders
        self.is_train = train  # training set or test set
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

        # print number of samples
        print 'Train samples: ', len(self.train)
        print 'Valid samples: ', len(self.valid)

    def _augmentation(self, inp, target):
        print(self.is_train)
        scale_factor = self.scale_factor
        if self.is_train:
            return
        else:
            print('valid')
            s = torch.randn(1).mul_(self.scale_factor).add_(1).clamp(1-self.scale_factor,1+self.scale_factor)[0]
            r = torch.randn(1).mul_(self.rot_factor).clamp(-2*self.rot_factor,2*self.rot_factor)[0]
            print(r)

            # Color
            inp[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp(0, 1)
            inp[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp(0, 1)
            inp[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp(0, 1)

            # Scale/rotation
            # if random.uniform(0, 1) <= 0.5:
            #     r = 0
            inp_res,out_res = self.inp_res, self.out_res
            inp = utils.crop(inp, torch.Tensor(((inp_res+1)/2,(inp_res+1)/2)), inp_res*s/200, r, inp_res)
            target = utils.crop(target, torch.Tensor(((out_res+1)/2,(out_res+1)/2)), out_res*s/200, r, out_res)
            return (inp, target)


    def __getitem__(self, index):
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
        img = utils.load_image(img_path) # CxHxW
        inp = utils.crop(img, c, s, 0, self.inp_res)
        target = torch.zeros(nparts, self.out_res, self.out_res)

        for i in xrange(0, nparts):
            if pts[i][0] > 0:
                tpts = utils.transform([pts[i][0], pts[i][1]], c, s, 0, self.out_res)
                utils.drawGaussian(target[i], tpts, self.gsize)

                # # debug
                # vinp = utils.downsample(inp, 4)
                # vinp  = (vinp*0.5 + target[i].expand(3, 64, 64)*0.5)
                # utils.imshow(vinp)
                # plt.pause(0.05)
        inp, target = self._augmentation(inp, target)
        print(inp.size())
        print(target.size())
        utils.imshow(inp)


        for i in xrange(0, nparts):
            if pts[i][0] > 0:
                # debug
                vinp = utils.downsample(inp, 4)
                vinp  = (vinp*0.5 + target[i].expand(3, 64, 64)*0.5)
                utils.imshow(vinp)
                plt.pause(0.05)

        return inp, target

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)


mpii = Mpii('mpii_annotations.json', 'mpii/images', train=False)
mpii.__getitem__(10)