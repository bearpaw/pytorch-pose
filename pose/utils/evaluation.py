from __future__ import absolute_import

import math
import numpy as np
import matplotlib.pyplot as plt
from random import randint

from .misc import *
from .transforms import transform, transform_preds

__all__ = ['accuracy', 'AverageMeter']

def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)
    preds = idx.repeat(1, 1, 2)

    preds[:,:,0] = preds[:,:,0] % scores.size(3)
    preds[:,:,1] = preds[:,:,1] / scores.size(2)

    pred_mask = maxval.gt(0).repeat(1, 1, 2).long()
    preds *= pred_mask
    return preds

def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size())
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 0 and target[n, c, 1] > 0:
                dists[n,c,:] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[n,c,:] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum()*1.0 / dists.ne(-1).sum()
    else:
        return -1

def accuracy(output, target, idxs, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    preds   = get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[:, idxs[i]-1, :])
        if acc[i+1] >= 0: 
            avg_acc = avg_acc + acc[i+1]
            cnt += 1
            
    if cnt != 0:  
        acc[0] = avg_acc / cnt
    return acc

def final_preds(output, center, scale, res):
    coords = get_preds(output).float() + 0.7

    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 0 and px < res[0]-1 and py > 0 and py < res[1]-1:
                diff = torch.Tensor([hm[py][px+1]-hm[py][px-1], hm[py+1][px]-hm[py-1][px]])
                coords[n][p] += diff.sign() * .25
    output = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        output[i] = transform_preds(coords[i], center[i], scale[i], res)

    if output.dim() < 3:
        output = output.view(1, output.size())

    return output

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count