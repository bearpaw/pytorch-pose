from __future__ import absolute_import

import numpy as np
from random import randint
import matplotlib.pyplot as plt

from .misc import *

__all__ = ['accuracy']

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

# print('eval')
# sigma = 1
# n = 1
# c = 16

# idx = [1,2,3,4,5,6,11,12,15,16]

# scores = torch.zeros([n, c, 64, 64])
# target = torch.zeros([n, c, 64, 64])

# for i in range(n):
#     for j in range(c):        
#         pt = np.zeros(2, dtype='int')
#         pt[0] = randint(0, 63)
#         pt[1] = randint(0, 63)
#         # print(pt)
#         scores[i, j, :, :] = draw_gaussian2(to_torch(scores[i, j, :, :]), pt, sigma)
#         gt = np.zeros(2, dtype='int')
#         gt[0] = pt[0] + randint(0,2)
#         gt[1] = pt[1] + randint(0,4)
#         target[i, j, :, :] = draw_gaussian2(to_torch(target[i, j, :, :]), gt, sigma)

# acc = accuracy(scores, target, idx)
# print(acc)
# preds = get_preds(scores)
# print(preds)

# plt.imshow(to_numpy(scores[0,0,:,:]))
# plt.show()

# #######################
# preds = torch.rand(4, 16, 2)
# target = torch.rand(4, 16, 2)
# normalize = torch.ones(preds.size(0))*64/10

# dists = calc_dists(preds, target, normalize)
# print(dists)

# a = dist_acc(dists)
# print(a)