# from pose.utils import evaluation
from pose import *
from pose.utils.misc import *
import matplotlib.pyplot as plt
import torch
import numpy as np
from random import randint


sigma = 1
n = 1
c = 16

idx = [1,2,3,4,5,6,11,12,15,16]

scores = torch.zeros([n, c, 64, 64])
target = torch.zeros([n, c, 64, 64])

win = None
for i in range(n):
    for j in range(c):        
        pt = np.zeros(2, dtype='int')
        pt[0] = randint(0, 63)
        pt[1] = randint(0, 63)
        # print(pt)
        scores[i, j, :, :] = draw_gaussian2(to_torch(scores[i, j, :, :]), pt, sigma)
        if win == None:
            win = plt.imshow(scores[i,j,:,:].numpy())
        else:
            win.set_data(scores[i,j,:,:].numpy())
        plt.pause(.3)
        plt.draw()
        # gt = np.zeros(2, dtype='int')
        # gt[0] = pt[0] + randint(0,2)
        # gt[1] = pt[1] + randint(0,4)
        # target[i, j, :, :] = draw_gaussian2(to_torch(target[i, j, :, :]), gt, sigma)

#--------------------------------------
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