# Author: Tao Hu <taohu620@gmail.com>
import torch
import torch.nn as nn
import torch.nn.functional as func

class HSM(nn.Module):
    def __init__(self, beta, gamma):
        super(HSM, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target):
        delta = torch.abs(input - target)
        loss = (1-torch.exp(-self.beta*delta))*torch.pow(delta,self.gamma)
        loss = loss.mean()
        return loss