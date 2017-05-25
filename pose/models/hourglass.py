'''
Hourglass network inserted in the pre-activated Resnet 
Use lr=0.01 for current version
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F
import math
# from .preresnet import BasicBlock, Bottleneck

__all__ = ['HourglassNet', 'hg1', 'hg2', 'hg4', 'hg8']

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        # self.residual = self._make_layer(block, num_blocks, planes)
        # self.bn = nn.BatchNorm2d(planes)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)
        # self.debug = self._make_residual(block, num_blocks, planes)
        # print(self.hg)
        # print(self.hg)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)


    def _hour_glass(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, layers, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 64
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, 32, layers[0])
        self.layer2 = self._make_residual(block, 32, layers[1])
        self.layer3 = self._make_residual(block, self.num_feats, layers[2])
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, fc, score, fc_, score_ = [],  [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, 64, 4))
            fc.append(self._make_fc(ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=False))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=False))
        self.hg = nn.ModuleList(hg)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc)
        self.score_ = nn.ModuleList(score_)

        # # initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, planes):
        bn = nn.BatchNorm2d(planes)
        conv = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        return nn.Sequential(
                bn,
                self.relu,
                conv
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    

        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.maxpool(x)
        x = self.layer3(x)  

        y = x.clone()
        for i in range(self.num_stacks):
            x = self.hg[i](x)
            x = self.fc[i](x)
            score = self.score[i](x)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](x)
                score_ = self.score_[i](score)
                x = y + fc_ + score_
                y = x.clone()

        return out

def hg1(**kwargs):
    model = HourglassNet(Bottleneck, [1, 1, 1], num_stacks=1, num_blocks=8, **kwargs)
    return model

def hg2(**kwargs):
    model = HourglassNet(Bottleneck, [1, 1, 1], num_stacks=2, num_blocks=4, **kwargs)
    return model

def hg4(**kwargs):
    model = HourglassNet(Bottleneck, [1, 1, 1], num_stacks=4, num_blocks=2, **kwargs)
    return model

def hg8(**kwargs):
    model = HourglassNet(Bottleneck, [1, 1, 1], num_stacks=8, num_blocks=1, **kwargs)
    return model