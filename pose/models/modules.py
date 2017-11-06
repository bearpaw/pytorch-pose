'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['Bottleneck', 'PyramidResidual']


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
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


class PyramidResidual(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, basewidth=9, cardinality=4):
        super(PyramidResidual, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.basewidth = basewidth
        self.cardinality = cardinality

        # build feature pyramid
        D = int(math.floor( planes * 2.0 / self.basewidth))
        C = cardinality
        # import pdb; pdb.set_trace()
        self.prepyra = nn.Sequential(
            nn.Conv2d(inplanes, D, kernel_size=1, bias=True),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
        )

        # upsampling is dynamically adjust to input resolution during forward
        sc = 2 ** (1.0 / C)
        pyra = []
        for c in range(0, C):
            scaled = 1 / (sc ** (c+1))
            s = nn.Sequential(
                nn.FractionalMaxPool2d(2, output_ratio=(scaled, scaled)),
                nn.Conv2d(D, D, kernel_size=3, padding=1, bias=True),
            )
            pyra.append(s)
        self.pyra = nn.ModuleList(pyra)
        self.postpyra = nn.Sequential(
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            nn.Conv2d(D, planes, kernel_size=1),
        )

    def forward(self, x):
        residual = x

        # main branch with original resolution
        main = self.bn1(x)
        main = self.relu(main)
        main = self.conv1(main)

        main = self.bn2(main)
        main = self.relu(main)
        main = self.conv2(main)

        # feature pyramid
        prepyra = self.prepyra(x)
        height, width = prepyra.size(2), prepyra.size(3)
        pyra = F.upsample(self.pyra[0](prepyra), size=(height, width), mode='bilinear')
        for c in range(1, self.cardinality):
            pyra += F.upsample(self.pyra[c](prepyra), size=(height, width), mode='bilinear')
        pyra = self.postpyra(pyra)

        # combine main branch and pyramid
        out = main + pyra

        # mapping back to the original channel
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
