from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict

import torch
import torch.nn as nn
from torchvision.models.resnet import model_zoo
from torchvision.models.resnet import model_urls
from torchvision.models.resnet import BasicBlock, Bottleneck

# Specification
resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
    34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
    50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
    101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
    152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')
}

class ResNetBackbone(nn.Module):

    def __init__(self, block, layers, in_channel=3):
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class DeconvHead(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters, kernel_size, conv_kernel_size, num_joints, depth_dim,
                 with_bias_end=True):
        super(DeconvHead, self).__init__()

        conv_num_filters = num_joints * depth_dim

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert conv_kernel_size == 1 or conv_kernel_size == 3, 'Only support kenerl 1 and 3'
        if conv_kernel_size == 1:
            pad = 0
        elif conv_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(
                nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=output_padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        if with_bias_end:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=True))
        else:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=False))
            self.features.append(nn.BatchNorm2d(conv_num_filters))
            self.features.append(nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        for i, l in enumerate(self.features):
            x = l(x)
        return x


class ResPoseNet(nn.Module):
    def __init__(self, backbone, head):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


# Helper functions
def get_default_network_config():
    config = edict()
    config.from_model_zoo = True
    config.pretrained = ''
    config.num_layers = 50
    config.num_deconv_layers = 3
    config.num_deconv_filters = 256
    config.num_deconv_kernel = 4
    config.final_conv_kernel = 1
    config.depth_dim = 1
    config.input_channel = 3
    return config

def init_pose_net(pose_net, name):
    org_resnet = model_zoo.load_url(model_urls[name])
    # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
    org_resnet.pop('fc.weight', None)
    org_resnet.pop('fc.bias', None)
    pose_net.backbone.load_state_dict(org_resnet)
    print("Init Network from model zoo")

# create network
def pose_resnet(**kwargs):

    cfg = get_default_network_config()

    block_type, layers, channels, name = resnet_spec[kwargs['resnet_layers']]
    backbone_net = ResNetBackbone(block_type, layers)
    head_net = DeconvHead(
        channels[-1], cfg.num_deconv_layers,
        cfg.num_deconv_filters, cfg.num_deconv_kernel,
        cfg.final_conv_kernel, kwargs['num_classes'], cfg.depth_dim
    )
    pose_net = ResPoseNet(backbone_net, head_net)
    init_pose_net(pose_net, name)
    return pose_net
