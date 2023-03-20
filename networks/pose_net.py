# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from collections import OrderedDict


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetPoseEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetPoseEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
    
class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1, num_ep=0):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.num_ep = num_ep

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        
        if self.num_ep > 0:
            self.convs["epconv"] = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ELU(inplace=True),
            nn.Conv2d(16, self.num_ep, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ELU(inplace=True)
            )
        
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256 + num_ep, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features, input_grids=None):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        
        if self.num_ep > 0:
            grids_ep = self.convs["epconv"](input_grids)
            dgrid = F.interpolate(grids_ep, size=(cat_features.shape[2], cat_features.shape[3]), align_corners=True, mode='bilinear')
            cat_features = torch.cat([cat_features, dgrid], dim=1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
    


def conv_elu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, pad=1):
   if batchNorm:
       return nn.Sequential(
           nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad,
                     bias=False),
           nn.BatchNorm2d(out_planes),
           nn.ELU(inplace=True)
       )
   else:
       return nn.Sequential(
           nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad,
                     bias=True),
           nn.ELU(inplace=True)
       )


class deconv(nn.Module):
   def __init__(self, in_planes, out_planes):
       super(deconv, self).__init__()
       self.elu = nn.ELU(inplace=True)
       self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

   def forward(self, x, ref):
       x = F.interpolate(x, size=(ref.size(2), ref.size(3)), mode='nearest')
       x = self.elu(self.conv1(x))
       return x


def conv_gep(in_planes, int_planes, out_planes):
   return nn.Sequential(
       nn.Conv2d(in_planes, int_planes, kernel_size=1, stride=1, padding=0, bias=True),
       nn.ELU(inplace=True),
       nn.Conv2d(int_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
   )


class residual_block(nn.Module):
   def __init__(self, in_planes, kernel_size=3):
       super(residual_block, self).__init__()
       self.elu = nn.ELU(inplace=True)
       self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
       self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

   def forward(self, x):
       x = self.elu(self.conv2(self.elu(self.conv1(x))) + x)
       return x


class PladeBackbone(nn.Module):
   def __init__(self, batchNorm=True, no_in=3, no_ep=8):
       super(PladeBackbone, self).__init__()
       self.batchNorm = batchNorm

       # Encoder
       # Encode pixel position from 2 to no_ep
       self.conv_ep1 = conv_gep(2, 16, no_ep)
       self.conv_ep2 = conv_gep(2, 16, no_ep)
       self.conv_ep3 = conv_gep(2, 16, no_ep)
       self.conv_ep4 = conv_gep(2, 16, no_ep)
       self.conv_ep5 = conv_gep(2, 16, no_ep)
       self.conv_ep6 = conv_gep(2, 16, no_ep)

       # Two input layers at full and half resolution
       self.conv0 = conv_elu(self.batchNorm, no_in, 64, kernel_size=3)
       self.conv0_1 = residual_block(64)
       self.conv0l = conv_elu(self.batchNorm, no_in, 64, kernel_size=3)
       self.conv0l_1 = residual_block(64)

       # Strided convs of encoder
       self.conv1 = conv_elu(self.batchNorm, 64 + no_ep, 128, pad=1, stride=2)
       self.conv1_1 = residual_block(128)
       self.conv2 = conv_elu(self.batchNorm, 128 + 64 + no_ep, 256, pad=1, stride=2)
       self.conv2_1 = residual_block(256)
       self.conv3 = conv_elu(self.batchNorm, 256 + no_ep, 256, pad=1, stride=2)
       self.conv3_1 = residual_block(256)
       self.conv4 = conv_elu(self.batchNorm, 256 + no_ep, 256, pad=1, stride=2)
       self.conv4_1 = residual_block(256)
       self.conv5 = conv_elu(self.batchNorm, 256 + no_ep, 256, pad=1, stride=2)
       self.conv5_1 = residual_block(256)
       self.conv6 = conv_elu(self.batchNorm, 256 * 2 + no_ep, 256, pad=1, stride=2)
       self.conv6_1 = residual_block(256)
       self.elu = nn.ELU(inplace=True)

       # Initialize conv layers
       for m in self.modules():
           if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
               nn.init.kaiming_normal_(m.weight.data)  # initialize weigths with normal distribution
               if m.bias is not None:
                   m.bias.data.zero_()  # initialize bias as zero
           elif isinstance(m, nn.BatchNorm2d):
               m.weight.data.fill_(1)
               m.bias.data.zero_()

   def forward(self, x, y, grid):
       _, _, H, W = x.shape

       ######################################### Left Encoder section##################################################
       # Early feature extraction at target resolution
       out_conv0 = self.conv0_1(self.conv0(x))

       # One strided conv encoder stage
       out_conv1 = self.conv1_1(self.conv1(torch.cat((out_conv0, self.conv_ep1(grid)), 1)))

       # Early geature extraction at half resolution
       out_conv0lr = self.conv0l_1(self.conv0l(
           F.interpolate(x, size=(out_conv1.shape[2], out_conv1.shape[3]), mode='bilinear', align_corners=True)))

       # Deep feature extraction
       dgrid = F.interpolate(grid, size=(out_conv1.shape[2], out_conv1.shape[3]), align_corners=True, mode='bilinear')
       out_conv2 = self.conv2_1(self.conv2(torch.cat((out_conv1, out_conv0lr, self.conv_ep2(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv2.shape[2], out_conv2.shape[3]), align_corners=True, mode='bilinear')
       out_conv3 = self.conv3_1(self.conv3(torch.cat((out_conv2, self.conv_ep3(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv3.shape[2], out_conv3.shape[3]), align_corners=True, mode='bilinear')
       out_conv4 = self.conv4_1(self.conv4(torch.cat((out_conv3, self.conv_ep4(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv4.shape[2], out_conv4.shape[3]), align_corners=True, mode='bilinear')
       out_conv5 = self.conv5_1(self.conv5(torch.cat((out_conv4, self.conv_ep5(dgrid)), 1)))

       ######################################### Right Encoder section##################################################
       # Early feature extraction at target resolution
       out_conv0r = self.conv0_1(self.conv0(y))

       # One strided conv encoder stage
       out_conv1r = self.conv1_1(self.conv1(torch.cat((out_conv0r, self.conv_ep1(grid)), 1)))

       # Early geature extraction at half resolution
       out_conv0lrr = self.conv0l_1(self.conv0l(
           F.interpolate(y, size=(out_conv1r.shape[2], out_conv1r.shape[3]), mode='bilinear', align_corners=True)))

       # Deep feature extraction
       dgrid = F.interpolate(grid, size=(out_conv1r.shape[2], out_conv1r.shape[3]), align_corners=True, mode='bilinear')
       out_conv2r = self.conv2_1(self.conv2(torch.cat((out_conv1r, out_conv0lrr, self.conv_ep2(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv2r.shape[2], out_conv2r.shape[3]), align_corners=True, mode='bilinear')
       out_conv3r = self.conv3_1(self.conv3(torch.cat((out_conv2r, self.conv_ep3(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv3r.shape[2], out_conv3r.shape[3]), align_corners=True, mode='bilinear')
       out_conv4r = self.conv4_1(self.conv4(torch.cat((out_conv3r, self.conv_ep4(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv4r.shape[2], out_conv4r.shape[3]), align_corners=True, mode='bilinear')
       out_conv5r = self.conv5_1(self.conv5(torch.cat((out_conv4r, self.conv_ep5(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv5.shape[2], out_conv5.shape[3]), align_corners=True, mode='bilinear')
       out_conv6 = self.conv6_1(self.conv6(torch.cat((out_conv5, out_conv5r, self.conv_ep6(dgrid)), 1)))

       return out_conv6


class PladePoseNet(nn.Module):
   def __init__(self, batchNorm, num_ep):
       super(PladePoseNet, self).__init__()
       self.backbone = PladeBackbone(batchNorm, no_in=3, no_ep=num_ep)
       self.relu = nn.ReLU()

       # An additional 1x1 conv layer on the logits (not shown in paper). Its contribution should not be much.
       self.convs = OrderedDict()
       self.convs[("pose", 0)] = nn.Conv2d(256, 256, 3, 1, 1)
       self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, 1, 1)
       self.convs[("pose", 2)] = nn.Conv2d(256, 6, 1)
       self.net = nn.ModuleList(list(self.convs.values()))
       
       for k, v in self.convs.items():
           nn.init.kaiming_normal_(v.weight.data)  # initialize weigths with normal distribution
           v.bias.data.zero_()  # initialize bias as zero

   def forward(self, x, y, in_grid):
       B, C, H, W = x.shape

       # Synthesize zoomed image
       dlog = self.backbone(x, y, in_grid)
       
       out = dlog
       for i in range(3):
           out = self.convs[("pose", i)](out)
           if i != 2:
               out = self.relu(out)

       out = out.mean(3).mean(2)

       out = 0.01 * out.view(-1, 1, 1, 6)

       axisangle = out[..., :3]
       translation = out[..., 3:]

       return axisangle, translation