# Copyright (C) 2021  Juan Luis Gonzalez Bello (juanluisgb@kaist.ac.kr)
# This software is not for commercial use
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# def FAL_net2B_gep(data=None, no_levels=49):
#     model = FAL_net(batchNorm=False, no_levels=no_levels)
#     if data is not None:
#         model.load_state_dict(data['state_dict'])
#     return model


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


class BackBone(nn.Module):
    def __init__(self, batchNorm=False, no_in=3, no_out=64):
        super(BackBone, self).__init__()
        self.batchNorm = batchNorm

        # Encoder
        self.batchNorm = batchNorm
        self.conv0 = conv_elu(self.batchNorm, no_in, 32, kernel_size=3)
        self.conv0_1 = residual_block(32)
        self.conv1 = conv_elu(self.batchNorm, 32, 64, pad=1, stride=2)
        self.conv1_1 = residual_block(64)
        self.conv2 = conv_elu(self.batchNorm, 64, 128, pad=1, stride=2)
        self.conv2_1 = residual_block(128)
        self.conv3 = conv_elu(self.batchNorm, 128, 256, pad=1, stride=2)
        self.conv3_1 = residual_block(256)
        self.conv4 = conv_elu(self.batchNorm, 256, 256, pad=1, stride=2)
        self.conv4_1 = residual_block(256)
        self.conv5 = conv_elu(self.batchNorm, 256, 256, pad=1, stride=2)
        self.conv5_1 = residual_block(256)
        self.conv6 = conv_elu(self.batchNorm, 256, 512, pad=1, stride=2)
        self.conv6_1 = residual_block(512)
        self.elu = nn.ELU(inplace=True)

        # i and up convolutions
        self.deconv6 = deconv(512, 256)
        self.iconv6 = conv_elu(self.batchNorm, 256 + 256, 256)
        self.deconv5 = deconv(256, 128)
        self.iconv5 = conv_elu(self.batchNorm, 128 + 256, 256)
        self.deconv4 = deconv(256, 128)
        self.iconv4 = conv_elu(self.batchNorm, 128 + 256, 256)
        self.deconv3 = deconv(256, 128)
        self.iconv3 = conv_elu(self.batchNorm, 128 + 128, 128)
        self.deconv2 = deconv(128, 64)
        self.iconv2 = conv_elu(self.batchNorm, 64 + 64, 64)
        self.deconv1 = deconv(64, 64)
        self.iconv1 = nn.Conv2d(32 + 64, no_out, kernel_size=3, stride=1, padding=1, bias=False)

        # Initialize conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        _, _, H, W = x.shape

        # Encoder section
        out_conv0 = self.conv0_1(self.conv0(x))
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        out_deconv6 = self.deconv6(out_conv6, out_conv5)
        concat6 = torch.cat((out_deconv6, out_conv5), 1)
        iconv6 = self.iconv6(concat6)

        out_deconv5 = self.deconv5(iconv6, out_conv4)
        concat5 = torch.cat((out_deconv5, out_conv4), 1)
        iconv5 = self.iconv5(concat5)

        out_deconv4 = self.deconv4(iconv5, out_conv3)
        concat4 = torch.cat((out_deconv4, out_conv3), 1)
        iconv4 = self.iconv4(concat4)

        out_deconv3 = self.deconv3(iconv4, out_conv2)
        concat3 = torch.cat((out_deconv3, out_conv2), 1)
        iconv3 = self.iconv3(concat3)

        out_deconv2 = self.deconv2(iconv3, out_conv1)
        concat2 = torch.cat((out_deconv2, out_conv1), 1)
        iconv2 = self.iconv2(concat2)

        out_deconv1 = self.deconv1(iconv2, out_conv0)
        concat1 = torch.cat((out_deconv1, out_conv0), 1)
        dlog = self.iconv1(concat1)

        return dlog


class FalNet(nn.Module):
    def __init__(self, batchNorm, height, width, no_levels, disp_min, disp_max):
        super(FalNet, self).__init__()
        self.no_levels = no_levels
        self.no_fac = 1
        self.backbone = BackBone(batchNorm, no_in=3, no_out=self.no_levels)
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.height = height
        self.width = width

        # An additional 1x1 conv layer on the logits (not shown in paper). Its contribution should not be much.
        self.conv0 = nn.Conv2d(self.no_levels, self.no_fac * self.no_levels, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv0.weight.data)  # initialize weigths with normal distribution
        self.conv0.bias.data.zero_()  # initialize bias as zero
        
        self.normalize = torchvision.transforms.Normalize(mean=[0.411, 0.432, 0.45],
                                std=[1, 1, 1])
        
        self.disp_layered = disp_max * (disp_min / disp_max)**(torch.arange(no_levels) / (no_levels-1)) # N
        
        self.disp_layered = self.disp_layered[None, :, None, None].expand(-1, -1, self.height, self.width).cuda()

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, input_left):
        input_left = self.normalize(input_left)
        self.outputs = {}
        B, C, H, W = input_left.shape

        # Synthesize zoomed image
        dlog = self.backbone(input_left)

        # Get shifted dprob
        self.outputs["logits"] = self.conv0(dlog)
        # print(self.outputs["logits"][0, :, 100, 300])
        self.outputs["probability"] = self.softmax(self.outputs["logits"])
        # print(self.outputs["probability"][0, :, 100, 300])
        self.outputs["disp_layered"] = self.disp_layered.expand(B, -1, -1, -1)
        self.outputs["padding_mask"] = padding_mask = torch.ones_like(self.disp_layered)
        self.outputs["disp"] = (self.outputs["probability"] * self.outputs["disp_layered"]).sum(1, True)
        self.outputs["depth"] = 0.1 * 0.58 * self.width / self.outputs["disp"]

        return self.outputs