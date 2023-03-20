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

from layers import create_camera_plane


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
    def __init__(self, batchNorm=False, no_in=3, no_out=64, no_ep=8):
        super(BackBone, self).__init__()
        self.batchNorm = batchNorm
        self.no_ep = no_ep

        # Encoder
        # Encode pixel position from 2 to no_ep
        if no_ep > 0:
            self.conv_ep1 = conv_elu(self.batchNorm, 2, 16, kernel_size=1, pad=0)
            self.conv_ep2 = conv_elu(self.batchNorm, 16, no_ep, kernel_size=1, pad=0)

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
        self.conv6 = conv_elu(self.batchNorm, 256 + no_ep, 256, pad=1, stride=2)
        self.conv6_1 = residual_block(256)
        self.elu = nn.ELU(inplace=True)

        # Decoder
        # i and up convolutions
        self.deconv6 = deconv(256, 128)
        self.iconv6 = conv_elu(self.batchNorm, 256 + 128, 256)
        self.deconv5 = deconv(256, 128)
        self.iconv5 = conv_elu(self.batchNorm, 128 + 256, 256)
        self.deconv4 = deconv(256, 128)
        self.iconv4 = conv_elu(self.batchNorm, 128 + 256, 256)
        self.deconv3 = deconv(256, 128)
        self.iconv3 = conv_elu(self.batchNorm, 128 + 256, 256)
        self.deconv2 = deconv(256, 128)
        self.iconv2 = conv_elu(self.batchNorm, 128 + 128, 128)
        self.deconv1 = deconv(128, 64)
        self.iconv1 = nn.Conv2d(64 + 64, no_out, kernel_size=3, stride=1, padding=1, bias=False)

        # Initialize conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, grid):
        _, _, H, W = x.shape

        # Encoder section
        # Early feature extraction at target resolution
        out_conv0 = self.conv0_1(self.conv0(x))

        # One strided conv encoder stage
        if self.no_ep > 0:
            grid = self.conv_ep2(self.conv_ep1(grid))
            out_conv1 = self.conv1_1(self.conv1(torch.cat((out_conv0, grid), 1)))

            # Early geature extraction at half resolution
            out_conv0lr = self.conv0l_1(self.conv0l(
                F.interpolate(x, size=(out_conv1.shape[2], out_conv1.shape[3]), mode='bilinear', align_corners=True)))

            # Deep feature extraction
            dgrid = F.interpolate(grid, size=(out_conv1.shape[2], out_conv1.shape[3]), align_corners=True, mode='bilinear')
            out_conv2 = self.conv2_1(self.conv2(torch.cat((out_conv1, out_conv0lr, dgrid), 1)))
            dgrid = F.interpolate(grid, size=(out_conv2.shape[2], out_conv2.shape[3]), align_corners=True, mode='bilinear')
            out_conv3 = self.conv3_1(self.conv3(torch.cat((out_conv2, dgrid), 1)))
            dgrid = F.interpolate(grid, size=(out_conv3.shape[2], out_conv3.shape[3]), align_corners=True, mode='bilinear')
            out_conv4 = self.conv4_1(self.conv4(torch.cat((out_conv3, dgrid), 1)))
            dgrid = F.interpolate(grid, size=(out_conv4.shape[2], out_conv4.shape[3]), align_corners=True, mode='bilinear')
            out_conv5 = self.conv5_1(self.conv5(torch.cat((out_conv4, dgrid), 1)))
            dgrid = F.interpolate(grid, size=(out_conv5.shape[2], out_conv5.shape[3]), align_corners=True, mode='bilinear')
            out_conv6 = self.conv6_1(self.conv6(torch.cat((out_conv5, dgrid), 1)))
        else:
            out_conv1 = self.conv1_1(self.conv1(out_conv0))
            # Early geature extraction at half resolution
            out_conv0lr = self.conv0l_1(self.conv0l(
                F.interpolate(x, size=(out_conv1.shape[2], out_conv1.shape[3]), mode='bilinear', align_corners=True)))
            out_conv2 = self.conv2_1(self.conv2(torch.cat((out_conv1, out_conv0lr), 1)))
            out_conv3 = self.conv3_1(self.conv3(out_conv2))
            out_conv4 = self.conv4_1(self.conv4(out_conv3))
            out_conv5 = self.conv5_1(self.conv5(out_conv4))
            out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # Decoder section
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

        return dlog, concat1


class PladeNet(nn.Module):
    def __init__(self, batchNorm,
                 no_levels,
                 disp_min, 
                 disp_max, 
                 num_ep=0, 
                 xz_levels=0, 
                 xz_min=0.1852, xz_max=0.3704, 
                 use_mixture_loss=False, 
                 render_probability=False, 
                 plane_residual=False):
        super(PladeNet, self).__init__()
        self.no_levels = no_levels
        self.xz_levels = xz_levels
        self.no_fac = 1
        self.num_ep = num_ep
        if render_probability:
            self.backbone = BackBone(batchNorm, no_in=3, no_out=self.no_levels + self.xz_levels - 1, no_ep=num_ep)
        else:
            self.backbone = BackBone(batchNorm, no_in=3, no_out=self.no_levels + self.xz_levels, no_ep=num_ep)
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.disp_min = disp_min
        self.disp_max = disp_max
        self.xz_min = xz_min
        self.xz_max = xz_max
        self.use_mixture_loss = use_mixture_loss
        self.render_probability = render_probability
        self.plane_residual = plane_residual

        # An additional 1x1 conv layer on the logits (not shown in paper). Its contribution should not be much.
        if render_probability:
            self.conv0 = nn.Conv2d(self.no_levels + self.xz_levels - 1, self.no_fac * self.no_levels + self.xz_levels - 1, kernel_size=1, stride=1, padding=0,
                               bias=True)
        else:
            self.conv0 = nn.Conv2d(self.no_levels + self.xz_levels, self.no_fac * self.no_levels + self.xz_levels, kernel_size=1, stride=1, padding=0,
                                bias=True)
        nn.init.kaiming_normal_(self.conv0.weight.data)  # initialize weigths with normal distribution
        self.conv0.bias.data.zero_()  # initialize bias as zero
        
        if self.use_mixture_loss:
            self.conv_sigma = nn.Conv2d(64 + 64, self.no_levels + self.xz_levels, kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.kaiming_normal_(self.conv_sigma.weight.data)  # initialize weigths with normal distribution
            
        if self.plane_residual:
            self.conv_residual = nn.Conv2d(64 + 64, self.no_levels + self.xz_levels, kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.kaiming_normal_(self.conv_residual.weight.data)  # initialize weigths with normal distribution
        
        self.normalize = torchvision.transforms.Normalize(mean=[0.411, 0.432, 0.45],
                                std=[1, 1, 1])
        
        # self.disp_layered = disp_max * (disp_min / disp_max)**(torch.arange(no_levels) / (no_levels-1)) # N
        
        # self.disp_layered = self.disp_layered[None, :, None, None].expand(-1, -1, self.height, self.width).cuda()
        
        # self.camera_plane = create_camera_plane(height=self.height, width=self.width)
        # if self.xz_levels > 0:
        #     y_coords = self.camera_plane[:, 1:2, :, :]
        #     y_coords[y_coords<1e-7] = 1e-7
        #     self.xz_layer = torch.linspace(xz_min, xz_max, self.xz_levels)
        #     self.xz_layer = self.xz_layer[None, :, None, None].expand(-1, -1, self.height, self.width).cuda()
        #     self.xz_layer = self.xz_layer / y_coords
        #     self.xz_layer = 0.1 * 0.58 * self.width / self.xz_layer
            
        #     self.disp_layered = torch.cat([self.disp_layered, self.xz_layer], dim=1)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, input_left, input_grids=None):
        input_left = self.normalize(input_left)
        self.outputs = {}
        B, C, H, W = input_left.shape
        
        # Synthesize zoomed image
        dlog, features = self.backbone(input_left, input_grids)
        
        disp_levels = torch.arange(self.no_levels).cuda()[None, :, None, None]
        disp_levels = disp_levels.expand(B, -1, -1, -1)
        if self.plane_residual:
            residual_levels = self.sigmoid(self.conv_residual(features)) - 0.5 #B, N, 1, 1
            disp_levels = disp_levels + residual_levels[:, :self.no_levels, ...]
        disp_layered = self.disp_max * (self.disp_min / self.disp_max)**(disp_levels / (self.no_levels-1)) # B, N, 1, 1
        disp_layered = disp_layered.expand(-1, -1, H, W)
        padding_mask = torch.ones_like(disp_layered)
        if self.xz_levels > 0:
            ground_levels = torch.arange(self.xz_levels).cuda()[None, :, None, None]
            ground_levels = ground_levels.expand(B, -1, -1, -1)
            if self.plane_residual:
                ground_levels = ground_levels + residual_levels[:, -self.xz_levels:, ...]
            ground_layered = self.xz_min + (self.xz_max - self.xz_min) * ground_levels / (self.xz_levels-1)
            ground_layered = ground_layered.expand(-1, -1, H, W)
            y_grids = input_grids[:, 1:, ...].clone()
            xz_padding_mask = (y_grids>=1e-7).expand(-1, self.xz_levels, -1, -1)
            y_grids[y_grids<1e-7] = 1e-7
            ground_layered = ground_layered * 1.92 / (y_grids / 2.)
            ground_layered = (input_grids[:, :1, :, -1:] - input_grids[:, :1, :, :1]) / 2. * ground_layered
            ground_layered = 0.1 * 0.58 * W / ground_layered
            disp_layered = torch.cat([disp_layered, ground_layered], dim=1)
            padding_mask = torch.cat([padding_mask, xz_padding_mask], dim=1)
        self.outputs["disp_layered"] = disp_layered
        self.outputs["padding_mask"] = padding_mask

        # Get shifted dprob
        self.outputs["logits"] = self.conv0(dlog)
        # print(self.outputs["logits"][0, :, 100, 300])
        if self.render_probability:
            depth_layered = 0.1 * 0.58 * W / disp_layered
            dists = depth_layered[:, 1:, ...] - depth_layered[:, :-1, ...]
            # dists = torch.cat([dists, 1e10 * torch.ones_like(dists[:, :1])], dim=1)
            camera_plane = create_camera_plane(height=H, width=W)
            dists = dists * torch.linalg.norm(camera_plane, dim=1, keepdim=True)
            self.outputs["dists"] = dists
            alpha = 1. - torch.exp(-F.relu(self.outputs["logits"]) * dists)
            ones = torch.ones_like(alpha[:, :1, ...])
            alpha = torch.cat([alpha, ones], dim=1)
            probability = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1, ...]), 1.-alpha+1e-10], dim=1), dim=1)[:, :-1, ...]
            self.outputs["probability"] = probability
            self.outputs["logits"] = torch.cat([self.outputs["logits"], ones], dim=1)
        else:
            self.outputs["probability"] = self.softmax(self.outputs["logits"])
            
        if self.use_mixture_loss:
            sigma = self.sigmoid(self.conv_sigma(features))
            sigma = torch.clamp(sigma, 0.01, 1.)
            self.outputs["sigma"] = sigma
            self.outputs["pi"] = pi = self.outputs["probability"]
            weights = pi / sigma
            weights = weights / weights.sum(1, True)
            self.outputs["probability"] = weights
            candidates_idx = weights.argmax(1, True)
            #self.outputs["disp"] = torch.gather(self.outputs["disp_layered"], 1, candidates_idx)#
        
        # candidates_idx = self.outputs["probability"].argmax(1, True)
        # self.outputs["disp"] = torch.gather(self.outputs["disp_layered"], 1, candidates_idx)
        
        self.outputs["disp"] = (self.outputs["probability"] * self.outputs["disp_layered"]).sum(1, True)
        
        self.outputs["depth"] = 0.1 * 0.58 * W / self.outputs["disp"]

        return self.outputs