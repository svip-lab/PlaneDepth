# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .denseaspp import DenseAspp

from collections import OrderedDict
from layers import *

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, 
                 no_levels=49, 
                 disp_min=2, 
                 disp_max=300, 
                 num_ep=0,
                 pe_type="neural",
                 use_skips=True, 
                 use_denseaspp=True, 
                 xz_levels=0, 
                 xz_min=0.1852, xz_max=0.3704, #xz_min=0.2315, xz_max=0.3426,#debugxz_min=0.001, xz_max=0.3704,#debug
                 yz_levels=0,
                 yz_min=0.1, yz_max=10.,
                 use_mixture_loss=False, 
                 render_probability=False,
                 plane_residual=False):
        super(DepthDecoder, self).__init__()

        self.no_levels = no_levels
        self.xz_levels = xz_levels
        self.yz_levels = yz_levels
        self.all_levels = self.no_levels + self.xz_levels + self.yz_levels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.disp_min = disp_min
        self.disp_max = disp_max
        self.xz_min = xz_min
        self.xz_max = xz_max
        self.yz_min = yz_min
        self.yz_max = yz_max
        self.num_ep = num_ep
        self.pe_type = pe_type
        self.use_mixture_loss = use_mixture_loss
        self.render_probability = render_probability
        self.plane_residual = plane_residual

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        self.use_denseaspp = use_denseaspp

        print("use {} xy planes, {} xz planes and {} yz planes.".format(self.no_levels, self.xz_levels, self.yz_levels))
        
        # decoder
        self.convs = OrderedDict()
        
        if self.num_ep > 0:
            if self.pe_type == "neural":
                self.convs["epconv"] = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ELU(inplace=True),
                nn.Conv2d(16, self.num_ep, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ELU(inplace=True)
                )
            elif self.pe_type == "frequency":
                self.convs["epconv"] = get_embedder((self.num_ep//2 - 1)//2)
        
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1]+self.num_ep if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            if i > 0:
                num_ch_in += self.num_ep
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        if use_denseaspp:
            print("use DenseAspp Block")
            self.convs["denseaspp"] = DenseAspp()
            
        #debug +14
        if render_probability:
            self.convs["dispconv"] = Conv3x3(self.num_ch_dec[0], self.all_levels - 1)#+14)
        else:
            self.convs["dispconv"] = Conv3x3(self.num_ch_dec[0], self.all_levels)#+14)
        
        if self.use_mixture_loss:
            print("use mixture Lap loss")
            self.convs["sigmaconv"] = Conv3x3(self.num_ch_dec[0], self.all_levels)#+14)
            
        
            
        if self.plane_residual:
            print("use plane residual")
            self.convs["residualconv"] = nn.Sequential(nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0], 1),
                                                       nn.AdaptiveAvgPool2d((1, 1)),
                                                       nn.Conv2d(self.num_ch_dec[0], self.all_levels, 1))


        # self.convs["angleconv"] = nn.Conv2d(self.num_ch_dec[0], 1, 3)
        

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        

    def forward(self, input_features, input_grids=None):
        self.outputs = {}
        
        if self.num_ep > 0:
            grids_ep = self.convs["epconv"](input_grids)

        # decoder
        x = input_features[-1]
        if self.num_ep > 0:
            dgrid = F.interpolate(grids_ep, size=(x.shape[2], x.shape[3]), align_corners=True, mode='bilinear')
            x = torch.cat([x, dgrid], dim=1)
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            if self.num_ep > 0 and i > 0:
                dgrid = F.interpolate(grids_ep, size=(x.shape[2], x.shape[3]), align_corners=True, mode='bilinear')
                x = torch.cat([x, dgrid], dim=1)
            x = self.convs[("upconv", i, 1)](x)
            
            if i == 4 and self.use_denseaspp:
                x = self.convs["denseaspp"](x)
                
        # angle = (self.sigmoid(self.convs["angleconv"](x).mean(dim=-1).mean(dim=-1)) - 0.5) * 0.75 * np.pi

        B, _, H, W = x.shape
        disp_levels = torch.arange(self.no_levels).cuda()[None, :, None, None]
        disp_levels = disp_levels.expand(B, -1, -1, -1)
        if self.plane_residual:
            residual_levels = self.sigmoid(self.convs["residualconv"](x)) - 0.5 #B, N, 1, 1
            disp_levels = disp_levels + residual_levels[:, :self.no_levels, ...]
        disp_layered = self.disp_max * (self.disp_min / self.disp_max)**(disp_levels / (self.no_levels-1)) # B, N, 1, 1
        distance = 0.1 * 0.58 * W / disp_layered[:, :, 0, 0]
        norm = torch.tensor([0, 0, 1]).cuda()[None, None, :].expand(B, self.no_levels, -1)
        disp_layered = disp_layered.expand(-1, -1, H, W)
        padding_mask = torch.ones_like(disp_layered)
        if self.xz_levels > 0:
            ground_levels = torch.arange(self.xz_levels).cuda()[None, :, None, None]
            ground_levels = ground_levels.expand(B, -1, -1, -1)
            if self.plane_residual:
                ground_levels = ground_levels + residual_levels[:, self.no_levels:self.no_levels+self.xz_levels, ...]
            ground_layered = self.xz_min + (self.xz_max - self.xz_min) * ground_levels / (self.xz_levels-1)
            h = ground_layered[:, :, 0, 0]
            ground_layered = ground_layered.expand(-1, -1, H, W)
            y_grids = input_grids[:, 1:, ...].clone()
            
            xz_padding_mask = (y_grids>=1e-7).expand(-1, self.xz_levels, -1, -1)
            
            y_grids[y_grids<1e-7] = 1e-7
            ground_layered = ground_layered * 1.92 / (y_grids / 2.)
            ground_layered = (input_grids[:, :1, :, -1:] - input_grids[:, :1, :, :1]) / 2. * ground_layered
            
            # angle
            # y = torch.linspace(-1, 1, 384).cuda()
            # y_idv =  (y[None, None, :] + torch.tan(-angle)[..., None])
            # y_idv[y_idv<1e-7] = 1e-7
            # ground_layered = h[:, :, None].expand(-1, -1, 384) /y_idv
            # ground_layered = ground_layered[..., None].expand(-1, -1, -1, 1280)
            
            ground_layered = 0.1 * 0.58 * W / ground_layered
            disp_layered = torch.cat([disp_layered, ground_layered], dim=1)
            padding_mask = torch.cat([padding_mask, xz_padding_mask], dim=1)
            
            ######debug#######
            # yz_levels = torch.arange(7).cuda()[None, :, None, None]
            # yz_levels = yz_levels.expand(B, -1, -1, -1)
            # yz_disp_min = 0.1 * 0.58 * 1280 / 10
            # yz_disp_max = 0.1 * 0.58 * 1280 / 0.05
            # yz_disp_layered = yz_disp_max * (yz_disp_min / yz_disp_max)**(yz_levels / 6) # B, N, 1, 1
            # yz_depth_layered = 0.1 * 0.58 * 1280 / yz_disp_layered
            # yz_depth_layered = yz_depth_layered.expand(-1, -1, H, W)
            # x_grids_0 = input_grids[:, :1, ...].clone()
            # yz_padding_mask_0 = (x_grids_0>=1e-7).expand(-1, 7, -1, -1)
            # x_grids_0[x_grids_0<1e-7] = 1e-7
            # yz_depth_layered_0 = yz_depth_layered * 0.58 / (x_grids_0 / 2.)
            # yz_depth_layered_0 = (input_grids[:, :1, :, -1:] - input_grids[:, :1, :, :1]) / 2. * yz_depth_layered_0
            
            # x_grids_1 = input_grids[:, :1, ...].clone()
            # yz_padding_mask_1 = (x_grids_1<=-1e-7).expand(-1, 7, -1, -1)
            # x_grids_1[x_grids_1>-1e-7] = -1e-7
            # yz_depth_layered_1 = -yz_depth_layered * 0.58 / (x_grids_1 / 2.)
            # yz_depth_layered_1 = (input_grids[:, :1, :, -1:] - input_grids[:, :1, :, :1]) / 2. * yz_depth_layered_1
            # yz_layered = torch.cat([yz_depth_layered_0, yz_depth_layered_1], dim=1)
            # yz_layered = 0.1 * 0.58 * W / yz_layered
            # disp_layered = torch.cat([disp_layered, yz_layered], dim=1)
            # padding_mask = torch.cat([padding_mask, yz_padding_mask_0, yz_padding_mask_1], dim=1)
            ######debug#######
            
            # original:
            # dgx = input_grids[:, 0, 0, -1] - input_grids[:, 0, 0, 0]
            # dgy = input_grids[:, 1, -1, 0] - input_grids[:, 1, 0, 0]
            # gy_min = input_grids[:, 1, 0, 0]
            # ground_angle = torch.arctan(-(gy_min + 0.5 * dgy) / (1.92 * dgy))
            # norm_angle = ground_angle + np.pi / 2
            # xz_norm = torch.stack([torch.zeros_like(norm_angle), torch.sin(norm_angle), torch.cos(norm_angle)], dim=1)
            # xz_norm = xz_norm[:, None, :].expand(-1, self.xz_levels, -1)
            # norm = torch.cat([norm, xz_norm], dim=1)
            # xz_distance = h * torch.sin(norm_angle)[:, None] * dgx[:, None] / dgy[:, None]
            # distance = torch.cat([distance, xz_distance], dim=1)
            # paper:
            gyc = (input_grids[:, 1, -1, 0] + input_grids[:, 1, 0, 0]) / 2
            py = (gyc + 1) * H / 2
            fs = (input_grids[:, 0, 0, -1] - input_grids[:, 0, 0, 0]) / 2.
            py_cy_fys = (py - H/2) / (H * 1.92 * fs)
            xz_norm = torch.stack([torch.zeros_like(py_cy_fys), torch.ones_like(py_cy_fys), py_cy_fys*torch.ones_like(py_cy_fys)], dim=1)
            xz_normalize = 1 / ((1+py_cy_fys**2)**0.5)
            xz_norm = xz_norm * xz_normalize[:, None]
            xz_distance = h * xz_normalize[:, None]
            xz_norm = xz_norm[:, None, :].expand(-1, self.xz_levels, -1)
            norm = torch.cat([norm, xz_norm], dim=1)
            distance = torch.cat([distance, xz_distance], dim=1)
            
        if self.yz_levels > 0:
            yz_levels = torch.arange(self.yz_levels//2).cuda()[None, :, None, None] #1, 0.5N, 1, 1
            yz_levels = torch.cat([yz_levels, yz_levels], dim=1) #1, N, 1, 1
            yz_levels = yz_levels.expand(B, -1, -1, -1) #B, N, 1, 1
            if self.plane_residual:
                yz_levels = yz_levels + residual_levels[:, -self.yz_levels:, ...]
            yz_disp_max = 1. / self.yz_min
            yz_disp_min = 1. / self.yz_max
            yz_disp_layered = yz_disp_max * (yz_disp_min / yz_disp_max)**(yz_levels / (0.5*self.yz_levels-1)) # B, N, 1, 1
            yz_layered = 1. / yz_disp_layered
            h = yz_layered[:, :, 0, 0]
            
            yz_layered_r = yz_layered[:, :self.yz_levels//2, ...].expand(-1, -1, H, W)
            x_grids_r = input_grids[:, :1, ...].clone()
            yz_padding_mask_r = (x_grids_r>=1e-7).expand(-1, self.yz_levels // 2, -1, -1)
            x_grids_r[x_grids_r<1e-7] = 1e-7
            yz_layered_r = yz_layered_r * 0.58 / (x_grids_r / 2.)
            yz_layered_r = (input_grids[:, :1, :, -1:] - input_grids[:, :1, :, :1]) / 2. * yz_layered_r
            
            yz_layered_l = yz_layered[:, -self.yz_levels//2:, ...].expand(-1, -1, H, W)
            x_grids_l = input_grids[:, :1, ...].clone()
            yz_padding_mask_l = (x_grids_l<=-1e-7).expand(-1, self.yz_levels // 2, -1, -1)
            x_grids_l[x_grids_l>-1e-7] = -1e-7
            yz_layered_l = -yz_layered_l * 0.58 / (x_grids_l / 2.)
            yz_layered_l = (input_grids[:, :1, :, -1:] - input_grids[:, :1, :, :1]) / 2. * yz_layered_l
            
            yz_layered = torch.cat([yz_layered_r, yz_layered_l], dim=1)
            yz_layered = 0.1 * 0.58 * W / yz_layered
            disp_layered = torch.cat([disp_layered, yz_layered], dim=1)
            padding_mask = torch.cat([padding_mask, yz_padding_mask_r, yz_padding_mask_l], dim=1)
            
            # All normals face outward
            gxc = (input_grids[:, 0, 0, -1] + input_grids[:, 0, 0, 0]) / 2
            px = (gxc + 1) * W / 2
            fs = (input_grids[:, 0, 0, -1] - input_grids[:, 0, 0, 0]) / 2.
            px_cx_fxs = (px - W/2) / (W * 0.58 * fs)
            yz_norm = torch.stack([torch.ones_like(px_cx_fxs), torch.zeros_like(px_cx_fxs), px_cx_fxs*torch.ones_like(px_cx_fxs)], dim=1)
            yz_normalize = 1 / ((1+px_cx_fxs**2)**0.5)
            yz_norm = yz_norm * yz_normalize[:, None]
            yz_distance = h * yz_normalize[:, None]
            yz_norm_r = yz_norm[:, None, :].expand(-1, self.yz_levels // 2, -1)
            yz_norm_l = -yz_norm[:, None, :].expand(-1, self.yz_levels // 2, -1)
            norm = torch.cat([norm, yz_norm_r, yz_norm_l], dim=1)
            distance = torch.cat([distance, yz_distance], dim=1)

        self.outputs["distance"] = distance
        self.outputs["norm"] = norm
        self.outputs["disp_layered"] = disp_layered
        self.outputs["padding_mask"] = padding_mask
        logits = self.convs["dispconv"](x)
        logits = logits * padding_mask
        self.outputs["logits"] = logits
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
            sigma = self.sigmoid(self.convs["sigmaconv"](x))
            sigma = torch.clamp(sigma, 0.01, 1.)
            self.outputs["sigma"] = sigma
            self.outputs["pi"] = pi = self.outputs["probability"]
            weights = pi / sigma
            weights = weights * padding_mask
            weights = weights / weights.sum(1, True)
            self.outputs["probability"] = weights
            candidates_idx = weights.argmax(1, True)
            #self.outputs["disp"] = torch.gather(self.outputs["disp_layered"], 1, candidates_idx)#
            
        self.outputs["disp"] = (self.outputs["probability"] * self.outputs["disp_layered"]).sum(1, True)
        #print(self.outputs["probability"][0, :, 200, 600].max())
        self.outputs["depth"] = 0.1 * 0.58 * W / self.outputs["disp"]

        return self.outputs
    
    
class DepthDecoderContinuous(nn.Module):
    def __init__(self, num_ch_enc, 
                 no_levels=49, 
                 disp_min=2, 
                 disp_max=300, 
                 num_ep=0,
                 pe_type="neural",
                 use_skips=True, 
                 use_denseaspp=True, 
                 xz_levels=0, 
                 xz_min=0.1852, xz_max=0.3704, #xz_min=0.2315, xz_max=0.3426,#
                 use_mixture_loss=False, 
                 render_probability=False,
                 plane_residual=False):
        super(DepthDecoderContinuous, self).__init__()

        self.no_levels = no_levels
        self.xz_levels = xz_levels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.disp_min = disp_min
        self.disp_max = disp_max
        self.xz_min = xz_min
        self.xz_max = xz_max
        self.num_ep = num_ep
        self.pe_type = pe_type
        self.use_mixture_loss = use_mixture_loss
        self.render_probability = render_probability
        self.plane_residual = plane_residual

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        self.use_denseaspp = use_denseaspp

        print("use {} xy plane and {} xz plane.".format(self.no_levels, self.xz_levels))
        
        # decoder
        self.convs = OrderedDict()
        
        if self.num_ep > 0:
            if self.pe_type == "neural":
                self.convs["epconv"] = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ELU(inplace=True),
                nn.Conv2d(16, self.num_ep, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ELU(inplace=True)
                )
            elif self.pe_type == "frequency":
                self.convs["epconv"] = get_embedder((self.num_ep//2 - 1)//2)
        
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1]+self.num_ep if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            if i > 0:
                num_ch_in += self.num_ep
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        if use_denseaspp:
            print("use DenseAspp Block")
            self.convs["denseaspp"] = DenseAspp()
            
        self.convs["dispconv"] = Conv3x3(self.num_ch_dec[0], self.no_levels + self.xz_levels)
            
        if render_probability:
            self.convs["piconv"] = Conv3x3(self.num_ch_dec[0], self.no_levels + self.xz_levels - 1)
        else:
            self.convs["piconv"] = Conv3x3(self.num_ch_dec[0], self.no_levels + self.xz_levels)
        
        if self.use_mixture_loss:
            print("use mixture Lap loss")
            self.convs["sigmaconv"] = Conv3x3(self.num_ch_dec[0], self.no_levels + self.xz_levels)
            
        if self.plane_residual:
            print("use plane residual")
            self.convs["residualconv"] = nn.Sequential(nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0], 1),
                                                       nn.AdaptiveAvgPool2d((1, 1)),
                                                       nn.Conv2d(self.num_ch_dec[0], self.no_levels + self.xz_levels, 1))

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        

    def forward(self, input_features, input_grids=None):
        self.outputs = {}
        
        if self.num_ep > 0:
            grids_ep = self.convs["epconv"](input_grids)

        # decoder
        x = input_features[-1]
        if self.num_ep > 0:
            dgrid = F.interpolate(grids_ep, size=(x.shape[2], x.shape[3]), align_corners=True, mode='bilinear')
            x = torch.cat([x, dgrid], dim=1)
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            if self.num_ep > 0 and i > 0:
                dgrid = F.interpolate(grids_ep, size=(x.shape[2], x.shape[3]), align_corners=True, mode='bilinear')
                x = torch.cat([x, dgrid], dim=1)
            x = self.convs[("upconv", i, 1)](x)
            
            if i == 4 and self.use_denseaspp:
                x = self.convs["denseaspp"](x)

        B, _, H, W = x.shape

        disp_levels = self.sigmoid(self.convs["dispconv"](x))#B, N, H, W
        self.outputs["disp_levels"] = disp_levels
        disp_layered = self.disp_max * (self.disp_min / self.disp_max)**disp_levels # B, N, H, W

        self.outputs["disp_layered"] = disp_layered
        logits = self.convs["piconv"](x)
        self.outputs["logits"] = logits
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
            sigma = self.sigmoid(self.convs["sigmaconv"](x))
            sigma = torch.clamp(sigma, 0.01, 1.)
            self.outputs["sigma"] = sigma
            self.outputs["pi"] = pi = self.outputs["probability"]
            weights = pi / sigma
            # weights = weights * padding_mask
            weights = weights / weights.sum(1, True)
            self.outputs["probability"] = weights
            candidates_idx = weights.argmax(1, True)
            #self.outputs["disp"] = torch.gather(self.outputs["disp_layered"], 1, candidates_idx)#
            
        self.outputs["disp"] = (self.outputs["probability"] * self.outputs["disp_layered"]).sum(1, True)
        self.outputs["depth"] = 0.1 * 0.58 * W / self.outputs["disp"]

        return self.outputs
