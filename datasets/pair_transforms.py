import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torchvision

class ToTensor(nn.Module):

    def __init__(self):
        super().__init__()
        self.to_tensor = torchvision.transforms.ToTensor()

    def forward(self, inputs):
        for k in list(inputs):
            if "color" in k:
                inputs[k] = self.to_tensor(inputs[k])
        return inputs

class RandomResizeCrop(nn.Module):

    def __init__(self, target_size=(192, 640), factor=(1, 1)):
        super().__init__()
        self.factor = factor
        self.target_size = target_size

    def forward(self, inputs):
        _, H, W = inputs[("color", "r", -1)].shape
        
        min_factor = max(max((self.target_size[0] + 1) / H, (self.target_size[1] + 1) / W), self.factor[0])  # plus one to ensure
        factor = np.random.uniform(low=min_factor, high=self.factor[1])
        h0 = random.randint(0, int(H * factor - self.target_size[0]))
        w0 = random.randint(0, int(W * factor - self.target_size[1]))
        
        grid = torch.meshgrid(torch.linspace(-1, 1, int(W * factor)), torch.linspace(-1, 1, int(H * factor)), indexing="xy")
        grid = torch.stack(grid, dim=0)
        inputs["grid"] = grid[:, h0: h0 + self.target_size[0], w0: w0 + self.target_size[1]].clone()
        
        for k in list(inputs):
            if "color" in k:
                input = inputs[k]
                n, im, i = k
                input = F.interpolate(input[None, ...], scale_factor=factor, mode="bicubic", align_corners=True, recompute_scale_factor=False)[0]
                input = input.clamp(min=0., max=1.)
                input = input[:, h0: h0 + self.target_size[0], w0: w0 + self.target_size[1]]
                inputs[(n, im)] = input
                inputs[(n + "_aug", im)] = input.clone()
                inputs[k] = F.interpolate(inputs[k][None, ...], size=self.target_size, mode="bicubic", align_corners=True)[0].clamp(min=0., max=1.)
                
        return inputs
    
class Resize(nn.Module):

    def __init__(self, target_size=(192, 640)):
        super().__init__()
        self.target_size = target_size
        grid = torch.meshgrid(torch.linspace(-1, 1, target_size[1]), torch.linspace(-1, 1, target_size[0]), indexing="xy")
        self.grid = torch.stack(grid, dim=0)

    def forward(self, inputs):
        _, H, W = inputs[("color", "r", -1)].shape
        inputs["grid"] = self.grid.clone()
        for k in list(inputs):
            if "color" in k:
                input = inputs[k]
                n, im, i = k
                input = F.interpolate(input[None, ...], size=self.target_size, mode="bicubic", align_corners=True)[0]
                input = input.clamp(min=0., max=1.)
                inputs[(n, im)] = input
                inputs[(n + "_aug", im)] = input
                inputs[k] = F.interpolate(inputs[k][None, ...], size=self.target_size, mode="bicubic", align_corners=True)[0].clamp(min=0., max=1.)
                
        return inputs
    
class RandomGamma(nn.Module):
    def __init__(self, min=1, max=1, p=0.5):
        super().__init__()
        self.min = min
        self.max = max
        self.A = 1.
        self.p = p

    def forward(self, inputs):
        if random.random() < self.p:
            factor = random.uniform(self.min, self.max)
            for k in list(inputs):
                if "color_aug" in k:
                    inputs[k] = self.A * ((inputs[k] / self.A) ** factor)
            return inputs
        else:
            return inputs


class RandomBrightness(nn.Module):
    def __init__(self, min=1, max=1, p=0.5):
        super().__init__()
        self.min = min
        self.max = max
        self.p = p

    def forward(self, inputs):
        if random.random() < self.p:
            factor = random.uniform(self.min, self.max)
            for k in list(inputs):
                if "color_aug" in k:
                    inputs[k] = inputs[k] * factor
                    inputs[k][inputs[k] > 1] = 1
            return inputs
        else:
            return inputs


class RandomColorBrightness(nn.Module):
    def __init__(self, min=1, max=1, p=0.5):
        super().__init__()
        self.min = min
        self.max = max
        self.p = p

    def forward(self, inputs):
        if random.random() < self.p:
            for c in range(3):
                for k in list(inputs):
                    if "color_aug" in k:
                        factor = random.uniform(self.min, self.max)
                        inputs[k][c, :, :] = inputs[k][c, :, :] * factor
                        inputs[k][inputs[k] > 1] = 1
            return inputs
        else:
            return inputs