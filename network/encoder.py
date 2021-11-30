import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from torch.autograd import Function
import numpy as np
import math

# from torch_geometric.nn.pool import voxel_grid
import logging

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, VAE=False):
        super().__init__()
        self.c_dim = c_dim
        self.vae = VAE

        hidden_dim=128

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)

        if self.vae:
            self.fc_mean = nn.Linear(hidden_dim, c_dim)
            self.fc_std = nn.Linear(hidden_dim, c_dim)
            torch.nn.init.constant_(self.fc_mean.weight,0)
            torch.nn.init.constant_(self.fc_mean.bias, 0)
            torch.nn.init.constant_(self.fc_std.weight, 0)
            torch.nn.init.constant_(self.fc_std.bias, -10)
            self.fc_c = None
        else:
            self.fc_c = nn.Linear(hidden_dim, c_dim)
            self.fc_mean = None
            self.fc_std = None
        
        self.actvn = nn.ReLU()
        self.pool = maxpool

    def name(self):
        if self.vae:
            return "ResNetPointNet_VAE"
        else:
            return "ResNetPointNet"

    def forward(self, p):
        batch_size, T, D = p.size()

        # # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)


        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        if self.vae:
            c_mean = self.fc_mean(self.actvn(net))
            c_std = self.fc_std(self.actvn(net))
            return c_mean,c_std
        else:
            c = self.fc_c(self.actvn(net))
            return c, None
