import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from torch.autograd import Function
import numpy as np
import math
from .onet_layers import CResnetBlockConv1d, CBatchNorm1d

# from occupancy network
class DecoderONet(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(
            
            self,
            latent_size,
            insize=3,
            outsize=1
            ):
        super().__init__()

        hidden_size = 512
        norm_method = "batch_norm"

        self.fc_p = nn.Conv1d(insize, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(latent_size, hidden_size, norm_method=norm_method)
        self.block1 = CResnetBlockConv1d(latent_size, hidden_size, norm_method=norm_method)
        self.block2 = CResnetBlockConv1d(latent_size, hidden_size, norm_method=norm_method)
        self.block3 = CResnetBlockConv1d(latent_size, hidden_size, norm_method=norm_method)
        self.block4 = CResnetBlockConv1d(latent_size, hidden_size, norm_method=norm_method)

        self.bn = CBatchNorm1d(latent_size, hidden_size, norm_method=norm_method)

        self.fc_out = nn.Conv1d(hidden_size, outsize, 1)

        self.actvn = F.relu

    def name(self):
        return "DecoderONet"



    def forward(self, latent, non_mnfld_pnts):


        non_mnfld_pnts = non_mnfld_pnts.transpose(1, 2)
        batch_size, D, T = non_mnfld_pnts.size()
        net = self.fc_p(non_mnfld_pnts)

        latent = latent.unsqueeze(2)

        net = self.block0(net, latent)
        net = self.block1(net, latent)
        net = self.block2(net, latent)
        net = self.block3(net, latent)
        net = self.block4(net, latent)

        out = self.fc_out(self.actvn(self.bn(net, latent)))

        return out.squeeze(1)