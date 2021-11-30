import torch
import torch.nn as nn
from torch import distributions as dist

from .encoder import ResnetPointnet
from .decoder import DecoderONet


class Network(nn.Module):
    def __init__(self,latent_size=256, VAE=False, decoder_insize=3, outsize=1):
        super().__init__()

        self.encoder = ResnetPointnet(dim=decoder_insize, c_dim=latent_size, VAE=VAE)
        self.decoder = DecoderONet(latent_size=latent_size, insize=decoder_insize, outsize=outsize)
        self.vae = VAE

    def name(self):
        return f"{self.encoder.name()}_{self.decoder.name()}"

    def forward(self, non_mnfld_pnts,mnfld_pnts, return_latent_reg=False):


        if self.vae:
            q_latent_mean,q_latent_std = self.encoder(mnfld_pnts)
            q_z = dist.Normal(q_latent_mean, torch.exp(q_latent_std))
            latent = q_z.rsample()
            latent_reg = 1.0e-3*(q_latent_mean.abs().mean(dim=-1) + (q_latent_std + 1).abs().mean(dim=-1))
        else:
            latent, _ = self.encoder(mnfld_pnts)
            latent_reg = None
        
        nonmanifold_pnts_pred = self.decoder(latent, non_mnfld_pnts)

        if return_latent_reg:
            return nonmanifold_pnts_pred, latent_reg
        return nonmanifold_pnts_pred


    def get_latent(self, mnfld_pnts, rand_predict=True):

        if self.vae:
            if rand_predict:
                q_latent_mean,q_latent_std = self.encoder(mnfld_pnts)
                q_z = dist.Normal(q_latent_mean, torch.exp(q_latent_std))
                latent = q_z.rsample()
            else:
                latent,_ = self.encoder(mnfld_pnts)
        else:
            latent, _ = self.encoder(mnfld_pnts)

        return latent

    def predict_from_latent(self, latent, non_mnfld_pnts, with_sigmoid=False):

        nonmanifold_pnts_pred = self.decoder(latent, non_mnfld_pnts)

        if with_sigmoid:
            nonmanifold_pnts_pred = torch.sigmoid(nonmanifold_pnts_pred) * 2 - 1

        return nonmanifold_pnts_pred