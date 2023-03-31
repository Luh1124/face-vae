from typing import List
from torch import tensor as Tensor
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from modules import (
    ConvBlock2D, 
    ConvBlock3D,
    DownBlock2D, 
    DownBlock3D, 
    UpBlock2D, 
    UpBlock3D, 
    SameBlock2D, 
    SameBlock3D, 
    ResBlock2D, 
    ResBlock3D, 
    ResBottleneck,
    LinearELR
    )
from utils import (
    out2heatmap,
    heatmap2kp,
    kp2gaussian_2d,
    kp2gaussian_3d,
    create_heatmap_representations,
    create_sparse_motions,
    create_deformed_source_image
)


class VAE(nn.Module):
    def __init__(self, in_channels=256, latent_dim=128, ch_dim=512, encoder_depth=3, decoder_depth=3):
        super().__init__()
        self.input_channel = in_channels
        self.latent_dim = latent_dim
        self.ch_dim = ch_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.epoch = 0
        self.step = 0

        modules = []
        # Build Encoder
        input_dim = self.input_channel
        for _ in range(self.encoder_depth):
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, self.ch_dim),
                    nn.BatchNorm1d(self.ch_dim),
                    nn.LeakyReLU(0.2),
                    # nn.ReLU()
                    )
            )
            input_dim = self.ch_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.ch_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.ch_dim, self.latent_dim)
        
        # Build Decoder
        modules = []
        input_dim = self.latent_dim
        for _ in range(self.decoder_depth):
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, self.ch_dim),
                    nn.BatchNorm1d(self.ch_dim),
                    nn.LeakyReLU(0.2),
                    # nn.ReLU()
                    )
            )
            input_dim = self.ch_dim

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(self.ch_dim, self.input_channel),
            # nn.ReLU()
        )
        
    def encode(self, input):

        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z):

        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        return self.decode(z), input, mu, log_var

    def loss_function(self, weight, *args):

        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        dim = log_var.shape[1]
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        recons_loss = F.mse_loss(recons[:, :32], input[:, :32]) * 5 + F.mse_loss(recons[:, 32:], input[:, 32:])
        loss = recons_loss + weight * kld_loss

        return {'recon': recons, 'loss': loss, 'rec_loss':recons_loss, 'kld':kld_loss}

    def sample(self, num_samples, current_device):

        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x):
        mu, _ = self.encode(x)

        return self.decode(mu)