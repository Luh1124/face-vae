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


class flatten_vae_nl(nn.Module):
    def __init__(self, down_seq = [16*4*4, 256], 
                 up_seq = [256], 
                 vae_seq = [256, 256], use_weight_norm = False, lin=LinearELR) -> None:
        super().__init__()                           
                                       
    def forward(self, x, train_vae):
        b = x.shape[0]
        mu = x[:, :16, ...].flatten(start_dim=1)
        logstd = x[:, 16:, ...].flatten(start_dim=1) * (0 if not train_vae else 1)
        z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device) * (0 if not train_vae else 1)
        x_de = z.view(b, 16, 4, 4)
        x_hat = x_de
        if train_vae:
            return mu, logstd, x_hat
        else:
            return None, None, x_hat


class EFE_conv5(nn.Module): 
    ## contrastive + simple conv thinner channels
    # experssion features extractor
    # def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024], 
    #                 up_seq=[1024, 512, 256, 128, 64, 32], 
    #                 mix_seq = [30, 15],
    #                 D=16, K=15, n_res=3, scale_factor=0.25) -> None:
    def __init__(self, use_weight_norm=False, down_seq=[3, 32, 64, 128, 256, 32], 
                    up_seq=[256, 256, 128, 64, 32, 32],
                    # mix_seq = [30, 15],
                    # contra_seq = [256, 512, 1024, 2048],
                    D=16, K=15, n_res=3, scale_factor=0.25, 
                    use_vae=True) -> None: 
    # [N,3,128,128]
    # [N,64,64,64]
    # [N,128,32,32]
    # [N,256,16,16]
    # [N,256 * 16,8,8]
    # [N,256,16,8,8]
    # [N,128,16,16,16]
    # [N,64,16,32,32]
    # [N,32,16,64,64]
    # [N,20,16,128,128] (heatmap)
    # [N,20,3] (key points)
        super().__init__()
        self.down = nn.Sequential(*[SameBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) if i==0 else DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1] // 2, up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[SameBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) if i==(len(up_seq)-2) else UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        # self.out_conv = nn.Conv3d(up_seq[-1], K, 7, 1, 3)
        self.mix = nn.Sequential(*[ResBlock3D(2*K, use_weight_norm) for _ in range(n_res)])
        self.mix_out = SameBlock3D(2*K, K, use_weight_norm)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor
        if use_vae:
            self.vae = flatten_vae_nl()
        else: 
            self.vae = None
        
    def encoder(self, x):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        x = self.down(x)

        return x

    def decoder(self, feature, kpc):
        x = self.mid_conv(feature)
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W)
        x = self.up(x)
        x = self.out_conv(x) # [N K 16 64 64]
        xc = kp2gaussian_3d(kpc, spatial_size=x.shape[2:])
        x = torch.cat((x, xc), dim=1)
        x = self.mix(x)
        x = self.mix_out(x)
        heatmap = out2heatmap(x)
        kp = heatmap2kp(heatmap)

        return kp
    
    def forward(self, x, x_a=None, kpc=None, train_vae=None):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        x = self.down(x)
        x_z = x
        if x_a is not None:
            x_c = x
            x_a = F.interpolate(x_a, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
            x_a_c = self.down(x_a)
        else:
            x_c = None
            x_a_c = None
        
        if self.vae is not None:
            x_vae = x
            x_mu, x_logstd, x_hat = self.vae(x_vae, train_vae)
            x_z = x_hat
        else:
            x_mu = None
            x_logstd = None
            x_hat = None
            x_vae = None

        x = self.mid_conv(x_z)
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W)
        x = self.up(x)
        x = self.out_conv(x) # [N K 16 64 64]
        xc = kp2gaussian_3d(kpc, spatial_size=x.shape[2:])
        x = torch.cat((x, xc), dim=1)
        x = self.mix(x)
        x = self.mix_out(x)
        heatmap = out2heatmap(x)
        # res kpc
        # kp = heatmap2kp(heatmap) + kpc
        kp = heatmap2kp(heatmap)
        return kp, x_c, x_a_c, (x_mu, x_logstd), (x_vae, x_hat)


class VAE_ori(nn.Module): 
    def __init__(self, lin=LinearELR, training=True) -> None:
        super().__init__()
        self.training = training
        self.encoder = nn.Sequential(*[lin(256, 256, norm="demod", act=nn.LeakyReLU(0.2)) for i in range(3)])
        self.mu_fc = lin(256, 256)
        self.logstd_fc = lin(256, 256)
        self.decoder = nn.Sequential(*[lin(256, 256, norm="demod", act=nn.LeakyReLU(0.2)) for i in range(3)])
                                       
    def forward(self, x):
        shape = x.shape
        x_fl = x.flatten(start_dim=1)
        x_en = self.encoder(x_fl)
        mu = self.mu_fc(x_en)
        logstd = self.logstd_fc(x_en)
        if self.training:
            z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
        else:
            z = mu
        x_de = self.decoder(z)
        x_hat = x_de.view(shape)
        return mu, logstd, x_hat

class VAE_v2(nn.Module): 
    def __init__(self, lin=LinearELR, training=True) -> None:
        super().__init__()
        self.training = training
        self.encoder = nn.Sequential(*[lin(256, 256, norm="demod", act=nn.ReLU()) for i in range(3)])
        self.mu_fc = lin(256, 256)
        self.logvar_fc = lin(256, 256)
        self.decoder = nn.Sequential(*[lin(256, 256, norm="demod", act=nn.ReLU()) for i in range(3)])
                                       
    def forward(self, x):
        shape = x.shape
        x_fl = x.flatten(start_dim=1)
        x_en = self.encoder(x_fl)
        mu = self.mu_fc(x_en)
        logvar = self.logvar_fc(x_en)
        if self.training:
            z = mu + torch.exp(0.5*logvar) * torch.randn_like(logvar, device=logvar.device)
        else:
            z = mu
        x_de = self.decoder(z)
        x_hat = x_de.view(shape)
        return mu, logvar, x_hat

from typing import List
from torch import tensor as Tensor


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
            nn.ReLU()
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


class Audio_encoder(nn.Module):
    def __init__(self, use_weight_norm=False):
        super(Audio_encoder, self).__init__()
        self.audio_encoder = nn.Sequential(
            ConvBlock2D("CNA",1,64,3,1,1,use_weight_norm),
            ConvBlock2D("CNA",64,128,3,1,1,use_weight_norm),
            # nn.MaxPool2d(3, stride=(1,2),return_indices=True, ceil_mode=True),
            DownBlock2D(128, 128, use_weight_norm),
            ConvBlock2D("CNA",128,256,3,1,1,use_weight_norm),
            ConvBlock2D("CNA",256,256,3,1,1,use_weight_norm),
            ConvBlock2D("CNA",256,512,3,1,1,use_weight_norm),
            DownBlock2D(512, 512, use_weight_norm),
            # nn.MaxPool2d(3, stride=(2,2),return_indices=True, ceil_mode=True)
            )
        
        self.audio_encoder_fc = nn.Sequential(
            nn.Linear(512*7*3,2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        feature = self.audio_encoder(input)

        feature = feature.view(feature.size(0),-1)
        x = self.audio_encoder_fc(feature)
        return x

    

class Audio_decoder(nn.Module):
    # 256
    def __init__(self, use_weight_norm=False):
        super(Audio_decoder, self).__init__()
        self.audio_decoder_fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(True),
            nn.Linear(2048, 512*7*3),
            nn.BatchNorm1d(512*7*3),
            nn.LeakyReLU(True),
            )
        
        self.audio_decoder = nn.Sequential(
            ConvBlock2D("CNA",512,512,3,1,1,use_weight_norm),
            UpBlock2D(512,512,use_weight_norm),
            ConvBlock2D("CNA",512,256,3,1,1,use_weight_norm),
            ConvBlock2D("CNA",256,256,3,1,1,use_weight_norm),
            ConvBlock2D("CNA",256,128,3,1,1,use_weight_norm),
            UpBlock2D(128,128,use_weight_norm),
            ConvBlock2D("CNA",128,64,3,1,1,use_weight_norm),
            ConvBlock2D("CNA",64,1,3,1,1,use_weight_norm),
            )


    def forward(self, x):
        feature = self.audio_decoder_fc(x)
        feature = feature.view(feature.size(0), 512, 7, 3)
        x = self.audio_decoder(feature)
        return x
        

class AudioVAE(nn.Module):
    def __init__(self, category="linear") -> None:
        super().__init__()
        self.audio_encoder = Audio_encoder()
        self.audio_decoder = Audio_decoder()
        self.fc_mu = nn.Linear(256, 128)
        self.fc_var = nn.Linear(256, 128)
        self.AM = AffineNet(category=category)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu
        
    def encode(self, input):
        x = self.audio_encoder(input)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var


    def forward(self, input, use_vae=True):
        mu, log_var = self.encode(input)
        if use_vae:
            y = self.reparameterize(mu, log_var)
        else: 
            y = mu

        z = self.AM(y)

        return self.audio_decoder(y), z, mu, log_var


class AffineNet(nn.Module):
    def __init__(self, inch=128, outch=128, category="linear") -> None:
        super().__init__()
        if category == "linear":
            # self.Affinelayer_mu = nn.Linear(inch ,outch, bias=True)
            # self.Affinelayer_logvar = nn.Linear(inch ,outch, bias=True)
            self.Affinelayer = nn.Linear(inch ,outch, bias=True)

        elif category == "nonlinear":
            # self.Affinelayer_mu = nn.Sequential(nn.Linear(inch ,outch, bias=True),
            #                                  nn.ReLU(),
            #                                  nn.Linear(inch ,outch, bias=True))  
            # self.Affinelayer_logvar = nn.Sequential(nn.Linear(inch ,outch, bias=True),
            #                                  nn.ReLU(),
            #                                  nn.Linear(inch ,outch, bias=True)) 
            self.Affinelayer = nn.Sequential(nn.Linear(inch ,outch, bias=True),
                                             nn.ReLU(),
                                             nn.Linear(inch ,outch, bias=True))
        elif category == "direct":
            # self.Affinelayer_mu = nn.Identity()
            # self.Affinelayer_logvar = nn.Identity()
            self.Affinelayer = nn.Identity()

    def forward(self, x):
        # mu = self.Affinelayer_mu(mu)
        # logvar = self.Affinelayer_logvar(logvar)
        x = self.Affinelayer(x)
        return x