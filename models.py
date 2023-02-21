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
    LinearELR,
    )
from utils import (
    out2heatmap,
    heatmap2kp,
    kp2gaussian_2d,
    kp2gaussian_3d,
    create_heatmap_representations,
    create_sparse_motions,
    create_deformed_source_image,
)
from models_utils import LinearELR, Conv2dELR, UpSampleBlock3d, ConvTranspose3dELR


# Positional encoding
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 6

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class EFE_lin_conv(nn.Module):
    # experssion features extractor
    # def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024], 
    #                 up_seq=[1024, 512, 256, 128, 64, 32], 
    #                 mix_seq = [30, 15],
    #                 D=16, K=15, n_res=3, scale_factor=0.25) -> None:
    def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024, 2048],
                     vae_seq = [2048, 4096, 4096], mid_seq = [2048, 2048],
                    #  up_seq = [2048,1024,512,256,128,64,32],
                     up_seq = [2048, 2048, 2048, 2048],
                     D=16, K=15, n_res=3, scale_factor=0.25) -> None:
    # [N,3,256,256] 
    # [N,3,64,64]
    # [N,64,32,32]
    # [N,128,16,16]
    # [N,256,8,8]
    # [N,512,4,4]
    # [N,1024,2,2]
    # [N,2048,1,1] -> vae contrastive
    # [N,2048*16,1,1]
    # [N,2048,16,1,1]
    # [N,1024,16,2,2] 
    # [N,512,16,4,4]
    # [N,256,16,8,8]
    # [N,128,16,16,16]
    # [N,64,16,32,32]
    # [N,32,16,64,64]
    # [N,K,16,64,64]
    # cat kpc2gauss [N,2*K,16,64,64] 
    # [N,2*K,16,64,64] -> [N,K,16,64,64]
    # cat([N,2048] [N, embding(N,60,3)])
        super().__init__()
        
        self.K, self.C, self.D = K, up_seq[0], D
        
        def encoder():
            down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])


        self.vae_encoder = None
        if vae_seq is not None:
            self.vae_encoder = nn.Sequential(*[LinearELR(vae_seq[i], vae_seq[i+1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(vae_seq) - 1)])
            self.mu = LinearELR(vae_seq[-1],vae_seq[-1]//2)
            self.logstd = LinearELR(vae_seq[-1],vae_seq[-1]//2)
        self.mid_map = nn.Sequential(*[LinearELR(mid_seq[i], mid_seq[i+1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(mid_seq) - 1)])
        self.mid_cat = nn.Sequential(*[LinearELR(mid_seq[i] + (self.K*63 if i==0 else 0), mid_seq[i+1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(mid_seq) - 1)])
        # self.mid_conv = nn.Conv2d(mid_seq[-1], up_seq[0] * D, 1, 1, 0)

        self.up = nn.Sequential(*[LinearELR(up_seq[i], up_seq[i+1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(up_seq) - 1)])
        # self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i+1], use_weight_norm) for i in range(len(up_seq) - 1)])
        
        self.final_linear = LinearELR(up_seq[-1], K*3)
        # self.final_conv = SameBlock3D(up_seq[-1], K, use_weight_norm)
        
        self.scale_factor = scale_factor

        self.get_embeding, _ = get_embedder(10)
        # self.mix = nn.Sequential(*[ResBlock3D(2*K, use_weight_norm) for _ in range(n_res)])
        # self.mix_out = SameBlock3D(2*K, K, use_weight_norm)

    def forward(self, x, x_a=None, kpc=None):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        x = self.down(x).squeeze(-1).squeeze(-1)
        x_z = x
        if x_a is not None:
            x_c = x
            x_a_c = self.down(x_a).squeeze(-1).squeeze(-1)
        else:
            x_c = None
            x_a_c = None
        if self.vae_encoder is not None:
            x_vae = self.vae_encoder(x_c)
            x_mu = self.mu(x_vae)
            x_logstd = self.logstd(x_vae)
            x_z = x_mu + torch.exp(x_logstd) * torch.randn(*x_logstd.size(), device=x_logstd.device)        
        else:
            x_mu = None
            x_logstd = None

        x = self.mid_map(x_z)

        xc = self.get_embeding(kpc).reshape(-1, self.K*63)
        x = torch.cat((x, xc), dim=1)
        x = self.mid_cat(x)
        # x = self.mid_conv(x)
        # N, _, H, W = x.shape
        # x = x.view(N, self.C, self.D, H, W)

        x = self.up(x)

        x = self.final_linear(x)
        # x = self.final_conv(x) # [N K 16 64 64]
        # xc = kp2gaussian_3d(kpc, spatial_size=x.shape[2:])
        # x = torch.cat((x, xc), dim=1)
        # x = self.mix(x)
        # x = self.mix_out(x)
        # heatmap = out2heatmap(x)

        x = F.tanh(x)
        kp = x.view(-1, self.K, 3)
        # res kpc
        # kp = heatmap2kp(heatmap) + kpc
        # kp = heatmap2kp(heatmap)

        return kp, x_c, x_a_c, x_mu, x_logstd


class EFE_linear(nn.Module):
    # experssion features extractor
    # def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024], 
    #                 up_seq=[1024, 512, 256, 128, 64, 32], 
    #                 mix_seq = [30, 15],
    #                 D=16, K=15, n_res=3, scale_factor=0.25) -> None:
    def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024, 2048],
                    #  vae_seq = [2048, 2048, 1024], 
                     vae_seq = None,
                     mid_seq = [2048, 512],
                     cat_seq = [512, 512],
                     
                    #  up_seq = [2048,1024,512,256,128,64,32],
                     up_seq = [512, 512],
                     D=16, K=15, n_res=3, scale_factor=0.25) -> None:
    # [N,3,256,256] 
    # [N,3,64,64]
    # [N,64,32,32]
    # [N,128,16,16]
    # [N,256,8,8]
    # [N,512,4,4]
    # [N,1024,2,2]
    # [N,2048,1,1] -> vae contrastive
    # [N,2048*16,1,1]
    # [N,2048,16,1,1]
    # [N,1024,16,2,2] 
    # [N,512,16,4,4]
    # [N,256,16,8,8]
    # [N,128,16,16,16]
    # [N,64,16,32,32]
    # [N,32,16,64,64]
    # [N,K,16,64,64]
    # cat kpc2gauss [N,2*K,16,64,64] 
    # [N,2*K,16,64,64] -> [N,K,16,64,64]
    # cat([N,2048] [N, embding(N,60,3)])
        super().__init__()
        
        self.K, self.C, self.D = K, up_seq[0], D
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.vae_encoder = None
        if vae_seq is not None:
            self.vae_encoder = nn.Sequential(*[LinearELR(vae_seq[i], vae_seq[i+1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(vae_seq) - 1)])
            self.mu = LinearELR(vae_seq[-1],vae_seq[-1]//2)
            self.logstd = LinearELR(vae_seq[-1],vae_seq[-1]//2)
        self.mid_map = nn.Sequential(*[LinearELR(mid_seq[i], mid_seq[i+1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(mid_seq) - 1)])
        self.mid_cat = nn.Sequential(*[LinearELR(cat_seq[i] + (self.K*63 if i==0 else 0), cat_seq[i+1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(mid_seq) - 1)])
        # self.mid_conv = nn.Conv2d(mid_seq[-1], up_seq[0] * D, 1, 1, 0)

        self.up = nn.Sequential(*[LinearELR(up_seq[i], up_seq[i+1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(up_seq) - 1)])
        # self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i+1], use_weight_norm) for i in range(len(up_seq) - 1)])
        
        self.final_linear = LinearELR(up_seq[-1], K*3)
        # self.final_conv = SameBlock3D(up_seq[-1], K, use_weight_norm)
        
        self.scale_factor = scale_factor

        self.get_embeding, _ = get_embedder(10)
        # self.mix = nn.Sequential(*[ResBlock3D(2*K, use_weight_norm) for _ in range(n_res)])
        # self.mix_out = SameBlock3D(2*K, K, use_weight_norm)

    def forward(self, x, x_a=None, kpc=None):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        x = self.down(x).squeeze(-1).squeeze(-1)
        x_z = x
        x_a = None
        if x_a is not None:
            x_c = x
            x_a_c = self.down(x_a).squeeze(-1).squeeze(-1)
        else:
            x_c = None
            x_a_c = None
        if self.vae_encoder is not None:
            x_vae = self.vae_encoder(x_c)
            x_mu = self.mu(x_vae)
            x_logstd = self.logstd(x_vae)
            x_z = x_mu + torch.exp(x_logstd) * torch.randn(*x_logstd.size(), device=x_logstd.device)        
        else:
            x_mu = None
            x_logstd = None

        x = self.mid_map(x_z)

        xc = self.get_embeding(kpc).reshape(-1, self.K*63)
        x = torch.cat((x, xc), dim=1)
        x = self.mid_cat(x)
        # x = self.mid_conv(x)
        # N, _, H, W = x.shape
        # x = x.view(N, self.C, self.D, H, W)

        x = self.up(x)

        x = self.final_linear(x)
        # x = self.final_conv(x) # [N K 16 64 64]
        # xc = kp2gaussian_3d(kpc, spatial_size=x.shape[2:])
        # x = torch.cat((x, xc), dim=1)
        # x = self.mix(x)
        # x = self.mix_out(x)
        # heatmap = out2heatmap(x)

        x = F.tanh(x)
        kp = x.view(-1, self.K, 3)
        # res kpc
        # kp = heatmap2kp(heatmap) + kpc
        # kp = heatmap2kp(heatmap)

        return kp, x_c, x_a_c, x_mu, x_logstd


class EFE_conv(nn.Module):
    # experssion features extractor
    # def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024], 
    #                 up_seq=[1024, 512, 256, 128, 64, 32], 
    #                 mix_seq = [30, 15],
    #                 D=16, K=15, n_res=3, scale_factor=0.25) -> None:
    def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256], 
                    up_seq=[256, 128, 64, 32], 
                    mix_seq = [30, 15],
                    contra_seq = [256, 512, 1024, 2048],
                    D=16, K=15, n_res=3, scale_factor=0.25) -> None:
    # [N,3,256,256]
    # [N,64,128,128]
    # [N,128,64,64]
    # [N,256,32,32]
    # [N,256 * 16,32,32]
    # [N,256,16,32,32]
    # [N,128,16,64,64]
    # [N,64,16,128,128]
    # [N,32,16,256,256]
    # [N,20,16,256,256] (heatmap)
    # [N,20,3] (key points)
        super().__init__()
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        # self.out_conv = nn.Conv3d(up_seq[-1], K, 7, 1, 3)
        self.mix = nn.Sequential(*[ResBlock3D(2*K, use_weight_norm) for _ in range(n_res)])
        self.mix_out = SameBlock3D(2*K, K, use_weight_norm)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor
        self.contrastive_conv = nn.Sequential(*[nn.Conv2d(contra_seq[i], contra_seq[i + 1], 3, 2, 1) for i in range(len(contra_seq) - 1)])

    def forward(self, x, x_a=None, kpc=None):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        x = self.down(x)
        x_z = x
        if x_a is not None:
            x_c = self.contrastive_conv(x)
            x_c = x_c.view(x.shape[0], -1)

            x_a = F.interpolate(x_a, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
            x_a_c = self.down(x_a)
            x_a_c = self.contrastive_conv(x_a_c)
            x_a_c = x_a_c.view(x.shape[0], -1)
        else:
            x_c = None
            x_a_c = None
        # if self.vae_encoder is not None:
        #     x_vae = self.vae_encoder(x_c)
        #     x_mu = self.mu(x_vae)
        #     x_logstd = self.logstd(x_vae)
        #     x_z = x_mu + torch.exp(x_logstd) * torch.randn(*x_logstd.size(), device=x_logstd.device)        
        # else:
        #     x_mu = None
        #     x_logstd = None

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
        return kp, x_c, x_a_c, None, None


class EFE_conv2(nn.Module):
    # experssion features extractor
    # def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024], 
    #                 up_seq=[1024, 512, 256, 128, 64, 32], 
    #                 mix_seq = [30, 15],
    #                 D=16, K=15, n_res=3, scale_factor=0.25) -> None:
    def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256], 
                    up_seq=[256, 128, 64, 32], 
                    mix_seq = [30, 15],
                    contra_seq = [256, 512, 1024, 2048],
                    D=16, K=15, n_res=3, scale_factor=0.25) -> None:
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
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        # self.out_conv = nn.Conv3d(up_seq[-1], K, 7, 1, 3)
        self.mix = nn.Sequential(*[ResBlock3D(2*K, use_weight_norm) for _ in range(n_res)])
        self.mix_out = SameBlock3D(2*K, K, use_weight_norm)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor

    def forward(self, x, x_a=None, kpc=None):
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
        
        # if self.vae_encoder is not None:
        #     x_vae = self.vae_encoder(x_c)
        #     x_mu = self.mu(x_vae)
        #     x_logstd = self.logstd(x_vae)
        #     x_z = x_mu + torch.exp(x_logstd) * torch.randn(*x_logstd.size(), device=x_logstd.device)        
        # else:
        #     x_mu = None
        #     x_logstd = None

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
        return kp, x_c, x_a_c, None, None


class local_vae(nn.Module):
    # VAE
    # [N, 256, 8, 8]
    # [N, 512, 4, 4]
    # [N, 512, 2, 2]
    # [N, 512, 1, 1] -> flatten
    # [N, 512*1*1] -> 2 fc
    # u = [N, 256] var = [N, 256]
    # z = u + var * epsion  [N, 256*1*1] -> fc
    # [N, 512*1*1]  -> view 
    # [N, 512, 1, 1]
    # [N, 512, 2 ,2]
    # [N, 512, 4, 4]
    # [N, 256, 8, 8]
    def __init__(self, down_seq = [128, 128], 
                 up_seq = [128, 128], 
                 vae_seq = [512,256], use_weight_norm = False, lin=LinearELR) -> None:
        super().__init__()
        self.up_seq = up_seq
        self.encoder = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.decoder = nn.Sequential(*[UpBlock2D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        # self.map_fc1 = nn.Linear(vae_dim, up_seq[0])
        self.map_fc1=lin(128*4*4, vae_seq[0], norm="demod", act=nn.LeakyReLU(0.2))
        # self.mu_fc = lin(vae_seq[0], vae_seq[1])
        # self.logstd_fc = lin(vae_seq[0], vae_seq[1])
        # self.map_fc2 = lin(vae_seq[1], 128*4*4, norm="demod", act=nn.LeakyReLU(0.2))
        self.map_fc2 = lin(vae_seq[0], 128*4*4, norm="demod", act=nn.LeakyReLU(0.2))
                                       
                                       
    def forward(self, x):
        b = x.shape[0]
        x_en = self.encoder(x)
        x_fl = self.map_fc1(x_en.flatten(start_dim=1)) # 2048 -> 512
        # mu = self.mu_fc(x_fl) * 0.1 #
        # logstd = self.logstd_fc(x_fl) * 0.01
        # z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
        # x_de = self.map_fc2(z).view(b, self.up_seq[0], 4, 4)
        x_de = self.map_fc2(x_fl).view(b, self.up_seq[0], 4, 4)
        x_hat = self.decoder(x_de)
        # return mu, logstd, x_hat
        return None, None, x_hat
    
class flatten_vae(nn.Module):
    # VAE
    # [N, 256, 2, 2]-> flatten
    # [N, 128*8*8]
    # [N, 128*2*2=8192]
    # [N, 512] -> 2 fc 
    # u = [N, 256] var = [N, 256]
    # z = u + var * epsion  [N, 256*1*1] -> fc
    # [N, 8192=128*8*8]  -> view 
    # [N, 128, 8, 8]
    def __init__(self, down_seq = [16*4*4, 256], 
                 up_seq = [256], 
                 vae_seq = [256, 256], use_weight_norm = False, lin=LinearELR) -> None:
        super().__init__()
        # self.up_seq = up_seq
        # self.encoder = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        # self.decoder = nn.Sequential(*[UpBlock2D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        # self.map_fc1 = nn.Linear(vae_dim, up_seq[0])
        # self.map_fc1=lin(128*8*8, vae_seq[0], norm="demod", act=nn.LeakyReLU(0.2))
        self.encoder = nn.Sequential(*[lin(down_seq[i], down_seq[i + 1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(down_seq) - 1)])
        self.mu_fc = lin(vae_seq[0], vae_seq[1])
        self.logstd_fc = lin(vae_seq[0], vae_seq[1])
        # self.map_fc2 = lin(vae_seq[1], 128*8*8, norm="demod", act=nn.LeakyReLU(0.2))
        # self.decoder = nn.Sequential(*[lin(up_seq[i], up_seq[i + 1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(up_seq) - 1)])                               
                                       
    def forward(self, x, train_vae):
        shape = x.shape
        # x_en = self.encoder(x)
        x_en=x
        x_fl = self.encoder(x_en.flatten(start_dim=1)) # 256 -> 256
        mu = self.mu_fc(x_fl) * 0.1
        logstd = self.logstd_fc(x_fl) * 0.01 * (0 if not train_vae else 1)
        z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device) * (0 if not train_vae else 1)
        x_de = z.view(shape)
        x_hat = x_de
        if train_vae:
            return mu, logstd, x_hat
        else:
            return None, None, x_hat


class flatten_vae_nl(nn.Module):
    # VAE
    # [N, 256, 2, 2]-> flatten
    # [N, 128*8*8]
    # [N, 128*2*2=8192]
    # [N, 512] -> 2 fc 
    # u = [N, 256] var = [N, 256]
    # z = u + var * epsion  [N, 256*1*1] -> fc
    # [N, 8192=128*8*8]  -> view 
    # [N, 128, 8, 8]
    def __init__(self, down_seq = [16*4*4, 256], 
                 up_seq = [256], 
                 vae_seq = [256, 256], use_weight_norm = False, lin=LinearELR) -> None:
        super().__init__()
        # self.up_seq = up_seq
        # self.encoder = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        # self.decoder = nn.Sequential(*[UpBlock2D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        # self.map_fc1 = nn.Linear(vae_dim, up_seq[0])
        # self.map_fc1=lin(128*8*8, vae_seq[0], norm="demod", act=nn.LeakyReLU(0.2))
        # self.encoder = nn.Sequential(*[lin(down_seq[i], down_seq[i + 1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(down_seq) - 1)])
        # self.mu_fc = lin(vae_seq[0], vae_seq[1])
        # self.logstd_fc = lin(vae_seq[0], vae_seq[1])
        # self.map_fc2 = lin(vae_seq[1], 128*8*8, norm="demod", act=nn.LeakyReLU(0.2))
        # self.decoder = nn.Sequential(*[lin(up_seq[i], up_seq[i + 1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(up_seq) - 1)])                               
                                       
    def forward(self, x, train_vae):
        b = x.shape[0]
        # x_en = self.encoder(x)
        # x_en=x
        # x_fl = self.encoder(x_en.flatten(start_dim=1)) # 256 -> 256
        # mu = self.mu_fc(x_fl) * 0.1
        # logstd = self.logstd_fc(x_fl) * 0.01
        # mu = self.mu_fc(x_fl)
        # logstd = self.logstd_fc(x_fl)
        mu = x[:, :16, ...].flatten(start_dim=1)
        logstd = x[:, 16:, ...].flatten(start_dim=1) * (0 if not train_vae else 1)
        z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device) * (0 if not train_vae else 1)
        # z = mu + torch.exp(logstd)
        # x_de = self.decoder(z).view(shape)
        x_de = z.view(b, 16, 4, 4)
        # x_hat = self.decoder(x_de)
        x_hat = x_de
        if train_vae:
            return mu, logstd, x_hat
        else:
            return None, None, x_hat


class EFE_conv3(nn.Module): 
    ## contrastive+local_vae
    # experssion features extractor
    # def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024], 
    #                 up_seq=[1024, 512, 256, 128, 64, 32], 
    #                 mix_seq = [30, 15],
    #                 D=16, K=15, n_res=3, scale_factor=0.25) -> None:
    def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256], 
                    up_seq=[256, 128, 64, 32], 
                    mix_seq = [30, 15],
                    contra_seq = [256, 512, 1024, 2048],
                    D=16, K=15, n_res=3, scale_factor=0.25, 
                    use_vae = True) -> None:
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
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        # self.out_conv = nn.Conv3d(up_seq[-1], K, 7, 1, 3)
        self.mix = nn.Sequential(*[ResBlock3D(2*K, use_weight_norm) for _ in range(n_res)])
        self.mix_out = SameBlock3D(2*K, K, use_weight_norm)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor
        if use_vae:
            self.vae = local_vae()
        
    def forward(self, x, x_a=None, kpc=None):
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
            x_mu, x_logstd, x_hat = self.vae(x_vae)
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


class EFE_conv4(nn.Module): 
    ## contrastive+ simple conv thinner channels
    # experssion features extractor
    # def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024], 
    #                 up_seq=[1024, 512, 256, 128, 64, 32], 
    #                 mix_seq = [30, 15],
    #                 D=16, K=15, n_res=3, scale_factor=0.25) -> None:
    def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 128, 256, 256], 
                    up_seq=[256, 256, 128, 128, 64, 32],
                    # mix_seq = [30, 15],
                    # contra_seq = [256, 512, 1024, 2048],
                    D=16, K=15, n_res=3, scale_factor=0.25, 
                    use_vae = True) -> None:
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
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        # self.out_conv = nn.Conv3d(up_seq[-1], K, 7, 1, 3)
        self.mix = nn.Sequential(*[ResBlock3D(2*K, use_weight_norm) for _ in range(n_res)])
        self.mix_out = SameBlock3D(2*K, K, use_weight_norm)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor
        if use_vae:
            self.vae = flatten_vae()
        else: 
            self.vae = None
        
    def forward(self, x, x_a=None, kpc=None):
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
            x_mu, x_logstd, x_hat = self.vae(x_vae)
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

class EFE_conv5(nn.Module): 
    ## contrastive + simple conv thinner channels
    # experssion features extractor
    # def __init__(self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024], 
    #                 up_seq=[1024, 512, 256, 128, 64, 32], 
    #                 mix_seq = [30, 15],
    #                 D=16, K=15, n_res=3, scale_factor=0.25) -> None:
    def __init__(self, use_weight_norm=False, down_seq=[3, 32, 64, 128, 256, 16], 
                    up_seq=[256, 256, 128, 64, 32, 32],
                    # mix_seq = [30, 15],
                    # contra_seq = [256, 512, 1024, 2048],
                    D=16, K=15, n_res=3, scale_factor=0.25) -> None: 
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
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[SameBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) if i==(len(up_seq)-2) else UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        # self.out_conv = nn.Conv3d(up_seq[-1], K, 7, 1, 3)
        self.mix = nn.Sequential(*[ResBlock3D(2*K, use_weight_norm) for _ in range(n_res)])
        self.mix_out = SameBlock3D(2*K, K, use_weight_norm)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor
        
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

        x = self.mid_conv(x_z)
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W)
        x = self.up(x)
        x = self.out_conv(x) # [N K 16 64 64]
        kpc_ng = kpc.detach()
        xc = kp2gaussian_3d(kpc_ng, spatial_size=x.shape[2:])
        x = torch.cat((x, xc), dim=1)
        x = self.mix(x)
        x = self.mix_out(x)
        # x = torch.nn.functional.tanh(x)
        heatmap = out2heatmap(x)
        # res kpc
        # kp = 0.3*heatmap2kp(heatmap) + kpc
        kp = heatmap2kp(heatmap)
        delta = kp - kpc_ng
        return delta, x_c, x_a_c, None, None


class flatten_vae6(nn.Module):
    # VAE
    # [N, 256, 2, 2]-> flatten
    # [N, 128*8*8]
    # [N, 128*2*2=8192]
    # [N, 512] -> 2 fc 
    # u = [N, 256] var = [N, 256]
    # z = u + var * epsion  [N, 256*1*1] -> fc
    # [N, 8192=128*8*8]  -> view 
    # [N, 128, 8, 8]
    def __init__(self, down_seq=[16*4*4, 256], 
                 up_seq=[256, 16*4*4], 
                 vae_seq=[256, 256], training=True, lin=LinearELR) -> None:
        super().__init__()
        self.encoder = nn.Sequential(*[lin(down_seq[i], down_seq[i + 1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(down_seq) - 1)])
        self.decoder = nn.Sequential(*[lin(up_seq[i], up_seq[i + 1], norm="demod", act=nn.LeakyReLU(0.2)) for i in range(len(up_seq) - 1)])
        self.mu_fc = lin(vae_seq[0], vae_seq[1])
        self.logstd_fc = lin(vae_seq[0], vae_seq[1])
        self.training = training
                                       
    def forward(self, x):
        shape = x.shape
        x_en = self.encoder(x.flatten(start_dim=1)) # 256 -> 256
        mu = self.mu_fc(x_en) * 0.1
        logstd = self.logstd_fc(x_en) * 0.01
        if self.training:
            z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
        else:
            z = mu
        x_de = self.decoder(z)
        x_hat = x_de.view(shape)
        return mu, logstd, x_hat


class EFE_conv6(nn.Module): 
    def __init__(self, use_weight_norm=False,
                # mix_seq = [30, 15],
                # contra_seq = [256, 512, 1024, 2048],
                D=16, K=15, scale_factor=0.25,
                use_vae = True, demod=True, lin=LinearELR, conv=Conv2dELR, up3d=UpSampleBlock3d) -> None:
        super().__init__()
        self.scale_factor = scale_factor

        down_seq = [3, 32, 64, 128, 256, 16]
        layers = [conv(down_seq[0], down_seq[1], 1, 1, 1, norm="demod" if demod else None, act=nn.LeakyReLU(0.2)),      # 64
            conv(down_seq[1], down_seq[2], 4, 2, 1, norm="demod" if demod else None, act=nn.LeakyReLU(0.2)),            # 32
            conv(down_seq[2], down_seq[3], 4, 2, 1, norm="demod" if demod else None, act=nn.LeakyReLU(0.2)),            # 16
            conv(down_seq[3], down_seq[4], 4, 2, 1, norm="demod" if demod else None, act=nn.LeakyReLU(0.2)),            # 8
            conv(down_seq[4], down_seq[5], 4, 2, 1, norm="demod" if demod else None, act=nn.LeakyReLU(0.2)),            # 4
            ]
        self.efe_encoder = nn.Sequential(*layers)

        # 64 64 32 16 8
        down_kpc = [K, 32, 64, 128, 128]
        # 15 16 64 64
        self.kpc_encoder_64 = ConvBlock3D('CNA', down_kpc[0], down_kpc[1], 1, 1, 0, use_weight_norm, nonlinearity_type="leakyrelu")
        self.kpc_encoder_32 = ConvBlock3D('CNA', down_kpc[1], down_kpc[2], 4, 2, 1, use_weight_norm, nonlinearity_type="leakyrelu")
        self.kpc_encoder_16 = ConvBlock3D('CNA', down_kpc[2], down_kpc[3], 4, 2, 1, use_weight_norm, nonlinearity_type="leakyrelu")
        self.kpc_encoder_8  = ConvBlock3D('CNA', down_kpc[3], down_kpc[4], 4, 2, 1, use_weight_norm, nonlinearity_type="leakyrelu")

        # 16*4*4 256*16*4*4 128*32*8*8 128*64*16*16 64*128*32*32 32*256*64*64 K*256*64*64  
        up_seq=[256, 128, 128, 64, 32, K]
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.efe_decoder8 = up3d(in_channels=up_seq[0], out_channels=up_seq[1])
        self.efe_decoder16 = up3d(in_channels=up_seq[1], out_channels=up_seq[2])
        self.efe_decoder32 = up3d(in_channels=up_seq[2], out_channels=up_seq[3])
        self.efe_decoder64 = up3d(in_channels=up_seq[3], out_channels=up_seq[4])
        self.efe_out = SameBlock3D(in_channels=up_seq[4], out_channels=up_seq[5], use_weight_norm=use_weight_norm)

        if use_vae:
            self.vae = flatten_vae6()
        else:
            self.vae = None

        self.C, self.D = up_seq[0], D
        
    def forward(self, x, x_a, kpc=None):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        x = self.efe_encoder(x) # b 16 4 4
        x_z = x
        if x_a is not None:
            x_c = x
            x_a = F.interpolate(x_a, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
            x_a_c = self.efe_encoder(x_a)
        else:
            x_c = None
            x_a_c = None
        
        if self.vae is not None:
            x_vae = x
            x_mu, x_logstd, x_hat = self.vae(x_vae)
            x_z = x_hat
        else:
            x_mu = None
            x_logstd = None
            x_hat = None
            x_vae = None

        x = self.mid_conv(x_z) # b 256*16 4 4
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W) # b 256 16 4 4
        
        xc = kp2gaussian_3d(kpc, spatial_size=(256, 64, 64))
        xc_64 = self.kpc_encoder_64(xc)
        xc_32 = self.kpc_encoder_32(xc_64)
        xc_16 = self.kpc_encoder_16(xc_32)
        xc_8 = self.kpc_encoder_8(xc_16)
        
        x = self.efe_decoder8(x, xc_8)
        x = self.efe_decoder16(x, xc_16)
        x = self.efe_decoder32(x, xc_32)
        x = self.efe_decoder64(x, xc_64)
        x = self.efe_out(x)
        
        heatmap = out2heatmap(x)
        # res kpc
        # kp = heatmap2kp(heatmap) + kpc
        kp = heatmap2kp(heatmap)
        return kp, x_c, x_a_c, (x_mu, x_logstd), (x_vae, x_hat)

class AFE(nn.Module):
    # 3D appearance features extractor
    # [N,3,256,256]
    # [N,64,256,256]
    # [N,128,128,128]
    # [N,256,64,64]
    # [N,512,64,64]
    # [N,32,16,64,64]
    def __init__(self, use_weight_norm=False, down_seq=[64, 128, 256], n_res=6, C=32, D=16):
        super().__init__()
        self.in_conv = ConvBlock2D("CNA", 3, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], C * D, 1, 1, 0)
        self.res = nn.Sequential(*[ResBlock3D(C, use_weight_norm) for _ in range(n_res)])
        self.C, self.D = C, D

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.mid_conv(x)
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W)
        x = self.res(x)
        return x


class CKD(nn.Module):
    # Canonical keypoints detector
    # [N,3,256,256]
    # [N,64,128,128]
    # [N,128,64,64]
    # [N,256,32,32]
    # [N,512,16,16]
    # [N,1024,8,8]
    # [N,16384,8,8]
    # [N,1024,16,8,8]
    # [N,512,16,16,16]
    # [N,256,16,32,32]
    # [N,128,16,64,64]
    # [N,64,16,128,128]
    # [N,32,16,256,256]
    # [N,20,16,256,256] (heatmap)
    # [N,20,3] (key points)
    def __init__(
        self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024], up_seq=[1024, 512, 256, 128, 64, 32], D=16, K=15, scale_factor=0.25
    ):
        super().__init__()
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        # self.out_conv = nn.Conv3d(up_seq[-1], K, 7, 1, 3)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        x = self.down(x)
        x = self.mid_conv(x)
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W)
        x = self.up(x)
        x = self.out_conv(x)
        heatmap = out2heatmap(x)
        kp = heatmap2kp(heatmap)
        return kp


class HPE_EDE(nn.Module):
    # Head pose estimator && expression deformation estimator
    # [N,3,256,256]
    # [N,64,64,64]
    # [N,256,64,64]
    # [N,512,32,32]
    # [N,1024,16,16]
    # [N,2048,8,8]
    # [N,2048]
    # [N,66] [N,66] [N,66] [N,3] [N,60]
    # [N,] [N,] [N,] [N,3] [N,20,3]
    def __init__(self, use_weight_norm=False, n_filters=[64, 256, 512, 1024, 2048], n_blocks=[3, 3, 5, 2], n_bins=66, K=15):
        super().__init__()
        self.pre_layers = nn.Sequential(ConvBlock2D("CNA", 3, n_filters[0], 7, 2, 3, use_weight_norm), nn.MaxPool2d(3, 2, 1))
        res_layers = []
        for i in range(len(n_filters) - 1):
            res_layers.extend(self._make_layer(i, n_filters[i], n_filters[i + 1], n_blocks[i], use_weight_norm))
        self.res_layers = nn.Sequential(*res_layers)
        self.fc_yaw = nn.Linear(n_filters[-1], n_bins)
        self.fc_pitch = nn.Linear(n_filters[-1], n_bins)
        self.fc_roll = nn.Linear(n_filters[-1], n_bins)
        self.fc_t = nn.Linear(n_filters[-1], 2)
        self.fc_scale = nn.Linear(n_filters[-1], 3)
        self.n_bins = n_bins
        self.idx_tensor = torch.FloatTensor(list(range(self.n_bins))).unsqueeze(0).cuda()

    def _make_layer(self, i, in_channels, out_channels, n_block, use_weight_norm):
        stride = 1 if i == 0 else 2
        return [ResBottleneck(in_channels, out_channels, stride, use_weight_norm)] + [
            ResBottleneck(out_channels, out_channels, 1, use_weight_norm) for _ in range(n_block)
        ]

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = torch.mean(x, (2, 3))
        yaw, pitch, roll, t, scale = self.fc_yaw(x), self.fc_pitch(x), self.fc_roll(x), self.fc_t(x), self.fc_scale(x)
        yaw = torch.softmax(yaw, dim=1)
        pitch = torch.softmax(pitch, dim=1)
        roll = torch.softmax(roll, dim=1)
        yaw = (yaw * self.idx_tensor).sum(dim=1)
        pitch = (pitch * self.idx_tensor).sum(dim=1)
        roll = (roll * self.idx_tensor).sum(dim=1)
        yaw = (yaw - self.n_bins // 2) * 3 * np.pi / 180
        pitch = (pitch - self.n_bins // 2) * 3 * np.pi / 180
        roll = (roll - self.n_bins // 2) * 3 * np.pi / 180
        zero = torch.zeros((t.shape[0], 1)).to(t.device)
        t = torch.cat((t, zero),dim=1)  # [b 2] -> [b 3]
        scale = scale.view(x.shape[0], 1, 3)
        return yaw, pitch, roll, t, scale


class MFE(nn.Module):
    # Motion field estimator
    # (4+1)x(20+1)=105
    # [N,105,16,64,64]
    # ...
    # [N,32,16,64,64]
    # [N,137,16,64,64]
    # 1.
    # [N,21,16,64,64] (mask)
    # 2.
    # [N,2192,64,64]
    # [N,1,64,64] (occlusion)
    def __init__(self, use_weight_norm=False, down_seq=[80, 64, 128, 256, 512, 1024], up_seq=[1024, 512, 256, 128, 64, 32], K=15, D=16, C1=32, C2=4):
        super().__init__()
        self.compress = nn.Conv3d(C1, C2, 1, 1, 0)
        self.down = nn.Sequential(*[DownBlock3D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.mask_conv = nn.Conv3d(down_seq[0] + up_seq[-1], K + 1, 7, 1, 3)
        self.occlusion_conv = nn.Conv2d((down_seq[0] + up_seq[-1]) * D, 1, 7, 1, 3)
        self.C, self.D = down_seq[0] + up_seq[-1], D

    def forward(self, fs, kp_s, kp_d, Rs, Rd):
        # the original fs is compressed to 4 channels using a 1x1x1 conv
        fs_compressed = self.compress(fs)
        N, _, D, H, W = fs.shape
        # [N,21,1,16,64,64]
        heatmap_representation = create_heatmap_representations(fs_compressed, kp_s, kp_d)
        # [N,21,16,64,64,3]
        sparse_motion = create_sparse_motions(fs_compressed, kp_s, kp_d, Rs, Rd)
        # [N,21,4,16,64,64]
        deformed_source = create_deformed_source_image(fs_compressed, sparse_motion)
        input = torch.cat([heatmap_representation, deformed_source], dim=2).view(N, -1, D, H, W)
        output = self.down(input)
        output = self.up(output)
        x = torch.cat([input, output], dim=1)
        mask = self.mask_conv(x)
        # [N,21,16,64,64,1]
        mask = F.softmax(mask, dim=1).unsqueeze(-1)
        # [N,16,64,64,3]
        deformation = (sparse_motion * mask).sum(dim=1)
        occlusion = self.occlusion_conv(x.view(N, -1, H, W))
        occlusion = torch.sigmoid(occlusion)
        return deformation, occlusion, mask


class Generator(nn.Module):
    # Generator
    # [N,32,16,64,64]
    # [N,512,64,64]
    # [N,256,64,64]
    # [N,128,128,128]
    # [N,64,256,256]
    # [N,3,256,256]
    def __init__(self, use_weight_norm=True, n_res=6, up_seq=[256, 128, 64], D=16, C=32):
        super().__init__()
        self.in_conv = ConvBlock2D("CNA", C * D, up_seq[0], 3, 1, 1, use_weight_norm, nonlinearity_type="leakyrelu")
        self.mid_conv = nn.Conv2d(up_seq[0], up_seq[0], 1, 1, 0)
        self.res = nn.Sequential(*[ResBlock2D(up_seq[0], use_weight_norm) for _ in range(n_res)])
        self.up = nn.Sequential(*[UpBlock2D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv2d(up_seq[-1], 3, 7, 1, 3)

    def forward(self, fs, deformation, occlusion):
        N, _, D, H, W = fs.shape
        fs = F.grid_sample(fs, deformation, align_corners=True).view(N, -1, H, W)
        fs = self.in_conv(fs)
        fs = self.mid_conv(fs)
        fs = fs * occlusion
        fs = self.res(fs)
        fs = self.up(fs)
        fs = self.out_conv(fs)
        fs = torch.sigmoid(fs)
        return fs


class Discriminator(nn.Module):
    # Patch Discriminator

    def __init__(self, use_weight_norm=True, down_seq=[64, 128, 256, 512], K=15):
        super().__init__()
        layers = []
        layers.append(ConvBlock2D("CNA", 3 + K, down_seq[0], 3, 2, 1, use_weight_norm, "instance", "leakyrelu"))
        layers.extend(
            [
                ConvBlock2D("CNA", down_seq[i], down_seq[i + 1], 3, 2 if i < len(down_seq) - 2 else 1, 1, use_weight_norm, "instance", "leakyrelu")
                for i in range(len(down_seq) - 1)
            ]
        )
        layers.append(ConvBlock2D("CN", down_seq[-1], 1, 3, 1, 1, use_weight_norm, activation_type="none"))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, kp):
        heatmap = kp2gaussian_2d(kp.detach()[:, :, :2], x.shape[2:])
        x = torch.cat([x, heatmap], dim=1)
        res = [x]
        for layer in self.layers:
            x = res[-1]
            res.append(layer(x))
        output = res[-1]
        features = res[1:-1]
        return output, features
