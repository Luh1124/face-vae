import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from typing import Optional, Union
import math

class _ConvBlock(nn.Module):
    def __init__(self, pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, dim, activation_type, nonlinearity_type):
        # the default weight norm is spectral norm
        # pattern: C for conv, N for activation norm(SyncBatchNorm), A for nonlinearity(ReLU)
        super().__init__()
        norm_channels = out_channels if pattern.find("C") < pattern.find("N") else in_channels
        weight_norm = spectral_norm if use_weight_norm else lambda x: x
        base_conv = nn.Conv2d if dim == 2 else nn.Conv3d

        def _get_activation():
            if activation_type == "batch":
                return nn.SyncBatchNorm(norm_channels)
            elif activation_type == "instance":
                return nn.InstanceNorm2d(norm_channels, affine=True) if dim == 2 else nn.InstanceNorm3d(norm_channels, affine=True)
            elif activation_type == "none":
                return nn.Identity()

        def _get_nonlinearity():
            if nonlinearity_type == "relu":
                return nn.ReLU(inplace=True)
            elif nonlinearity_type == "leakyrelu":
                return nn.LeakyReLU(0.2, inplace=True)

        mappings = {
            "C": weight_norm(base_conv(in_channels, out_channels, kernel_size, stride, padding)),
            "N": _get_activation(),
            "A": _get_nonlinearity(),
        }
        module_list = []
        for c in pattern:
            module_list.append(mappings[c])
        self.layers = nn.Sequential(*module_list)

    def forward(self, x):
        return self.layers(x)


class ConvBlock2D(_ConvBlock):
    def __init__(
        self, pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, activation_type="batch", nonlinearity_type="relu"
    ):
        super().__init__(pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, 2, activation_type, nonlinearity_type)


class ConvBlock3D(_ConvBlock):
    def __init__(
        self, pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, activation_type="batch", nonlinearity_type="relu"
    ):
        super().__init__(pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, 3, activation_type, nonlinearity_type)


class _DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight_norm, base_conv, base_pooling, kernel_size):
        super().__init__()
        self.layers = nn.Sequential(base_conv("CNA", in_channels, out_channels, 3, 1, 1, use_weight_norm), base_pooling(kernel_size))

    def forward(self, x):
        return self.layers(x)


class DownBlock2D(_DownBlock):
    def __init__(self, in_channels, out_channels, use_weight_norm):
        super().__init__(in_channels, out_channels, use_weight_norm, ConvBlock2D, nn.AvgPool2d, (2, 2))


class DownBlock3D(_DownBlock):
    def __init__(self, in_channels, out_channels, use_weight_norm):
        super().__init__(in_channels, out_channels, use_weight_norm, ConvBlock3D, nn.AvgPool3d, (1, 2, 2))


class _UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight_norm, base_conv, scale_factor):
        super().__init__()
        self.layers = nn.Sequential(nn.Upsample(scale_factor=scale_factor), base_conv("CNA", in_channels, out_channels, 3, 1, 1, use_weight_norm))

    def forward(self, x):
        return self.layers(x)


class UpBlock2D(_UpBlock):
    def __init__(self, in_channels, out_channels, use_weight_norm):
        super().__init__(in_channels, out_channels, use_weight_norm, ConvBlock2D, (2, 2))


class UpBlock3D(_UpBlock):
    def __init__(self, in_channels, out_channels, use_weight_norm):
        super().__init__(in_channels, out_channels, use_weight_norm, ConvBlock3D, (1, 2, 2))


class _SameBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight_norm, base_conv):
        super().__init__()
        self.layers = base_conv("CNA", in_channels, out_channels, 1, 1, 0, use_weight_norm)

    def forward(self, x):
        return self.layers(x)


class SameBlock2D(_SameBlock):
    def __init__(self, in_channels, out_channels, use_weight_norm):
        super().__init__(in_channels, out_channels, use_weight_norm, ConvBlock2D)


class SameBlock3D(_SameBlock):
    def __init__(self, in_channels, out_channels, use_weight_norm):
        super().__init__(in_channels, out_channels, use_weight_norm, ConvBlock3D)


class _ResBlock(nn.Module):
    def __init__(self, in_channels, use_weight_norm, base_block):
        super().__init__()
        self.layers = nn.Sequential(
            base_block("NAC", in_channels, in_channels, 3, 1, 1, use_weight_norm),
            base_block("NAC", in_channels, in_channels, 3, 1, 1, use_weight_norm),
        )

    def forward(self, x):
        return x + self.layers(x)


class ResBlock2D(_ResBlock):
    def __init__(self, in_channels, use_weight_norm):
        super().__init__(in_channels, use_weight_norm, ConvBlock2D)


class ResBlock3D(_ResBlock):
    def __init__(self, in_channels, use_weight_norm):
        super().__init__(in_channels, use_weight_norm, ConvBlock3D)


class ResBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_weight_norm):
        super().__init__()
        self.down_sample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.down_sample = ConvBlock2D("CN", in_channels, out_channels, 1, stride, 0, use_weight_norm)
        self.layers = nn.Sequential(
            ConvBlock2D("CNA", in_channels, out_channels // 4, 1, 1, 0, use_weight_norm),
            ConvBlock2D("CNA", out_channels // 4, out_channels // 4, 3, stride, 1, use_weight_norm),
            ConvBlock2D("CN", out_channels // 4, out_channels, 1, 1, 0, use_weight_norm),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.down_sample(x) + self.layers(x))


class LinearELR(nn.Module):
    """Linear layer with equalized learning rate from stylegan2"""
    def __init__(self, inch, outch, lrmult=1., norm : Optional[str]=None, act=None):
        super(LinearELR, self).__init__()

        # compute gain from activation fn
        try:
            if isinstance(act, nn.LeakyReLU):
                actgain = nn.init.calculate_gain("leaky_relu", act.negative_slope)
            elif isinstance(act, nn.ReLU):
                actgain = nn.init.calculate_gain("relu")
            else:
                actgain = nn.init.calculate_gain(act)
        except:
            actgain = 1.

        initgain = 1. / math.sqrt(inch)

        self.weight = nn.Parameter(torch.randn(outch, inch) / lrmult)
        self.weightgain = actgain

        if norm == None:
            self.weightgain = self.weightgain * initgain * lrmult

        self.bias = nn.Parameter(torch.full([outch], 0.))

        self.norm : Optional[str] = norm
        self.act = act

        self.fused = False

    def extra_repr(self):
        return 'inch={}, outch={}, norm={}, act={}'.format(
            self.weight.size(1), self.weight.size(0), self.norm, self.act
        )

    def getweight(self):
        if self.fused:
            return self.weight
        else:
            weight = self.weight
            if self.norm is not None:
                if self.norm == "demod":
                    weight = F.normalize(weight, dim=1)
            return weight

    def fuse(self):
        if not self.fused:
            with torch.no_grad():
                self.weight.data = self.getweight() * self.weightgain
        self.fused = True

    def forward(self, x):
        if self.fused:
            weight = self.getweight()

            out = torch.addmm(self.bias[None], x, weight.t())
            if self.act is not None:
                out = self.act(out)
            return out
        else:
            weight = self.getweight()

            if self.act is None:
                out = torch.addmm(self.bias[None], x, weight.t(), alpha=self.weightgain)
                return out
            else:
                out = F.linear(x, weight * self.weightgain, bias=self.bias)
                out = self.act(out)
                return out