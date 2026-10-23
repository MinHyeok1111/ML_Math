"""Layers shared by the convolutional models and the ViT (provided)."""
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth.

    During training, the whole residual branch of each sample is dropped with
    probability p and the surviving branches are scaled by 1 / (1 - p). In
    evaluation mode the layer is the identity.
    """

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


def drop_path_rates(num_blocks, max_rate):
    """One DropPath rate per block, rising linearly from 0 to max_rate."""
    return [max_rate * i / max(num_blocks - 1, 1) for i in range(num_blocks)]


class Downsample(nn.Module):
    """Entry layer of a convolutional stage.

    The first stage keeps the 32x32 resolution with a 3x3 stride-1 convolution;
    every later stage halves the resolution with a 2x2 stride-2 convolution.
    Both are followed by BatchNorm.
    """

    def __init__(self, in_dim, out_dim, first=False):
        super().__init__()
        if first:
            self.conv = nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1, bias=False)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, 2, stride=2, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        return self.norm(self.conv(x))
