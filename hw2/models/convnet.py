"""ConvNet: a residual network of dense 3x3 convolutions (provided).

Layout for a 32x32 input:

    stage   resolution   channels   blocks
      0       32x32          40        2
      1       16x16          80        2
      2        8x8          160        2
      3        4x4          304        2

4.81 M parameters, 241.0 M multiply-accumulates per image.
"""
import torch.nn as nn
import torch.nn.functional as F

from models.layers import Downsample, DropPath, drop_path_rates


class ConvBlock(nn.Module):
    """x -> conv kxk -> BN -> ReLU -> conv kxk -> BN -> DropPath -> (+ x) -> ReLU"""

    def __init__(self, dim, kernel_size=3, drop_path=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=padding, bias=False)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=padding, bias=False)
        self.norm2 = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + self.drop_path(h))


class ConvNet(nn.Module):
    def __init__(self, blocks=(2, 2, 2, 2), dims=(40, 80, 160, 304), kernel_size=3,
                 drop_path=0.1, num_classes=10):
        super().__init__()
        rates = drop_path_rates(sum(blocks), drop_path)
        self.stages = nn.ModuleList()
        index = 0
        for i, (depth, dim) in enumerate(zip(blocks, dims)):
            layers = [Downsample(3 if i == 0 else dims[i - 1], dim, first=(i == 0))]
            for _ in range(depth):
                layers.append(ConvBlock(dim, kernel_size, rates[index]))
                index += 1
            self.stages.append(nn.Sequential(*layers))
        self.norm = nn.BatchNorm1d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.head(self.norm(x.mean((-2, -1))))
