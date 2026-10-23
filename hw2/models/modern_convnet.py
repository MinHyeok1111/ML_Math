"""Problem 2. ModernConvNet.

A ConvNet whose blocks separate spatial mixing (a depthwise convolution) from
channel mixing (a pointwise MLP). Layout for a 32x32 input:

    stage   resolution   channels   blocks
      0       32x32          48        2
      1       16x16          96        2
      2        8x8          192        6
      3        4x4          384        2

4.74 M parameters, 244.1 M multiply-accumulates per image.

Implement FeedForward (2-a), ModernConvBlock (2-b) and the stage construction
in ModernConvNet (2-c). Keep the attribute names given in the docstrings:
check.py and the grading load weights through them.
"""
import torch.nn as nn

from models.layers import Downsample, DropPath, drop_path_rates


# -----------------------------------------------------------------------------
# 2-a. Pointwise MLP
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Channel MLP applied independently at every pixel.

        x -> 1x1 conv (dim -> expand * dim) -> GELU -> 1x1 conv (expand * dim -> dim)

    Both convolutions have a bias. Attribute names: fc1, act, fc2.

    Input and output: (B, dim, H, W).
    """

    def __init__(self, dim, expand=4):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim * expand, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(dim * expand, dim, kernel_size=1, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# -----------------------------------------------------------------------------
# 2-b. Block
# -----------------------------------------------------------------------------
class ModernConvBlock(nn.Module):
    """Residual block.

        x -> depthwise kxk conv -> BatchNorm -> FeedForward -> DropPath -> (+ x)

    The depthwise convolution has one kxk filter per channel (groups=dim),
    padding (k - 1) // 2 and no bias. There is no activation after the sum.
    Attribute names: dwconv, norm, ffn, drop_path.

    Input and output: (B, dim, H, W).
    """

    def __init__(self, dim, kernel_size=3, expand=4, drop_path=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.ffn = FeedForward(dim, expand)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.ffn(self.norm(self.dwconv(x))))


# -----------------------------------------------------------------------------
# 2-c. Network
# -----------------------------------------------------------------------------
class ModernConvNet(nn.Module):
    def __init__(self, blocks=(2, 2, 6, 2), dims=(48, 96, 192, 384), kernel_size=3,
                 expand=4, drop_path=0.1, num_classes=10):
        """Build self.stages, one nn.Sequential per stage:

            stage i = Downsample(in_dim, dims[i], first=(i == 0)),
                      followed by blocks[i] ModernConvBlocks of width dims[i]

        in_dim is 3 for the first stage and dims[i - 1] otherwise. The blocks
        take their DropPath rates from `rates` in order: the first block of the
        network uses rates[0], the last uses rates[-1].
        """
        super().__init__()
        rates = drop_path_rates(sum(blocks), drop_path)
        self.stages = nn.ModuleList()
        index = 0
        for i, (depth, dim) in enumerate(zip(blocks, dims)):
            in_dim = 3 if i == 0 else dims[i - 1]
            layers = [Downsample(in_dim, dim, first=(i == 0))]
            for _ in range(depth):
                layers.append(ModernConvBlock(dim, kernel_size, expand, rates[index]))
                index += 1
            self.stages.append(nn.Sequential(*layers))
        self.norm = nn.BatchNorm1d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.head(self.norm(x.mean((-2, -1))))
