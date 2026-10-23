"""Problem 3. Vision Transformer.

A 32x32 image is cut into 4x4 patches, giving 64 tokens of width 256 that pass
through 6 pre-norm Transformer blocks with 4 attention heads. The tokens are
averaged and classified.

4.77 M parameters, 315.4 M multiply-accumulates per image.

Implement PatchEmbedding (3-a) and Attention.forward (3-b). Keep the attribute
names given in the docstrings: check.py and the grading load weights through
them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import DropPath, drop_path_rates


# -----------------------------------------------------------------------------
# 3-a. Patch embedding
# -----------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Image -> sequence of patch tokens with a learned position embedding.

    1. Cut (B, C, H, W) into non-overlapping patch x patch squares. The
       resulting N = (H / patch) * (W / patch) tokens are ordered row by row,
       left to right within a row.
    2. Flatten each square into a vector of length patch * patch * C, in
       (row, column, channel) order: the C values of the top-left pixel come
       first, then the next pixel of the same row.
    3. Project every vector to `dim` with one nn.Linear (with bias).
    4. Add a learned position embedding of shape (1, N, dim), initialised with
       nn.init.trunc_normal_(std=0.02).

    Attribute names: proj (the nn.Linear), pos (the nn.Parameter).
    Do not use nn.Conv2d or nn.Unfold.

    Input: (B, C, H, W). Output: (B, N, dim).
    """

    def __init__(self, dim, patch=4, img_size=32, in_channels=3):
        super().__init__()
        self.patch = patch
        self.num_tokens = (img_size // patch) ** 2
        self.proj = nn.Linear(patch * patch * in_channels, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch
        # unfold: (B, C, H, W) -> (B, C, num_h, p, num_w, p)
        patches = x.unfold(2, p, p).unfold(3, p, p)
        # permute to (B, num_h, num_w, p, p, C)
        patches = patches.permute(0, 2, 3, 4, 5, 1)
        # reshape to (B, N, p*p*C)
        patches = patches.reshape(B, -1, p * p * C)
        # project: (B, N, dim)
        out = self.proj(patches)
        # add position embedding
        out = out + self.pos
        return out


# -----------------------------------------------------------------------------
# 3-b. Multi-head self-attention
# -----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads != 0:
            raise ValueError('dim must be divisible by heads')
        self.heads = heads
        self.head_dim = dim // heads
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x):
        """Multi-head self-attention.

        1. Q, K, V = wq(x), wk(x), wv(x), each (B, N, dim).
        2. Split each into `heads` heads of width head_dim = dim / heads:
           (B, N, dim) -> (B, heads, N, head_dim). Head h takes channels
           h * head_dim ... (h + 1) * head_dim - 1.
        3. For every head, softmax(Q K^T / sqrt(head_dim)) V, with the softmax
           taken over the keys.
        4. Concatenate the heads back to (B, N, dim) and apply wo.

        Write the attention with matrix products and softmax. Do not call
        torch.nn.functional.scaled_dot_product_attention or
        nn.MultiheadAttention.

        Input and output: (B, N, dim).
        """
        B, N, dim = x.shape
        heads = self.heads
        head_dim = self.head_dim

        # Q, K, V: (B, N, dim) -> (B, heads, N, head_dim) -> (B, heads, head_dim, N) for K transpose
        q = self.wq(x).view(B, N, heads, head_dim).transpose(1, 2)       # (B, heads, N, head_dim)
        k = self.wk(x).view(B, N, heads, head_dim).transpose(1, 2)       # (B, heads, N, head_dim)
        v = self.wv(x).view(B, N, heads, head_dim).transpose(1, 2)       # (B, heads, N, head_dim)

        # attention scores: (B, heads, N, N)
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        # apply attention: (B, heads, N, head_dim)
        out = (attn @ v).transpose(1, 2).reshape(B, N, dim)

        # output projection
        out = self.wo(out)
        return out


# -----------------------------------------------------------------------------
# Provided
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Token MLP: Linear (dim -> expand * dim) -> GELU -> Linear (expand * dim -> dim)."""

    def __init__(self, dim, expand=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expand)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expand, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm Transformer block: x + Attention(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim, heads, expand=4, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, expand)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.ffn(self.norm2(x)))


class ViT(nn.Module):
    def __init__(self, depth=6, dim=256, heads=4, patch=4, expand=4, drop_path=0.1,
                 img_size=32, num_classes=10):
        super().__init__()
        self.embed = PatchEmbedding(dim, patch, img_size)
        rates = drop_path_rates(depth, drop_path)
        self.blocks = nn.Sequential(*[Block(dim, heads, expand, rates[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.blocks(self.embed(x))
        return self.head(self.norm(x).mean(1))
