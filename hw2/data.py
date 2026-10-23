"""CIFAR-10 held on the GPU, augmented one batch at a time (provided).

A Colab session has two CPU cores, too few to augment images for a T4 one at a
time. The whole training set (150 MB as uint8) is therefore copied to the GPU
once, and every transform below is a batched tensor operation. Together they
reproduce this torchvision.transforms.v2 pipeline:

    RandomCrop(32, padding=4) -> RandomHorizontalFlip() -> RandAugment(2, 9)
    -> Normalize -> RandomErasing(p=0.25) -> MixUp(alpha=0.8) or CutMix(alpha=1.0)

RandAugment draws its two operations per image, exactly as torchvision does;
images that drew the same operation are transformed together.
"""
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from torchvision.datasets import CIFAR10

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10


class CIFAR10GPU:
    """CIFAR-10 as uint8 tensors on `device`.

    x_train (50000, 3, 32, 32), y_train (50000,), x_test (10000, 3, 32, 32),
    y_test (10000,).
    """

    def __init__(self, root, device):
        train = CIFAR10(root, train=True, download=True)
        test = CIFAR10(root, train=False, download=True)
        self.device = torch.device(device)
        self.x_train = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous().to(self.device)
        self.y_train = torch.tensor(train.targets, device=self.device)
        self.x_test = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous().to(self.device)
        self.y_test = torch.tensor(test.targets, device=self.device)
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1) * 255

    def normalize(self, x):
        return (x.float() - self.mean) / self.std

    def train_batches(self, batch_size, generator, augment):
        """Yield (images, labels) for one epoch in a random order.

        The last incomplete batch is dropped. With augment=True every image is
        cropped, flipped, RandAugmented and randomly erased; MixUp and CutMix
        are applied afterwards by the training loop.
        """
        n = self.x_train.shape[0]
        order = torch.randperm(n, device=self.device, generator=generator)
        for start in range(0, n - batch_size + 1, batch_size):
            index = order[start:start + batch_size]
            x = self.x_train[index]
            if augment:
                x = crop_flip(x, generator)
                x = rand_augment(x, generator)
                x = random_erasing(self.normalize(x), generator)
            else:
                x = self.normalize(x)
            yield x, self.y_train[index]

    def test_batches(self, batch_size):
        for start in range(0, self.x_test.shape[0], batch_size):
            yield (self.normalize(self.x_test[start:start + batch_size]),
                   self.y_test[start:start + batch_size])


# -----------------------------------------------------------------------------
# Crop and flip
# -----------------------------------------------------------------------------
def crop_flip(x, generator, pad=4):
    """Zero-pad by `pad`, take a random 32x32 crop per image, flip half of them."""
    B, _, H, W = x.shape
    dev = x.device
    x = F.pad(x, (pad, pad, pad, pad))
    dy = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=generator)
    dx = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=generator)
    rows = torch.arange(H, device=dev).view(1, H) + dy.view(-1, 1)
    cols = torch.arange(W, device=dev).view(1, W) + dx.view(-1, 1)
    batch = torch.arange(B, device=dev).view(-1, 1, 1)
    x = x[batch, :, rows.unsqueeze(-1), cols.unsqueeze(1)].permute(0, 3, 1, 2).contiguous()
    flip = torch.rand(B, device=dev, generator=generator) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)


# -----------------------------------------------------------------------------
# RandAugment (torchvision.transforms.v2.RandAugment, num_magnitude_bins=31)
# -----------------------------------------------------------------------------
_OPS = ('Identity', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'Brightness',
        'Color', 'Contrast', 'Sharpness', 'Posterize', 'Solarize', 'AutoContrast', 'Equalize')
_SIGNED = {'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'Brightness', 'Color',
           'Contrast', 'Sharpness'}
_BINS = 31


def _magnitude(name, magnitude, size):
    level = magnitude / (_BINS - 1)
    return {'ShearX': 0.3 * level, 'ShearY': 0.3 * level,
            'TranslateX': 150.0 / 331.0 * size * level,
            'TranslateY': 150.0 / 331.0 * size * level,
            'Rotate': 30.0 * level, 'Brightness': 0.9 * level, 'Color': 0.9 * level,
            'Contrast': 0.9 * level, 'Sharpness': 0.9 * level,
            'Posterize': int(round(8 - magnitude / ((_BINS - 1) / 4))),
            'Solarize': 1.0 - level}.get(name, 0.0)


def _apply(x, name, m):
    """One RandAugment operation on a uint8 batch (nearest interpolation, fill 0)."""
    nearest = TF.InterpolationMode.NEAREST
    if name == 'Identity':
        return x
    if name == 'ShearX':
        return TF.affine(x, angle=0.0, translate=[0, 0], scale=1.0,
                         shear=[math.degrees(math.atan(m)), 0.0], interpolation=nearest,
                         center=[0, 0])
    if name == 'ShearY':
        return TF.affine(x, angle=0.0, translate=[0, 0], scale=1.0,
                         shear=[0.0, math.degrees(math.atan(m))], interpolation=nearest,
                         center=[0, 0])
    if name == 'TranslateX':
        return TF.affine(x, angle=0.0, translate=[int(m), 0], scale=1.0, shear=[0.0, 0.0],
                         interpolation=nearest)
    if name == 'TranslateY':
        return TF.affine(x, angle=0.0, translate=[0, int(m)], scale=1.0, shear=[0.0, 0.0],
                         interpolation=nearest)
    if name == 'Rotate':
        return TF.rotate(x, angle=m, interpolation=nearest)
    if name == 'Brightness':
        return TF.adjust_brightness(x, 1.0 + m)
    if name == 'Color':
        return TF.adjust_saturation(x, 1.0 + m)
    if name == 'Contrast':
        return TF.adjust_contrast(x, 1.0 + m)
    if name == 'Sharpness':
        return TF.adjust_sharpness(x, 1.0 + m)
    if name == 'Posterize':
        return TF.posterize(x, bits=int(m))
    if name == 'Solarize':
        return TF.solarize(x, threshold=255.0 * m)
    if name == 'AutoContrast':
        return TF.autocontrast(x)
    return TF.equalize(x)


def rand_augment(x, generator, num_ops=2, magnitude=9):
    B, size = x.shape[0], x.shape[-1]
    for _ in range(num_ops):
        op = torch.randint(len(_OPS), (B,), device=x.device, generator=generator)
        negative = torch.rand(B, device=x.device, generator=generator) <= 0.5
        out = x.clone()
        for k, name in enumerate(_OPS):
            m = _magnitude(name, magnitude, size)
            chosen = op == k
            groups = ((chosen & ~negative, m), (chosen & negative, -m)) if name in _SIGNED \
                else ((chosen, m),)
            for mask, value in groups:
                index = mask.nonzero().squeeze(1)
                if index.numel():
                    out[index] = _apply(x[index], name, value)
        x = out
    return x


# -----------------------------------------------------------------------------
# RandomErasing (torchvision defaults: scale (0.02, 0.33), ratio (0.3, 3.3), value 0)
# -----------------------------------------------------------------------------
def random_erasing(x, generator, p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), attempts=10):
    """Erase one random rectangle in a fraction p of the normalised images.

    Each image takes the first of 10 sampled rectangles that fits inside it.
    """
    B, _, H, W = x.shape
    dev = x.device
    area = H * W * (scale[0] + (scale[1] - scale[0]) *
                    torch.rand(B, attempts, device=dev, generator=generator))
    log_ratio = math.log(ratio[0]) + (math.log(ratio[1]) - math.log(ratio[0])) * \
        torch.rand(B, attempts, device=dev, generator=generator)
    h = torch.sqrt(area * torch.exp(log_ratio)).round().long()
    w = torch.sqrt(area / torch.exp(log_ratio)).round().long()
    fits = (h < H) & (w < W)
    first = fits.float().argmax(1, keepdim=True)
    h, w = h.gather(1, first).squeeze(1), w.gather(1, first).squeeze(1)
    erase = fits.any(1) & (torch.rand(B, device=dev, generator=generator) < p)
    top = (torch.rand(B, device=dev, generator=generator) * (H - h + 1).clamp(min=1)).long()
    left = (torch.rand(B, device=dev, generator=generator) * (W - w + 1).clamp(min=1)).long()
    r = torch.arange(H, device=dev).view(1, H, 1)
    c = torch.arange(W, device=dev).view(1, 1, W)
    box = ((r >= top.view(-1, 1, 1)) & (r < (top + h).view(-1, 1, 1)) &
           (c >= left.view(-1, 1, 1)) & (c < (left + w).view(-1, 1, 1)) & erase.view(-1, 1, 1))
    return x.masked_fill(box.unsqueeze(1), 0.0)


# -----------------------------------------------------------------------------
# MixUp / CutMix
# -----------------------------------------------------------------------------
def mix_batch(x, y, label_smoothing, rng, generator, num_classes=NUM_CLASSES):
    """MixUp (alpha 0.8) or CutMix (alpha 1.0), chosen with equal probability per batch.

    Returns the mixed images and soft targets. The targets are label-smoothed
    one-hot vectors mixed with the same weight as the images, which is what
    nn.CrossEntropyLoss(label_smoothing=...) computes from mixed one-hot targets.
    `rng` is a numpy Generator; it draws the mixing weight.
    """
    B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
    off = label_smoothing / num_classes
    target = torch.full((B, num_classes), off, device=x.device)
    target.scatter_(1, y[:, None], 1.0 - label_smoothing + off)
    partner = torch.randperm(B, device=x.device, generator=generator)
    if rng.random() < 0.5:
        lam = float(rng.beta(0.8, 0.8))
        x = lam * x + (1.0 - lam) * x[partner]
    else:
        lam = float(rng.beta(1.0, 1.0))
        cut_h, cut_w = int(H * math.sqrt(1.0 - lam)), int(W * math.sqrt(1.0 - lam))
        cy, cx = int(rng.integers(H)), int(rng.integers(W))
        y0, y1 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
        x0, x1 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[partner][:, :, y0:y1, x0:x1]
        lam = 1.0 - (y1 - y0) * (x1 - x0) / (H * W)
    return x, lam * target + (1.0 - lam) * target[partner]
