"""Seeding, model size, result files and learning-curve figures (provided)."""
import json
import os
import random

import numpy as np
import torch

# categorical series colours, in this fixed order; text and grid stay neutral
SERIES = ('#2a78d6', '#eb6834', '#1baf7a')
INK, INK_SOFT, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3de', '#fcfcfb'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, image_size=32):
    """Multiply-accumulates of one forward pass on a single image.

    torch.utils.flop_counter counts matrix products and convolutions;
    attention computed with explicit matrix products is included.
    """
    from torch.utils.flop_counter import FlopCounterMode
    device = next(model.parameters()).device
    training = model.training
    model.eval()
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size, device=device))
    model.train(training)
    return counter.get_total_flops() / 2


def describe(model):
    return (f'{count_parameters(model) / 1e6:.2f} M parameters, '
            f'{count_macs(model) / 1e6:.1f} M MACs per image')


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(obj, handle, indent=1)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def plot_curves(series, path, train=True, title=None):
    """Loss and accuracy per epoch, one colour per run.

    series: list of (label, history) with history holding the lists
    train_loss, train_acc, test_loss, test_acc. Test curves are solid, training
    curves dashed (drawn only when train=True).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150, facecolor=SURFACE)
    panels = (('loss', 'Loss' if train else 'Test loss', 1.0),
              ('acc', 'Accuracy (%)' if train else 'Test accuracy (%)', 100.0))
    for ax, (key, name, scale) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for colour, (label, history) in zip(SERIES, series):
            epochs = np.arange(1, len(history[f'test_{key}']) + 1)
            ax.plot(epochs, np.asarray(history[f'test_{key}']) * scale, color=colour,
                    linewidth=1.6)
            if train:
                ax.plot(epochs, np.asarray(history[f'train_{key}']) * scale, color=colour,
                        linewidth=1.6, linestyle=(0, (4, 2)))
        ax.set_title(name, color=INK, fontsize=11, loc='left')
        ax.set_xlabel('epoch', color=INK_SOFT, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=INK_SOFT, labelsize=8, length=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
    handles = [Line2D([], [], color=colour, linewidth=2.0, label=label)
               for colour, (label, _) in zip(SERIES, series)]
    if train:
        handles += [Line2D([], [], color=INK_SOFT, linewidth=1.6, label='test'),
                    Line2D([], [], color=INK_SOFT, linewidth=1.6, linestyle=(0, (4, 2)),
                           label='train')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
    if title:
        fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.07, 1, 0.96 if title else 1))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
