"""Problem 3. Vision Transformer.

Implement PatchEmbedding and Attention.forward in models/vit.py (3-a, 3-b) and
pass check.py first. This script then trains the ViT with the full recipe and
compares it with the two convolutional networks.

Run:
    python p3_vit.py

Output, in results/:
    p3_vit.json, p3_vit_curves.png
    p3_compare.png         test curves of ConvNet, ModernConvNet and ViT
"""
import argparse
import os

import engines
import utils
from data import CIFAR10GPU
from models import ViT

RUN = 'p3_vit'


def main(args):
    device = utils.get_device()
    data = CIFAR10GPU(args.data, device)
    utils.set_seed(args.seed)
    model = ViT()
    print(f'\nViT: {utils.describe(model)}', flush=True)
    engines.fit(model, data, RUN, 'full', args)

    rows = [('ConvNet', utils.load_json(os.path.join(args.out, 'p1_convnet_full.json'))),
            ('ModernConvNet', utils.load_json(os.path.join(args.out, 'p2_modern_convnet.json'))),
            ('ViT', utils.load_json(os.path.join(args.out, f'{RUN}.json')))]
    rows = [(name, r) for name, r in rows if r]
    engines.summary([r for _, r in rows])
    path = utils.plot_curves([(name, r['history']) for name, r in rows],
                             os.path.join(args.out, 'p3_compare.png'), train=False)
    print(f'wrote {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem 3: ViT')
    main(engines.add_arguments(parser).parse_args())
