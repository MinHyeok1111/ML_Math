"""Problem 2. ModernConvNet.

Implement FeedForward, ModernConvBlock and ModernConvNet in
models/modern_convnet.py (2-a to 2-c) and pass check.py first. This script then
trains the network with the full recipe and compares it with the ConvNet of
Problem 1.

Run:
    python p2_modern_convnet.py

Output, in results/:
    p2_modern_convnet.json, p2_modern_convnet_curves.png
    p2_compare.png         test curves of ConvNet and ModernConvNet
"""
import argparse
import os

import engines
import utils
from data import CIFAR10GPU
from models import ModernConvNet

RUN = 'p2_modern_convnet'


def main(args):
    device = utils.get_device()
    data = CIFAR10GPU(args.data, device)
    utils.set_seed(args.seed)
    model = ModernConvNet()
    print(f'\nModernConvNet: {utils.describe(model)}', flush=True)
    engines.fit(model, data, RUN, 'full', args)

    rows = [('ConvNet', utils.load_json(os.path.join(args.out, 'p1_convnet_full.json'))),
            ('ModernConvNet', utils.load_json(os.path.join(args.out, f'{RUN}.json')))]
    rows = [(name, r) for name, r in rows if r]
    engines.summary([r for _, r in rows])
    if len(rows) == 2:
        path = utils.plot_curves([(name, r['history']) for name, r in rows],
                                 os.path.join(args.out, 'p2_compare.png'), train=False)
        print(f'wrote {path}')
    else:
        print('run p1_convnet.py --recipe full to add the ConvNet to the comparison')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem 2: ModernConvNet')
    main(engines.add_arguments(parser).parse_args())
