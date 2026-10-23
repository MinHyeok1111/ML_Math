"""Problem 1. ConvNet: augmentation and regularization.

Trains the provided ConvNet twice with the same optimizer and schedule:

    --recipe none   no augmentation, no label smoothing, no stochastic depth
    --recipe full   the recipe used in every later problem

There is nothing to implement in this problem.

Run:
    python p1_convnet.py --recipe none
    python p1_convnet.py --recipe full

Output, in results/:
    p1_convnet_none.json, p1_convnet_none_curves.png
    p1_convnet_full.json, p1_convnet_full_curves.png
    p1_compare.png         both runs, written once both exist
"""
import argparse
import os

import engines
import utils
from data import CIFAR10GPU
from models import ConvNet

RUNS = {'none': 'p1_convnet_none', 'full': 'p1_convnet_full'}


def main(args):
    device = utils.get_device()
    data = CIFAR10GPU(args.data, device)
    recipes = ('none', 'full') if args.recipe == 'both' else (args.recipe,)
    for recipe in recipes:
        utils.set_seed(args.seed)
        model = ConvNet(drop_path=engines.RECIPES[recipe]['drop_path'])
        print(f'\nConvNet, recipe "{recipe}": {utils.describe(model)}', flush=True)
        engines.fit(model, data, RUNS[recipe], recipe, args)

    results = {recipe: utils.load_json(os.path.join(args.out, f'{run}.json'))
               for recipe, run in RUNS.items()}
    engines.summary([r for r in results.values() if r])
    if all(results.values()):
        path = utils.plot_curves([('no augmentation', results['none']['history']),
                                  ('full recipe', results['full']['history'])],
                                 os.path.join(args.out, 'p1_compare.png'))
        print(f'wrote {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem 1: ConvNet')
    parser.add_argument('--recipe', choices=('none', 'full', 'both'), default='both')
    main(engines.add_arguments(parser).parse_args())
