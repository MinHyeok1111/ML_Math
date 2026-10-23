"""Training and evaluation shared by Problems 1-3 (provided).

Every model is trained with the same recipe. Do not change it: the questions
compare models, and the comparison only holds if the recipe is identical.

    optimizer        AdamW, learning rate 1e-3, weight decay 0.05, batch 256
    schedule         120 epochs; linear warm-up from 0 over 3 epochs, then cosine to 0
    'full' recipe    crop + flip + RandAugment(2, 9) + RandomErasing(0.25) + MixUp/CutMix,
                     label smoothing 0.1, stochastic depth up to 0.1
    'none' recipe    no augmentation, no label smoothing, no stochastic depth (Problem 1 only)

The scripts evaluate on the 10,000 CIFAR-10 test images after every epoch and
save a checkpoint, so an interrupted run resumes where it stopped.
"""
import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from data import mix_batch

# GradScaler skips the first optimizer step while it calibrates the loss scale;
# the scheduler then warns about the order of the two calls. The warning is harmless.
warnings.filterwarnings('ignore', message=r'Detected call of `lr_scheduler\.step\(\)`')

RECIPES = {
    'full': {'augment': True, 'label_smoothing': 0.1, 'drop_path': 0.1},
    'none': {'augment': False, 'label_smoothing': 0.0, 'drop_path': 0.0},
}
EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, WARMUP_EPOCHS = 120, 256, 1e-3, 0.05, 3


def add_arguments(parser):
    parser.add_argument('--data', default='data', help='CIFAR-10 is downloaded here')
    parser.add_argument('--out', default='results', help='result files and figures')
    parser.add_argument('--checkpoints', default='checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='keep 120 for the results you report')
    parser.add_argument('--restart', action='store_true',
                        help='ignore an existing checkpoint and train from scratch')
    return parser


def cosine_with_warmup(optimizer, total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def evaluate(model, data, amp, batch_size=500):
    """Test loss (without label smoothing) and test accuracy."""
    model.eval()
    loss = correct = total = 0.0
    with torch.autocast(data.device.type, enabled=amp):
        for x, y in data.test_batches(batch_size):
            logits = model(x).float()
            loss += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return loss / total, correct / total


def _rng_state(generator, rng):
    state = {'torch': torch.get_rng_state(), 'generator': generator.get_state(),
             'numpy': rng.bit_generator.state}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state, generator, rng):
    torch.set_rng_state(state['torch'])
    generator.set_state(state['generator'])
    rng.bit_generator.state = state['numpy']
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def fit(model, data, run, recipe, args):
    """Train `model` with the given recipe and write results/<run>.json and
    results/<run>_curves.png. Returns the result dictionary."""
    settings = RECIPES[recipe]
    device = data.device
    amp = device.type == 'cuda'
    model = model.to(device)
    if amp:
        model = model.to(memory_format=torch.channels_last)

    steps_per_epoch = data.x_train.shape[0] // BATCH_SIZE
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = cosine_with_warmup(optimizer, args.epochs * steps_per_epoch,
                                   WARMUP_EPOCHS * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=settings['label_smoothing'])
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    config = {'run': run, 'recipe': recipe, 'seed': args.seed, 'epochs': args.epochs}
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
               'seconds': []}
    start = 0
    checkpoint_path = os.path.join(args.checkpoints, f'{run}.pt')
    if os.path.exists(checkpoint_path) and not args.restart:
        # random-number states must stay on the CPU; load_state_dict moves the rest
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint['config'] != config:
            raise RuntimeError(f'{checkpoint_path} was saved with {checkpoint["config"]}; '
                               f'pass --restart to discard it')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        _set_rng_state(checkpoint['rng'], generator, rng)
        history, start = checkpoint['history'], checkpoint['epoch']
        print(f'  resuming {run} from epoch {start}', flush=True)

    for epoch in range(start, args.epochs):
        model.train()
        tic = time.time()
        loss_sum = correct = seen = 0.0
        for x, y in data.train_batches(BATCH_SIZE, generator, settings['augment']):
            target = y
            if settings['augment']:
                x, target = mix_batch(x, y, settings['label_smoothing'], rng, generator)
            if amp:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, enabled=amp):
                logits = model(x)
                # mixed targets are already smoothed soft labels
                loss = F.cross_entropy(logits, target) if settings['augment'] \
                    else criterion(logits, target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.numel()
        test_loss, test_acc = evaluate(model, data, amp)
        history['train_loss'].append(loss_sum / seen)
        history['train_acc'].append(correct / seen)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['seconds'].append(time.time() - tic)
        print(f'  [{run}] epoch {epoch + 1:3d}/{args.epochs}  '
              f'train loss {loss_sum / seen:.4f} acc {correct / seen * 100:6.2f}  |  '
              f'test loss {test_loss:.4f} acc {test_acc * 100:6.2f}  '
              f'({history["seconds"][-1]:.1f} s)', flush=True)

        os.makedirs(args.checkpoints, exist_ok=True)
        torch.save({'config': config, 'epoch': epoch + 1, 'history': history,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'rng': _rng_state(generator, rng)}, checkpoint_path + '.tmp')
        os.replace(checkpoint_path + '.tmp', checkpoint_path)

    result = dict(config, model=type(model).__name__,
                  params=utils.count_parameters(model), macs=utils.count_macs(model),
                  final_test_acc=history['test_acc'][-1], best_test_acc=max(history['test_acc']),
                  final_train_acc=history['train_acc'][-1],
                  seconds_per_epoch=float(np.median(history['seconds'])),
                  device=torch.cuda.get_device_name(0) if amp else 'cpu', history=history)
    utils.save_json(os.path.join(args.out, f'{run}.json'), result)
    utils.plot_curves([(run, history)], os.path.join(args.out, f'{run}_curves.png'))
    return result


def summary(results):
    """Print one row per result: size, accuracy and speed."""
    print(f'\n{"run":<22}{"params":>9}{"MACs":>10}{"train acc":>11}{"test acc":>10}'
          f'{"s/epoch":>9}')
    for r in results:
        print(f'{r["run"]:<22}{r["params"] / 1e6:>8.2f}M{r["macs"] / 1e6:>9.1f}M'
              f'{r["final_train_acc"] * 100:>10.2f}%{r["final_test_acc"] * 100:>9.2f}%'
              f'{r["seconds_per_epoch"]:>9.1f}')
