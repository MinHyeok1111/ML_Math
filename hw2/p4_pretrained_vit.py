"""Problem 4. Pretrained ViT.

A ViT-S/16 pretrained on ImageNet-21k (timm model vit_small_patch16_224.augreg_in21k,
21.7 M parameters) is adapted to CIFAR-10 in two ways:

    linear probe    the backbone stays frozen; a single linear layer is trained on its features
    fine-tuning     every parameter is trained, for 2 epochs

The 32x32 CIFAR-10 images are resized to 224x224, the input size of the
pretrained model.

Implement extract_features (4-a), train_linear_probe (4-b) and
configure_finetune (4-c), and pass check.py first.

Run:
    python p4_pretrained_vit.py

Output, in results/: p4_pretrained_vit.json
"""
import argparse
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from data import CIFAR10GPU, NUM_CLASSES

# GradScaler skips the first optimizer step while it calibrates the loss scale;
# the scheduler then warns about the order of the two calls. The warning is harmless.
warnings.filterwarnings('ignore', message=r'Detected call of `lr_scheduler\.step\(\)`')

MODEL_NAME = 'vit_small_patch16_224.augreg_in21k'
PROBE = {'epochs': 20, 'lr': 1e-3, 'batch_size': 512}
FINETUNE = {'epochs': 2, 'lr': 1e-4, 'weight_decay': 0.01, 'batch_size': 64,
            'label_smoothing': 0.1}


def load_backbone(num_classes):
    """The pretrained model and its input configuration (size, mean, std).

    num_classes=0 removes the classifier, so the model returns one feature
    vector per image; num_classes=10 attaches a new, randomly initialised head.
    """
    import timm
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    cfg = model.pretrained_cfg
    return model, {'size': cfg['input_size'][-1], 'mean': cfg['mean'], 'std': cfg['std']}


def preprocess(images, config):
    """uint8 (B, 3, 32, 32) -> float (B, 3, size, size): bilinear resize, then normalise."""
    x = images.float() / 255.0
    x = F.interpolate(x, size=config['size'], mode='bilinear', align_corners=False)
    mean = torch.tensor(config['mean'], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(config['std'], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# -----------------------------------------------------------------------------
# 4-a. Features of the frozen backbone
# -----------------------------------------------------------------------------
def extract_features(backbone, images, config, batch_size=128):
    """Features of uint8 images as an (N, D) float32 tensor.

    Put the backbone in evaluation mode and run it without gradient tracking,
    batch_size images at a time, on preprocess(images[i:i + batch_size], config).
    On a GPU, run the forward pass under torch.autocast. Return the features of
    all images concatenated in order, converted to float32.
    """
    backbone.eval()
    features = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]
            preprocessed = preprocess(batch, config)
            with torch.autocast(batch.device.type, enabled=batch.device.type == 'cuda'):
                feat = backbone(preprocessed)
            features.append(feat.float())
    return torch.cat(features, dim=0)


# -----------------------------------------------------------------------------
# 4-b. Linear probe
# -----------------------------------------------------------------------------
def train_linear_probe(train_x, train_y, test_x, test_y, epochs, lr, batch_size, seed):
    """Train one nn.Linear(D, 10) on fixed features; return (head, test accuracy).

    Use AdamW with the given learning rate and no weight decay, cross-entropy
    without label smoothing, and mini-batches of batch_size drawn in a new
    random order every epoch (use a torch.Generator seeded with `seed` on the
    device of the features). Test accuracy is a float in [0, 1].
    """
    D = train_x.shape[1]
    head = nn.Linear(D, NUM_CLASSES)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator(device=train_x.device)
    generator.manual_seed(seed)

    for _ in range(epochs):
        order = torch.randperm(train_x.shape[0], device=train_x.device, generator=generator)
        for start in range(0, train_x.shape[0], batch_size):
            idx = order[start:start + batch_size]
            logits = head(train_x[idx])
            loss = criterion(logits, train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        logits = head(test_x)
        accuracy = (logits.argmax(1) == test_y).float().mean().item()
    return head, accuracy


# -----------------------------------------------------------------------------
# 4-c. Fine-tuning setup
# -----------------------------------------------------------------------------
def configure_finetune(model, lr, weight_decay, total_steps):
    """(optimizer, scheduler) for fine-tuning every parameter of the model.

    optimizer: AdamW over all parameters, with the given learning rate and
    weight decay. scheduler: cosine decay from lr to 0 over total_steps
    (torch.optim.lr_scheduler.CosineAnnealingLR), stepped once per batch.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    return optimizer, scheduler


# -----------------------------------------------------------------------------
# Provided
# -----------------------------------------------------------------------------
@torch.no_grad()
def test_accuracy(model, data, config, batch_size=250):
    model.eval()
    correct = 0
    with torch.autocast(data.device.type, enabled=data.device.type == 'cuda'):
        for start in range(0, data.x_test.shape[0], batch_size):
            logits = model(preprocess(data.x_test[start:start + batch_size], config))
            correct += (logits.argmax(1) == data.y_test[start:start + batch_size]).sum().item()
    return correct / data.x_test.shape[0]


def finetune(model, data, config, seed):
    """Train every parameter for FINETUNE['epochs'] epochs; return the test accuracy per epoch."""
    s = FINETUNE
    device = data.device
    amp = device.type == 'cuda'
    n = data.x_train.shape[0]
    steps_per_epoch = n // s['batch_size']
    optimizer, scheduler = configure_finetune(model, s['lr'], s['weight_decay'],
                                              s['epochs'] * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    history = []
    for epoch in range(s['epochs']):
        model.train()
        tic = time.time()
        order = torch.randperm(n, device=device, generator=generator)
        for start in range(0, n - s['batch_size'] + 1, s['batch_size']):
            index = order[start:start + s['batch_size']]
            x = preprocess(data.x_train[index], config)
            flip = torch.rand(x.shape[0], device=device, generator=generator) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
            with torch.autocast(device.type, enabled=amp):
                loss = F.cross_entropy(model(x), data.y_train[index],
                                       label_smoothing=s['label_smoothing'])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        history.append(test_accuracy(model, data, config))
        print(f'  [fine-tune] epoch {epoch + 1}/{s["epochs"]}  test acc {history[-1] * 100:.2f}  '
              f'({time.time() - tic:.0f} s)', flush=True)
    return history


def main(args):
    device = utils.get_device()
    data = CIFAR10GPU(args.data, device)

    print('\nLinear probe', flush=True)
    utils.set_seed(args.seed)
    tic = time.time()
    backbone, config = load_backbone(num_classes=0)
    backbone = backbone.to(device)
    train_x = extract_features(backbone, data.x_train, config)
    test_x = extract_features(backbone, data.x_test, config)
    print(f'  features: train {tuple(train_x.shape)}, test {tuple(test_x.shape)}', flush=True)
    head, probe_acc = train_linear_probe(train_x, data.y_train, test_x, data.y_test,
                                         seed=args.seed, **PROBE)
    probe_minutes = (time.time() - tic) / 60
    print(f'  [linear probe] test acc {probe_acc * 100:.2f}', flush=True)
    del backbone, train_x, test_x
    if device == 'cuda':
        torch.cuda.empty_cache()

    print('\nFine-tuning', flush=True)
    utils.set_seed(args.seed)
    tic = time.time()
    model, config = load_backbone(num_classes=NUM_CLASSES)
    model = model.to(device)
    history = finetune(model, data, config, args.seed)
    finetune_minutes = (time.time() - tic) / 60

    result = {'run': 'p4_pretrained_vit', 'model': MODEL_NAME, 'seed': args.seed,
              'backbone_params': utils.count_parameters(model) - utils.count_parameters(model.head),
              'probe_params': utils.count_parameters(head), 'finetune_params': utils.count_parameters(model),
              'linear_probe_acc': probe_acc, 'finetune_acc': history[-1], 'finetune_history': history,
              'probe_minutes': probe_minutes, 'finetune_minutes': finetune_minutes}
    utils.save_json(os.path.join(args.out, 'p4_pretrained_vit.json'), result)

    scratch = utils.load_json(os.path.join(args.out, 'p3_vit.json'))
    print(f'\n{"model":<28}{"trained params":>16}{"test acc":>10}')
    if scratch:
        print(f'{"ViT from scratch (P3)":<28}{scratch["params"] / 1e6:>15.2f}M'
              f'{scratch["final_test_acc"] * 100:>9.2f}%')
    print(f'{"ViT-S/16 linear probe":<28}{result["probe_params"] / 1e6:>15.4f}M'
          f'{probe_acc * 100:>9.2f}%')
    print(f'{"ViT-S/16 fine-tuned":<28}{result["finetune_params"] / 1e6:>15.2f}M'
          f'{history[-1] * 100:>9.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem 4: pretrained ViT')
    parser.add_argument('--data', default='data')
    parser.add_argument('--out', default='results')
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())
