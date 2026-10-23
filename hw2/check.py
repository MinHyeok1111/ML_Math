"""Self-check for Homework 2.

Tests every function you implement on small random inputs, on the CPU, without
CIFAR-10 and without the pretrained model. It finishes in a few seconds. It is
a debugging aid, not the grader.

    Passing every check does not mean full marks: the written answers and the
    trained results carry as many points as the code.
    Failing a check does not mean zero: if your choice is defensible, write the
    reason in the report and you can still receive most of the credit.

torch.nn.functional.scaled_dot_product_attention appears in this file only, as
the reference for your attention.

Run:
    python check.py
"""
import argparse
import os
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

RESULTS = []
BUDGET = {'ConvNet': (4.81e6, 241.0e6), 'ModernConvNet': (4.74e6, 244.1e6),
          'ViT': (4.77e6, 315.4e6)}


def record(name, detail, passed):
    passed = bool(passed)
    RESULTS.append((name, detail, passed))
    print(f'  [{"PASS" if passed else "FAIL"}] {name:<36s} {detail}')


def skip(name, reason):
    RESULTS.append((name, reason, None))
    print(f'  [SKIP] {name:<36s} {reason}')


def guarded(name):
    """Turn an unfinished or broken part into a SKIP instead of a crash."""
    def wrapper(function):
        def inner(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except NotImplementedError:
                skip(name, 'not implemented yet')
            except Exception as error:  # noqa: BLE001 - a student aid, not a library
                skip(name, f'raised {type(error).__name__}: {error}')
                if os.environ.get('HW2_TRACEBACK'):
                    traceback.print_exc()
        return inner
    return wrapper


def max_error(a, b):
    return (a.double() - b.double()).abs().max().item()


def budget_check(name, build):
    @guarded(f'{name} size')
    def run():
        torch.manual_seed(0)
        model = build()
        params, macs = utils.count_parameters(model), utils.count_macs(model)
        target_params, target_macs = BUDGET[name]
        ok = abs(params / target_params - 1) <= 0.05 and abs(macs / target_macs - 1) <= 0.05
        record(f'{name} size', f'{params / 1e6:.2f} M params, {macs / 1e6:.1f} M MACs '
               f'(expected {target_params / 1e6:.2f} M, {target_macs / 1e6:.1f} M, +-5%)', ok)
    run()


# -----------------------------------------------------------------------------
# Problem 1
# -----------------------------------------------------------------------------
def check_problem1():
    from models.convnet import ConvNet
    print('\nProblem 1. ConvNet (provided)')
    budget_check('ConvNet', ConvNet)


# -----------------------------------------------------------------------------
# Problem 2
# -----------------------------------------------------------------------------
def check_problem2():
    import models.modern_convnet as m
    print('\nProblem 2. ModernConvNet')

    @guarded('2-a FeedForward')
    def feedforward():
        torch.manual_seed(0)
        ffn = m.FeedForward(16, expand=4)
        x = torch.randn(2, 16, 5, 5)
        y = ffn(x)
        reference = F.conv2d(F.gelu(F.conv2d(x, ffn.fc1.weight, ffn.fc1.bias)),
                             ffn.fc2.weight, ffn.fc2.bias)
        params = utils.count_parameters(ffn)
        expected = 16 * 64 + 64 + 64 * 16 + 16
        error = max_error(y, reference)
        record('2-a FeedForward', f'shape {tuple(y.shape)}, max err={error:.1e}, '
               f'{params} params (expected {expected})',
               y.shape == x.shape and error < 1e-5 and params == expected)

    @guarded('2-b ModernConvBlock')
    def block():
        torch.manual_seed(0)
        blk = m.ModernConvBlock(16, kernel_size=3, expand=4, drop_path=0.0).eval()
        x = torch.randn(2, 16, 8, 8)
        y = blk(x)
        dw = blk.dwconv
        depthwise = (dw.groups == 16 and tuple(dw.kernel_size) == (3, 3) and dw.bias is None
                     and tuple(dw.padding) == (1, 1))
        branch = blk.ffn(blk.norm(F.conv2d(x, dw.weight, None, 1, 1, 1, 16)))
        error = max_error(y, x + branch)
        with torch.no_grad():
            blk.ffn.fc2.weight.zero_()
            blk.ffn.fc2.bias.zero_()
        identity = max_error(blk(x), x)
        record('2-b ModernConvBlock', f'shape {tuple(y.shape)}, depthwise={depthwise}, '
               f'max err={error:.1e}, zero branch -> identity err={identity:.1e}',
               y.shape == x.shape and depthwise and error < 1e-5 and identity == 0.0)

    @guarded('2-c ModernConvNet')
    def network():
        torch.manual_seed(0)
        net = m.ModernConvNet().eval()
        y = net(torch.randn(2, 3, 32, 32))
        blocks = [b for b in net.modules() if isinstance(b, m.ModernConvBlock)]
        widths = [b.dwconv.in_channels for b in blocks]
        expected = [48] * 2 + [96] * 2 + [192] * 6 + [384] * 2
        rates = [b.drop_path.p for b in blocks]
        record('2-c ModernConvNet', f'output {tuple(y.shape)}, {len(blocks)} blocks, '
               f'drop path {rates[0]:.2f} -> {rates[-1]:.2f}',
               y.shape == (2, 10) and widths == expected
               and abs(rates[0]) < 1e-12 and abs(rates[-1] - 0.1) < 1e-12)

    feedforward()
    block()
    network()
    budget_check('ModernConvNet', m.ModernConvNet)


# -----------------------------------------------------------------------------
# Problem 3
# -----------------------------------------------------------------------------
def check_problem3():
    import models.vit as v
    print('\nProblem 3. ViT')

    @guarded('3-a PatchEmbedding')
    def patch_embedding():
        torch.manual_seed(0)
        embed = v.PatchEmbedding(32, patch=4, img_size=32)
        x = torch.randn(2, 3, 32, 32)
        y = embed(x)
        patches = x.unfold(2, 4, 4).unfold(3, 4, 4)                 # (B, C, 8, 8, 4, 4)
        patches = patches.permute(0, 2, 3, 4, 5, 1).reshape(2, 64, 48)
        reference = patches @ embed.proj.weight.T + embed.proj.bias + embed.pos
        learned = isinstance(embed.pos, nn.Parameter) and embed.pos.requires_grad \
            and tuple(embed.pos.shape) == (1, 64, 32)
        error = max_error(y, reference)
        record('3-a PatchEmbedding', f'output {tuple(y.shape)}, learned position={learned}, '
               f'max err={error:.1e}',
               tuple(y.shape) == (2, 64, 32) and learned and error < 1e-5)

    @guarded('3-b Attention')
    def attention():
        torch.manual_seed(0)
        attn = v.Attention(32, heads=4).eval()
        x = torch.randn(2, 10, 32)
        y = attn(x)

        def heads(t):
            return t.view(2, 10, 4, 8).transpose(1, 2)
        q, k, w = heads(attn.wq(x)), heads(attn.wk(x)), heads(attn.wv(x))
        out = F.scaled_dot_product_attention(q, k, w).transpose(1, 2).reshape(2, 10, 32)
        error = max_error(y, attn.wo(out))
        record('3-b Attention', f'shape {tuple(y.shape)}, max err={error:.1e} '
               '(scaled_dot_product_attention)', y.shape == x.shape and error < 1e-5)

    patch_embedding()
    attention()
    budget_check('ViT', v.ViT)


# -----------------------------------------------------------------------------
# Problem 4
# -----------------------------------------------------------------------------
def check_problem4():
    import p4_pretrained_vit as p4
    print('\nProblem 4. Pretrained ViT (a stand-in backbone; nothing is downloaded)')
    config = {'size': 8, 'mean': (0.5, 0.5, 0.5), 'std': (0.25, 0.25, 0.25)}

    @guarded('4-a extract_features')
    def features():
        torch.manual_seed(0)
        backbone = nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.BatchNorm2d(6),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        backbone.train()
        images = torch.randint(0, 256, (10, 3, 32, 32), dtype=torch.uint8)
        y = p4.extract_features(backbone, images, config, batch_size=4)
        evaluated = not backbone.training
        backbone.eval()
        with torch.no_grad():
            reference = backbone(p4.preprocess(images, config))
        error = max_error(y, reference)
        record('4-a extract_features', f'output {tuple(y.shape)} {str(y.dtype)[6:]}, '
               f'eval mode={evaluated}, no grad={not y.requires_grad}, max err={error:.1e}',
               tuple(y.shape) == (10, 6) and y.dtype == torch.float32 and evaluated
               and not y.requires_grad and error < 1e-5)

    @guarded('4-b train_linear_probe')
    def probe():
        torch.manual_seed(0)
        centres = torch.randn(10, 16) * 3
        train_y, test_y = torch.arange(2000) % 10, torch.arange(500) % 10
        train_x = centres[train_y] + torch.randn(2000, 16)
        test_x = centres[test_y] + torch.randn(500, 16)
        head, accuracy = p4.train_linear_probe(train_x, train_y, test_x, test_y,
                                               epochs=20, lr=1e-2, batch_size=256, seed=0)
        record('4-b train_linear_probe', f'test acc={accuracy:.3f} on separable features '
               '(expected > 0.95)',
               isinstance(head, nn.Linear) and head.in_features == 16
               and head.out_features == 10 and 0.95 < accuracy <= 1.0)

    @guarded('4-c configure_finetune')
    def finetune():
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2))
        optimizer, scheduler = p4.configure_finetune(model, lr=1e-4, weight_decay=0.01,
                                                     total_steps=10)
        covered = {id(p) for group in optimizer.param_groups for p in group['params']}
        all_params = all(id(p) in covered for p in model.parameters())
        start = optimizer.param_groups[0]['lr']
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        end = optimizer.param_groups[0]['lr']
        record('4-c configure_finetune', f'{type(optimizer).__name__}, all params={all_params}, '
               f'lr {start:.1e} -> {end:.1e} after 10 steps',
               isinstance(optimizer, torch.optim.AdamW) and all_params
               and abs(start - 1e-4) < 1e-12 and end < 1e-9
               and abs(optimizer.param_groups[0]['weight_decay'] - 0.01) < 1e-12)

    features()
    probe()
    finetune()


# -----------------------------------------------------------------------------
def main(args):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    print(f'self-check with torch {torch.__version__} on the CPU')
    check_problem1()
    check_problem2()
    check_problem3()
    check_problem4()
    passed = sum(1 for _, _, state in RESULTS if state is True)
    failed = sum(1 for _, _, state in RESULTS if state is False)
    skipped = sum(1 for _, _, state in RESULTS if state is None)
    print(f'\n{passed} passed, {failed} failed, {skipped} skipped out of {len(RESULTS)} checks')
    print('Paste this output into section 2 of your report.')


if __name__ == '__main__':
    main(argparse.ArgumentParser(description='Homework 2 self-check').parse_args())
