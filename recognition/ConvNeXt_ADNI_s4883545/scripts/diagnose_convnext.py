"""Read-only model diagnostics on training images. Never reads test images.

Compare the full custom architecture with torchvision using identical trained
weights, then measure initial gradients for two fixed input scalings. No
optimizer update or performance claim is made by this diagnostic.
"""

# Support direct script execution from any working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paths import PROJECT_ROOT

import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny

from dataset import ADNIDataset, load_splits
from modules import build_model
from train import seed_everything
from training_state import write_json


def reference_state(state):
    """Map our parameter names to torchvision 0.24's equivalent modules."""
    result = {}
    block_names = {'dwconv': '0', 'norm': '2', 'pwconv1': '3', 'pwconv2': '5'}
    for name, value in state.items():
        parts = name.split('.')
        if parts[0] == 'downsample_layers':
            target = f'features.{2 * int(parts[1])}.' + '.'.join(parts[2:])
        elif parts[0] == 'stages':
            prefix = f'features.{2 * int(parts[1]) + 1}.{parts[2]}'
            if parts[3] == 'gamma':
                target, value = prefix + '.layer_scale', value[:, None, None]
            else:
                target = prefix + f'.block.{block_names[parts[3]]}.' + '.'.join(parts[4:])
        elif parts[0] == 'norm':
            target = 'classifier.0.' + parts[1]
        elif parts[0] == 'head':
            target = 'classifier.2.' + parts[1]
        else:
            raise ValueError(f'Unmapped model parameter: {name}')
        result[target] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--splits', type=Path, default=PROJECT_ROOT / 'artifacts/splits_seed42')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh report path')
    seed_everything(42)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    splits, audit = load_splits(args.splits)
    loader = DataLoader(ADNIDataset(args.data_root, splits['train']), batch_size=16,
                        shuffle=True, generator=torch.Generator().manual_seed(42), num_workers=0)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    custom = build_model('convnext_tiny', **saved['config']['model_kwargs']).to(device).eval()
    custom.load_state_dict(saved['model_state_dict'], strict=True)
    reference = convnext_tiny(weights=None, num_classes=2).to(device).eval()
    reference.load_state_dict(reference_state(saved['model_state_dict']), strict=True)
    with torch.inference_mode():
        ours, official = custom(images), reference(images)
    torch.testing.assert_close(ours, official, atol=1e-5, rtol=1e-5)
    report = {'seed': 42, 'split': 'train', 'images': 16, 'test_images_read': 0,
              'split_sha256': audit['split_sha256'],
              'reference': 'Installed torchvision.models.convnext_tiny, weights=None, loaded with the same checkpoint',
              'reference_max_absolute_logit_error': (ours-official).abs().max().item(),
              'reference_agreement_passed': True, 'optimizer_updates': 0, 'gradient_diagnostics': {}}
    del custom, reference, saved
    for scaling in ('unit', 'centered', 'centered_bfloat16'):
        seed_everything(42)
        model = build_model('convnext_tiny', drop_path_rate=0.1).to(device).train()
        inputs = images if scaling == 'unit' else images * 2 - 1
        stats = {}
        def stem_hook(module, inputs, outputs):
            variances = outputs.detach().float().var(dim=1, correction=0)
            stats['stem_zero_variance_fraction'] = (variances == 0).float().mean().item()
        handle = model.downsample_layers[0][0].register_forward_hook(stem_hook)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=scaling == 'centered_bfloat16'):
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
        handle.remove()
        loss.backward()
        norms = {name: p.grad.detach().norm().item() for name, p in model.named_parameters()}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        stats.update(loss=loss.item(), total_gradient_norm=norm.item(),
                     largest_gradient_tensors=sorted(norms.items(), key=lambda item: item[1], reverse=True)[:5])
        report['gradient_diagnostics'][scaling] = stats
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    import json
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
