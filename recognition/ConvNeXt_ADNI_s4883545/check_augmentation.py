"""Preview training-only augmentation and check four real GPU optimizer steps.

This is a functional smoke check, not a model performance experiment. It reads
64 training and 32 validation images from the saved manifests, never test images.
No checkpoint is produced and no resulting weights are used by the full run.
"""

import argparse
import json
from pathlib import Path
import random

import torch

if __package__:
    from .dataset import ADNIDataset, AUGMENTATION_SETTINGS, load_splits
    from .modules import build_model
    from .train import make_loader, parse_args, run_epoch, seed_everything
else:
    from dataset import ADNIDataset, AUGMENTATION_SETTINGS, load_splits
    from modules import build_model
    from train import make_loader, parse_args, run_epoch, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise ValueError("Use a fresh diagnostic output directory")
    if options.output.resolve().is_relative_to(options.data_root.resolve()):
        raise ValueError("Save diagnostics outside the dataset")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This check requires CUDA with BF16 support")
    splits, audit = load_splits(options.splits)
    rng = random.Random(42)
    train_rows = rng.sample(splits["train"], 64)
    val_rows = rng.sample(splits["val"], 32)
    args = parse_args(["--data-root", str(options.data_root), "--splits", str(options.splits),
                       "--model", "convnext_tiny", "--batch-size", "16", "--lr", "0.00003",
                       "--weight-decay", "0.01", "--drop-path-rate", "0.1", "--workers", "0",
                       "--threads", "4", "--amp", "--amp-dtype", "bfloat16",
                       "--input-scaling", "centered", "--train-augmentation", "mild_affine"])
    seed_everything(42)
    torch.set_num_threads(4)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Show two images per class, with fresh independently sampled transforms.
    preview_rows = [r for label in (0, 1) for r in [x for x in train_rows if x["label"] == label][:2]]
    plain = ADNIDataset(options.data_root, preview_rows, "centered")
    augmented = ADNIDataset(options.data_root, preview_rows, "centered", "mild_affine")
    figure, axes = plt.subplots(len(preview_rows), 3, figsize=(8, 10), constrained_layout=True)
    for row_index, row in enumerate(preview_rows):
        for col in range(3):
            tensor, label = (plain if col == 0 else augmented)[row_index]
            assert tensor.shape == (3, 224, 224) and torch.isfinite(tensor).all()
            assert torch.equal(tensor[0], tensor[1]) and torch.equal(tensor[1], tensor[2])
            axes[row_index, col].imshow(tensor[0], cmap="gray", vmin=-1, vmax=1)
            axes[row_index, col].set_title(f"{'AD' if label else 'NC'} | " + ("original" if col == 0 else f"affine {col}"))
            axes[row_index, col].axis("off")
    figure.suptitle("Training images only: augmentation inspection")
    options.output.mkdir(parents=True, exist_ok=False)
    figure.savefig(options.output / "augmentation_preview.png", dpi=150)
    plt.close(figure)

    seed_everything(42)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = build_model("convnext_tiny", drop_path_rate=0.1).to(device)
    initial_head = model.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    torch.cuda.reset_peak_memory_stats(device)
    train_metrics = run_epoch(model, make_loader(args, train_rows, device, training=True), device,
                              optimizer, scaler, amp=True, amp_dtype=torch.bfloat16)
    allocated, reserved = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
    assert train_metrics["optimizer_steps"] == 4 and train_metrics["amp_skipped_updates"] == 0
    assert not torch.equal(initial_head, model.head.weight)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.isfinite(t).all() for t in model.state_dict().values())
    val_loader = make_loader(args, val_rows, device)
    assert val_loader.dataset.augmentation == "none"
    val_metrics = run_epoch(model, val_loader, device, amp=True, amp_dtype=torch.bfloat16)
    report = {"scope": "Functional smoke check, no accuracy claim or saved model",
              "train_images": 64, "validation_images": 32, "test_images_read": 0,
              "optimizer_steps": 4, "amp_skipped_updates": 0,
              "finite_gradients_and_weights": True, "weights_updated": True,
              "split_sha256": audit["split_sha256"], "seed": 42,
              "gpu": torch.cuda.get_device_name(device), "batch_size": 16, "amp_dtype": "bfloat16",
              "augmentation_settings": AUGMENTATION_SETTINGS["mild_affine"],
              "evaluation_augmentation": "none", "train_seconds": train_metrics["seconds"],
              "validation_seconds": val_metrics["seconds"],
              "peak_training_allocated_gib": allocated / 2**30, "peak_training_reserved_gib": reserved / 2**30,
              "budget_note": "Four batches include startup overhead and do not measure a full epoch."}
    (options.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
