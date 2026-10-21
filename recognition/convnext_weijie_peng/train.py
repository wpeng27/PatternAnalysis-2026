"""Train the small CNN on audited splits, without evaluating the test set.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import random
import time

# Required by CUDA's deterministic matrix operations. Set before CUDA initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def check_environment() -> dict:
    """Verify imports and an actual CUDA forward/backward computation."""
    import matplotlib
    import numpy
    import PIL
    import sklearn
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and PyTorch installation.")
    torch.manual_seed(42)
    layer = torch.nn.Conv2d(3, 8, kernel_size=3).to("cuda")
    inputs = torch.randn(2, 3, 224, 224, device="cuda")
    loss = layer(inputs).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or not torch.isfinite(layer.weight.grad).all():
        raise RuntimeError("CUDA check produced a non-finite loss or gradient.")
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "pillow": PIL.__version__,
        "numpy": numpy.__version__, "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__, "pytorch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "cuda_forward_backward": "passed", "model_training": "not performed",
    }


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """A top-level function remains picklable by Windows DataLoader workers."""
    import numpy as np
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False) -> dict:
    """Sample-weighted loss and accuracy, including the final partial batch."""
    import torch
    from torch import nn

    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, samples, skipped_updates = 0.0, 0, 0, 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at batch {step}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # AMP can overflow while its dynamic scale is being adjusted.
                # unscale_ records non-finite gradients, so scaler.step skips that
                # update. Full-precision non-finite gradients remain a hard error.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=not amp)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    skipped_updates += 1
            count = labels.numel()
            samples += count
            total_loss += loss.item() * count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            if step == 1 or step % 50 == 0 or step == len(loader):
                phase = "train" if training else "val"
                print(f"{phase} batch {step}/{len(loader)}: loss={total_loss / samples:.4f}, "
                      f"slice_accuracy={correct / samples:.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if samples == 0:
        raise ValueError("Cannot train or evaluate an empty loader")
    result = {"loss": total_loss / samples, "slice_accuracy": correct / samples,
              "samples": samples, "seconds": time.perf_counter() - start}
    if training:
        result.update({"optimizer_steps": len(loader) - skipped_updates,
                       "amp_skipped_updates": skipped_updates})
    return result


def train(args) -> None:
    import importlib.metadata
    import torch
    from torch.utils.data import DataLoader
    from dataset import ADNIDataset, CLASS_TO_INDEX, IMAGE_SIZE, load_splits, sha256
    from modules import build_model

    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        raise ValueError("Epochs, batch size and threads must be positive, workers nonnegative")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive, weight decay nonnegative")
    if args.output.exists():
        raise ValueError("Run output already exists. Choose a new directory to preserve results.")
    if args.output.resolve().is_relative_to(args.data_root.resolve()):
        raise ValueError("Save training artifacts outside the dataset")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu for CPU execution.")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    splits, audit = load_splits(args.splits)
    if audit["seed"] != args.seed:
        raise ValueError("Training seed must match the saved split seed")
    loaders = {}
    for split in ("train", "val"):
        loaders[split] = DataLoader(
            ADNIDataset(args.data_root, splits[split]), batch_size=args.batch_size,
            shuffle=(split == "train"), num_workers=args.workers,
            pin_memory=(device.type == "cuda"), drop_last=False,
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed),
            persistent_workers=(args.workers > 0),
        )
    # Preflight one real training image before allocating model/optimizer memory.
    image, label = loaders["train"].dataset[0]
    if (image.shape != (3, IMAGE_SIZE, IMAGE_SIZE) or image.dtype != torch.float32
            or not torch.equal(image[0], image[1]) or not torch.equal(image[1], image[2])
            or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1):
        raise RuntimeError("Image preprocessing preflight failed")
    print(f"Image preflight: {tuple(image.shape)}, {image.dtype}, identical grayscale channels", flush=True)
    model = build_model("small_cnn").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"model": "small_cnn", "pretrained": False, "effective_amp": amp,
                   "class_to_index": CLASS_TO_INDEX, "parameter_count": sum(p.numel() for p in model.parameters()),
                   "preprocessing": "Pillow L, bilinear 224x224, float32 /255, repeat three channels",
                   "augmentation": "none", "deterministic_algorithms": True,
                   "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                   "split_sha256": audit["split_sha256"], "python": platform.python_version(),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                   "pytorch_cuda": torch.version.cuda,
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("torch", "torchvision", "Pillow", "numpy", "scikit-learn", "matplotlib")},
                   "source_sha256": {p.name: sha256(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    history = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training = run_epoch(model, loaders["train"], device, optimizer, scaler, amp)
        # Capture training peaks before validation. These are PyTorch allocator peaks,
        # not whole-system GPU memory or an estimate of other models' requirements.
        allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        validation = run_epoch(model, loaders["val"], device, amp=amp)
        record = {"epoch": epoch, "train": training, "val": validation,
                  "peak_training_allocated_bytes": allocated, "peak_training_reserved_bytes": reserved}
        history.append(record)
        checkpoint = {"model_name": "small_cnn", "epoch": epoch,
                      "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "scaler_state_dict": scaler.state_dict(), "config": config, "history": history}
        torch.save(checkpoint, args.output / "last.pt")
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(checkpoint, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2), flush=True)
    print(f"Completed. Artifacts: {args.output}. Test set was not evaluated.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--splits", type=Path, default=Path(__file__).parent / "artifacts/splits_seed42")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts/runs/small_cnn_epoch1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.check_environment:
        print(json.dumps(check_environment(), indent=2))
    elif args.data_root is None:
        parser.error("Training requires --data-root. Use --check-environment for the setup check.")
    else:
        train(args)


if __name__ == "__main__":
    main()
