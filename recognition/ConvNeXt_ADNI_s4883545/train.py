"""Train either architecture or evaluate a saved checkpoint without retraining.

Training validates after each epoch and never evaluates the test set.
Checkpoint evaluation defaults to validation. Test evaluation must be selected
explicitly with --mode evaluate --eval-split test after model selection is final.
AI-assisted implementation.
"""

import argparse
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone

PROCESS_STARTED = time.perf_counter()

# Required by CUDA's deterministic matrix operations. Set before CUDA initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Support both direct script execution and python -m recognition.<project>.train.
if __package__:
    from .dataset import ADNIDataset, AUGMENTATION_SETTINGS, CLASS_TO_INDEX, IMAGE_SIZE, load_splits, sha256
    from .modules import MODEL_NAMES, build_model, weight_decay_groups, ModelEMA, checkpoint_weight_kind
    from .metrics import classification_metrics, aggregate_patients, SELECTION_RULE
    from .training_state import capture_random_state, restore_random_state, save_checkpoint, write_json, update_selection
    from .training_state import WarmupCosine, verify_cosine_checkpoint
else:
    from dataset import ADNIDataset, AUGMENTATION_SETTINGS, CLASS_TO_INDEX, IMAGE_SIZE, load_splits, sha256
    from modules import MODEL_NAMES, build_model, weight_decay_groups, ModelEMA, checkpoint_weight_kind
    from metrics import classification_metrics, aggregate_patients, SELECTION_RULE
    from training_state import capture_random_state, restore_random_state, save_checkpoint, write_json, update_selection
    from training_state import WarmupCosine, verify_cosine_checkpoint

PREPROCESSING = "Pillow L, bilinear 224x224, float32 /255, repeat three channels"
PREPROCESSING_BY_SCALING = {
    "unit": PREPROCESSING,
    "centered": "Pillow L, bilinear 224x224, float32 /255 then 2*x-1, repeat three channels",
}


def emit(message, log_file=None):
    print(message, flush=True)
    if log_file is not None:
        with Path(log_file).open("a", encoding="utf-8") as stream:
            stream.write(str(message) + "\n")


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


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False, phase=None,
              patient_rows=None, include_predictions=False, log_file=None, amp_dtype=None, ema=None,
              step_scheduler=None) -> dict:
    """Sample-weighted loss and accuracy, including the final partial batch."""
    import torch
    from torch import nn

    training = optimizer is not None
    phase = phase or ("train" if training else "val")
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, samples, skipped_updates = 0.0, 0, 0, 0
    all_labels, all_probabilities = [], []
    if patient_rows is not None:
        from torch.utils.data import SequentialSampler
        if training or not isinstance(loader.sampler, SequentialSampler):
            raise ValueError("Patient aggregation requires an ordered, non-training loader")
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
            with torch.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at batch {step}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # AMP can overflow while its dynamic scale is being adjusted.
                # unscale_ records non-finite gradients, so scaler.step skips that
                # update. Without active scaling (FP32/BF16), non-finite gradients
                # must fail before an optimizer update can corrupt the weights.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    skipped_updates += 1
                else:
                    if ema is not None:
                        ema.update(model)
                    if step_scheduler is not None:
                        step_scheduler.step()
            count = labels.numel()
            samples += count
            total_loss += loss.item() * count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            all_labels.extend(labels.detach().cpu().tolist())
            # Column 1 is AD. Never pass predicted classes or P(NC) to AD AUROC.
            all_probabilities.extend(logits.detach().float().softmax(dim=1)[:, CLASS_TO_INDEX["AD"]].cpu().tolist())
            if step == 1 or step % 50 == 0 or step == len(loader):
                updates = (f", optimizer_steps={step - skipped_updates}, amp_skipped={skipped_updates}"
                           if training else "")
                emit(f"{phase} batch {step}/{len(loader)}: loss={total_loss / samples:.4f}, "
                     f"slice_accuracy={correct / samples:.4f}{updates}", log_file)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if samples == 0:
        raise ValueError("Cannot train or evaluate an empty loader")
    result = classification_metrics(all_labels, all_probabilities, loss=total_loss / samples)
    result["slice_accuracy"] = result["accuracy"]  # Backward-compatible, explicit unit.
    if patient_rows is not None:
        result["patient"] = aggregate_patients(patient_rows, all_labels, all_probabilities)
    if include_predictions:
        result["predictions"] = {"labels": all_labels, "p_ad": all_probabilities}
    result["seconds"] = time.perf_counter() - start
    if training:
        result.update({"optimizer_steps": len(loader) - skipped_updates,
                       "amp_skipped_updates": skipped_updates,
                       "measurement_note": "Online training metrics across changing weights and active dropout"})
    return result


def prepare_run(args):
    """Check output paths and device, then configure reproducible execution."""
    import torch

    if args.output.exists():
        raise ValueError("Run output already exists. Choose a new directory to preserve results.")
    if not args.data_root.is_dir():
        raise ValueError(f"Dataset root does not exist: {args.data_root}")
    if args.output.resolve().is_relative_to(args.data_root.resolve()):
        raise ValueError("Save run artifacts outside the dataset")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        raise ValueError("Device must be cpu, cuda or cuda:N") from error
    if device.type not in {"cpu", "cuda"} or (device.type == "cpu" and device.index is not None):
        raise ValueError("Device must be cpu, cuda or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable. Use --device cpu for CPU execution.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {device.index}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    return device


def make_loader(args, rows, device, training=False):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        ADNIDataset(args.data_root, rows, input_scaling=args.input_scaling,
                    augmentation=args.train_augmentation if training else "none"), batch_size=args.batch_size,
        shuffle=training, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed),
        # Restart workers at each epoch boundary so their seeding can be restored
        # from generator state during resume. Random transforms use torch RNG.
        persistent_workers=False,
    )


def train(args) -> None:
    import importlib.metadata
    import torch

    segment_started = time.perf_counter()
    device = prepare_run(args)
    splits, audit = load_splits(args.splits)
    if audit["seed"] != args.seed:
        raise ValueError("Training seed must match the saved split seed")
    loaders = {split: make_loader(args, splits[split], device, training=(split == "train"))
               for split in ("train", "val")}
    # Preflight one real training image before allocating model/optimizer memory.
    # Random augmentation in preflight must not change model initialization or
    # consume the first training transform. PIL RandomAffine draws on CPU RNG.
    with torch.random.fork_rng(devices=[]):
        image, label = loaders["train"].dataset[0]
    if (image.shape != (3, IMAGE_SIZE, IMAGE_SIZE) or image.dtype != torch.float32
            or not torch.equal(image[0], image[1]) or not torch.equal(image[1], image[2])
            or not torch.isfinite(image).all()
            or image.min() < (-1 if args.input_scaling == "centered" else 0) or image.max() > 1):
        raise RuntimeError("Image preprocessing preflight failed")
    print(f"Image preflight: {tuple(image.shape)}, {image.dtype}, identical grayscale channels, "
          f"input_scaling={args.input_scaling}", flush=True)
    model_kwargs = {"drop_path_rate": args.drop_path_rate}
    model = build_model(args.model, **model_kwargs).to(device)
    ema = ModelEMA(model, args.ema_decay) if args.ema else None
    parameter_groups, group_metadata = weight_decay_groups(model, args.weight_decay, args.weight_decay_policy)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    amp = (args.amp is not False) and device.type == "cuda"
    amp_dtype = getattr(torch, args.amp_dtype)
    if amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16 AMP")
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    if args.lr_schedule == "warmup_cosine":
        scheduler = WarmupCosine(optimizer, args.schedule_epochs * len(loaders["train"]),
                                 args.warmup_epochs * len(loaders["train"]))
        scheduler_settings = dict(scheduler.settings)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
            threshold=args.min_delta, threshold_mode="abs", min_lr=1e-6)
        scheduler_settings = {"name": "ReduceLROnPlateau", "metric": "val.loss", "mode": "min",
                              "factor": args.lr_factor, "patience": args.lr_patience,
                              "threshold": args.min_delta, "threshold_mode": "abs", "min_lr": 1e-6}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"model": args.model, "pretrained": False, "effective_amp": amp,
                   "class_to_index": CLASS_TO_INDEX, "parameter_count": sum(p.numel() for p in model.parameters()),
                   "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "preprocessing": PREPROCESSING_BY_SCALING[args.input_scaling],
                   "augmentation": args.train_augmentation,
                   "augmentation_settings": AUGMENTATION_SETTINGS[args.train_augmentation],
                   "deterministic_algorithms": True,
                   "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                   "split_sha256": audit["split_sha256"], "python": platform.python_version(),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                   "pytorch_cuda": torch.version.cuda,
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("torch", "torchvision", "Pillow", "numpy", "scikit-learn", "matplotlib")},
                   "source_sha256": {p.name: sha256(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}})
    config.update({"checkpoint_version": 2, "selection_rule": SELECTION_RULE,
                   "validation_weights": "ema" if ema is not None else "online",
                   "ema_settings": {"enabled": args.ema, "decay": args.ema_decay,
                                    "initialization": "Copy fresh online model", "warmup": False,
                                    "update": "After each successful optimizer step only",
                                    "dtype": "float32", "buffers": "Average floating, copy integer"},
                   "model_kwargs": model_kwargs,
                   "optimizer_parameter_groups": group_metadata,
                   "optimizer_settings": {"name": "AdamW", "lr": args.lr,
                                          "weight_decay": args.weight_decay, "betas": [0.9, 0.999],
                                          "eps": 1e-8, "parameter_groups":
                                          "All parameters, including biases and normalization" if args.weight_decay_policy == "all"
                                          else "No decay for biases and 1D parameters, configured decay for remaining weights"},
                   "scheduler_settings": scheduler_settings,
                   "regularization_settings": {"weight_decay": args.weight_decay, "drop_path_rate": args.drop_path_rate,
                                               "weight_decay_policy": args.weight_decay_policy,
                                               "classifier_dropout": 0.2 if args.model == "small_cnn" else 0.0,
                                               "label_smoothing": 0.0, "augmentation": args.train_augmentation,
                                               "gradient_clip_norm": 5.0},
                   "model_settings": {"architecture": str(model), "input_shape": [3, IMAGE_SIZE, IMAGE_SIZE], "num_classes": 2},
                   "selection_metric": "val.loss", "selection_unit": "slice",
                   "early_stopping_patience": args.patience, "early_stopping_min_delta": args.min_delta,
                   "scheduler": "Linear warmup and cosine decay per successful optimizer update" if args.lr_schedule == "warmup_cosine"
                                else "ReduceLROnPlateau on validation slice cross-entropy",
                   "scheduler_factor": args.lr_factor, "scheduler_patience": args.lr_patience,
                   "precision": ("CUDA float16 autocast with GradScaler" if args.amp_dtype == "float16"
                                 else "CUDA bfloat16 autocast without gradient scaling") if amp else "float32",
                   "device_type": device.type, "command": [sys.executable, *sys.argv], "pid": os.getpid(),
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "patient_metrics_basis": "Mean P(AD) with manifest label consistency verified",
                   "timing_definition": "Epoch seconds include training, validation and metrics, exclude checkpoint disk writes",
                   "memory_definition": "Peak PyTorch allocated/reserved bytes across full training forward/backward/optimizer steps"})
    history = []
    selection = {"best_loss": None, "best_epoch": None, "bad_epochs": 0, "stopped": False}
    start_epoch = 1
    best_source = None
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get("checkpoint_version") != 2:
            raise ValueError("This checkpoint lacks complete resume state. Old checkpoints remain evaluation-only.")
        previous = saved["config"]
        checkpoint_weight_kind(saved)
        if previous.get("lr_schedule", "plateau") != args.lr_schedule:
            raise ValueError("Resume setting differs from the original run: lr_schedule")
        if args.lr_schedule == "warmup_cosine":
            if any(previous.get(k) != config[k] for k in ("schedule_epochs", "warmup_epochs", "scheduler_settings")):
                raise ValueError("Resume cosine schedule settings/horizon differ from original run")
            verify_cosine_checkpoint(saved, config)
        if previous.get("ema", False) != args.ema or (args.ema and previous.get("ema_decay") != args.ema_decay):
            raise ValueError("Resume setting differs from the original run: ema/ema_decay")
        if previous.get("weight_decay_policy", "all") != args.weight_decay_policy:
            raise ValueError("Resume setting differs from the original run: weight_decay_policy")
        if (previous.get("optimizer_parameter_groups") is not None
                and previous["optimizer_parameter_groups"] != group_metadata):
            raise ValueError("Resume optimizer parameter groups differ from the original run")
        if previous.get("amp_dtype", "float16") != args.amp_dtype:
            raise ValueError("Resume setting differs from the original run: amp_dtype")
        if previous.get("model_kwargs", {"drop_path_rate": 0.0}) != model_kwargs:
            raise ValueError("Resume setting differs from the original run: model_kwargs")
        if (previous.get("augmentation", "none") != args.train_augmentation
                or previous.get("augmentation_settings", AUGMENTATION_SETTINGS["none"]) != config["augmentation_settings"]):
            raise ValueError("Resume setting differs from the original run: augmentation")
        required_same = ("model", "batch_size", "lr", "weight_decay", "seed", "workers", "threads",
                         "effective_amp", "class_to_index", "preprocessing", "split_sha256", "device_type", "model_settings",
                         "device_name", "packages", "selection_rule", "early_stopping_patience",
                         "early_stopping_min_delta", "scheduler_factor", "scheduler_patience")
        for key in required_same:
            if previous.get(key) != config.get(key):
                raise ValueError(f"Resume setting differs from the original run: {key}")
        selection = dict(saved["selection_state"])
        if selection["stopped"]:
            raise ValueError("This run already met its predefined early-stopping rule")
        start_epoch = saved["epoch"] + 1
        if args.epochs < start_epoch:
            raise ValueError("--epochs is the total target and must exceed the resumed epoch")
        best_source = args.resume.parent / "best.pt"
        if not best_source.is_file() or sha256(best_source.read_bytes()) != saved["best_checkpoint_sha256"]:
            raise ValueError("Resume requires the unchanged companion best.pt beside last.pt")
        model.load_state_dict(saved["training_model_state_dict"] if ema is not None else saved["model_state_dict"])
        if ema is not None:
            ema.model.load_state_dict(saved["model_state_dict"], strict=True)
            ema.updates = saved["ema_updates"]
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        history = list(saved["history"])
        restore_random_state(saved["random_state"], loaders)
        config["resumed_from"] = {"path": str(args.resume.resolve()), "epoch": saved["epoch"],
                                  "sha256": sha256(args.resume.read_bytes())}
    args.output.mkdir(parents=True, exist_ok=False)
    if best_source is not None:
        shutil.copy2(best_source, args.output / "best.pt")
    write_json(args.output / "config.json", config)
    log_file = args.output / "training.log"
    emit(f"PID {os.getpid()}. {SELECTION_RULE}. NC=0, AD=1. This run evaluates training and validation only.", log_file)
    startup_seconds = time.perf_counter() - segment_started
    process_startup_seconds = time.perf_counter() - PROCESS_STARTED
    status = {"state": "running", "pid": os.getpid(), "start_epoch": start_epoch,
              "target_epochs": args.epochs, "last_completed_epoch": start_epoch - 1,
              "startup_seconds": startup_seconds, "process_startup_seconds": process_startup_seconds,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "status.json", status)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            emit(f"Epoch {epoch}/{args.epochs}", log_file)
            epoch_started = time.perf_counter()
            current_lr = optimizer.param_groups[0]["lr"]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            training = run_epoch(model, loaders["train"], device, optimizer, scaler, amp, log_file=log_file,
                                 amp_dtype=amp_dtype, ema=ema,
                                 step_scheduler=scheduler if args.lr_schedule == "warmup_cosine" else None)
            allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            evaluation_model = ema.model if ema is not None else model
            validation = run_epoch(evaluation_model, loaders["val"], device, amp=amp,
                                   patient_rows=splits["val"], log_file=log_file, amp_dtype=amp_dtype)
            if args.lr_schedule == "plateau":
                scheduler.step(validation["loss"])
            improved = update_selection(selection, validation["loss"], epoch, args.patience, args.min_delta)
            record = {"epoch": epoch, "train": training, "val": validation,
                      "validation_weights": config["validation_weights"],
                      "ema_updates": ema.updates if ema is not None else 0,
                      "scheduler_updates": scheduler.completed_updates if args.lr_schedule == "warmup_cosine" else None,
                      "learning_rate": current_lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
                      "epoch_seconds": time.perf_counter() - epoch_started,
                      "peak_training_allocated_bytes": allocated, "peak_training_reserved_bytes": reserved,
                      "selected_as_best": improved, "bad_epochs": selection["bad_epochs"]}
            history.append(record)
            checkpoint = {"checkpoint_version": 2, "model_name": args.model, "epoch": epoch,
                          "model_state_dict": evaluation_model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                          "evaluation_weights": config["validation_weights"],
                          "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                          "random_state": capture_random_state(loaders), "selection_state": dict(selection),
                          "config": config, "history": history}
            if ema is not None:
                checkpoint.update(training_model_state_dict=model.state_dict(), ema_updates=ema.updates)
            if improved:
                save_checkpoint(args.output / "best.pt", checkpoint)
            checkpoint["best_checkpoint_sha256"] = sha256((args.output / "best.pt").read_bytes())
            save_checkpoint(args.output / "last.pt", checkpoint)
            write_json(args.output / "history.json", history)
            status.update(last_completed_epoch=epoch, best_epoch=selection["best_epoch"],
                          segment_seconds=time.perf_counter() - segment_started)
            write_json(args.output / "status.json", status)
            emit(f"Epoch {epoch} done: train_loss={training['loss']:.5f}, val_loss={validation['loss']:.5f}, "
                 f"val_slice_accuracy={validation['accuracy']:.4f}, val_AD_AUROC={validation['ad_auroc']:.4f}, "
                 f"seconds={record['epoch_seconds']:.2f}, best_epoch={selection['best_epoch']}", log_file)
            if selection["stopped"]:
                break
        status.update(state="completed", stop_reason="early_stopping" if selection["stopped"] else "epoch_limit",
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      segment_seconds=time.perf_counter() - segment_started,
                      process_seconds=time.perf_counter() - PROCESS_STARTED,
                      test_evaluated=False)
        write_json(args.output / "status.json", status)
        emit(f"Completed: {status['stop_reason']}. Best epoch {selection['best_epoch']}. Artifacts: {args.output}", log_file)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}",
                      segment_seconds=time.perf_counter() - segment_started)
        write_json(args.output / "status.json", status)
        emit(status["error"], log_file)
        raise


def evaluate(args) -> dict:
    """Load weights and evaluate one split, without an optimizer or weight writes.

    Checkpoint loading follows PyTorch's state_dict workflow:
    https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    import torch

    device = prepare_run(args)
    splits, audit = load_splits(args.splits)
    raw = args.checkpoint.read_bytes()
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    evaluation_weights = checkpoint_weight_kind(checkpoint)
    if checkpoint.get("model_name") != args.model:
        raise ValueError("Selected --model does not match the checkpoint model")
    saved_scaling = config.get("input_scaling", "unit")
    if saved_scaling not in PREPROCESSING_BY_SCALING:
        raise ValueError("Unsupported checkpoint input scaling")
    if args.input_scaling is not None and args.input_scaling != saved_scaling:
        raise ValueError("Evaluation input scaling must match the checkpoint configuration")
    args.input_scaling = saved_scaling
    if (config.get("class_to_index") != CLASS_TO_INDEX
            or config.get("preprocessing") != PREPROCESSING_BY_SCALING[saved_scaling]):
        raise ValueError("Checkpoint class encoding or preprocessing does not match this loader")
    if config.get("split_sha256") != audit["split_sha256"]:
        raise ValueError("Saved splits do not match the checkpoint's original split hashes")
    model_kwargs = config.get("model_kwargs", {"drop_path_rate": 0.0})
    if args.drop_path_rate is not None and args.drop_path_rate != model_kwargs["drop_path_rate"]:
        raise ValueError("Evaluation drop-path rate must match the checkpoint configuration")
    model = build_model(args.model, **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader = make_loader(args, splits[args.eval_split], device)
    requested_amp = config.get("effective_amp", False) if args.amp is None else args.amp
    amp = requested_amp and device.type == "cuda"
    amp_dtype_name = args.amp_dtype or config.get("amp_dtype", "float16")
    amp_dtype = getattr(torch, amp_dtype_name)
    if amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16 AMP")
    metrics = run_epoch(model, loader, device, amp=amp, phase=args.eval_split,
                        patient_rows=splits[args.eval_split], include_predictions=args.save_predictions,
                        amp_dtype=amp_dtype)
    predictions = metrics.pop("predictions", None)
    result = {"mode": "evaluate", "model": args.model, "split": args.eval_split,
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(raw),
              "checkpoint_epoch": checkpoint["epoch"], "split_sha256": audit["split_sha256"],
              "evaluation_weights": evaluation_weights, "ema_decay": config.get("ema_decay") if config.get("ema") else None,
              "model_kwargs": model_kwargs,
              "input_scaling": saved_scaling, "preprocessing": config["preprocessing"],
              "training_augmentation": config.get("augmentation", "none"), "evaluation_augmentation": "none",
              "device": str(device), "batch_size": args.batch_size, "effective_amp": amp,
              "amp_dtype": amp_dtype_name,
              "seed": args.seed, "python": platform.python_version(), "torch": str(torch.__version__),
              "pytorch_cuda": torch.version.cuda, "metrics": metrics, "retrained": False}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if predictions is not None:
        import csv
        with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("path", "patient_id", "label", "p_ad", "predicted", "confidence"))
            writer.writeheader()
            for row, label, probability in zip(splits[args.eval_split], predictions["labels"], predictions["p_ad"]):
                writer.writerow({"path": row["path"], "patient_id": row["patient_id"], "label": label,
                                 "p_ad": probability, "predicted": int(probability > 0.5),
                                 "confidence": max(probability, 1 - probability)})
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--check-environment", action="store_true", help="Run only the CUDA dependency check")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train", help="Train or load a checkpoint")
    parser.add_argument("--model", choices=MODEL_NAMES, default="small_cnn", help="Implemented model architecture")
    parser.add_argument("--data-root", type=Path, help="Directory containing train/AD, train/NC, test/AD, test/NC")
    parser.add_argument("--splits", type=Path, default=Path(__file__).parent / "artifacts/splits_seed42",
                        help="Existing audited split lists, reused without resplitting")
    parser.add_argument("--output", "--output-dir", dest="output", type=Path,
                        help="New output directory (default: artifacts/runs/<model>_<mode>)")
    parser.add_argument("--epochs", type=int, default=1, help="Total epoch target, including previously completed epochs on resume")
    parser.add_argument("--resume", type=Path, help="Version-2 last.pt with companion best.pt, resume at next epoch into a new output")
    parser.add_argument("--patience", type=int, default=5, help="Early stop after this many non-improving validation losses")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Required absolute validation-loss decrease for selection")
    parser.add_argument("--lr-patience", type=int, default=2, help="ReduceLROnPlateau patience")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Learning-rate multiplier on a plateau")
    parser.add_argument("--lr-schedule", choices=("plateau", "warmup_cosine"), default="plateau",
                        help="Plateau after validation, or warmup/cosine per successful optimizer update")
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Cosine only: warmup duration in training-loader epochs (default 3)")
    parser.add_argument("--schedule-epochs", type=int, default=None,
                        help="Cosine only: fixed total LR horizon (defaults to --epochs), preserve on resume")
    parser.add_argument("--batch-size", type=int, default=32, help="Training/evaluation batch size")
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--weight-decay-policy", choices=("all", "exclude_1d_bias"), default="all",
                        help="Training optimizer: all parameters, or zero decay for biases and 1D tensors")
    parser.add_argument("--ema", action="store_true", help="Train with an FP32 EMA shadow and select using EMA validation loss")
    parser.add_argument("--ema-decay", type=float, default=None, help="Fixed EMA decay per successful optimizer step (train default 0.999)")
    parser.add_argument("--drop-path-rate", type=float, default=None,
                        help="ConvNeXt maximum stochastic-depth probability (train default 0, evaluate inherits checkpoint)")
    parser.add_argument("--input-scaling", choices=("unit", "centered"), default=None,
                        help="unit=[0,1], centered=2*x-1 (train default unit, evaluation inherits checkpoint)")
    parser.add_argument("--train-augmentation", choices=tuple(AUGMENTATION_SETTINGS), default="none",
                        help="Training only: mild_affine rotates +/-7 degrees, shifts +/-5%% and scales 0.95-1.05")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, must match split seed when training")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers, zero disables subprocesses")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--device", default="cuda", help="cpu, cuda or cuda:N")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Mixed precision on CUDA (default: on in training, checkpoint setting in evaluation)")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default=None,
                        help="Autocast dtype (train default float16, evaluation inherits checkpoint)")
    parser.add_argument("--checkpoint", type=Path, help="Saved checkpoint, required in evaluate mode")
    parser.add_argument("--eval-split", choices=("val", "test"), help="Evaluate mode only (default: val)")
    parser.add_argument("--save-predictions", action="store_true", help="Save per-slice P(AD) in evaluate mode")
    args = parser.parse_args(argv)
    if args.lr_schedule == "warmup_cosine":
        if args.mode != "train":
            parser.error("Learning-rate schedules are training-only")
        args.warmup_epochs = 3 if args.warmup_epochs is None else args.warmup_epochs
        args.schedule_epochs = args.epochs if args.schedule_epochs is None else args.schedule_epochs
        if args.warmup_epochs < 0 or args.schedule_epochs - args.warmup_epochs < 2:
            parser.error("Cosine horizon needs at least two epochs after nonnegative warmup")
        if args.epochs > args.schedule_epochs:
            parser.error("Training epochs cannot exceed the fixed cosine horizon")
        if args.lr < 1e-6:
            parser.error("Cosine peak LR must be at least its minimum LR of 1e-6")
    elif args.warmup_epochs is not None or args.schedule_epochs is not None:
        parser.error("--warmup-epochs and --schedule-epochs require --lr-schedule warmup_cosine")
    if args.mode == "evaluate" and (args.ema or args.ema_decay is not None):
        parser.error("EMA evaluation weights are inherited from the checkpoint, omit --ema and --ema-decay")
    if args.ema_decay is not None and not args.ema:
        parser.error("--ema-decay requires --ema")
    if args.ema_decay is None:
        args.ema_decay = 0.999
    if not math.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1:
        parser.error("EMA decay must be finite and in [0, 1)")
    if args.mode == "evaluate" and args.train_augmentation != "none":
        parser.error("Augmentation is training-only. Evaluation always uses unchanged images.")
    if args.mode == "train" and args.input_scaling is None:
        args.input_scaling = "unit"
    if args.mode == "train" and args.amp_dtype is None:
        args.amp_dtype = "float16"
    if args.mode == "train" and args.drop_path_rate is None:
        args.drop_path_rate = 0.0
    if args.drop_path_rate is not None:
        if not math.isfinite(args.drop_path_rate) or not 0 <= args.drop_path_rate < 1:
            parser.error("Drop-path rate must be finite and in [0, 1)")
        if args.model == "small_cnn" and args.drop_path_rate != 0:
            parser.error("Nonzero --drop-path-rate requires --model convnext_tiny")
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.workers < 0:
        parser.error("Epochs, batch size and threads must be positive, workers nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Learning rate must be finite and positive, weight decay finite and nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be between 0 and 2**32 - 1")
    if args.patience < 1 or args.lr_patience < 0 or not 0 < args.lr_factor < 1:
        parser.error("Patience must be positive, LR patience nonnegative and LR factor between zero and one")
    if not math.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("Minimum loss decrease must be finite and nonnegative")
    if not args.check_environment:
        if args.data_root is None:
            parser.error("--data-root is required for training and evaluation")
        if args.mode == "evaluate" and args.checkpoint is None:
            parser.error("--checkpoint is required in evaluate mode")
        if args.mode == "train" and (args.checkpoint is not None or args.eval_split is not None):
            parser.error("--checkpoint and --eval-split require --mode evaluate")
        if args.mode == "evaluate" and args.resume is not None:
            parser.error("--resume is only for training, use --checkpoint for evaluation")
        if args.mode == "train" and args.save_predictions:
            parser.error("--save-predictions is only for evaluation")
    if args.mode == "evaluate" and args.eval_split is None:
        args.eval_split = "val"
    if args.output is None:
        args.output = Path(__file__).parent / "artifacts/runs" / f"{args.model}_{args.mode}"
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.check_environment:
        print(json.dumps(check_environment(), indent=2))
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
