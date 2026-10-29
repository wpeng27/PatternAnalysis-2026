"""Measure warmed, synchronized CUDA inference on audited validation images only.

Two scopes: GPU-resident input to logits, and local JPEG path to CPU probabilities.
Wall timing synchronizes CUDA before/after each batch, following PyTorch guidance:
https://docs.pytorch.org/docs/2.9/notes/cuda.html#asynchronous-execution
No training, test evaluation, compilation or changes to selected checkpoints.
AI-assisted implementation with explicit timing and statistical checks.
"""

# Support direct script execution from any working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paths import source_path

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

# train sets CUBLAS_WORKSPACE_CONFIG before CUDA initialization.
from train import seed_everything, PREPROCESSING_BY_SCALING
from dataset import ADNIDataset, load_splits
from modules import build_model, checkpoint_weight_kind
from scripts.compare_validation import verify_run, file_hash


def summarize(milliseconds, batch_size):
    """Throughput uses total images / total batch time, not mean reciprocals."""
    values = np.asarray(milliseconds, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(values <= 0) or batch_size < 1):
        raise ValueError("Need positive finite timing samples and a positive batch size")
    return {"batches": len(values), "batch_size": batch_size,
            "mean_batch_ms": float(values.mean()), "median_batch_ms": float(np.median(values)),
            "p95_batch_ms": float(np.quantile(values, .95)), "min_batch_ms": float(values.min()),
            "max_batch_ms": float(values.max()), "std_batch_ms": float(values.std(ddof=0)),
            "amortized_ms_per_image": float(values.mean() / batch_size),
            "images_per_second": float(len(values) * batch_size * 1000 / values.sum())}


def cyclic_indices(step, batch_size, count):
    if step < 0 or batch_size < 1 or count < batch_size:
        raise ValueError("Need nonnegative step and enough images for a full batch")
    return [(step * batch_size + offset) % count for offset in range(batch_size)]


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gpu_snapshot():
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv"], capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def measure(model, dataset, gpu_batches, batch_size, precision, scope, device,
            warmup=20, iterations=100, repeats=3):
    import torch

    amp = precision != "float32"
    dtype = getattr(torch, precision)

    def invoke(step):
        if scope == "jpeg_to_probabilities":
            indices = cyclic_indices(step, batch_size, len(dataset))
            images = torch.stack([dataset[i][0] for i in indices]).to(device)
        else:
            images = gpu_batches[step % len(gpu_batches)]
        with torch.autocast(device_type="cuda", enabled=amp, dtype=dtype):
            logits = model(images)
        return logits.float().softmax(dim=1).cpu() if scope == "jpeg_to_probabilities" else logits

    if model.training:
        raise ValueError("Latency must be measured in model.eval() mode")
    samples, groups = [], []
    with torch.inference_mode():
        for repeat in range(repeats):
            for i in range(warmup):
                output = invoke(i)
            torch.cuda.synchronize(device)
            times = []
            for i in range(iterations):
                # Synchronization drains previous work before the timed interval.
                torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                output = invoke(i + repeat * iterations)
                torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                times.append(elapsed_ms)
                # Check outside timing, without retaining all GPU outputs.
                if output.shape != (batch_size, 2) or not torch.isfinite(output).all().item():
                    raise RuntimeError("Invalid inference output")
            groups.append(summarize(times, batch_size))
            samples.extend(times)
            print(f"{scope}, {precision}, batch {batch_size}, repeat {repeat+1}/{repeats}: "
                  f"median {groups[-1]['median_batch_ms']:.3f} ms", flush=True)
    return {"scope": scope, "precision": precision, "batch_size": batch_size,
            "warmup_batches_per_repeat": warmup, "iterations_per_repeat": iterations,
            "repeats": repeats, "summary": summarize(samples, batch_size),
            "repeat_summaries": groups, "raw_batch_ms": samples}


def reproduction_command(settings):
    """PowerShell command preserving the actual models and all timing settings."""
    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    command = ["py", "-3.13", quote(Path(__file__).resolve())]
    for name in ("cnn_run", "convnext_run", "splits", "data_root", "output"):
        path = Path(settings[name]).resolve()
        if name == "output":
            path = path.with_name(path.name + "_recheck")
        command.extend(("--" + name.replace("_", "-"), quote(path)))
    command.extend(("--batch-sizes", *(str(value) for value in settings["batch_sizes"])))
    for name in ("samples", "warmup", "iterations", "repeats", "seed", "threads"):
        command.extend(("--" + name, str(settings[name])))
    return " ".join(command)


def write_report(output, result):
    lines = ["# Measured inference latency", "",
             f"Completed {result['finished_utc']} (UTC). GPU: {result['environment']['gpu']}. "
             f"Python {result['environment']['python']}, PyTorch {result['environment']['torch']}, "
             f"CUDA runtime {result['environment']['cuda']}, CPU threads {result['settings']['threads']}.", "",
             f"Same {result['validation_subset']['slices']} validation slices "
             f"from {result['validation_subset']['patients']} patients for both models, sampled without replacement "
             f"using seed {result['settings']['seed']}. Sampled slices are reused cyclically. No test images or metrics.", "",
             "## Timing protocol", "",
             f"Each configuration: {result['settings']['repeats']} repeats of "
             f"{result['settings']['warmup']} warm-up batches followed by "
             f"{result['settings']['iterations']} timed batches. CUDA is synchronized before timing and before "
             "stopping the wall clock. Each batch is processed serially with no overlap. Statistics pool the timed "
             "batches, with per-repeat summaries and raw durations in latency.json. p95 is the empirical 95th percentile, "
             "not an uncertainty interval. No slow samples are removed.", "",
             "- **GPU-resident forward:** preprocessed input is already on the GPU. Includes Python dispatch, "
             "autocast context, model forward to logits and final synchronization. Excludes loading, preprocessing, "
             "host/device transfer, softmax and output transfer. This is synchronized wall latency, not kernel-only CUDA-event time.",
             "- **JPEG to probabilities:** includes path resolution, local file read and SHA-256 verification, JPEG decode, "
             "grayscale conversion, bilinear resize to 224x224, checkpoint scaling, three-channel repetition, serial CPU "
             "batch stacking, blocking GPU transfer, forward, FP32 softmax and transfer of both probabilities to CPU. "
             "No worker processes or pinned-memory prefetching. File reads may hit the OS cache, so this is a warmed local "
             "pipeline, not cold disk latency. No model loading, startup, plotting, metric calculation or network service overhead.",
             "- Both use eval() and inference_mode(), contiguous input tensors, deterministic algorithms, "
             "cuDNN benchmarking disabled and TF32 disabled. No torch.compile or CUDA graphs. Original checkpoint "
             "AMP is benchmarked alongside common FP32. FP32 timings are a comparison, not a replacement evaluation "
             "or a change to frozen confidence thresholds.", "",
             "Synchronized timing follows [PyTorch 2.9 CUDA guidance](https://docs.pytorch.org/docs/2.9/notes/cuda.html#asynchronous-execution).", "",
             "## Checkpoint identities", "",
             "| Model | Evaluation weights | Selected epoch | Checkpoint SHA-256 |",
             "|---|---|---:|---|"]
    for model in result["models"]:
        lines.append(f"| {model['model']} | {model['evaluation_weights']} | {model['selected_epoch']} | "
                     f"`{model['checkpoint_sha256']}` |")
    lines += ["", "EMA checkpoints are timed using their stored evaluation weights, not the online training weights. "
              "EMA averaging is a training operation and is not performed during inference.", "",
             "## Results", "",
             "| Model | Precision | Batch | Scope | Median batch ms | p95 batch ms | Mean ms/image | Images/s |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for model in result["models"]:
        for measurement in model["measurements"]:
            s = measurement["summary"]
            scope = "Forward" if measurement["scope"] == "gpu_resident_forward" else "JPEG to probabilities"
            lines.append(f"| {model['model']} ({model['evaluation_weights']}) | {measurement['precision']} | {s['batch_size']} | {scope} | "
                         f"{s['median_batch_ms']:.3f} | {s['p95_batch_ms']:.3f} | "
                         f"{s['amortized_ms_per_image']:.3f} | {s['images_per_second']:.1f} |")
    lines += ["", "Mean ms/image divides mean batch latency by batch size. At batch 16 it is amortized processing "
              "cost, not the latency experienced by one image waiting for a batch. Images/s = total images divided by "
              "total timed batch seconds, without overlap. Throughput excludes the untimed checks and warm-up. "
              "Use batch-one latency for single-image response time.", "",
              "## Checks and limits", "",
              "Both checkpoint identities and original validation records were verified before benchmarking. "
              "The actual model parameter counts match the saved counts. Every timed output has two finite values "
              "per image. JPEG-pipeline probabilities match probabilities from the preloaded input at the same precision "
              "in an untimed check. Model parameter/buffer hashes and checkpoint file hashes are unchanged after timing.", "",
              "This measures these implementations on this Windows GPU session. Model settings and normalization differ. "
              "Recorded AMP uses FP16 for CNN and BF16 for ConvNeXt, so the common FP32 rows provide a precision-matched "
              "comparison. Configuration order is fixed and no exclusive-GPU lock was acquired. Background load, "
              "clock changes, thermals, caching and Python overhead can affect measurements. GPU snapshots before and "
              "after each model are recorded, not continuous monitoring. Peak training memory and parameter-count "
              "measurements remain the previously recorded values.", "",
              "## Reproduce", "", "Run from the repository root with a fresh output directory:", "", "```powershell",
              result["reproduce_command"], "```", "",
              "Default batch sizes are 1 and 16, with 128 sampled slices, 20 warm-up batches, 100 timed batches "
              "and 3 repeats per setting. See latency.json for the exact executed arguments, timestamps, device/driver "
              "snapshots, source hashes, sample manifest identities, checkpoint provenance and raw timings."]
    (output / "latency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import torch
    import torchvision
    import PIL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-run", type=Path, required=True)
    parser.add_argument("--convnext-run", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    if (min(args.batch_sizes + [args.samples, args.warmup, args.iterations, args.repeats, args.threads]) < 1
            or max(args.batch_sizes) > args.samples or len(set(args.batch_sizes)) != len(args.batch_sizes)
            or any(args.samples % b for b in args.batch_sizes)):
        parser.error("Need positive counts, distinct batch sizes and samples divisible by every batch size")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable, GPU latency cannot be measured")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    # PyTorch 2.9 API: avoid deprecated allow_tf32 setters and their warnings.
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    device = torch.device("cuda:0")
    splits, audit = load_splits(args.splits)
    if args.samples > len(splits["val"]):
        parser.error("More samples requested than validation images")
    indices = np.random.default_rng(args.seed).choice(len(splits["val"]), size=args.samples, replace=False)
    rows = [splits["val"][int(i)] for i in indices]
    start_time = time.perf_counter()
    result = {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "test_evaluated": False, "training_performed": False,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                              "pillow": PIL.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
                              "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(device),
                              "gpu_total_gib": torch.cuda.get_device_properties(device).total_memory / 2**30},
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "execution": {"eval_mode": True, "inference_mode": True, "deterministic_algorithms": True,
                            "cudnn_benchmark": False, "tf32": False, "compiled": False, "workers": 0,
                            "timing": "perf_counter_ns, synchronize CUDA before and after every batch"},
              "source_sha256": {name: file_hash(source_path(name)) for name in
                                ("benchmark_latency.py", "modules.py", "dataset.py", "train.py", "compare_validation.py")},
              "split_sha256": audit["split_sha256"],
              "validation_subset": {"slices": len(rows), "patients": len({r['patient_id'] for r in rows}),
                                    "nc_slices": sum(r['label'] == 0 for r in rows),
                                    "ad_slices": sum(r['label'] == 1 for r in rows),
                                    "paths": [r['path'] for r in rows]}, "models": []}
    args.output.mkdir(parents=True, exist_ok=False)
    result["reproduce_command"] = reproduction_command(result["settings"])
    (args.output / "latency.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for run, name in ((args.cnn_run, "small_cnn"), (args.convnext_run, "convnext_tiny")):
        verified, _ = verify_run(run, run.with_name(run.name + "_validation"), name, splits["val"], audit)
        checkpoint = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
        config = checkpoint["config"]
        weight_kind = checkpoint_weight_kind(checkpoint)
        scaling = config.get("input_scaling", "unit")
        if config["preprocessing"] != PREPROCESSING_BY_SCALING[scaling]:
            raise ValueError("Unknown checkpoint preprocessing")
        model = build_model(name, **config.get("model_kwargs", {"drop_path_rate": 0.0}))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval().to(device)
        if sum(p.numel() for p in model.parameters()) != config["parameter_count"]:
            raise ValueError("Parameter count mismatch")
        del checkpoint
        original_state = state_hash(model)
        dataset = ADNIDataset(args.data_root, rows, input_scaling=scaling, augmentation="none")
        # Untimed preload verifies file hashes and warms the OS file cache.
        cpu_pool = torch.stack([dataset[i][0] for i in range(len(dataset))])
        gpu_pool = cpu_pool.to(device)
        recorded_precision = config.get("amp_dtype", "float16") if config["effective_amp"] else "float32"
        if recorded_precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Selected precision BF16 unsupported on this GPU")
        entry = {"model": name, "checkpoint_sha256": verified["checkpoint_sha256"],
                 "checkpoint": str((run / "best.pt").resolve()), "evaluation_weights": weight_kind,
                 "ema_decay": config.get("ema_decay") if weight_kind == "ema" else None,
                 "model_kwargs": config.get("model_kwargs", {"drop_path_rate": 0.0}),
                 "selected_epoch": verified["selected_epoch"], "recorded_precision": recorded_precision,
                 "preprocessing": config["preprocessing"], "verification": verified["verified"],
                 "gpu_before": gpu_snapshot(), "measurements": []}
        print(f"Benchmarking {name}, {weight_kind} weights, epoch {entry['selected_epoch']}, "
              f"recorded precision {recorded_precision}", flush=True)
        for batch_size in args.batch_sizes:
            batches = list(gpu_pool.split(batch_size))
            for precision in dict.fromkeys((recorded_precision, "float32")):
                with torch.inference_mode(), torch.autocast("cuda", enabled=precision != "float32", dtype=getattr(torch, precision)):
                    a = model(batches[0]).float().softmax(dim=1).cpu()
                    b = model(torch.stack([dataset[i][0] for i in range(batch_size)]).to(device)).float().softmax(dim=1).cpu()
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                for scope in ("gpu_resident_forward", "jpeg_to_probabilities"):
                    entry["measurements"].append(measure(model, dataset, batches, batch_size, precision, scope, device,
                                                         args.warmup, args.iterations, args.repeats))
        if original_state != state_hash(model) or file_hash(run / "best.pt") != verified["checkpoint_sha256"]:
            raise RuntimeError("Model/checkpoint changed during inference")
        entry.update(gpu_after=gpu_snapshot(), model_state_unchanged=True, checkpoint_file_unchanged=True,
                     preloaded_and_pipeline_probabilities_equal=True)
        result["models"].append(entry)
        (args.output / "latency.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        del model, gpu_pool, cpu_pool, batches, a, b
        torch.cuda.empty_cache()
    result.update(state="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                  benchmark_process_seconds=time.perf_counter() - start_time)
    (args.output / "latency.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(args.output, result)
    print(f"Completed. Report: {args.output / 'latency_report.md'}", flush=True)


if __name__ == "__main__":
    main()
