"""Load an ADNI checkpoint, export predictions and draw an input/prediction grid.

Examples (from this project directory, using local checkpoints and images):
  py -3.13 predict.py --checkpoint artifacts/runs/RUN/best.pt --images IMAGE.jpeg --output artifacts/demo
  py -3.13 predict.py --checkpoint artifacts/runs/RUN/best.pt --data-root ../../ADNI/AD_NC --splits artifacts/splits_seed42 --limit 12 --output artifacts/demo_val

Model, class mapping, normalization and CUDA autocast precision come from the
checkpoint. CPU runs use FP32 and may give slightly different probabilities.
Manifest mode defaults to validation. --split test is an explicit evaluation,
never a fresh holdout after prior test exposure. Individual images have unknown
labels: folder names are not used to manufacture ground truth or metrics.
AI-assisted implementation for Weijie Peng's COMP3710 project.
"""

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .dataset import ADNIDataset, CLASS_TO_INDEX, load_splits, preprocess_image, sha256
    from .metrics import classification_metrics, aggregate_patients
    from .modules import MODEL_NAMES, build_model, checkpoint_weight_kind
    from .train import PREPROCESSING_BY_SCALING, seed_everything
else:
    from dataset import ADNIDataset, CLASS_TO_INDEX, load_splits, preprocess_image, sha256
    from metrics import classification_metrics, aggregate_patients
    from modules import MODEL_NAMES, build_model, checkpoint_weight_kind
    from train import PREPROCESSING_BY_SCALING, seed_everything


def spaced_indices(count, limit):
    """Deterministic, evenly spaced manifest positions, not outcome selection."""
    size = min(count, limit)
    if size == 1:
        return [0]
    return [index * (count - 1) // (size - 1) for index in range(size)]


class ImageFiles(Dataset):
    """Unlabelled image input using exactly the audited loader's transform."""

    def __init__(self, paths, scaling):
        self.paths = [path.resolve(strict=True) for path in paths]
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("Repeated image paths are not allowed")
        self.rows = [{"path": str(path), "patient_id": "", "label": None,
                      "file_sha256": sha256(path.read_bytes())} for path in self.paths]
        self.scaling = scaling

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        raw = self.paths[index].read_bytes()
        if sha256(raw) != self.rows[index]["file_sha256"]:
            raise ValueError("Image changed after prediction input selection")
        return preprocess_image(raw, self.scaling), -1


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as error:
        raise ValueError("Device must be auto, cpu, cuda or cuda:N") from error
    if device.type not in {"cpu", "cuda"} or (device.type == "cpu" and device.index is not None):
        raise ValueError("Device must be auto, cpu, cuda or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is unavailable")
        torch.cuda.set_device(device)
    return device


def load_predictor(path, device):
    """Strict state loading. model_state_dict is the saved evaluation/EMA model."""
    raw = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    config = checkpoint["config"]
    name = checkpoint["model_name"]
    if name not in MODEL_NAMES or config.get("model", name) != name:
        raise ValueError("Unsupported or inconsistent checkpoint model identity")
    scaling = config.get("input_scaling", "unit")
    if (scaling not in PREPROCESSING_BY_SCALING
            or config.get("class_to_index") != CLASS_TO_INDEX
            or config.get("preprocessing") != PREPROCESSING_BY_SCALING[scaling]):
        raise ValueError("Checkpoint class mapping or preprocessing is incompatible")
    kind = checkpoint_weight_kind(checkpoint)
    kwargs = config.get("model_kwargs", {"drop_path_rate": 0.0})
    model = build_model(name, **kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite parameters or buffers")
    model.to(device).eval().requires_grad_(False)
    dtype_name = config.get("amp_dtype", "float16")
    if dtype_name not in {"float16", "bfloat16"}:
        raise ValueError("Unsupported checkpoint AMP dtype")
    amp = bool(config.get("effective_amp", False)) and device.type == "cuda"
    if amp and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Checkpoint requires BF16 CUDA support. Use --device cpu for FP32.")
    identity = {"model": name, "model_kwargs": kwargs, "checkpoint": str(path.resolve()),
                "checkpoint_sha256": sha256(raw), "checkpoint_epoch": checkpoint["epoch"],
                "evaluation_weights": kind, "ema_decay": config.get("ema_decay") if kind == "ema" else None,
                "class_to_index": CLASS_TO_INDEX, "input_scaling": scaling,
                "preprocessing": config["preprocessing"], "evaluation_augmentation": "none",
                "effective_amp": amp, "amp_dtype": dtype_name,
                "effective_precision": dtype_name if amp else "float32"}
    return model, config, identity


def load_policy(path, identity, config):
    if path is None:
        return None
    raw = path.read_bytes()
    policy = json.loads(raw)
    if (policy.get("state") != "verified_validation_confidence_analysis"
            or policy.get("evaluation_unit") != "slice"
            or policy.get("class_to_index") != CLASS_TO_INDEX
            or policy.get("split_sha256") != config.get("split_sha256")):
        raise ValueError("Review policy is incompatible with the checkpoint's validation protocol")
    matches = [entry for entry in policy["models"] if entry["model"] == identity["model"]
               and entry["checkpoint_sha256"] == identity["checkpoint_sha256"]]
    if len(matches) != 1 or matches[0].get("status") != "selected":
        raise ValueError("Review policy does not identify this checkpoint")
    threshold = matches[0]["threshold"]
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0.5 <= threshold <= 1:
        raise ValueError("Review threshold must be finite and in [0.5, 1]")
    return {"path": str(path.resolve()), "sha256": sha256(raw), "threshold": threshold,
            "source_experiment_status": policy.get("experiment_status"),
            "rule": "Accept confidence >= threshold, otherwise refer",
            "note": "Validation-selected research policy, no guarantee for new images or changed precision"}


def prediction_record(row, probability, threshold=None):
    predicted = int(probability > 0.5)
    confidence = max(probability, 1 - probability)
    label = row["label"]
    names = {value: key for key, value in CLASS_TO_INDEX.items()}
    return {"path": row["path"], "patient_id": row["patient_id"],
            "file_sha256": row["file_sha256"], "label": label,
            "true_class": names[label] if label is not None else None,
            "p_nc": 1 - probability, "p_ad": probability,
            "predicted": predicted, "predicted_class": names[predicted],
            "confidence": confidence, "correct": predicted == label if label is not None else None,
            "review_decision": ("accept" if confidence >= threshold else "refer")
            if threshold is not None else "not_applied"}


def save_demo(path, previews, predictions, identity):
    """Display actual preprocessed inputs, with scaling inverted for viewing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = min(4, len(previews))
    rows = math.ceil(len(previews) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(max(7, 4 * columns), 5 * rows + 1), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (index, gray) in zip(axes.flat, sorted(previews.items())):
        result = predictions[index]
        if identity["input_scaling"] == "centered":
            gray = (gray + 1) / 2
        axis.imshow(gray.numpy(), cmap="gray", vmin=0, vmax=1)
        truth = result["true_class"] or "unknown"
        title = (f"{Path(result['path']).name}\nTrue: {truth} | Predicted: {result['predicted_class']}\n"
                 f"P(AD)={result['p_ad']:.6f} | Confidence={result['confidence']:.4f}")
        if result["review_decision"] != "not_applied":
            title += f"\nReview: {result['review_decision']}"
        axis.set_title(title, fontsize=10,
                       color="firebrick" if result["correct"] is False else "black")
    figure.suptitle(f"{identity['model']} | {identity['evaluation_weights']} weights | "
                   f"epoch {identity['checkpoint_epoch']}\n224 x 224 grayscale inputs\n"
                   "Illustrative subset, not an accuracy estimate", fontsize=12)
    # Reserve explicit space for four-line captions even when axis decorations
    # are hidden. tight_layout can otherwise overlap adjacent image captions.
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02,
                           top=1 - 1.7 / (5 * rows + 1), wspace=0.12, hspace=0.45)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def predict(args):
    if args.output.exists():
        raise ValueError("Output already exists. Choose a fresh directory to preserve results.")
    if args.data_root is not None and args.output.resolve().is_relative_to(args.data_root.resolve()):
        raise ValueError("Save predictions outside the dataset")
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    model, config, identity = load_predictor(args.checkpoint, device)
    seed_everything(config.get("seed", 42))
    policy = load_policy(args.review_policy, identity, config)
    audit = None
    if args.splits is not None:
        splits, audit = load_splits(args.splits)
        if audit["split_sha256"] != config.get("split_sha256"):
            raise ValueError("Manifest hashes differ from checkpoint splits")
        all_rows = splits[args.split]
        indices = spaced_indices(len(all_rows), args.limit or len(all_rows))
        rows = [all_rows[index] for index in indices]
        dataset = ADNIDataset(args.data_root, rows, input_scaling=identity["input_scaling"])
        total = len(all_rows)
    else:
        dataset = ImageFiles(args.images, identity["input_scaling"])
        rows, total = dataset.rows, len(dataset)
    if not rows:
        raise ValueError("No images selected")
    batch_size = args.batch_size or config.get("batch_size", 16)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda", drop_last=False)
    display = set(spaced_indices(len(rows), args.max_display))
    predictions, previews, total_loss = [], {}, 0.0
    threshold = policy["threshold"] if policy else None
    with torch.inference_mode():
        for step, (images, labels) in enumerate(loader, start=1):
            offset = len(predictions)
            for index in range(len(images)):
                if offset + index in display:
                    previews[offset + index] = images[index, 0].clone()
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=identity["effective_amp"],
                                dtype=getattr(torch, identity["amp_dtype"])):
                logits = model(images)
                loss = (torch.nn.functional.cross_entropy(logits, labels.to(device))
                        if audit is not None else None)
            if logits.shape != (len(images), 2) or not torch.isfinite(logits).all():
                raise ValueError("Model must produce two finite logits per image")
            if loss is not None:
                total_loss += loss.item() * len(images)
            probabilities = logits.float().softmax(dim=1)[:, CLASS_TO_INDEX["AD"]].cpu().tolist()
            predictions.extend(prediction_record(row, prob, threshold)
                               for row, prob in zip(rows[offset:offset + len(images)], probabilities))
            if step == 1 or step % 50 == 0 or step == len(loader):
                print(f"Prediction batch {step}/{len(loader)}: {len(predictions)}/{len(rows)} images", flush=True)
    metrics = None
    if audit is not None:
        labels = [row["label"] for row in rows]
        probabilities = [row["p_ad"] for row in predictions]
        metrics = classification_metrics(labels, probabilities, loss=total_loss / len(rows))
        # Partial patient slices must not be presented as full patient evaluation.
        if len(rows) == total:
            metrics["patient"] = aggregate_patients(rows, labels, probabilities)
    result = {"state": "completed", **identity, "device": str(device), "batch_size": batch_size,
              "python": platform.python_version(), "torch": str(torch.__version__),
              "input_mode": "audited_manifest" if audit else "unlabelled_images",
              "split": args.split if audit else None, "split_sha256": audit["split_sha256"] if audit else None,
              "data_root": str(args.data_root.resolve()) if audit else None,
              "manifest_directory": str(args.splits.resolve()) if audit else None,
              "available_images": total, "predicted_images": len(rows),
              "selection": "all" if total == len(rows) else "evenly spaced manifest positions",
              "metrics_scope": ("complete_split" if total == len(rows) else "demonstration_subset") if audit else None,
              "metrics": metrics, "review_policy": policy,
              "referred_images": sum(row["review_decision"] == "refer" for row in predictions) if policy else None,
              "decision_rule": "AD if P(AD)>0.5, otherwise NC (including ties)",
              "evaluation_unit": "slice", "retrained": False,
              "test_inference_performed": args.split == "test" if audit else None,
              "evaluation_note": "Demonstration only, does not supersede frozen or exploratory experiment records",
              "demo_indices": sorted(display), "outputs": ["predictions.csv", "prediction.json", "demo.png"]}
    if sha256(args.checkpoint.read_bytes()) != identity["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during prediction")
    args.output.mkdir(parents=True, exist_ok=False)
    save_demo(args.output / "demo.png", previews, predictions, identity)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (args.output / "prediction.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(predictions)} predictions and demo.png to {args.output}", flush=True)
    if metrics:
        print(f"{result['metrics_scope']} slice accuracy: {metrics['accuracy']:.4%}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, nargs="+", help="Individual unlabelled image files")
    source.add_argument("--splits", type=Path, help="Audited split directory containing audit.json and CSVs")
    parser.add_argument("--data-root", type=Path, help="Required with --splits")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None,
                        help="Manifest partition (default val). Test must be explicitly selected.")
    parser.add_argument("--limit", type=int, help="Limit manifest input to evenly spaced positions for a small demo")
    parser.add_argument("--review-policy", type=Path, help="Existing selected_policies.json matching this checkpoint")
    parser.add_argument("--batch-size", type=int, help="Defaults to saved training batch size")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-display", type=int, default=12, help="Number of evenly spaced images in demo.png (1-64)")
    parser.add_argument("--output", "--output-dir", type=Path, required=True, help="Fresh output directory")
    args = parser.parse_args(argv)
    if args.splits is not None and args.data_root is None:
        parser.error("--splits requires --data-root")
    if args.images is not None and any(value is not None for value in (args.data_root, args.split, args.limit)):
        parser.error("--data-root, --split and --limit require --splits")
    for name in ("batch_size", "threads", "limit", "max_display"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_display > 64:
        parser.error("--max-display must be at most 64")
    args.split = args.split or "val"
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        predict(args)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
