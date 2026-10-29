"""Controlled validation sensitivity checks, without training or test inference.

Perturbations are diagnostics, not certified label-preserving transformations.
Checkpoint identities and the existing review thresholds remain fixed.
AI-assisted implementation.
"""

# Support direct script execution from any working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paths import source_path

import argparse
import csv
from datetime import datetime, timezone
import json
import time

import numpy as np

# Sets deterministic CUDA environment before CUDA initialization.
from train import seed_everything, PREPROCESSING_BY_SCALING
from dataset import ADNIDataset, load_splits
from modules import build_model, checkpoint_weight_kind
from scripts.compare_validation import verify_run, file_hash
from scripts.benchmark_latency import state_hash
from metrics import classification_metrics

import torch
import torch.nn.functional as F


CONDITIONS = (
    "identity", "brightness_0.9", "brightness_1.1", "contrast_0.9", "contrast_1.1",
    "gamma_0.9", "gamma_1.1", "left_4", "right_4", "up_4", "down_4", "crop_4",
)
DEFINITIONS = {
    "input": "Float32 grayscale [0,1] after existing Pillow bilinear resize to 224x224, before model scaling",
    "identity": "No change",
    "brightness": "clip(factor*x,0,1), factors 0.9 and 1.1",
    "contrast": "For x>0, clip(mu+factor*(x-mu),0,1), mu=per-image mean of nonzero pixels. Keep original zero pixels zero. This is not a brain mask.",
    "gamma": "x**gamma for gamma 0.9 and 1.1, no clipping or requantization",
    "translation": "Integer shift 4 pixels left/right/up/down, zero fill, no wraparound or interpolation",
    "crop": "Remove four pixels at every edge of resized image, then bilinear 216x216 to 224x224, align_corners=False, antialias=True",
    "normalization": "After perturbation use each checkpoint's saved scaling, then repeat grayscale into three equal channels",
    "precision": "Full validation uses saved evaluation AMP precision and batch size. Repeat five selected cases in FP32 and native AMP with the same five-case batch.",
    "review_candidates": "Existing EMA confidence rule alone, OR model disagreement, OR slice/patient-mean disagreement, OR either disagreement. No fitted thresholds.",
    "interpretation": "One-factor-at-a-time exploratory sensitivity analysis after case inspection. Changes show sensitivity to these interventions, not original error causation, anatomical mechanisms or proven generalization gains.",
}


def perturb(x, condition):
    """Return a new tensor, preserving finite [0,1] single-channel input."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if x.ndim != 4 or x.shape[1] != 1 or not x.is_floating_point():
        raise ValueError("Expected floating N,1,H,W grayscale tensor")
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Input must be finite and in [0,1]")
    if condition == "identity":
        return x.clone()
    if condition.startswith("brightness_"):
        return (x * float(condition.split("_")[1])).clamp(0, 1)
    if condition.startswith("contrast_"):
        mask = x > 0
        mean = x.sum(dim=(-2, -1), keepdim=True) / mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
        value = (mean + float(condition.split("_")[1]) * (x - mean)).clamp(0, 1)
        return torch.where(mask, value, x)
    if condition.startswith("gamma_"):
        return x.pow(float(condition.split("_")[1]))
    if min(x.shape[-2:]) <= 8:
        raise ValueError("Spatial perturbations require height and width >8")
    if condition == "crop_4":
        return F.interpolate(x[..., 4:-4, 4:-4], size=x.shape[-2:], mode="bilinear",
                             align_corners=False, antialias=True)
    y = torch.zeros_like(x)
    if condition == "left_4":
        y[..., :-4] = x[..., 4:]
    elif condition == "right_4":
        y[..., 4:] = x[..., :-4]
    elif condition == "up_4":
        y[..., :-4, :] = x[..., 4:, :]
    elif condition == "down_4":
        y[..., 4:, :] = x[..., :-4, :]
    return y


def compare_probabilities(labels, base, candidate, threshold):
    y, base, candidate = np.asarray(labels), np.asarray(base), np.asarray(candidate)
    if (y.shape != base.shape or base.shape != candidate.shape or y.ndim != 1 or not len(y)
            or not np.isin(y, [0, 1]).all() or not np.isfinite(base).all()
            or not np.isfinite(candidate).all() or np.any((base < 0) | (base > 1))
            or np.any((candidate < 0) | (candidate > 1)) or not .5 <= threshold <= 1):
        raise ValueError("Need aligned binary labels, valid probabilities and confidence threshold")
    before, after = base > .5, candidate > .5
    good, now_good = before == y, after == y
    accepted = np.maximum(candidate, 1 - candidate) >= threshold
    delta = np.abs(candidate - base)
    metrics = classification_metrics(y.tolist(), candidate.tolist())
    metrics.pop("loss")  # No logits-based CE is collected in this diagnostic.
    return {"metrics": metrics, "class_flips": int((before != after).sum()),
            "correct_to_wrong": int((good & ~now_good).sum()),
            "wrong_to_correct": int((~good & now_good).sum()),
            "mean_absolute_p_ad_change": float(delta.mean()),
            "max_absolute_p_ad_change": float(delta.max()),
            "mean_change_original_correct": float(delta[good].mean()) if good.any() else None,
            "mean_change_original_wrong": float(delta[~good].mean()) if (~good).any() else None,
            "accepted": int(accepted.sum()), "accepted_errors": int((accepted & ~now_good).sum()),
            "accepted_accuracy": float(now_good[accepted].mean()) if accepted.any() else None,
            "referral_action_changes": int((accepted != (np.maximum(base, 1 - base) >= threshold)).sum())}


def routing_results(rows, ema, cnn, threshold):
    y = np.array([row["label"] for row in rows])
    ema, cnn = np.asarray(ema), np.asarray(cnn)
    patient = np.array([r["patient_id"] for r in rows])
    patient_pred = np.empty(len(rows), dtype=bool)
    for pid in set(patient):
        mask = patient == pid
        if len(set(y[mask])) != 1:
            raise ValueError("Patient labels disagree")
        patient_pred[mask] = ema[mask].mean() > .5
    confidence_flag = np.maximum(ema, 1 - ema) < threshold
    disagreement = (ema > .5) != (cnn > .5)
    consistency = (ema > .5) != patient_pred
    wrong = (ema > .5) != y
    output = {}
    for name, referred in {
            "current": confidence_flag, "or_model_disagreement": confidence_flag | disagreement,
            "or_patient_disagreement": confidence_flag | consistency,
            "or_either_disagreement": confidence_flag | disagreement | consistency}.items():
        output[name] = {"referred": int(referred.sum()), "review_workload": float(referred.mean()),
                        "referred_errors": int((wrong & referred).sum()),
                        "referred_correct": int((~wrong & referred).sum()),
                        "fraction_errors_referred": float((wrong & referred).sum() / wrong.sum()) if wrong.any() else None,
                        "coverage": float((~referred).mean()),
                        "accepted_accuracy": float((~wrong)[~referred].mean()) if (~referred).any() else None,
                        "accepted_false_negatives": int((wrong & ~referred & (y == 1)).sum()),
                        "accepted_false_positives": int((wrong & ~referred & (y == 0)).sum())}
    both_confident_wrong = wrong & ((cnn > .5) != y) & (np.maximum(ema, 1-ema) >= .95) & (np.maximum(cnn, 1-cnn) >= .95)
    output["both_wrong_confidence_at_least_095"] = {"slices": int(both_confident_wrong.sum()),
                                                   "patients": len(set(patient[both_confident_wrong])),
                                                   "false_negatives": int((both_confident_wrong & (y == 1)).sum())}
    return output


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@torch.inference_mode()
def infer(model, pool, condition, scaling, batch, dtype):
    probabilities = []
    for offset in range(0, len(pool), batch):
        x = perturb(pool[offset:offset + batch], condition)
        if scaling == "centered":
            x = x.mul(2).sub(1)
        x = x.repeat(1, 3, 1, 1).to("cuda")
        with torch.autocast("cuda", enabled=dtype != "float32", dtype=getattr(torch, dtype)):
            logits = model(x)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite output logits")
        probabilities.extend(logits.float().softmax(1)[:, 1].cpu().tolist())
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True, help="Verified EMA/CNN confidence analysis.json")
    parser.add_argument("--cases", type=Path, required=True, help="Failure-case evidence.json")
    parser.add_argument("--output", type=Path, required=True, help="Fresh artifact directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a fresh directory")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required for this full validation diagnostic")
    torch.set_num_threads(4)
    seed_everything(42)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    splits, audit = load_splits(Path(analysis["inputs"]["splits"]))
    rows = splits["val"]
    indices = [{r["path"]: i for i, r in enumerate(rows)}[c["path"]] for c in cases]
    labels = [r["label"] for r in rows]
    verified, original = {}, {}
    for entry in analysis["models"]:
        name = entry["model"]
        run = Path(entry["verification"]["run"])
        v, p = verify_run(run, run.with_name(run.name + "_validation"), name, rows, audit)
        for key in ("checkpoint_sha256", "prediction_sha256"):
            if v[key] != entry["verification"][key]:
                raise ValueError("Confidence analysis provenance mismatch")
        verified[name], original[name] = v, p
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "conditions": CONDITIONS,
            "definitions": DEFINITIONS, "slices": len(rows), "patients": len({r['patient_id'] for r in rows}),
            "case_paths": [c["path"] for c in cases], "test_evaluated": False, "training_performed": False,
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "seed": 42,
            "analysis_sha256": file_hash(args.analysis), "cases_sha256": file_hash(args.cases),
            "source_sha256": {f: file_hash(source_path(f)) for f in
                              (Path(__file__).name, "dataset.py", "modules.py", "metrics.py", "compare_validation.py")},
            "split_sha256": audit["split_sha256"],
            "models": {k: {"checkpoint_sha256": v["checkpoint_sha256"], "run": v["run"]} for k, v in verified.items()},
            "thresholds": {m["model"]: m["selected"]["threshold"] for m in analysis["models"]},
            "native_identity_check": "Same predictions and max P(AD) difference <=1e-6 against saved full-validation export, otherwise abort"}
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "status.json", {"state": "running", "phase": "load_images"})
    def log(message):
        print(message, flush=True)
        with (args.output / "run.log").open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    log(f"Plan saved. Loading {len(rows)} audited validation images.")
    dataset = ADNIDataset(Path(analysis["inputs"]["data_root"]), rows, "unit", "none")
    pool = torch.empty((len(rows), 1, 224, 224), dtype=torch.float32)
    for i in range(len(rows)):
        image, label = dataset[i]
        assert label == labels[i] and torch.equal(image[0], image[1]) and torch.equal(image[1], image[2])
        pool[i] = image[:1]
    result = {"plan": plan, "models": {}, "routing": routing_results(rows, original["convnext_tiny"],
              original["small_cnn"], plan["thresholds"]["convnext_tiny"])}
    for name, v in verified.items():
        run = Path(v["run"])
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        kind = checkpoint_weight_kind(checkpoint)
        config = checkpoint["config"]
        scaling = config.get("input_scaling", "unit")
        assert config["preprocessing"] == PREPROCESSING_BY_SCALING[scaling]
        model = build_model(name, **config.get("model_kwargs", {}))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval().requires_grad_(False).cuda()
        del checkpoint
        old_state = state_hash(model)
        batch = v["settings"]["evaluation_batch_size"]
        dtype = config.get("amp_dtype", "float16") if config["effective_amp"] else "float32"
        summary = {"evaluation_weights": kind, "batch_size": batch, "precision": dtype,
                   "conditions": {}, "cases": [], "selected_epoch": v["selected_epoch"]}
        with (args.output / f"{name}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["condition", "path", "patient_id", "label", "p_ad", "predicted", "confidence"])
            for condition in CONDITIONS:
                tick = time.perf_counter()
                probabilities = infer(model, pool, condition, scaling, batch, dtype)
                if condition == "identity":
                    difference = float(np.max(np.abs(np.asarray(probabilities) - original[name])))
                    if difference > 1e-6 or not np.array_equal(np.asarray(probabilities) > .5, np.asarray(original[name]) > .5):
                        raise RuntimeError(f"Original evaluation not reproduced for {name}: {difference}")
                    summary["identity_max_probability_difference"] = difference
                values = compare_probabilities(labels, original[name], probabilities, plan["thresholds"][name])
                values["seconds"] = time.perf_counter() - tick
                summary["conditions"][condition] = values
                for row, p in zip(rows, probabilities):
                    writer.writerow([condition, row["path"], row["patient_id"], row["label"], p, int(p > .5), max(p, 1-p)])
                f.flush()
                log(f"{name} {condition}: accuracy={values['metrics']['accuracy']:.6f}, flips={values['class_flips']}, corrected={values['wrong_to_correct']}, harmed={values['correct_to_wrong']}, seconds={values['seconds']:.2f}")
                write_json(args.output / "status.json", {"state": "running", "model": name, "completed_condition": condition})
        for condition in CONDITIONS:
            native = infer(model, pool[indices], condition, scaling, len(indices), dtype)
            fp32 = infer(model, pool[indices], condition, scaling, len(indices), "float32")
            summary["cases"].append({"condition": condition, "native_p_ad": native, "fp32_p_ad": fp32})
        assert state_hash(model) == old_state and file_hash(run / "best.pt") == v["checkpoint_sha256"]
        summary["state_and_checkpoint_unchanged"] = True
        result["models"][name] = summary
        write_json(args.output / "results.partial.json", result)
        del model
        torch.cuda.empty_cache()
    assert file_hash(args.analysis) == plan["analysis_sha256"] and file_hash(args.cases) == plan["cases_sha256"]
    result.update(state="completed", seconds=time.perf_counter() - start, completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "results.json", result)
    write_json(args.output / "status.json", {k: result[k] for k in ("state", "seconds", "completed_utc")})
    log(f"Completed all conditions in {result['seconds']:.2f} seconds. No training or test evaluation.")


if __name__ == "__main__":
    main()
