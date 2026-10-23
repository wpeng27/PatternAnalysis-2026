"""Summarise measured throughput or a completed baseline without reading test images.

The budget projection uses measured epochs and is explicitly an estimate.
Final reports require a completed run and separate best-checkpoint validation.
"""

import argparse
import json
from pathlib import Path
import statistics

if __package__:
    from .dataset import sha256
    from .modules import build_model
    from .training_state import write_json
else:
    from dataset import sha256
    from modules import build_model
    from training_state import write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def completed_run(run):
    status = read_json(run / "status.json")
    if status["state"] != "completed":
        raise ValueError("Run is not complete. Check status.json and training.log before reporting results.")
    history, config = read_json(run / "history.json"), read_json(run / "config.json")
    if not history or history[-1]["epoch"] != status["last_completed_epoch"]:
        raise ValueError("Completed status and history disagree")
    return history, config, status


def budget_report(run, max_epochs):
    history, config, status = completed_run(run)
    if len(history) < 3 or status["start_epoch"] != 1:
        raise ValueError("Use a completed trial of at least three epochs starting at epoch one")
    if max_epochs < len(history):
        raise ValueError("Projected epoch limit cannot be smaller than the completed trial")
    times = [r["epoch_seconds"] for r in history]
    warm_mean = statistics.mean(times[1:])
    warm_max = max(times[1:])
    remaining = max_epochs - len(history)
    # Actual trial wall time includes startup and checkpoint overhead. Continuation
    # adds two startups: a one-epoch resume demonstration and the full continuation.
    # 25% headroom on remaining epoch time is a planning allowance, not a guarantee.
    estimate = status["process_seconds"] + 2 * status["process_startup_seconds"] + remaining * warm_mean
    conservative = status["process_seconds"] + 2 * status["process_startup_seconds"] + remaining * warm_max * 1.25
    report = {
        "kind": "Measured trial with estimated continuation budget", "trial_run": str(run.resolve()),
        "gpu": config["device_name"], "batch_size": config["batch_size"], "precision": config["precision"],
        "workers": config["workers"], "epoch_seconds_including_validation": times,
        "process_startup_seconds_measured": status["process_startup_seconds"],
        "first_epoch_extra_seconds_vs_warm_mean": max(0, times[0] - warm_mean),
        "warm_epoch_mean_seconds": warm_mean, "warm_epoch_max_seconds": warm_max,
        "trial_process_seconds_measured": status["process_seconds"],
        "peak_allocated_gib_measured": max(r["peak_training_allocated_bytes"] or 0 for r in history) / 2**30,
        "peak_reserved_gib_measured": max(r["peak_training_reserved_bytes"] or 0 for r in history) / 2**30,
        "bounded_total_epochs": max_epochs, "remaining_epochs_at_most": remaining,
        "planned_continuation_process_starts": 2,
        "early_stopping_patience": config["early_stopping_patience"],
        "estimated_total_seconds_including_trial": estimate,
        "conservative_estimated_total_seconds_including_trial": conservative,
        "projection_caveat": "Estimates, not guarantees. Cache, startup and other GPU work can change runtime. Early stopping may shorten it.",
        "test_evaluated": False,
    }
    write_json(run / "throughput_budget.json", report)
    return report


def segment_lineage(run):
    """Sum actual segment runtimes without counting inherited history twice."""
    segments, seen = [], set()
    while True:
        current = run.resolve()
        if current in seen:
            raise ValueError("Cycle in run lineage")
        seen.add(current)
        _, config, status = completed_run(current)
        segments.append({"run": str(current), "process_seconds": status["process_seconds"],
                         "start_epoch": status["start_epoch"], "last_completed_epoch": status["last_completed_epoch"]})
        if "resumed_from" not in config:
            return list(reversed(segments))
        run = Path(config["resumed_from"]["path"]).parent


def final_report(run, evaluation_path):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history, config, status = completed_run(run)
    evaluation = read_json(evaluation_path)
    best_path = run / "best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    selected = min(history, key=lambda record: record["val"]["loss"])
    if config["early_stopping_min_delta"] != 0:
        raise ValueError("This summary expects strict minimum-loss selection with min_delta=0")
    if best["epoch"] != selected["epoch"] or last["selection_state"]["best_epoch"] != selected["epoch"]:
        raise ValueError("Best checkpoint does not follow the predefined selection rule")
    if (evaluation["split"] != "val" or evaluation["retrained"]
            or evaluation["checkpoint_sha256"] != sha256(best_path.read_bytes())
            or evaluation["checkpoint_epoch"] != best["epoch"]):
        raise ValueError("Validation result is not from this unchanged best checkpoint")
    if abs(evaluation["metrics"]["loss"] - selected["val"]["loss"]) > 1e-6:
        raise ValueError("Reloaded validation loss does not reproduce the selected epoch")
    model = build_model(best["model_name"], **best["config"].get("model_kwargs", {}))
    model.load_state_dict(best["model_state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite model state")
    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, (("loss", "Slice cross-entropy"), ("accuracy", "Slice accuracy"),
                                           ("macro_f1", "Slice macro-F1"), ("ad_auroc", "Slice AD AUROC"))):
        axis.plot(epochs, [r["train"][key] for r in history], label="Train (online)", marker=".")
        axis.plot(epochs, [r["val"][key] for r in history], label="Validation", marker=".")
        axis.axvline(best["epoch"], color="gray", linestyle="--", linewidth=1, label="Selected epoch")
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.2)
        if key != "loss":
            axis.set_ylim(0, 1.02)
        axis.legend(fontsize=8)
    figure.suptitle(f"{best['model_name']} | selected epoch {best['epoch']} by validation loss")
    figure.savefig(run / "learning_curves.png", dpi=160)
    plt.close(figure)
    segments = segment_lineage(run)
    summary = {
        "state": "completed_and_best_checkpoint_reloaded", "stop_reason": status["stop_reason"],
        "epochs_completed": len(history), "best_epoch": best["epoch"], "selection_rule": config["selection_rule"],
        "checkpoint_sha256": sha256(best_path.read_bytes()), "class_to_index": config["class_to_index"],
        "parameter_count": config["parameter_count"], "gpu": config["device_name"],
        "batch_size": config["batch_size"], "precision": config["precision"],
        "validation": evaluation["metrics"], "test_evaluated": False,
        "epoch_seconds_including_validation": [r["epoch_seconds"] for r in history],
        "epoch_timing_definition": config["timing_definition"], "segments_measured": segments,
        "total_measured_training_process_seconds": sum(segment["process_seconds"] for segment in segments),
        "best_reload_validation_seconds": evaluation["metrics"]["seconds"],
        "peak_training_allocated_gib": max(r["peak_training_allocated_bytes"] or 0 for r in history) / 2**30,
        "peak_training_reserved_gib": max(r["peak_training_reserved_bytes"] or 0 for r in history) / 2**30,
        "memory_definition": config["memory_definition"],
        "notes": ["Training curves are online metrics, not an evaluation of the final weights on train.",
                  "Patient labels are checked for manifest consistency, not independently clinically verified.",
                  "Validation results are used for model selection and are not held-out test estimates."],
    }
    write_json(run / "baseline_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--budget-epochs", type=int)
    parser.add_argument("--validation", type=Path, help="evaluation.json from reloaded best checkpoint")
    args = parser.parse_args()
    if (args.budget_epochs is None) == (args.validation is None):
        parser.error("Choose either --budget-epochs or --validation")
    result = budget_report(args.run, args.budget_epochs) if args.budget_epochs is not None else final_report(args.run, args.validation)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
