"""Plot completed training histories without training or opening image data.

Writes PNG and SVG metric/precision/recall curves and numeric inspection evidence.
Training metrics are online measurements with changing weights and regularization,
whereas validation is measured once after each epoch. Loss axes use a labelled
log scale to retain large validation spikes without hiding smaller losses.
AI-assisted implementation.
"""

# Support direct script execution from any working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


METRICS = (("loss", "Cross-entropy (log scale)"), ("accuracy", "Slice accuracy"),
           ("macro_f1", "Slice macro-F1"), ("ad_auroc", "Slice AD AUROC: P(AD)"))


def read_run(run):
    def read(name):
        return json.loads((run / name).read_text(encoding="utf-8"))
    history, config, status = read("history.json"), read("config.json"), read("status.json")
    if status["state"] != "completed" or not history:
        raise ValueError(f"Expected a completed, non-empty run: {run}")
    epochs = [r["epoch"] for r in history]
    if epochs != list(range(1, status["last_completed_epoch"] + 1)):
        raise ValueError("History must contain consecutive epochs, including inherited resume history")
    if config["selection_metric"] != "val.loss" or config["early_stopping_min_delta"] != 0:
        raise ValueError("These plots expect strict minimum validation slice loss selection")
    for row in history:
        for split in ("train", "val"):
            for key, _ in METRICS:
                value = row[split][key]
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Missing/non-finite {split}.{key} at epoch {row['epoch']}")
                if (key == "loss" and value <= 0) or (key != "loss" and not 0 <= value <= 1):
                    raise ValueError("Metric outside plotting domain")
            for label in ("NC", "AD"):
                for key in ("precision", "recall"):
                    value = row[split]["per_class"][label][key]
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Invalid per-class metric")
    selected = min(history, key=lambda r: r["val"]["loss"])
    # This command uses saved selection evidence, not an inferred accuracy peak.
    summary = read("baseline_summary.json")
    if not selected["epoch"] == status["best_epoch"] == summary["best_epoch"]:
        raise ValueError("History, completed status and verified summary disagree on selection")
    if summary["state"] != "completed_and_best_checkpoint_reloaded":
        raise ValueError("Generate a verified report_run.py summary first")
    if abs(summary["validation"]["loss"] - selected["val"]["loss"]) > 1e-6:
        raise ValueError("Verified validation result does not match selected history epoch")
    return history, config, selected, summary


def decorate(axis, epochs, selected_epoch, title, probability=True):
    axis.axvline(selected_epoch, color="#666666", linestyle="--", linewidth=1.3,
                 label=f"Selected epoch {selected_epoch}")
    axis.set(title=title, xlabel="Epoch", xlim=(0.5, epochs[-1] + 0.5))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    axis.grid(alpha=0.22, which="major")
    if probability:
        axis.set_ylim(0, 1.02)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(fontsize=9, loc="best")


def save_figure(figure, destination, title):
    figure.suptitle(title, fontsize=16, y=0.98)
    figure.text(0.5, 0.02, "Train: online metrics with changing weights and regularization. "
                "Validation: end-of-epoch, unaugmented.\n"
                "Selection: lowest validation slice cross-entropy, earliest exact tie. No test data.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.075, 1, 0.94), h_pad=2)
    figure.savefig(destination.with_suffix(".png"), dpi=180)
    figure.savefig(destination.with_suffix(".svg"))
    plt.close(figure)


def plot_run(run, output):
    history, config, selected, summary = read_run(run)
    destination = output / run.name
    destination.mkdir()
    epochs = [r["epoch"] for r in history]
    train_label = "Train (online, augmented)" if config.get("augmentation", "none") != "none" else "Train (online)"
    name = {"small_cnn": "Small CNN", "convnext_tiny": "ConvNeXt-Tiny"}[config["model"]]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (key, title) in zip(axes.flat, METRICS):
        for split, label, color in (("train", train_label, "#1675ad"), ("val", "Validation", "#d25b19")):
            axis.plot(epochs, [r[split][key] for r in history], label=label, color=color, marker=".", markersize=6)
        axis.scatter(selected["epoch"], selected["val"][key], color="#d25b19", edgecolor="black", s=55, zorder=4)
        if key == "loss":
            axis.set_yscale("log")
        decorate(axis, epochs, selected["epoch"], title, probability=key != "loss")
    save_figure(figure, destination / "learning_curves", f"{name} | {len(history)} epochs | selected epoch {selected['epoch']}")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, label in enumerate(("NC", "AD")):
        for col, metric in enumerate(("precision", "recall")):
            axis = axes[row, col]
            for split, legend, color in (("train", train_label, "#1675ad"), ("val", "Validation", "#d25b19")):
                axis.plot(epochs, [r[split]["per_class"][label][metric] for r in history],
                          label=legend, color=color, marker=".", markersize=6)
            axis.scatter(selected["epoch"], selected["val"]["per_class"][label][metric],
                         color="#d25b19", edgecolor="black", s=55, zorder=4)
            decorate(axis, epochs, selected["epoch"], f"{label} {metric}")
    save_figure(figure, destination / "per_class_curves", f"{name} | slice precision and recall | NC=0, AD=1")
    peak = max(history, key=lambda r: r["val"]["loss"])
    best_accuracy = max(history, key=lambda r: r["val"]["accuracy"])
    evidence = {
        "run": str(run.resolve()), "model": config["model"], "epochs": epochs,
        "history_sha256": hashlib.sha256((run / "history.json").read_bytes()).hexdigest(),
        "checkpoint_sha256": summary["checkpoint_sha256"], "selection_rule": config["selection_rule"],
        "selected_epoch": selected["epoch"], "selected": selected, "first": history[0], "last": history[-1],
        "maximum_validation_loss_epoch": peak["epoch"], "maximum_validation_loss": peak["val"]["loss"],
        "maximum_validation_accuracy_epoch": best_accuracy["epoch"],
        "maximum_validation_accuracy": best_accuracy["val"]["accuracy"],
        "last_five_validation_loss_range": [min(r["val"]["loss"] for r in history[-5:]),
                                             max(r["val"]["loss"] for r in history[-5:])],
        "selected_online_train_minus_val_accuracy_pp": 100 * (selected["train"]["accuracy"] - selected["val"]["accuracy"]),
        "learning_rate_changes": [{"after_epoch": r["epoch"], "from": r["learning_rate"], "to": r["next_learning_rate"]}
                                  for r in history if r["learning_rate"] != r["next_learning_rate"]],
        "augmentation": config.get("augmentation", "none"), "evaluation_unit": "slice",
        "notes": ["No smoothing or missing-point removal. Loss axes are logarithmic.",
                  "Train/validation gaps are descriptive, not an isolated estimate of overfitting.",
                  "These configurations differ in augmentation, precision and optimization, as well as architecture.",
                  "Selection was checked against previously verified summary/status, without reloading weights or training."]}
    (destination / "curve_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return {"run": run.name, "selected_epoch": selected["epoch"], "output": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True, help="Completed run, repeat for multiple models")
    parser.add_argument("--output", type=Path, required=True, help="New directory for PNG/SVG plots and evidence")
    args = parser.parse_args()
    if len({r.name for r in args.run}) != len(args.run):
        parser.error("Run directory names must be distinct")
    # Validate every source before creating output files.
    for run in args.run:
        read_run(run)
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps([plot_run(run, args.output) for run in args.run], indent=2))


if __name__ == "__main__":
    main()
