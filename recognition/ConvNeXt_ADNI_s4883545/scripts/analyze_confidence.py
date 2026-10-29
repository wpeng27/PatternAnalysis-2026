"""Validation-only confidence analysis using verified checkpoint prediction exports.

Confidence = max(P(AD), 1-P(AD)), not a calibrated probability of correctness.
Calibration terminology follows Guo et al. (ICML 2017):
https://proceedings.mlr.press/v70/guo17a.html
No temperature scaling, inference, training or test evaluation is performed.
AI-assisted implementation, verified with hand-calculated and boundary tests.
"""

# Support direct script execution from any working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paths import source_path

import argparse
from collections import defaultdict
import csv
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.compare_validation import verify_run, file_hash
from dataset import load_splits


SELECTION_RULE = ("Maximize validation slice coverage subject to accepted accuracy >= target "
                  "and coverage >= minimum. Accept confidence >= threshold, include all ties. "
                  "Choose the lowest threshold if coverage ties. No feasible threshold => no policy.")


def arrays(labels, p_ad):
    y, p = np.asarray(labels), np.asarray(p_ad, dtype=np.float64)
    if (y.ndim != 1 or not len(y) or p.shape != y.shape or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Need nonempty aligned binary labels and finite P(AD) in [0,1]")
    predicted = (p > 0.5).astype(int)
    return y, p, np.maximum(p, 1 - p), predicted == y


def coverage_curve(labels, p_ad):
    """Every attainable nonempty threshold set. Never split equal-confidence cases."""
    y, p, confidence, correct = arrays(labels, p_ad)
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    ends = np.r_[np.flatnonzero(sorted_confidence[:-1] != sorted_confidence[1:]), len(y) - 1]
    cumulative_correct = np.cumsum(correct[order])
    points = []
    for end in ends:
        n, good = int(end + 1), int(cumulative_correct[end])
        points.append({"threshold": float(sorted_confidence[end]), "accepted": n,
                       "correct": good, "errors": n - good, "coverage": n / len(y),
                       "accuracy": good / n})
    # The lowest admissible threshold represents accepting all cases, even when
    # the minimum observed confidence is greater than 0.5.
    if points[-1]["threshold"] != 0.5:
        points.append(dict(points[-1], threshold=0.5))
    return points


def select_threshold(curve, target=0.90, minimum_coverage=0.50):
    if not 0 < target <= 1 or not 0 < minimum_coverage <= 1:
        raise ValueError("Accuracy target and minimum coverage must lie in (0,1]")
    feasible = [point for point in curve if point["accuracy"] >= target
                and point["coverage"] >= minimum_coverage]
    return min(feasible, key=lambda point: (-point["coverage"], point["threshold"])) if feasible else None


def threshold_summary(labels, p_ad, threshold):
    y, p, confidence, correct = arrays(labels, p_ad)
    if not np.isfinite(threshold) or not 0.5 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0.5,1]")
    accepted = confidence >= threshold
    predicted = (p > 0.5).astype(int)
    result = {"threshold": float(threshold), "accepted": int(accepted.sum()),
              "referred": int((~accepted).sum()), "coverage": float(accepted.mean()),
              "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
              "referred_accuracy": float(correct[~accepted].mean()) if (~accepted).any() else None,
              "accepted_errors": int((accepted & ~correct).sum()),
              "referred_errors": int((~accepted & ~correct).sum()),
              "fraction_of_all_errors_referred": (float((~accepted & ~correct).sum() / (~correct).sum())
                                                   if (~correct).any() else None),
              "per_class": {}}
    for label, name in ((0, "NC"), (1, "AD")):
        actual = y == label
        true_positive = int((accepted & actual & (predicted == label)).sum())
        predicted_count = int((accepted & (predicted == label)).sum())
        kept = int((actual & accepted).sum())
        result["per_class"][name] = {
            "total": int(actual.sum()), "accepted": kept,
            "coverage": float(kept / actual.sum()) if actual.any() else None,
            "accepted_precision": true_positive / predicted_count if predicted_count else None,
            "accepted_recall": true_positive / kept if kept else None,
            "accepted_errors": int((accepted & actual & ~correct).sum())}
    return result


def calibration_summary(labels, p_ad):
    y, p, confidence, correct = arrays(labels, p_ad)
    # Ten equal-width bins over the attainable binary-confidence interval.
    # Bins are [lower, upper), with the last including confidence == 1.
    edges = np.linspace(0.5, 1.0, 11)
    bin_ids = np.searchsorted(edges[1:-1], confidence, side="right")
    bins = []
    for index in range(10):
        mask = bin_ids == index
        bins.append({"lower": float(edges[index]), "upper": float(edges[index + 1]),
                     "count": int(mask.sum()),
                     "confidence": float(confidence[mask].mean()) if mask.any() else None,
                     "accuracy": float(correct[mask].mean()) if mask.any() else None})
    ece = sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"]) / len(y)
    return {"accuracy": float(correct.mean()), "mean_confidence": float(confidence.mean()),
            "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
            "mean_confidence_wrong": float(confidence[~correct].mean()) if (~correct).any() else None,
            "correctness_auroc": float(roc_auc_score(correct, confidence)) if len(np.unique(correct)) == 2 else None,
            "correctness_auroc_definition": "Positive=correct prediction, score=max(P(AD),1-P(AD)), not AD AUROC",
            "binary_brier": float(np.mean((p - y) ** 2)),
            "brier_definition": "Mean (P(AD)-label)^2, single binary probability, not sum over both classes",
            "ece": float(ece), "ece_definition": "10 equal-width bins over [0.5,1], weighted absolute accuracy-confidence gap",
            "bins": bins}


def clustered_intervals(rows, labels, p_ad, threshold, repeats=2000, seed=42):
    """Resample whole validation patients at a fixed already-selected threshold.

    Descriptive percentile intervals only. They do not correct threshold/model
    selection bias and are not a fresh held-out validation of a 90% guarantee.
    """
    _, _, confidence, correct = arrays(labels, p_ad)
    if len(rows) != len(correct) or repeats < 1:
        raise ValueError("Need aligned patient rows and positive bootstrap repeats")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["patient_id"]].append(i)
    counts = []
    for patient in sorted(groups):
        indices = groups[patient]
        accepted = confidence[indices] >= threshold
        counts.append([len(indices), int(accepted.sum()), int((accepted & correct[indices]).sum())])
    counts = np.array(counts)
    sampled = np.random.default_rng(seed).integers(0, len(counts), size=(repeats, len(counts)))
    totals = counts[sampled].sum(axis=1)
    valid = totals[:, 1] > 0
    accuracy = totals[valid, 2] / totals[valid, 1]
    return {"method": "Patient-cluster percentile bootstrap, fixed threshold, 95% descriptive interval",
            "seed": seed, "repeats": repeats, "patients": len(counts),
            "empty_acceptance_replicates": int((~valid).sum()),
            "accepted_accuracy": np.quantile(accuracy, [0.025, 0.975]).tolist() if len(accuracy) else None,
            "coverage": np.quantile(totals[:, 1] / totals[:, 0], [0.025, 0.975]).tolist(),
            "limitation": "Does not account for model/threshold selection on these same validation patients"}


def ranked_errors(rows, labels, p_ad, selected_threshold):
    y, p, confidence, correct = arrays(labels, p_ad)
    if len(rows) != len(y):
        raise ValueError("Need aligned error identities")
    order = sorted(np.flatnonzero(~correct), key=lambda i: (-confidence[i], rows[i]["path"]))
    errors = []
    for i in order:
        errors.append({"path": rows[i]["path"], "patient_id": rows[i]["patient_id"],
                       "file_sha256": rows[i].get("file_sha256"),
                       "true_class": "AD" if y[i] else "NC", "predicted_class": "AD" if p[i] > 0.5 else "NC",
                       "error_type": "false_negative_AD" if y[i] else "false_positive_AD",
                       "p_ad": float(p[i]), "confidence": float(confidence[i]),
                       "accepted_by_selected_rule": bool(confidence[i] >= selected_threshold)
                       if selected_threshold is not None else None})
    # Avoid presenting adjacent slices from one patient as independent examples.
    distinct, seen = [], set()
    for error in errors:
        if error["patient_id"] not in seen:
            distinct.append(error)
            seen.add(error["patient_id"])
        if len(distinct) == 5:
            break
    return errors, distinct


def save_plots(output, models, target, min_coverage, data_root=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for model, color in zip(models, ("tab:blue", "tab:orange")):
        points = model["curve"]
        axes[0].plot([100*p["coverage"] for p in points], [100*p["accuracy"] for p in points],
                     label=model["name"], color=color)
        policy = model["selected"]
        if policy:
            axes[0].scatter(100*policy["coverage"], 100*policy["accepted_accuracy"], color=color, marker="X", s=100)
        bins = [b for b in model["confidence"]["bins"] if b["count"]]
        axes[1].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-",
                     color=color, label=model["name"])
    axes[0].axhline(target*100, color="gray", linestyle="--", label="Accuracy target")
    axes[0].axvline(min_coverage*100, color="gray", linestyle=":", label="Minimum coverage")
    axes[0].set(xlabel="Accepted validation slices (%)", ylabel="Accepted-case accuracy (%)",
                title="Accuracy versus coverage (X = selected policy)", xlim=(0, 102), ylim=(0, 102))
    axes[1].plot([0.5, 1], [0.5, 1], "k--", label="Confidence = accuracy")
    axes[1].set(xlabel="Mean confidence in bin", ylabel="Observed accuracy in bin",
                title="Reliability: 10 bins over confidence [0.5, 1]", xlim=(0.49, 1.01), ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Validation only | Uncalibrated confidence | Primary unit: slice")
    fig.savefig(output / "accuracy_coverage.png", dpi=180)
    fig.savefig(output / "accuracy_coverage.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, model in zip(axes, models):
        for correct, label, color in ((True, "Correct", "tab:blue"), (False, "Wrong", "tab:red")):
            values = model["confidence_samples"][str(correct)]
            if values:
                ax.hist(values, bins=np.linspace(0.5, 1, 21), weights=np.ones(len(values))/len(values),
                        alpha=0.55, label=f"{label} (n={len(values)})", color=color)
        ax.set(title=model["name"], xlabel="Confidence", ylabel="Fraction within correctness group")
        ax.legend()
    fig.suptitle("Validation confidence distributions, each group normalized separately")
    fig.savefig(output / "confidence_distributions.png", dpi=180)
    plt.close(fig)

    if data_root is not None:
        from PIL import Image
        root = data_root.resolve()
        for model in models:
            examples = model["top_errors_distinct_patients"]
            if not examples:
                continue
            fig, axes = plt.subplots(1, len(examples), figsize=(3*len(examples), 4), squeeze=False,
                                     constrained_layout=True)
            for ax, error in zip(axes[0], examples):
                path = (root / error["path"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Image path escapes data root")
                if file_hash(path) != error["file_sha256"]:
                    raise ValueError("Error image differs from the audited validation file")
                with Image.open(path) as im:
                    ax.imshow(im.convert("L"), cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"True {error['true_class']} / predicted {error['predicted_class']}\n"
                             f"confidence {error['confidence']:.6f}\n"
                             f"patient {error['patient_id']}", fontsize=9)
                ax.axis("off")
            fig.suptitle(f"{model['name']}: highest-confidence validation errors, distinct patients\n"
                         "Folder labels, raw images. Visual appearance does not establish an error's cause.")
            fig.savefig(output / f"{model['model']}_confident_errors.png", dpi=160)
            plt.close(fig)


def write_report(output, result):
    percent = lambda value: "undefined" if value is None else f"{100*value:.2f}%"
    lines = ["# Validation confidence and review thresholds", "",
             f"Primary unit: slice. Same {result['validation_cases']} validation slices from "
             f"{result['validation_patients']} patients. NC=0, AD=1. "
             "Predict AD if P(AD)>0.5, otherwise NC. Confidence=max(P(AD),1-P(AD)). No calibration was fitted.", "",
             "## Checkpoint identity", "",
             "| Model | Selected epoch | Evaluation weights | Checkpoint SHA-256 |",
             "|---|---:|---|---|"]
    for model in result["models"]:
        verified = model["verification"]
        lines.append(f"| {model['name']} | {verified['selected_epoch']} | "
                     f"{verified['settings'].get('evaluation_weights', 'online')} | `{verified['checkpoint_sha256']}` |")
    lines += ["",
             "## Selection rule declared before analysis", "", result["selection_rule"], "",
             f"Target: accepted-case accuracy >= {percent(result['target_accuracy'])}, "
             f"coverage >= {percent(result['minimum_coverage'])}. These are proposed project targets, not task-sheet requirements. "
             "Thresholds range over 0.5 and every distinct observed confidence. Accept confidence >= threshold, "
             "refer confidence < threshold. Exact ties stay together. Use the exact thresholds in selected_policies.json, "
             "not rounded display values. No positive-coverage feasible threshold means no selected policy.", "",
             "Coverage = accepted slices / all validation slices. Accepted accuracy = correct accepted / accepted. "
             "Referred cases are excluded, not assumed to be corrected by a human. This is slice referral, not a validated patient workflow.", "",
             "## Confidence versus correctness", "",
             "| Model | All-case accuracy | Mean confidence, correct | Mean confidence, wrong | Correctness AUROC | Binary Brier | ECE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for model in result["models"]:
        c = model["confidence"]
        auc = "undefined" if c["correctness_auroc"] is None else f"{c['correctness_auroc']:.6f}"
        lines.append(f"| {model['name']} | {percent(c['accuracy'])} | {percent(c['mean_confidence_correct'])} | "
                     f"{percent(c['mean_confidence_wrong'])} | {auc} | {c['binary_brier']:.6f} | {c['ece']:.6f} |")
    lines += ["", "Correctness AUROC ranks correct versus incorrect predictions using confidence. It is distinct from "
              "AD AUROC. Binary Brier = mean (P(AD)-label)^2. ECE uses 10 equal-width bins over [0.5,1], "
              "weighted by count, with empty bins omitted and confidence 1 included in the last bin. "
              "Binning affects ECE. Raw confidence is not a clinical probability of correctness. "
              "Calibration terminology: [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html).", "",
              "![Accuracy versus coverage and reliability](accuracy_coverage.png)", "",
              "Lines connect attainable threshold points for readability. No interpolation is used to select a policy. "
              "Tiny accepted sets can have high accuracy and are constrained by the declared minimum coverage.", "",
              "![Confidence distributions](confidence_distributions.png)", "",
              "## Selected operating points", "",
              "| Model | Exact threshold | Accepted / referred | Coverage | Accepted accuracy | Errors accepted / referred |",
              "|---|---:|---:|---:|---:|---:|"]
    for model in result["models"]:
        s = model["selected"]
        if s:
            lines.append(f"| {model['name']} | {s['threshold']!r} | {s['accepted']} / {s['referred']} | "
                         f"{percent(s['coverage'])} | {percent(s['accepted_accuracy'])} | "
                         f"{s['accepted_errors']} / {s['referred_errors']} |")
        else:
            lines.append(f"| {model['name']} | No feasible policy | — | — | — | — |")
    for model in result["models"]:
        lines += ["", f"### {model['name']}", "",
                  "| Threshold | Coverage | Accepted accuracy | Accepted errors | NC coverage | AD coverage |",
                  "|---|---:|---:|---:|---:|---:|"]
        for s in model["fixed_thresholds"]:
            lines.append(f"| {s['threshold']:.2f} | {percent(s['coverage'])} | {percent(s['accepted_accuracy'])} | "
                         f"{s['accepted_errors']} | {percent(s['per_class']['NC']['coverage'])} | "
                         f"{percent(s['per_class']['AD']['coverage'])} |")
        high = next(s for s in model["fixed_thresholds"] if s["threshold"] == 0.95)
        retained_examples = sum(e["accepted_by_selected_rule"] is True for e in model["top_errors_distinct_patients"])
        lines += ["", f"At confidence >=0.95, {high['accepted_errors']} errors remain: "
                  f"{high['per_class']['NC']['accepted_errors']} NC-to-AD false positives and "
                  f"{high['per_class']['AD']['accepted_errors']} AD-to-NC false negatives. "
                  f"Of the {len(model['top_errors_distinct_patients'])} ranked examples below, "
                  f"{retained_examples} survive the selected policy (zero if no policy exists)."]
        s = model["selected"]
        if s:
            lines += ["", "At the selected threshold:", ""]
            for name, c in s["per_class"].items():
                lines.append(f"- {name}: {c['accepted']}/{c['total']} accepted ({percent(c['coverage'])}), "
                             f"accepted precision {percent(c['accepted_precision'])}, "
                             f"accepted recall {percent(c['accepted_recall'])}, accepted errors {c['accepted_errors']}.")
            ci = model["selected_cluster_intervals"]
            lines += ["", f"Referred-set model accuracy: {percent(s['referred_accuracy'])}. "
                      f"Fraction of all errors referred: {percent(s['fraction_of_all_errors_referred'])}.", "",
                      f"Patient-cluster bootstrap, {ci['repeats']} replicates, seed {ci['seed']}: "
                      f"fixed-threshold accepted-accuracy 95% interval {percent(ci['accepted_accuracy'][0])}–"
                      f"{percent(ci['accepted_accuracy'][1])}, coverage {percent(ci['coverage'][0])}–"
                      f"{percent(ci['coverage'][1])}. These descriptive intervals do not account for threshold or "
                      "model selection on the same validation patients. They are not a guarantee of the target on new patients."]
        lines += ["", "Highest-confidence wrong predictions from five distinct patients (deterministic ranking):", "",
                  "| Case | True | Predicted | Confidence | Accepted by selected rule |", "|---|---|---|---:|---|"]
        for error in model["top_errors_distinct_patients"]:
            lines.append(f"| `{error['path']}` | {error['true_class']} | {error['predicted_class']} | "
                         f"{error['confidence']:.9f} | {error['accepted_by_selected_rule']} |")
        if (output / f"{model['model']}_confident_errors.png").exists():
            lines += ["", f"![{model['name']} confident errors]({model['model']}_confident_errors.png)"]
    lines += ["", "## Interpretation limits and provenance", "",
              "Accepted recall above is conditional on retained true-class cases. It excludes referrals and must not "
              "be reported as overall clinical sensitivity. A threshold of 0.5 accepts everything and means the "
              "aggregate project target does not demand referral, not that all cases are trustworthy. Highly confident "
              "errors can survive high thresholds. Review based only on low confidence cannot catch those errors.", "",
              "Model/checkpoint identity, prediction alignment, metric definitions and all split hashes were verified "
              "using compare_validation.verify_run. The selected checkpoints are unchanged. Their validation results "
              "already influenced training choices. This analysis additionally selects thresholds on the same patients, "
              "so both accuracy and coverage are descriptive, selection-biased validation estimates. No independent "
              "clinical labels, uncertainty calibration, domain-shift test or statistical superiority claim is supplied. "
              "Settings including preprocessing, precision and training budgets differ between the two models.", "",
              "Full curves, reliability-bin counts, high-confidence error counts, class retention, bootstrap settings "
              "and provenance are saved in analysis.json. Ranked error identities stay local in ignored artifacts. "
              "Use selected_policies.json to preserve exact thresholds and checkpoint/prediction hashes. "
              "No test predictions or test metrics were used in this analysis.", "",
              ("These models were previously evaluated on test. This is a new validation-only threshold analysis after "
               "test exposure, not a new pre-test freeze. Earlier test results and referral policies remain separate. "
               "The newly selected policy has not been applied to test here."
               if result["experiment_status"] == "after_test_exposure" else
               "This report does not establish whether any separate test evaluation has occurred. "
               "The policy is selected from validation only."), "",
              "## Reproduce", "", "Run from the repository root with a new output directory:", "", "```powershell",
              result["reproduce_command"],
              "```", "", "Omit --data-root to regenerate the numeric analysis without the error-image panels. "
              "If custom targets are used, pass those original values instead. Numerical reports include exact "
              "targets and source hashes. Checkpoints and error images are local artifacts and must not be committed."]
    (output / "confidence_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-run", type=Path, required=True)
    parser.add_argument("--convnext-run", type=Path, required=True)
    parser.add_argument("--cnn-evaluation", type=Path)
    parser.add_argument("--convnext-evaluation", type=Path)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Optional: render verified validation error images")
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-coverage", type=float, default=0.50)
    parser.add_argument("--experiment-status", choices=("validation_analysis", "after_test_exposure"),
                        default="validation_analysis", help="Report context only, never reads test predictions")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    select_threshold([], args.target_accuracy, args.minimum_coverage)
    splits, audit = load_splits(args.splits)
    rows = splits["val"]
    labels = [row["label"] for row in rows]
    models = []
    for run, evaluation, model_name, title in (
            (args.cnn_run, args.cnn_evaluation, "small_cnn", "Small CNN"),
            (args.convnext_run, args.convnext_evaluation, "convnext_tiny", "ConvNeXt-Tiny")):
        verified, p_ad = verify_run(run, evaluation or run.with_name(run.name + "_validation"),
                                    model_name, rows, audit)
        if verified["settings"].get("evaluation_weights") == "ema":
            title += " (EMA)"
        curve = coverage_curve(labels, p_ad)
        choice = select_threshold(curve, args.target_accuracy, args.minimum_coverage)
        threshold = choice["threshold"] if choice else None
        selected = threshold_summary(labels, p_ad, threshold) if choice else None
        errors, distinct = ranked_errors(rows, labels, p_ad, threshold)
        _, _, confidence, correct = arrays(labels, p_ad)
        models.append({"model": model_name, "name": title, "verification": verified,
                       "confidence": calibration_summary(labels, p_ad), "curve": curve,
                       "selected": selected, "errors": errors, "top_errors_distinct_patients": distinct,
                       "selected_cluster_intervals": clustered_intervals(rows, labels, p_ad, threshold) if choice else None,
                       "fixed_thresholds": [threshold_summary(labels, p_ad, t) for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)],
                       "confidence_samples": {str(good): confidence[correct == good].tolist() for good in (True, False)}})
    result = {"state": "verified_validation_confidence_analysis", "test_evaluated": False,
              "experiment_status": args.experiment_status,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_unit": "slice", "selection_rule": SELECTION_RULE,
              "target_accuracy": args.target_accuracy, "minimum_coverage": args.minimum_coverage,
              "validation_cases": len(rows), "validation_patients": len({r['patient_id'] for r in rows}),
              "split_sha256": audit["split_sha256"], "source_sha256": file_hash(Path(__file__)),
              "verification_code_sha256": file_hash(source_path("compare_validation.py")), "models": models}
    result["inputs"] = {key: str(value.resolve()) if isinstance(value, Path) else value
                        for key, value in vars(args).items()}
    reproduce = [str(Path(__file__).resolve()), "--cnn-run", str(args.cnn_run.resolve()),
                 "--convnext-run", str(args.convnext_run.resolve()), "--splits", str(args.splits.resolve()),
                 "--target-accuracy", str(args.target_accuracy), "--minimum-coverage", str(args.minimum_coverage),
                 "--experiment-status", args.experiment_status, "--output",
                 str(args.output.resolve().with_name(args.output.name + "_recheck"))]
    for option, value in (("--cnn-evaluation", args.cnn_evaluation),
                          ("--convnext-evaluation", args.convnext_evaluation), ("--data-root", args.data_root)):
        if value is not None:
            reproduce.extend([option, str(value.resolve())])
    result["reproduce_command"] = "py -3.13 " + " ".join("'" + item.replace("'", "''") + "'" for item in reproduce)
    args.output.mkdir(parents=True, exist_ok=False)
    save_plots(args.output, models, args.target_accuracy, args.minimum_coverage, args.data_root)
    policies = {k: v for k, v in result.items() if k != "models"}
    policies["class_to_index"] = {"NC": 0, "AD": 1}
    policies["decision_rule"] = "Predict AD if P(AD)>0.5, else NC. Accept max(P(AD),1-P(AD))>=threshold, else refer."
    policies["models"] = []
    for model in models:
        with (args.output / f"{model['model']}_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(model["curve"][0]))
            writer.writeheader()
            writer.writerows(model["curve"])
        (args.output / f"{model['model']}_ranked_errors.json").write_text(
            json.dumps(model.pop("errors"), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        model.pop("confidence_samples")
        policies["models"].append({"model": model["model"], "status": "selected" if model["selected"] else "no_feasible_policy",
                                   "threshold": model["selected"]["threshold"] if model["selected"] else None,
                                   "checkpoint_sha256": model["verification"]["checkpoint_sha256"],
                                   "prediction_sha256": model["verification"]["prediction_sha256"],
                                   "settings": model["verification"]["settings"], "validation": model["selected"]})
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "selected_policies.json").write_text(json.dumps(policies, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(args.output, result)
    print(json.dumps({"output": str(args.output), "models": [{"model": m["model"], "selected": m["selected"],
                      "confidence": {k: v for k, v in m["confidence"].items() if k != "bins"}}
                     for m in models], "test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
