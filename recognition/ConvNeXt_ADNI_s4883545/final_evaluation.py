"""Freeze validation-selected artifacts, then evaluate the test set without tuning.

Separate freeze/evaluate commands preserve a hashed pre-test specification and
private snapshots of checkpoints, manifests, policies, configuration and code.
Only the existing train.py evaluation interface is invoked, never training.
AI-assisted implementation with tests for freeze integrity and metric separation.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

if __package__:
    from .compare_validation import verify_run, file_hash, read_json, align_predictions, require_metrics_match, metric_table
    from .dataset import load_splits, CLASS_TO_INDEX
    from .metrics import classification_metrics, aggregate_patients
    from .analyze_confidence import threshold_summary
else:
    from compare_validation import verify_run, file_hash, read_json, align_predictions, require_metrics_match, metric_table
    from dataset import load_splits, CLASS_TO_INDEX
    from metrics import classification_metrics, aggregate_patients
    from analyze_confidence import threshold_summary


SOURCE_FILES = ("final_evaluation.py", "train.py", "dataset.py", "modules.py", "metrics.py",
                "training_state.py", "compare_validation.py", "analyze_confidence.py")


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_freeze(root):
    """Reject changes to the freeze record or any artifact it identifies."""
    root = root.resolve()
    digest = (root / "freeze.sha256").read_text(encoding="ascii").strip()
    if file_hash(root / "freeze.json") != digest:
        raise ValueError("Freeze specification changed")
    record = read_json(root / "freeze.json")
    if record["state"] != "frozen_before_test" or record["test_evaluated_at_freeze"]:
        raise ValueError("Expected a pre-test freeze")
    for name, expected in record["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    return record, digest


def freeze(args):
    root = args.output.resolve()
    if root.exists():
        raise ValueError("Choose a new freeze directory")
    if root.is_relative_to(args.data_root.resolve()):
        raise ValueError("Freeze must be outside the dataset")
    splits, audit = load_splits(args.splits)
    policies = read_json(args.policies)
    if (policies["test_evaluated"] or policies["evaluation_unit"] != "slice"
            or policies["class_to_index"] != CLASS_TO_INDEX or policies["split_sha256"] != audit["split_sha256"]):
        raise ValueError("Policies must come from the same validation-only experiment")
    policy_by_model = {p["model"]: p for p in policies["models"]}
    if len(policy_by_model) != len(policies["models"]) or set(policy_by_model) != {"small_cnn", "convnext_tiny"}:
        raise ValueError("Expected one policy per model")
    records = []
    for name, run in (("small_cnn", args.cnn_run), ("convnext_tiny", args.convnext_run)):
        verified, probabilities = verify_run(run, run.with_name(run.name + "_validation"), name, splits["val"], audit)
        policy = policy_by_model[name]
        if (policy["status"] != "selected" or policy["checkpoint_sha256"] != verified["checkpoint_sha256"]
                or policy["prediction_sha256"] != verified["prediction_sha256"]
                or policy["settings"] != json.loads(json.dumps(verified["settings"]))):
            raise ValueError("Policy does not match the verified validation checkpoint")
        recomputed = threshold_summary([r["label"] for r in splits["val"]], probabilities, policy["threshold"])
        require_metrics_match(recomputed, policy["validation"])
        config = read_json(run / "config.json")
        records.append({"model": name, "original_run": str(run.resolve()),
                        "checkpoint": f"frozen/{name}/best.pt", "checkpoint_sha256": verified["checkpoint_sha256"],
                        "configuration": f"frozen/{name}/config.json", "epoch": verified["selected_epoch"],
                        "threshold": policy["threshold"], "validation": verified["metrics"],
                        "evaluation": {"batch_size": verified["settings"]["evaluation_batch_size"],
                                       "amp": config["effective_amp"], "amp_dtype": config.get("amp_dtype", "float16"),
                                       "preprocessing": config["preprocessing"], "input_scaling": config.get("input_scaling", "unit"),
                                       "seed": 42, "device": "cuda", "workers": 0, "threads": 4}})
    root.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_FILES:
        destination = root / "code" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(Path(__file__).with_name(name), destination)
    for name in ("audit.json", "train.csv", "val.csv", "test.csv"):
        destination = root / "frozen/splits" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.splits / name, destination)
    shutil.copy2(args.policies, root / "frozen/selected_policies.json")
    for entry in records:
        target = root / entry["checkpoint"]
        target.parent.mkdir(parents=True, exist_ok=False)
        shutil.copy2(Path(entry["original_run"]) / "best.pt", target)
        shutil.copy2(Path(entry["original_run"]) / "config.json", root / entry["configuration"])
        if file_hash(target) != entry["checkpoint_sha256"]:
            raise ValueError("Copied checkpoint digest mismatch")
    record = {"state": "frozen_before_test", "created_utc": datetime.now(timezone.utc).isoformat(),
              "test_evaluated_at_freeze": False, "data_root": str(args.data_root.resolve()),
              "class_to_index": CLASS_TO_INDEX, "primary_unit": "slice",
              "prediction_rule": "AD if P(AD)>0.5, otherwise NC, including ties",
              "review_rule": "Accept max(P(AD),1-P(AD))>=frozen threshold, otherwise refer",
              "checkpoint_selection": "Minimum validation slice cross-entropy, earliest exact ties",
              "minimum_unfiltered_test_accuracy": 0.80,
              "review_target_accuracy": policies["target_accuracy"], "review_minimum_coverage": policies["minimum_coverage"],
              "threshold_selection": policies["selection_rule"], "split_sha256": audit["split_sha256"],
              "test_cases": len(splits["test"]), "test_patients": len({r['patient_id'] for r in splits['test']}),
              "restriction": "No model, preprocessing, checkpoint, threshold or configuration tuning on test results",
              "models": records,
              "files_sha256": {str(p.relative_to(root)).replace('\\', '/'): file_hash(p)
                               for p in sorted(root.rglob('*')) if p.is_file()}}
    write_new_json(root / "freeze.json", record)
    (root / "freeze.sha256").write_text(file_hash(root / "freeze.json") + "\n", encoding="ascii")
    verify_freeze(root)
    print(json.dumps({"state": record["state"], "freeze_sha256": file_hash(root / "freeze.json"),
                      "test_cases": record["test_cases"], "models": [{k: m[k] for k in
                      ("model", "epoch", "checkpoint_sha256", "threshold", "evaluation")} for m in records]}, indent=2))


def summarize_test(rows, probabilities, metrics, threshold, minimum=0.80):
    labels = [r["label"] for r in rows]
    full = classification_metrics(labels, probabilities, loss=metrics["loss"])
    require_metrics_match(full, metrics)
    patient = aggregate_patients(rows, labels, probabilities)
    require_metrics_match(patient, metrics["patient"])
    review = threshold_summary(labels, probabilities, threshold)
    accepted_indices = [i for i, p in enumerate(probabilities) if max(p, 1-p) >= threshold]
    accepted_metrics = (classification_metrics([labels[i] for i in accepted_indices],
                                               [probabilities[i] for i in accepted_indices])
                        if accepted_indices else None)
    return {"unfiltered_slice": full, "unfiltered_patient": patient, "review": review,
            "accepted_slice": accepted_metrics,
            "accepted_loss_note": "Probability-based binary NLL, not independently recovered logits cross-entropy",
            "minimum_accuracy": minimum, "meets_minimum_unfiltered_slice_accuracy": full["accuracy"] >= minimum}


def write_report(root, result, frozen):
    cnn, convnext = result["models"]
    lines = ["# Frozen final test evaluation", "",
             f"Freeze created {frozen['created_utc']}. Evaluation started {result['started_utc']}. "
             f"Completed {result['finished_utc']}. Timestamps are UTC.", "",
             f"Freeze SHA-256: `{result['freeze_sha256']}`. Checkpoint, configuration, policy, manifest and source "
             "snapshots were hashed before test evaluation and rechecked afterwards. No retraining or tuning was performed.", "",
             f"Same {frozen['test_cases']} slices from {frozen['test_patients']} patients. NC=0, AD=1. "
             "Primary unit is slice, with all test cases included. Predict AD if P(AD)>0.5, ties NC. AD AUROC uses P(AD). "
             "Macro-F1 is the unweighted mean of NC and AD F1, with zero_division=0. Loss is sample-weighted logits cross-entropy.", "",
             "## Primary unfiltered test results", "", metric_table(cnn["unfiltered_slice"], convnext["unfiltered_slice"]), "",
             "Confusion matrices: rows=true, columns=predicted, order [NC,AD].", "",
             f"- CNN: `{cnn['unfiltered_slice']['confusion_matrix']}`",
             f"- ConvNeXt: `{convnext['unfiltered_slice']['confusion_matrix']}`", "",
             f"**ConvNeXt minimum 0.80 unfiltered test accuracy: {'MET' if convnext['meets_minimum_unfiltered_slice_accuracy'] else 'NOT MET'}.** "
             f"Measured {100*convnext['unfiltered_slice']['accuracy']:.4f}%. This decision uses all slices, "
             "not patient aggregation or confidence filtering.", "",
             "## Secondary patient aggregation", "",
             "Arithmetic mean of all slice P(AD) within each patient, then the same >0.5 decision rule. "
             "One vote per patient. Labels and identities were checked for consistency against the audited manifest, "
             "not independently clinically verified.", "",
             metric_table(cnn["unfiltered_patient"], convnext["unfiltered_patient"]), "",
             "## Unchanged validation-selected referral rules", "",
             "Acceptance uses confidence >= threshold, with equal confidence retained together. Referred cases are "
             "excluded from accepted accuracy, not assumed corrected by a human. No threshold search was run on test predictions.", "",
             "| Model | Frozen threshold | Accepted / referred | Coverage | Accepted accuracy | Accepted errors | Proposed 90% / 50% target |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for m in result["models"]:
        r = m["review"]
        accuracy = "undefined" if r["accepted_accuracy"] is None else f"{100*r['accepted_accuracy']:.2f}%"
        lines.append(f"| {m['model']} | {r['threshold']!r} | {r['accepted']} / {r['referred']} | "
                     f"{100*r['coverage']:.2f}% | {accuracy} | {r['accepted_errors']} | "
                     f"{'Met' if m['meets_proposed_review_target'] else 'Not met'} |")
    for m in result["models"]:
        r = m["review"]
        lines += ["", f"{m['model']}: accepted NC coverage {100*r['per_class']['NC']['coverage']:.2f}%, "
                  f"AD coverage {100*r['per_class']['AD']['coverage']:.2f}%. "
                  f"Accepted errors: {r['per_class']['NC']['accepted_errors']} NC-to-AD false positives and "
                  f"{r['per_class']['AD']['accepted_errors']} AD-to-NC false negatives. "
                  f"Referred errors: {r['referred_errors']}."]
    lines += ["", "## Validation and test remain separate", "",
              "| Model | Selected epoch | Validation slice accuracy | Test slice accuracy | Evaluation precision / batch |",
              "|---|---:|---:|---:|---|"]
    for original, m in zip(frozen["models"], result["models"]):
        e = original["evaluation"]
        lines.append(f"| {m['model']} | {original['epoch']} | {100*original['validation']['accuracy']:.2f}% | "
                     f"{100*m['unfiltered_slice']['accuracy']:.2f}% | {e['amp_dtype']} / {e['batch_size']} |")
    lines += ["", "Checkpoint selection and thresholds were fixed using validation. CNN uses unit input scaling and FP16, "
              "ConvNeXt uses centered scaling and BF16. Evaluation augmentation is disabled. The precision and batch "
              "settings preserve the validation protocol, rather than the separate common-FP32 latency experiment.", "",
              "Classification metrics were independently recomputed from aligned test exports. Patient aggregation "
              "and fixed-threshold acceptance were also recomputed. Saved slice loss comes from inference logits, "
              "not a reconstruction from exported probabilities. Raw JPEG hashes are verified by the dataset loader "
              "during evaluation. Frozen and original checkpoint hashes remain unchanged.", "",
              "The results describe one supplied test split. Correlated slices are not independent patients, and test "
              "performance is not a clinical deployment guarantee. Training configurations differ, so this is not an "
              "isolated architecture ablation. The test set is now exposed and must not become a tuning set. Any future "
              "test-guided changes require a new independent evaluation set for an unbiased claim.", "",
              "Full metrics, accepted-case metrics, hashes, commands and patient results are in final_results.json. "
              "Per-slice exports and logs are under test/<model> and logs/. All artifacts, checkpoints and patient "
              "records remain local and ignored by Git. Frozen source code is under code/."]
    (root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(root):
    root = root.resolve()
    if (root / "final_results.json").exists():
        raise ValueError("Final results already exist. Read them instead of repeating test evaluation")
    frozen, digest = verify_freeze(root)
    # Run this stage from the archived script so reporting code is also frozen.
    if Path(__file__).resolve() != root / "code/final_evaluation.py":
        subprocess.run([sys.executable, str(root / "code/final_evaluation.py"), "evaluate", "--output", str(root)], check=True)
        return
    splits, audit = load_splits(root / "frozen/splits")
    rows = splits["test"]
    result = {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "freeze_sha256": digest,
              "test_evaluated": True, "retrained": False, "tuned_on_test": False, "models": []}
    (root / "logs").mkdir(exist_ok=True)
    for model in frozen["models"]:
        verify_freeze(root)
        name, settings = model["model"], model["evaluation"]
        output = root / "test" / name
        command = [sys.executable, "-u", str(root / "code/train.py"), "--mode", "evaluate", "--model", name,
                   "--checkpoint", str(root / model["checkpoint"]), "--data-root", frozen["data_root"],
                   "--splits", str(root / "frozen/splits"), "--eval-split", "test", "--save-predictions",
                   "--batch-size", str(settings["batch_size"]), "--device", settings["device"],
                   "--workers", "0", "--threads", "4", "--seed", "42", "--amp" if settings["amp"] else "--no-amp",
                   "--amp-dtype", settings["amp_dtype"], "--output", str(output)]
        if not output.exists():
            write_new_json(root / "logs" / f"{name}_launch.json", {
                "started_utc": datetime.now(timezone.utc).isoformat(), "command": command, "freeze_sha256": digest})
            with (root / "logs" / f"{name}.log").open("x", encoding="utf-8") as log:
                print(f"Evaluating frozen {name} on {len(rows)} test slices", flush=True)
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        evaluation = read_json(output / "evaluation.json")
        if (evaluation["split"] != "test" or evaluation["mode"] != "evaluate" or evaluation["retrained"]
                or evaluation["model"] != name or evaluation["checkpoint_sha256"] != model["checkpoint_sha256"]
                or evaluation["checkpoint_epoch"] != model["epoch"] or evaluation["split_sha256"] != audit["split_sha256"]):
            raise ValueError("Test output provenance differs from freeze")
        for key in ("batch_size", "effective_amp", "amp_dtype", "preprocessing", "input_scaling", "seed"):
            expected = settings["amp"] if key == "effective_amp" else settings[key]
            if evaluation[key] != expected:
                raise ValueError(f"Test setting differs from freeze: {key}")
        if evaluation["evaluation_augmentation"] != "none":
            raise ValueError("Test augmentation is not permitted")
        with (output / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            probabilities = align_predictions(rows, list(csv.DictReader(stream)))
        summary = summarize_test(rows, probabilities, evaluation["metrics"], model["threshold"],
                                 frozen["minimum_unfiltered_test_accuracy"])
        review = summary["review"]
        summary.update(model=name, evaluation_seconds=evaluation["metrics"]["seconds"], command=command,
                       checkpoint_sha256=model["checkpoint_sha256"],
                       evaluation_sha256=file_hash(output / "evaluation.json"), prediction_sha256=file_hash(output / "predictions.csv"),
                       meets_proposed_review_target=(review["accepted_accuracy"] is not None
                            and review["accepted_accuracy"] >= frozen["review_target_accuracy"]
                            and review["coverage"] >= frozen["review_minimum_coverage"]))
        result["models"].append(summary)
        print(f"{name}: unfiltered test accuracy={summary['unfiltered_slice']['accuracy']:.6f}", flush=True)
    verify_freeze(root)
    for model in frozen["models"]:
        if file_hash(Path(model["original_run"]) / "best.pt") != model["checkpoint_sha256"]:
            raise ValueError("Original checkpoint changed")
    result.update(state="completed_verified_frozen_test_evaluation", finished_utc=datetime.now(timezone.utc).isoformat(),
                  freeze_unchanged=True, original_checkpoints_unchanged=True)
    write_new_json(root / "final_results.json", result)
    write_report(root, result, frozen)
    print(f"Completed frozen evaluation. Report: {root / 'final_report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser("freeze", help="Snapshot and hash settings before any test inference")
    for name in ("cnn-run", "convnext-run", "policies", "splits", "data-root", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    run = sub.add_parser("evaluate", help="Evaluate the existing freeze, never change settings")
    run.add_argument("--output", type=Path, required=True, help="Existing freeze directory")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
