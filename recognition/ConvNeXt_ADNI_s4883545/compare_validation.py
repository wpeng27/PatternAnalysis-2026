"""Verify and compare selected CNN/ConvNeXt validation results on identical cases.

Reads saved manifests, predictions, histories and local checkpoints only. No
image loading, inference, retraining or test evaluation. Classification metrics
are recomputed from P(AD). Slice cross-entropy is checked against selected history
because rounded exported probabilities cannot reproduce logits-based CE exactly.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

if __package__:
    from .dataset import CLASS_TO_INDEX, load_splits
    from .metrics import aggregate_patients, classification_metrics, SELECTION_RULE
else:
    from dataset import CLASS_TO_INDEX, load_splits
    from metrics import aggregate_patients, classification_metrics, SELECTION_RULE


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def align_predictions(rows, predictions):
    """Reject missing/duplicate/mislabelled cases, then align by audited path."""
    expected = {r["path"]: r for r in rows}
    by_path = {r["path"]: r for r in predictions}
    if len(expected) != len(rows) or len(by_path) != len(predictions):
        raise ValueError("Duplicate case path")
    if set(expected) != set(by_path):
        raise ValueError("Prediction cases differ from the validation manifest")
    probabilities = []
    for row in rows:
        prediction = by_path[row["path"]]
        if prediction["patient_id"] != row["patient_id"] or int(prediction["label"]) != int(row["label"]):
            raise ValueError("Prediction patient/label differs from the manifest")
        probability = float(prediction["p_ad"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid P(AD)")
        if int(prediction["predicted"]) != int(probability > 0.5):
            raise ValueError("Exported prediction does not use AD if P(AD)>0.5")
        confidence = float(prediction["confidence"])
        if not math.isfinite(confidence) or abs(confidence - max(probability, 1 - probability)) > 1e-12:
            raise ValueError("Exported confidence does not match P(AD)")
        probabilities.append(probability)
    return probabilities


def require_metrics_match(expected, recorded):
    """Check complete metric definitions and values, ignoring timing extras."""
    for key, value in expected.items():
        if key not in recorded:
            raise ValueError(f"Recorded metric missing: {key}")
        actual = recorded[key]
        if isinstance(value, dict):
            require_metrics_match(value, actual)
        elif isinstance(value, float):
            if not isinstance(actual, (int, float)) or not math.isfinite(actual) or abs(value - actual) > 1e-12:
                raise ValueError(f"Recomputed metric differs: {key}")
        elif value != actual:
            raise ValueError(f"Metric definition/value differs: {key}")


def verify_run(run, evaluation_dir, expected_model, rows, audit):
    import torch
    if __package__:
        from .modules import checkpoint_weight_kind
    else:
        from modules import checkpoint_weight_kind

    config = read_json(run / "config.json")
    status = read_json(run / "status.json")
    history = read_json(run / "history.json")
    evaluation = read_json(evaluation_dir / "evaluation.json")
    if status["state"] != "completed" or history[-1]["epoch"] != status["last_completed_epoch"]:
        raise ValueError("Run completion/history mismatch")
    if evaluation["split"] != "val" or evaluation["mode"] != "evaluate" or evaluation["retrained"]:
        raise ValueError("Expected checkpoint-only validation")
    if config["model"] != expected_model or evaluation["model"] != expected_model:
        raise ValueError("Model identity mismatch")
    for record in (config, evaluation):
        if record["split_sha256"] != audit["split_sha256"] or record["seed"] != audit["seed"]:
            raise ValueError("Split identity/seed mismatch")
    if config["class_to_index"] != CLASS_TO_INDEX or config["model_settings"]["input_shape"] != [3, 224, 224]:
        raise ValueError("Class mapping/input shape mismatch")
    metric_hash = file_hash(Path(__file__).with_name("metrics.py"))
    if config["source_sha256"]["metrics.py"] != metric_hash:
        raise ValueError("Recorded metric implementation differs from the recomputation code")
    if config["selection_rule"] != SELECTION_RULE or config["early_stopping_min_delta"] != 0:
        raise ValueError("Unexpected checkpoint selection rule")
    selected = min(history, key=lambda r: r["val"]["loss"])
    if not selected["epoch"] == status["best_epoch"] == evaluation["checkpoint_epoch"]:
        raise ValueError("Selected epoch mismatch")
    checkpoint_path = run / "best.pt"
    checkpoint_hash = file_hash(checkpoint_path)
    if evaluation["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Evaluation does not identify the current best checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weight_kind = checkpoint_weight_kind(checkpoint)
    if (evaluation.get("evaluation_weights", "online") != weight_kind
            or config.get("ema", False) != checkpoint["config"].get("ema", False)
            or config.get("validation_weights", "online") != weight_kind
            or (weight_kind == "ema" and config.get("ema_decay") != checkpoint["config"].get("ema_decay"))):
        raise ValueError("Evaluation/checkpoint EMA identity mismatch")
    if checkpoint["model_name"] != expected_model or checkpoint["epoch"] != selected["epoch"]:
        raise ValueError("Checkpoint model/epoch mismatch")
    for key in ("model", "class_to_index", "preprocessing", "seed", "split_sha256", "selection_rule",
                "batch_size", "lr", "weight_decay", "precision", "augmentation", "model_settings"):
        if checkpoint["config"][key] != config[key]:
            raise ValueError(f"Checkpoint/run setting mismatch: {key}")
    with (evaluation_dir / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        probabilities = align_predictions(rows, list(csv.DictReader(stream)))
    labels = [r["label"] for r in rows]
    # CE comes from logits in the original inference. Check it against history,
    # then pass it through while independently recomputing classification metrics.
    if abs(evaluation["metrics"]["loss"] - selected["val"]["loss"]) > 1e-6:
        raise ValueError("Reloaded slice loss does not match selected history")
    metrics = classification_metrics(labels, probabilities, loss=evaluation["metrics"]["loss"])
    require_metrics_match(metrics, evaluation["metrics"])
    require_metrics_match(metrics, selected["val"])
    patient = aggregate_patients(rows, labels, probabilities)
    require_metrics_match(patient, evaluation["metrics"]["patient"])
    require_metrics_match(patient, selected["val"]["patient"])
    if evaluation.get("evaluation_augmentation", "none") != "none":
        raise ValueError("Evaluation augmentation must be disabled")
    if evaluation.get("preprocessing", config["preprocessing"]) != config["preprocessing"]:
        raise ValueError("Evaluation preprocessing mismatch")
    groups = checkpoint["optimizer_state_dict"]["param_groups"]
    verify_optimizer_groups(checkpoint, config, selected["next_learning_rate"])
    group_settings = [{k: v for k, v in group.items() if k != "params"} for group in groups]
    scheduler = checkpoint["scheduler_state_dict"]
    schedule_kind = config.get("lr_schedule", "plateau")
    if checkpoint["config"].get("lr_schedule", "plateau") != schedule_kind:
        raise ValueError("Checkpoint scheduler identity mismatch")
    if schedule_kind == "warmup_cosine":
        if __package__:
            from .training_state import verify_cosine_checkpoint
        else:
            from training_state import verify_cosine_checkpoint
        scheduler_settings = verify_cosine_checkpoint(checkpoint, config)
    else:
        scheduler_settings = {key: scheduler[key] for key in
                              ("factor", "patience", "mode", "threshold", "threshold_mode", "min_lrs")}
    settings = {key: config[key] for key in ("pretrained", "parameter_count", "batch_size", "lr", "weight_decay",
                 "seed", "workers", "threads", "precision", "preprocessing", "augmentation", "model_settings",
                 "early_stopping_patience", "early_stopping_min_delta", "packages", "scheduler")}
    settings.update(total_epochs=len(history), selected_epoch=selected["epoch"],
                    evaluation_weights=weight_kind, ema_settings=config.get("ema_settings"),
                    weight_decay_policy=config.get("weight_decay_policy", "all"),
                    optimizer_parameter_groups=config.get("optimizer_parameter_groups"),
                    model_kwargs=config.get("model_kwargs"),
                    optimizer_config=config.get("optimizer_settings"),
                    selected_checkpoint_optimizer_groups=group_settings,
                    scheduler_settings=scheduler_settings,
                    augmentation_settings=config.get("augmentation_settings", {"name": config["augmentation"]}),
                    evaluation_batch_size=evaluation["batch_size"],
                    evaluation_amp_dtype=evaluation.get("amp_dtype", "not recorded, FP16 documented in run precision"),
                    regularization_settings=config.get("regularization_settings"),
                    metadata_note="Missing legacy structured fields remain null. Saved optimizer groups, scheduler "
                                  "state and architecture string are reported without inventing missing fields.")
    del checkpoint
    result = {"run": str(run.resolve()), "model": expected_model, "selected_epoch": selected["epoch"],
              "metrics": metrics, "patient": patient, "settings": settings,
              "checkpoint_sha256": checkpoint_hash, "metrics_code_sha256": metric_hash,
              "prediction_sha256": file_hash(evaluation_dir / "predictions.csv"),
              "evaluation_sha256": file_hash(evaluation_dir / "evaluation.json"),
              "config_sha256": file_hash(run / "config.json"),
              "verified": {"prediction_cases_match_manifest": True, "class_mapping_and_definitions_match": True,
                           "classification_metrics_recomputed": True, "patient_metrics_recomputed": True,
                           "slice_logit_loss_matches_selected_history": True, "checkpoint_hash_and_settings_match": True}}
    return result, probabilities


def verify_optimizer_groups(checkpoint, config, learning_rate):
    """Check actual decay rates and membership, including legacy single groups."""
    import torch
    if __package__:
        from .modules import build_model, weight_decay_groups
    else:
        from modules import build_model, weight_decay_groups
    policy = config.get("weight_decay_policy", "all")
    if checkpoint["config"].get("weight_decay_policy", "all") != policy:
        raise ValueError("Checkpoint/run weight-decay policy mismatch")
    with torch.random.fork_rng(devices=[]):
        model = build_model(config["model"], **config.get("model_kwargs", {}))
    expected, metadata = weight_decay_groups(model, config["weight_decay"], policy)
    for record in (config, checkpoint["config"]):
        recorded = record.get("optimizer_parameter_groups")
        if (recorded is not None or policy != "all") and recorded != metadata:
            raise ValueError("Optimizer parameter-group metadata mismatch")
    actual = checkpoint["optimizer_state_dict"]["param_groups"]
    ids = [index for group in actual for index in group["params"]]
    if len(actual) != len(expected) or len(ids) != len(set(ids)):
        raise ValueError("Optimizer group count or unique membership mismatch")
    for group, wanted in zip(actual, expected):
        if (not group.get("decoupled_weight_decay") or group["weight_decay"] != wanted["weight_decay"]
                or group["lr"] != learning_rate or len(group["params"]) != len(wanted["params"])
                or (policy != "all" and group.get("param_names") != wanted["param_names"])):
            raise ValueError("Saved AdamW group settings differ from the selected history/configuration")


def paired_counts(labels, cnn_probabilities, convnext_probabilities):
    counts = {"both_correct": 0, "cnn_only_correct": 0, "convnext_only_correct": 0, "both_wrong": 0}
    if not len(labels) == len(cnn_probabilities) == len(convnext_probabilities):
        raise ValueError("Paired predictions must be aligned")
    for label, a, b in zip(labels, cnn_probabilities, convnext_probabilities):
        ac, bc = int(a > 0.5) == label, int(b > 0.5) == label
        key = "both_correct" if ac and bc else "cnn_only_correct" if ac else "convnext_only_correct" if bc else "both_wrong"
        counts[key] += 1
    return counts


def metric_table(cnn, convnext):
    fields = [("Accuracy", lambda m: m["accuracy"], True),
              ("Macro-F1", lambda m: m["macro_f1"], False),
              ("AD AUROC", lambda m: m["ad_auroc"], False),
              ("Loss", lambda m: m["loss"], False)]
    for label in ("NC", "AD"):
        for metric in ("precision", "recall"):
            fields.append((f"{label} {metric}", lambda m, label=label, metric=metric: m["per_class"][label][metric], True))
    lines = ["| Metric | CNN | ConvNeXt |", "|---|---:|---:|"]
    for title, getter, percent in fields:
        values = [f"{100 * getter(m):.2f}%" if percent else f"{getter(m):.6f}" for m in (cnn, convnext)]
        lines.append(f"| {title} | {values[0]} | {values[1]} |")
    return "\n".join(lines)


def write_report(output, result):
    cnn, convnext = result["models"]
    def scaling(model):
        preprocessing = model["settings"]["preprocessing"]
        if "then 2*x-1" in preprocessing:
            return "[-1,1]"
        if "float32 /255, repeat three channels" in preprocessing:
            return "[0,1]"
        return preprocessing
    def regularization(model):
        settings = model["settings"]
        if settings["regularization_settings"] is not None:
            return json.dumps(settings["regularization_settings"])
        values = re.findall(r"Dropout\(p=([0-9.]+)", settings["model_settings"]["architecture"])
        return "Classifier dropout " + ", ".join(values) + " (recorded architecture)" if values else "See recorded architecture"
    shared_scheduler = cnn["settings"]["scheduler_settings"] == convnext["settings"]["scheduler_settings"]
    differences = {key: cnn["metrics"][key] - convnext["metrics"][key]
                   for key in ("accuracy", "macro_f1", "ad_auroc")}
    leader = "CNN" if differences["accuracy"] > 0 else "ConvNeXt" if differences["accuracy"] < 0 else "Neither model"
    all_primary = all(value > 0 for value in differences.values()) and all(
        cnn["metrics"]["per_class"][label][key] > convnext["metrics"]["per_class"][label][key]
        for label in ("NC", "AD") for key in ("precision", "recall"))
    lines = ["# Task 5.3: verified validation comparison", "",
             "Both prediction files were aligned to the same audited validation manifest. "
             "All classification metrics and patient aggregation were recomputed successfully. "
             "This comparison performed no inference, training or test evaluation.", "",
             "## Common evaluation protocol", "",
             f"- Same {result['validation_cases']} slices from {result['validation_patients']} patients. "
             f"Slice counts: {result['slice_class_counts']}. Patient counts: {result['patient_class_counts']}.",
             "- NC=0, AD=1. AD is predicted if P(AD)>0.5, with ties assigned to NC. No confidence filtering.",
             "- Primary unit: slice. Per-class precision/recall use zero_division=0. Macro-F1 is the unweighted mean of NC/AD F1.",
             "- AD AUROC uses continuous P(AD), never predicted labels or P(NC).",
             "- Secondary unit: patient. Average all slice P(AD) per patient, then apply the same decision rule. "
             "Each patient gets one vote after checking consistent labels and filename identity.",
             "- Checkpoint selection: minimum validation slice cross-entropy, earliest exact ties. "
             f"Selected CNN epoch {cnn['selected_epoch']}, ConvNeXt epoch {convnext['selected_epoch']}.",
             "- Slice loss is saved sample-weighted logits-based cross-entropy, verified against selected history. "
             "Patient loss is binary NLL of mean P(AD), clipped to [1e-15,1-1e-15], and was recomputed.",
             "- The metric implementation SHA-256 is identical in both training records and the current recomputation. "
             f"Train/validation/test manifest hashes and seed {cnn['settings']['seed']} match both checkpoints and evaluation records.", "",
             "## Primary slice results", "", metric_table(cnn["metrics"], convnext["metrics"]), "",
             "Confusion matrices use rows=true, columns=predicted, order [NC,AD].", "",
             f"- CNN: `{cnn['metrics']['confusion_matrix']}`",
             f"- ConvNeXt: `{convnext['metrics']['confusion_matrix']}`", "",
             "## Secondary patient results", "", metric_table(cnn["patient"], convnext["patient"]), "",
             f"- CNN confusion matrix: `{cnn['patient']['confusion_matrix']}`",
             f"- ConvNeXt confusion matrix: `{convnext['patient']['confusion_matrix']}`", "",
             "Patient labels are checked for manifest consistency, not independently clinically verified. "
             "Patient accuracy does not replace the primary slice or final-test result.", "",
             "## Settings and comparability", "",
             f"Recorded pretrained flags: CNN={cnn['settings']['pretrained']}, ConvNeXt={convnext['settings']['pretrained']}. "
             f"Both use seed {cnn['settings']['seed']} and input shape [3,224,224]. Validation augmentation is disabled. "
             "Saved optimizer groups confirm decoupled weight decay. Exact preprocessing strings and scheduler states "
             "are included in comparison.json.", "",
             f"Scheduler settings identical: {shared_scheduler}. CNN: `{cnn['settings']['scheduler_settings']}`. "
             f"ConvNeXt: `{convnext['settings']['scheduler_settings']}`.", "",
             "| Setting | CNN | ConvNeXt |", "|---|---|---|",
             f"| Parameters | {cnn['settings']['parameter_count']:,} | {convnext['settings']['parameter_count']:,} |",
             f"| Input scaling | {scaling(cnn)} | {scaling(convnext)} |",
             f"| Training augmentation | {json.dumps(cnn['settings']['augmentation_settings'])} | {json.dumps(convnext['settings']['augmentation_settings'])} |",
             f"| Training batch size | {cnn['settings']['batch_size']} | {convnext['settings']['batch_size']} |",
             f"| Evaluation batch size | {cnn['settings']['evaluation_batch_size']} | {convnext['settings']['evaluation_batch_size']} |",
             f"| Initial learning rate | {cnn['settings']['lr']} | {convnext['settings']['lr']} |",
             f"| Weight decay | {cnn['settings']['weight_decay']} | {convnext['settings']['weight_decay']} |",
             f"| Regularization | {regularization(cnn)} | {regularization(convnext)} |",
             f"| Early-stopping patience | {cnn['settings']['early_stopping_patience']} | {convnext['settings']['early_stopping_patience']} |",
             f"| Precision | {cnn['settings']['precision']} | {convnext['settings']['precision']} |",
             f"| Completed epochs | {cnn['settings']['total_epochs']} | {convnext['settings']['total_epochs']} |", "",
             "ConvNeXt's budget was extended from 20 to 30 epochs after validation inspection. "
             "Its configuration was selected after several validation-guided experiments. "
             "The baseline checkpoint has older metadata: missing structured optimizer/model_kwargs fields are not "
             "fabricated. Saved optimizer groups, scheduler state, precision and architecture string provide the reported evidence.", "",
             "These are comparable evaluation results for two trained configurations, not an isolated architecture ablation. "
             "Input scaling, augmentation, optimization, precision and training budget differ. No statistical-significance "
             "claim is made from this single split and seed. Slices from one patient are correlated.", "",
             "## Paired cases and conclusion", "",
             f"Slice correctness counts: `{result['paired_slice_correctness']}`.",
             f"Patient correctness counts: `{result['paired_patient_correctness']}`.", "",
             f"{leader} leads in validation slice accuracy by {abs(result['cnn_minus_convnext_accuracy_pp']):.2f} percentage points. "
             f"CNN exceeds ConvNeXt in all reported primary classification metrics: {all_primary}. "
             "This comparison supports conclusions about these recorded validation configurations, without establishing "
             "how either will perform on the final test set.", "",
             "## Provenance", "",
             f"Validation manifest SHA-256: `{result['split_sha256']['val']}`.", "",
             "Complete split hashes, case identity digest, checkpoint hashes, prediction-file hashes, recomputed metrics "
             "and checkpoint-derived settings are saved in `comparison.json`. Split hygiene was rechecked from saved "
             "manifest identities and exact-image hashes. Image bytes were not reopened, so this does not constitute a "
             "new raw-data audit or a near-duplicate check. The test set remains unevaluated."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-run", type=Path, required=True)
    parser.add_argument("--convnext-run", type=Path, required=True)
    parser.add_argument("--cnn-evaluation", type=Path)
    parser.add_argument("--convnext-evaluation", type=Path)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    splits, audit = load_splits(args.splits)
    rows = splits["val"]
    results, probabilities = [], []
    for run, evaluation, model in ((args.cnn_run, args.cnn_evaluation, "small_cnn"),
                                  (args.convnext_run, args.convnext_evaluation, "convnext_tiny")):
        evaluation = evaluation or run.with_name(run.name + "_validation")
        result, probability = verify_run(run, evaluation, model, rows, audit)
        results.append(result)
        probabilities.append(probability)
    patient_indices = defaultdict(list)
    for index, row in enumerate(rows):
        patient_indices[row["patient_id"]].append(index)
    patient_ids = sorted(patient_indices)
    patient_labels = [rows[patient_indices[patient][0]]["label"] for patient in patient_ids]
    patient_probabilities = [[sum(probs[i] for i in patient_indices[patient]) / len(patient_indices[patient])
                              for patient in patient_ids] for probs in probabilities]
    identities = [(r["path"], r["patient_id"], r["label"]) for r in rows]
    result = {"state": "verified_validation_comparison", "test_evaluated": False,
              "validation_cases": len(rows), "validation_patients": len(patient_ids),
              "slice_class_counts": dict(Counter(r["class_name"] for r in rows)),
              "patient_class_counts": dict(Counter("AD" if label else "NC" for label in patient_labels)),
              "split_counts": audit["counts"], "split_sha256": audit["split_sha256"],
              "validation_identity_sha256": hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest(),
              "models": results, "paired_slice_correctness": paired_counts([r["label"] for r in rows], *probabilities),
              "paired_patient_correctness": paired_counts(patient_labels, *patient_probabilities),
              "cnn_minus_convnext_accuracy_pp": 100 * (results[0]["metrics"]["accuracy"] - results[1]["metrics"]["accuracy"])}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(args.output, result)
    print(json.dumps({"output": str(args.output), "cases_verified": len(rows), "patients_verified": len(patient_ids),
                      "cnn_minus_convnext_accuracy_pp": result["cnn_minus_convnext_accuracy_pp"],
                      "paired_slice_correctness": result["paired_slice_correctness"]}, indent=2))


if __name__ == "__main__":
    main()
