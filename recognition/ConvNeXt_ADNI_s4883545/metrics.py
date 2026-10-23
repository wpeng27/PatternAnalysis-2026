"""Binary metrics: NC=0, AD=1, AD AUROC always uses P(AD).

AI-assisted implementation checked against hand-calculated examples.
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
"""

from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

CLASS_TO_INDEX = {"NC": 0, "AD": 1}
SELECTION_RULE = "Minimum validation slice cross-entropy loss, earliest epoch on exact ties"


def classification_metrics(labels, p_ad, unit="slice", loss=None):
    """One vote per evaluation unit. Tied probabilities predict NC, like argmax."""
    y = np.asarray(labels)
    probabilities = np.asarray(p_ad, dtype=np.float64)
    if y.ndim != 1 or probabilities.shape != y.shape or not len(y):
        raise ValueError("Nonempty, aligned one-dimensional labels and probabilities are required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("Labels must be NC=0/AD=1 and probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("P(AD) must lie in [0, 1]")
    predicted = (probabilities > 0.5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], zero_division=0)
    # Patient probability aggregation has no logits, so compute its own binary NLL.
    if loss is None:
        clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
        loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
    if not np.isfinite(loss):
        raise ValueError("Loss must be finite")
    return {
        "evaluation_unit": unit, "class_to_index": CLASS_TO_INDEX,
        "samples": len(y), "loss": float(loss), "accuracy": float(np.mean(y == predicted)),
        "macro_f1": float(np.mean(f1)),
        "ad_auroc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "auroc_score": "P(AD), positive class AD=1",
        "auroc_undefined_reason": None if len(np.unique(y)) == 2 else "Only one ground-truth class",
        "per_class": {name: {"precision": float(precision[index]), "recall": float(recall[index]),
                             "f1": float(f1[index]), "support": int(support[index])}
                      for name, index in CLASS_TO_INDEX.items()},
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted, class order [NC, AD]",
        "decision_rule": "AD if P(AD)>0.5, otherwise NC",
    }


def aggregate_patients(rows, labels, p_ad):
    """Mean P(AD) only after verifying identity/label alignment and consistency."""
    if len(rows) != len(labels) or len(rows) != len(p_ad):
        raise ValueError("Patient rows and slice predictions are not aligned")
    groups = defaultdict(list)
    paths = set()
    for row, label, probability in zip(rows, labels, p_ad):
        if row["path"] in paths:
            raise ValueError("Repeated image row in patient aggregation")
        paths.add(row["path"])
        if int(row["label"]) != int(label) or CLASS_TO_INDEX[row["class_name"]] != int(label):
            raise ValueError("Prediction labels do not match verified manifest labels")
        groups[row["patient_id"]].append((int(label), float(probability)))
    patient_labels, patient_probabilities = [], []
    for patient in sorted(groups):
        values = groups[patient]
        labels_for_patient = {label for label, _ in values}
        if len(labels_for_patient) != 1:
            raise ValueError(f"Conflicting labels for patient {patient}")
        patient_labels.append(values[0][0])
        patient_probabilities.append(sum(p for _, p in values) / len(values))
    result = classification_metrics(patient_labels, patient_probabilities, unit="patient")
    result["aggregation"] = "Arithmetic mean of all available slice P(AD), one vote per patient"
    result["label_verification"] = "Consistent audited folder labels and staff-confirmed filename IDs, not clinical verification"
    return result
