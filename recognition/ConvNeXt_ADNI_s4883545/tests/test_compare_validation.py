"""Guard against misleading comparisons from misaligned exported predictions."""

import copy
import unittest

from scripts.compare_validation import align_predictions, paired_counts, require_metrics_match
from metrics import classification_metrics


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"path": "train/NC/101_1.jpeg", "patient_id": "101", "label": 0},
                     {"path": "train/AD/202_1.jpeg", "patient_id": "202", "label": 1}]
        self.predictions = [dict(row, label=str(row["label"]), p_ad=str(p),
                                 predicted=str(int(p > 0.5)), confidence=str(max(p, 1-p)))
                            for row, p in zip(self.rows, [0.5, 0.8])]

    def test_reordered_exports_align_by_case_and_ties_predict_nc(self):
        self.assertEqual(align_predictions(self.rows, self.predictions[::-1]), [0.5, 0.8])

    def test_missing_duplicate_and_relabelled_cases_are_rejected(self):
        cases = [self.predictions[:1], self.predictions + self.predictions[:1]]
        for field, value in (("path", "train/AD/203_1.jpeg"), ("patient_id", "203"), ("label", "0")):
            changed = copy.deepcopy(self.predictions)
            changed[1][field] = value
            cases.append(changed)
        for predictions in cases:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                align_predictions(self.rows, predictions)

    def test_invalid_probabilities_predictions_or_confidence_are_rejected(self):
        for field, value in (("p_ad", "nan"), ("p_ad", "1.1"), ("predicted", "1"),
                             ("confidence", "0.9"), ("confidence", "nan")):
            changed = copy.deepcopy(self.predictions)
            changed[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                align_predictions(self.rows, changed)

    def test_metric_value_and_definition_drift_are_rejected(self):
        expected = classification_metrics([0, 1], [0.5, 0.8])
        require_metrics_match(expected, dict(expected, seconds=1.0))
        for key, value in (("accuracy", 0.5), ("auroc_score", "P(NC)"), ("evaluation_unit", "patient")):
            changed = copy.deepcopy(expected)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_metrics_match(expected, changed)

    def test_paired_correctness_counts_are_hand_checked(self):
        counts = paired_counts([0, 0, 1, 1], [0.1, 0.2, 0.2, 0.2], [0.1, 0.8, 0.8, 0.2])
        self.assertEqual(counts, {"both_correct": 1, "cnn_only_correct": 1,
                                  "convnext_only_correct": 1, "both_wrong": 1})
        with self.assertRaises(ValueError):
            paired_counts([0, 1], [0.1], [0.2])


if __name__ == "__main__":
    unittest.main()
