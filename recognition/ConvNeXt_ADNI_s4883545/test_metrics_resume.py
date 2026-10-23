"""Hand-calculated metric examples and exact epoch-boundary resume regression."""

from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image
import torch

from dataset import audit_and_split, save_splits
from metrics import aggregate_patients, classification_metrics
from train import parse_args, train
from training_state import update_selection


class MetricTests(unittest.TestCase):
    def test_known_labels_and_ad_auc_direction(self):
        labels, p_ad = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
        result = classification_metrics(labels, p_ad)
        self.assertEqual(result["evaluation_unit"], "slice")
        self.assertEqual(result["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["per_class"]["NC"]["precision"], 2 / 3)
        self.assertEqual(result["per_class"]["NC"]["recall"], 1)
        self.assertEqual(result["per_class"]["AD"]["precision"], 1)
        self.assertEqual(result["per_class"]["AD"]["recall"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], (0.8 + 2 / 3) / 2)
        self.assertEqual(result["ad_auroc"], 0.75)
        self.assertEqual(classification_metrics(labels, [1 - p for p in p_ad])["ad_auroc"], 0.25)
        self.assertAlmostEqual(result["loss"], -sum(math.log(p) for p in (0.9, 0.6, 0.35, 0.8)) / 4)

    def test_patient_aggregation_checks_labels_and_weights_patients_equally(self):
        labels, p_ad = [0, 0, 1], [0.1, 0.9, 0.8]
        rows = [{"path": str(i), "patient_id": patient, "label": label,
                 "class_name": "AD" if label else "NC"}
                for i, (patient, label) in enumerate(zip(["a", "a", "b"], labels))]
        result = aggregate_patients(rows, labels, p_ad)
        self.assertEqual(result["evaluation_unit"], "patient")
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["accuracy"], 1.0)  # Mean P(AD)=0.5 ties to NC.
        rows[1].update(label=1, class_name="AD")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            aggregate_patients(rows, [0, 1, 1], p_ad)

    def test_undefined_auc_and_invalid_probabilities(self):
        self.assertIsNone(classification_metrics([1, 1], [0.7, 0.9])["ad_auroc"])
        for probabilities in ([float("nan")], [1.1], [-0.1]):
            with self.assertRaises(ValueError):
                classification_metrics([0], probabilities)

    def test_selection_ties_and_patience(self):
        state = {"best_loss": None, "best_epoch": None, "bad_epochs": 0, "stopped": False}
        self.assertTrue(update_selection(state, 0.5, 1, patience=2))
        self.assertFalse(update_selection(state, 0.5, 2, patience=2))
        self.assertFalse(state["stopped"])
        self.assertFalse(update_selection(state, 0.6, 3, patience=2))
        self.assertEqual(state["best_epoch"], 1)
        self.assertTrue(state["stopped"])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "data"
        for split in ("train", "test"):
            for class_index, label in enumerate(("AD", "NC")):
                folder = self.root / split / label
                folder.mkdir(parents=True)
                first = 100 + 100 * class_index + (50 if split == "test" else 0)
                for patient in range(first, first + (2 if split == "train" else 1)):
                    rng = random.Random(patient)
                    Image.frombytes("L", (16, 16), bytes(rng.randrange(256) for _ in range(256))).save(folder / f"{patient}_1.jpeg")
        self.splits = self.base / "splits"
        rows, report = audit_and_split(self.root)
        save_splits(self.splits, rows, report)

    def run_training(self, name, epochs, resume=None, input_scaling="unit"):
        options = ["--data-root", str(self.root), "--splits", str(self.splits), "--output", str(self.base / name),
                   "--device", "cpu", "--workers", "0", "--threads", "2", "--batch-size", "2",
                   "--epochs", str(epochs), "--lr-patience", "0", "--no-amp", "--input-scaling", input_scaling]
        if resume is not None:
            options.extend(["--resume", str(resume)])
        with redirect_stdout(io.StringIO()):
            train(parse_args(options))
        return torch.load(self.base / name / "last.pt", map_location="cpu", weights_only=True)

    def test_centered_training_records_preprocessing_and_rejects_wrong_resume(self):
        saved = self.run_training("centered", 1, input_scaling="centered")
        self.assertEqual(saved["config"]["input_scaling"], "centered")
        self.assertIn("2*x-1", saved["config"]["preprocessing"])
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            self.run_training("wrong_scaling", 2, self.base / "centered/last.pt")
        self.assertFalse((self.base / "wrong_scaling").exists())

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_resume_matches_uninterrupted_model_optimizer_scheduler_and_rng(self):
        uninterrupted = self.run_training("continuous", 2)
        first = self.run_training("first", 1)
        self.assertEqual(first["epoch"], 1)
        resumed = self.run_training("resumed", 2, self.base / "first/last.pt")
        self.assertEqual(resumed["epoch"], 2)
        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                    "random_state", "selection_state"):
            self.assert_nested_equal(uninterrupted[key], resumed[key])
        self.assertEqual(uninterrupted["history"][1]["val"]["loss"], resumed["history"][1]["val"]["loss"])
        best = torch.load(self.base / "resumed/best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(best["epoch"], resumed["selection_state"]["best_epoch"])


if __name__ == "__main__":
    unittest.main()
