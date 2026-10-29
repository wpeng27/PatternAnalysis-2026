"""Hand-calculated metric examples and exact epoch-boundary resume regression."""

from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from dataset import audit_and_split, save_splits
from metrics import aggregate_patients, classification_metrics
from modules import build_model
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

    def run_training(self, name, epochs, resume=None, input_scaling="unit", augmentation="none",
                     weight_decay_policy="all", ema_decay=None, schedule_epochs=None, warmup_epochs=1):
        options = ["--data-root", str(self.root), "--splits", str(self.splits), "--output", str(self.base / name),
                   "--device", "cpu", "--workers", "0", "--threads", "2", "--batch-size", "2",
                   "--epochs", str(epochs), "--lr-patience", "0", "--no-amp", "--input-scaling", input_scaling,
                   "--train-augmentation", augmentation, "--weight-decay-policy", weight_decay_policy]
        if resume is not None:
            options.extend(["--resume", str(resume)])
        if ema_decay is not None:
            options.extend(["--ema", "--ema-decay", str(ema_decay)])
        if schedule_epochs is not None:
            options.extend(["--lr-schedule", "warmup_cosine", "--schedule-epochs", str(schedule_epochs),
                            "--warmup-epochs", str(warmup_epochs)])
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

    def test_augmented_preflight_keeps_initial_weights_identical(self):
        torch.manual_seed(42)
        expected = build_model("small_cnn").state_dict()
        def checked_build(*args, **kwargs):
            model = build_model(*args, **kwargs)
            self.assert_nested_equal(expected, model.state_dict())
            return model
        with patch("train.build_model", side_effect=checked_build) as mocked:
            self.run_training("preflight", 1, input_scaling="centered", augmentation="mild_affine")
        mocked.assert_called_once()

    def test_affine_resume_matches_uninterrupted_and_rejects_changed_augmentation(self):
        continuous = self.run_training("affine_continuous", 2, input_scaling="centered", augmentation="mild_affine")
        first = self.run_training("affine_first", 1, input_scaling="centered", augmentation="mild_affine")
        self.assertEqual(first["config"]["augmentation"], "mild_affine")
        self.assertEqual(first["config"]["augmentation_settings"]["degrees"], [-7.0, 7.0])
        resumed = self.run_training("affine_resumed", 2, self.base / "affine_first/last.pt",
                                    input_scaling="centered", augmentation="mild_affine")
        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                    "random_state", "selection_state"):
            self.assert_nested_equal(continuous[key], resumed[key])
        self.assertEqual(continuous["history"][1]["val"]["loss"], resumed["history"][1]["val"]["loss"])
        with self.assertRaisesRegex(ValueError, "augmentation"):
            self.run_training("wrong_augmentation", 2, self.base / "affine_first/last.pt", input_scaling="centered")
        self.assertFalse((self.base / "wrong_augmentation").exists())

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

    def test_grouped_decay_resume_is_exact_and_rejects_policy_change(self):
        continuous = self.run_training("grouped_continuous", 2, weight_decay_policy="exclude_1d_bias")
        first = self.run_training("grouped_first", 1, weight_decay_policy="exclude_1d_bias")
        resumed = self.run_training("grouped_resumed", 2, self.base / "grouped_first/last.pt",
                                    weight_decay_policy="exclude_1d_bias")
        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "random_state", "selection_state"):
            self.assert_nested_equal(continuous[key], resumed[key])
        self.assertEqual(continuous["history"][1]["val"]["loss"], resumed["history"][1]["val"]["loss"])
        self.assertEqual(first["config"]["weight_decay_policy"], "exclude_1d_bias")
        self.assertEqual([g["weight_decay"] for g in first["optimizer_state_dict"]["param_groups"]], [0.0001, 0.0])
        with self.assertRaisesRegex(ValueError, "weight_decay_policy"):
            self.run_training("bad_policy", 2, self.base / "grouped_first/last.pt")
        self.assertFalse((self.base / "bad_policy").exists())

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

    def test_ema_resume_restores_both_weights_and_matches_continuous(self):
        continuous = self.run_training("ema_continuous", 2, ema_decay=0.9, weight_decay_policy="exclude_1d_bias")
        first = self.run_training("ema_first", 1, ema_decay=0.9, weight_decay_policy="exclude_1d_bias")
        resumed = self.run_training("ema_resumed", 2, self.base / "ema_first/last.pt",
                                    ema_decay=0.9, weight_decay_policy="exclude_1d_bias")
        for key in ("model_state_dict", "training_model_state_dict", "ema_updates", "optimizer_state_dict",
                    "scheduler_state_dict", "scaler_state_dict", "random_state", "selection_state"):
            self.assert_nested_equal(continuous[key], resumed[key])
        self.assertEqual(continuous["history"][1]["val"]["loss"], resumed["history"][1]["val"]["loss"])
        self.assertEqual(first["evaluation_weights"], "ema")
        self.assertEqual(resumed["ema_updates"], 2)
        self.assertTrue(any(not torch.equal(first['model_state_dict'][k], v)
                            for k, v in first['training_model_state_dict'].items()))
        for name, decay in (("ema_disabled", None), ("ema_changed", 0.8)):
            with self.assertRaisesRegex(ValueError, "ema"):
                self.run_training(name, 2, self.base / "ema_first/last.pt", ema_decay=decay,
                                  weight_decay_policy="exclude_1d_bias")
            self.assertFalse((self.base / name).exists())
        from train import evaluate
        from scripts.compare_validation import verify_run
        from dataset import load_splits
        best = torch.load(self.base / "ema_resumed/best.pt", map_location="cpu", weights_only=True)
        args = parse_args(["--mode", "evaluate", "--checkpoint", str(self.base / "ema_resumed/best.pt"),
                           "--data-root", str(self.root), "--splits", str(self.splits), "--device", "cpu",
                           "--batch-size", "2", "--threads", "2", "--workers", "0", "--save-predictions",
                           "--output", str(self.base / "ema_evaluation")])
        with redirect_stdout(io.StringIO()):
            result = evaluate(args)
        self.assertEqual(result["evaluation_weights"], "ema")
        self.assertAlmostEqual(result["metrics"]["loss"], best['history'][best['epoch']-1]['val']['loss'])
        rows, audit = load_splits(self.splits)
        verified, _ = verify_run(self.base / "ema_resumed", self.base / "ema_evaluation", "small_cnn", rows['val'], audit)
        self.assertEqual(verified['settings']['evaluation_weights'], 'ema')

    def test_cosine_ema_resume_crosses_warmup_and_matches_continuous(self):
        opts = dict(ema_decay=0.9, weight_decay_policy="exclude_1d_bias", schedule_epochs=4)
        continuous = self.run_training("cosine_continuous", 4, **opts)
        self.run_training("cosine_first", 1, **opts)
        resumed = self.run_training("cosine_resumed", 4, self.base / "cosine_first/last.pt", **opts)
        for key in ("model_state_dict", "training_model_state_dict", "ema_updates", "optimizer_state_dict",
                    "scheduler_state_dict", "scaler_state_dict", "random_state", "selection_state"):
            self.assert_nested_equal(continuous[key], resumed[key])
        for a, b in zip(continuous['history'], resumed['history']):
            for key in ('learning_rate', 'next_learning_rate', 'scheduler_updates'):
                self.assertEqual(a[key], b[key])
            self.assertEqual(a['val']['loss'], b['val']['loss'])
        self.assertEqual(resumed['scheduler_state_dict']['completed_updates'], 4)
        self.assertEqual(resumed['history'][0]['learning_rate'], 1e-5)
        self.assertEqual(resumed['history'][-1]['learning_rate'], 1e-6)
        for name, changes in (('cosine_horizon', {'schedule_epochs': 5}),
                              ('cosine_warmup', {'warmup_epochs': 2}),
                              ('cosine_disabled', {'schedule_epochs': None})):
            with self.assertRaisesRegex(ValueError, 'schedule'):
                self.run_training(name, 4, self.base / 'cosine_first/last.pt', **(opts | changes))
            self.assertFalse((self.base/name).exists())
        from train import evaluate
        from scripts.compare_validation import verify_run
        from dataset import load_splits
        args = parse_args(['--mode','evaluate','--checkpoint',str(self.base/'cosine_resumed/best.pt'),
                           '--data-root',str(self.root),'--splits',str(self.splits),'--device','cpu',
                           '--batch-size','2','--threads','2','--workers','0','--save-predictions',
                           '--output',str(self.base/'cosine_evaluation')])
        with redirect_stdout(io.StringIO()):
            result = evaluate(args)
        rows, audit = load_splits(self.splits)
        verified, _ = verify_run(self.base/'cosine_resumed', self.base/'cosine_evaluation', 'small_cnn', rows['val'], audit)
        self.assertEqual(result['evaluation_weights'], 'ema')
        self.assertEqual(verified['settings']['scheduler_settings']['name'], 'WarmupCosine')


if __name__ == "__main__":
    unittest.main()
