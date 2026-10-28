"""Prediction contract checks using synthetic images and tiny CNN checkpoints."""

from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from dataset import ADNIDataset, CLASS_TO_INDEX, audit_and_split, load_splits, save_splits, preprocess_image
from modules import build_model
from predict import ImageFiles, load_predictor, load_policy, parse_args, predict, prediction_record, spaced_indices
from train import PREPROCESSING_BY_SCALING


class PredictionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "data"
        for split in ("train", "test"):
            for class_index, name in enumerate(("NC", "AD")):
                folder = self.root / split / name
                folder.mkdir(parents=True)
                first = 100 + class_index * 100 + (50 if split == "test" else 0)
                for patient in range(first, first + (3 if split == "train" else 1)):
                    rng = random.Random(patient)
                    for slice_id in (1, 2):
                        pixels = bytes(rng.randrange(256) for _ in range(13 * 17 * 3))
                        Image.frombytes("RGB", (13, 17), pixels).save(folder / f"{patient}_{slice_id}.jpeg")
        rows, report = audit_and_split(self.root)
        self.split_dir = self.base / "splits"
        save_splits(self.split_dir, rows, report)
        self.splits, self.audit = load_splits(self.split_dir)
        self.model = build_model("small_cnn")
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.classifier[-1].bias[1] = torch.log(torch.tensor(3.0))
        self.saved = {"model_name": "small_cnn", "epoch": 2, "model_state_dict": self.model.state_dict(),
                      "config": {"class_to_index": CLASS_TO_INDEX,
                                 "preprocessing": PREPROCESSING_BY_SCALING["centered"],
                                 "input_scaling": "centered", "model_kwargs": {"drop_path_rate": 0.0},
                                 "split_sha256": self.audit["split_sha256"], "seed": 42,
                                 "effective_amp": True, "amp_dtype": "float16", "batch_size": 3}}
        self.checkpoint = self.base / "fixture.pt"
        torch.save(self.saved, self.checkpoint)

    def args(self, extra=()):
        return parse_args(["--checkpoint", str(self.checkpoint), "--splits", str(self.split_dir),
                           "--data-root", str(self.root), "--output", str(self.base / "result"),
                           "--device", "cpu", "--max-display", "2", *extra])

    def test_shared_preprocessing_matches_pillow_reference_and_channels(self):
        row = self.splits["val"][0]
        path = self.root / row["path"]
        with Image.open(path) as source:
            reference = source.convert("L").resize((224, 224), Image.Resampling.BILINEAR)
            reference = torch.tensor(list(reference.getdata()), dtype=torch.float32).reshape(1, 224, 224) / 255
        for scaling in ("unit", "centered"):
            expected = reference if scaling == "unit" else reference * 2 - 1
            standalone = ImageFiles([path], scaling)[0][0]
            audited = ADNIDataset(self.root, [row], scaling)[0][0]
            self.assertEqual(standalone.shape, (3, 224, 224))
            self.assertTrue(torch.equal(standalone, expected.repeat(3, 1, 1)))
            self.assertTrue(torch.equal(standalone, audited))
        with self.assertRaises(ValueError):
            preprocess_image(path.read_bytes(), "unknown")

    def test_manifest_prediction_known_metrics_and_no_training(self):
        before = self.checkpoint.read_bytes()
        with patch("predict.build_model", return_value=self.model), \
             patch("torch.optim.AdamW", side_effect=AssertionError("Prediction must not train")), \
             redirect_stdout(io.StringIO()):
            result = predict(self.args())
        self.assertEqual(result["split"], "val")
        self.assertEqual(result["metrics_scope"], "complete_split")
        self.assertAlmostEqual(result["metrics"]["accuracy"], 0.5)
        self.assertEqual(result["metrics"]["ad_auroc"], 0.5)
        self.assertAlmostEqual(result["metrics"]["macro_f1"], 1 / 3)
        self.assertEqual(result["metrics"]["patient"]["samples"], 2)
        self.assertEqual(result["predicted_images"], 4)  # Includes final partial batch of one.
        self.assertEqual(result["effective_precision"], "float32")
        self.assertFalse(result["test_inference_performed"])
        self.assertFalse(self.model.training)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.parameters()))
        self.assertEqual(before, self.checkpoint.read_bytes())
        output = self.base / "result"
        with (output / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(abs(float(row["p_ad"]) - 0.75) < 1e-6 for row in rows))
        self.assertEqual(json.loads((output / "prediction.json").read_text()), result)
        with Image.open(output / "demo.png") as image:
            self.assertGreater(image.width, 100)
        with self.assertRaisesRegex(ValueError, "already exists"):
            predict(self.args())

    def test_limited_manifest_is_not_full_patient_evaluation(self):
        with redirect_stdout(io.StringIO()):
            result = predict(self.args(["--limit", "2"]))
        self.assertEqual(result["metrics_scope"], "demonstration_subset")
        self.assertEqual(result["predicted_images"], 2)
        self.assertNotIn("patient", result["metrics"])
        self.assertEqual(spaced_indices(4300, 12)[-1], 4299)

    def test_single_image_has_no_inferred_label_or_metrics(self):
        image = self.root / self.splits["val"][0]["path"]
        args = parse_args(["--checkpoint", str(self.checkpoint), "--images", str(image),
                           "--device", "cpu", "--output", str(self.base / "single")])
        with redirect_stdout(io.StringIO()):
            result = predict(args)
        self.assertIsNone(result["metrics"])
        self.assertIsNone(result["split"])
        self.assertEqual(result["predicted_images"], 1)
        with (args.output / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["true_class"], "")
        self.assertEqual(row["correct"], "")

    def test_ema_uses_evaluation_weights_not_online_weights(self):
        online = {name: tensor.clone() for name, tensor in self.saved["model_state_dict"].items()}
        online["classifier.5.bias"] = torch.tensor([10.0, -10.0])
        self.saved.update(evaluation_weights="ema", training_model_state_dict=online,
                          history=[{"train": {"optimizer_steps": 3}}], ema_updates=3)
        self.saved["config"].update(ema=True, ema_decay=0.999, validation_weights="ema")
        torch.save(self.saved, self.checkpoint)
        model, _, identity = load_predictor(self.checkpoint, torch.device("cpu"))
        self.assertEqual(identity["evaluation_weights"], "ema")
        self.assertTrue(torch.equal(model.classifier[-1].bias, self.model.classifier[-1].bias))
        self.assertFalse(torch.equal(model.classifier[-1].bias, online["classifier.5.bias"]))
        self.saved["ema_updates"] = 2
        torch.save(self.saved, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "update count"):
            load_predictor(self.checkpoint, torch.device("cpu"))

    def test_incompatible_checkpoint_manifest_and_changed_image_rejected(self):
        for key, value in (("class_to_index", {"AD": 0, "NC": 1}),
                           ("preprocessing", "wrong"), ("split_sha256", {})):
            with self.subTest(key=key):
                bad = {**self.saved, "config": {**self.saved["config"], key: value}}
                torch.save(bad, self.checkpoint)
                with self.assertRaises(ValueError):
                    predict(self.args())
                self.assertFalse(self.args().output.exists())
        torch.save(self.saved, self.checkpoint)
        path = self.root / self.splits["val"][0]["path"]
        image_dataset = ImageFiles([path], "centered")
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            image_dataset[0]
        with self.assertRaisesRegex(ValueError, "changed"):
            predict(self.args())

    def test_class_ties_threshold_ties_and_policy_checkpoint_identity(self):
        row = {"path": "image", "patient_id": "1", "label": 0, "file_sha256": "hash"}
        tie = prediction_record(row, 0.5, 0.5)
        self.assertEqual((tie["predicted_class"], tie["correct"], tie["review_decision"]), ("NC", True, "accept"))
        self.assertEqual(prediction_record(row, 0.75, 0.75)["review_decision"], "accept")
        self.assertEqual(prediction_record(row, 0.75, 0.7501)["review_decision"], "refer")
        _, config, identity = load_predictor(self.checkpoint, torch.device("cpu"))
        policy = {"state": "verified_validation_confidence_analysis", "evaluation_unit": "slice",
                  "class_to_index": CLASS_TO_INDEX, "split_sha256": config["split_sha256"],
                  "models": [{"model": "small_cnn", "checkpoint_sha256": identity["checkpoint_sha256"],
                              "status": "selected", "threshold": 0.75}]}
        path = self.base / "policy.json"
        path.write_text(json.dumps(policy))
        self.assertEqual(load_policy(path, identity, config)["threshold"], 0.75)
        policy["models"][0]["checkpoint_sha256"] = "wrong"
        path.write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, "identify this checkpoint"):
            load_policy(path, identity, config)

    def test_cli_rejects_ambiguous_or_invalid_arguments(self):
        for extra in (["--batch-size", "0"], ["--max-display", "65"], ["--limit", "0"],
                      ["--images", "x.jpeg"]):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(extra)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--checkpoint", "x.pt", "--splits", "splits", "--output", "out"])


if __name__ == "__main__":
    unittest.main()
