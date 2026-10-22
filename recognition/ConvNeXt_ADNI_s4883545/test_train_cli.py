"""Check CLI validation and checkpoint-only evaluation on synthetic images."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from dataset import CLASS_TO_INDEX, audit_and_split, load_splits, save_splits
from modules import build_model
from train import PREPROCESSING, evaluate, parse_args, prepare_run


class ArgumentTests(unittest.TestCase):
    def test_explicit_training_options_and_aliases(self):
        args = parse_args(["--model", "small_cnn", "--data-root", "ADNI/AD_NC",
                           "--epochs", "2", "--batch-size", "8", "--learning-rate", "0.0003",
                           "--seed", "42", "--device", "cpu", "--output-dir", "artifacts/run",
                           "--workers", "0", "--no-amp"])
        self.assertEqual((args.model, args.epochs, args.batch_size, args.lr), ("small_cnn", 2, 8, 0.0003))
        self.assertEqual((args.seed, args.device, args.output), (42, "cpu", Path("artifacts/run")))
        self.assertFalse(args.amp)

    def test_evaluation_defaults_to_validation(self):
        args = parse_args(["--mode", "evaluate", "--data-root", "ADNI/AD_NC", "--checkpoint", "best.pt"])
        self.assertEqual(args.eval_split, "val")

    def test_invalid_values_and_conflicting_modes_are_rejected(self):
        cases = [[], ["--mode", "evaluate"], ["--batch-size", "0"], ["--epochs", "0"],
                 ["--workers", "-1"], ["--seed", "-1"], ["--seed", str(2**32)],
                 ["--lr", "nan"], ["--lr", "inf"], ["--lr", "-1"], ["--weight-decay", "nan"],
                 ["--checkpoint", "best.pt"], ["--eval-split", "test"], ["--model", "unknown"]]
        for options in cases:
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                # The empty case verifies that a data root is required.
                argv = ["--data-root", "ADNI/AD_NC"] + options if options else []
                with self.assertRaises(SystemExit) as error:
                    parse_args(argv)
                self.assertEqual(error.exception.code, 2)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "data"
        for split in ("train", "test"):
            for index, label in enumerate(("AD", "NC")):
                directory = self.root / split / label
                directory.mkdir(parents=True)
                first = 100 + index * 100 + (50 if split == "test" else 0)
                for patient in range(first, first + (2 if split == "train" else 1)):
                    rng = random.Random(patient)
                    pixels = bytes(rng.randrange(256) for _ in range(16 * 16))
                    Image.frombytes("L", (16, 16), pixels).save(directory / f"{patient}_1.jpeg")
        self.split_dir = self.base / "splits"
        rows, report = audit_and_split(self.root)
        save_splits(self.split_dir, rows, report)
        _, audit = load_splits(self.split_dir)
        torch.manual_seed(42)
        model = build_model("small_cnn")
        self.checkpoint = self.base / "fixture.pt"
        self.saved = {"model_name": "small_cnn", "epoch": 1, "model_state_dict": model.state_dict(),
                      "config": {"class_to_index": CLASS_TO_INDEX, "preprocessing": PREPROCESSING,
                                 "split_sha256": audit["split_sha256"], "effective_amp": True}}
        torch.save(self.saved, self.checkpoint)

    def args(self, split="val"):
        return parse_args(["--mode", "evaluate", "--model", "small_cnn", "--data-root", str(self.root),
                           "--splits", str(self.split_dir), "--checkpoint", str(self.checkpoint),
                           "--eval-split", split, "--output", str(self.base / f"evaluation_{split}"),
                           "--device", "cpu", "--batch-size", "1", "--workers", "0"])

    def test_evaluation_loads_weights_without_optimizer_or_checkpoint_writes(self):
        original_bytes = self.checkpoint.read_bytes()
        for split in ("val", "test"):
            with self.subTest(split=split):
                # The test branch uses only synthetic fixtures, never ADNI test data.
                torch.manual_seed(99)
                model = build_model("small_cnn")
                args = self.args(split)
                with patch("train.build_model", return_value=model), \
                     patch("torch.optim.AdamW", side_effect=AssertionError("Evaluation must not train")), \
                     redirect_stdout(io.StringIO()):
                    result = evaluate(args)
                self.assertEqual(result["metrics"]["samples"], 2)
                self.assertEqual(result["split"], split)
                self.assertFalse(result["retrained"])
                self.assertFalse(result["effective_amp"])  # CPU disables CUDA AMP.
                self.assertFalse(model.training)
                self.assertTrue(all(torch.equal(value, self.saved["model_state_dict"][name])
                                    for name, value in model.state_dict().items()))
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertEqual(self.checkpoint.read_bytes(), original_bytes)
                self.assertEqual([p.name for p in args.output.iterdir()], ["evaluation.json"])
                written = json.loads((args.output / "evaluation.json").read_text(encoding="utf-8"))
                self.assertEqual(written, result)

    def test_incompatible_checkpoint_is_rejected_before_writing_results(self):
        for key, bad_value in (("split_sha256", {}), ("class_to_index", {"AD": 0, "NC": 1}),
                               ("preprocessing", "different")):
            with self.subTest(field=key):
                bad = {**self.saved, "config": {**self.saved["config"], key: bad_value}}
                torch.save(bad, self.checkpoint)
                args = self.args()
                with self.assertRaises(ValueError):
                    evaluate(args)
                self.assertFalse(args.output.exists())

    def test_cuda_alias_resolves_to_an_explicit_device_index(self):
        args = self.args()
        args.device = "cuda"
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.set_device") as set_device, patch("train.seed_everything"):
            self.assertEqual(prepare_run(args), torch.device("cuda:0"))
            set_device.assert_called_once_with(torch.device("cuda:0"))
            args.device = "cuda:2"
            with self.assertRaisesRegex(ValueError, "unavailable"):
                prepare_run(args)


if __name__ == "__main__":
    unittest.main()
