"""Regression tests use synthetic fixtures, never as model performance evidence."""

import copy
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import (ADNIDataset, assign_splits, audit_and_split, load_splits,
                     patient_and_slice, save_splits)
from modules import build_model
from train import run_epoch, seed_everything


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "data"
        for split in ("train", "test"):
            for class_index, label in enumerate(("AD", "NC")):
                folder = self.root / split / label
                folder.mkdir(parents=True)
                start = 100 + class_index * 100 + (50 if split == "test" else 0)
                for patient in range(start, start + (5 if split == "train" else 1)):
                    for slice_id in (1, 2):
                        rng = random.Random(patient * 10 + slice_id)
                        pixels = bytes(rng.randrange(256) for _ in range(16 * 20))
                        Image.frombytes("L", (16, 20), pixels).save(folder / f"{patient}_{slice_id}.jpeg")

    def test_strict_filename_parser(self):
        self.assertEqual(patient_and_slice("00392586_092.JPEG"), ("392586", 92))
        for name in ("patient_92.jpeg", "392586.jpeg", "392586_92_extra.jpeg", "392586_92.png"):
            with self.assertRaises(ValueError):
                patient_and_slice(name)

    def test_centered_scaling_is_exact_and_preserves_labels_and_channels(self):
        rows, report = audit_and_split(self.root)
        self.assertTrue(report["passed"])
        unit = ADNIDataset(self.root, rows)
        centered = ADNIDataset(self.root, rows, input_scaling="centered")
        for index in (0, len(rows) - 1):
            image, label = unit[index]
            changed, changed_label = centered[index]
            self.assertEqual(label, changed_label)
            self.assertTrue(torch.equal(changed, image * 2 - 1))
            self.assertTrue(torch.equal(changed[0], changed[1]) and torch.equal(changed[1], changed[2]))
            self.assertGreaterEqual(changed.min().item(), -1)
            self.assertLessEqual(changed.max().item(), 1)

    def test_split_reproducibility_and_loader(self):
        rows, report = audit_and_split(self.root)
        self.assertTrue(report["passed"], report["issues"])
        self.assertEqual(report["counts"]["val"]["AD"], {"images": 2, "patients": 1})
        self.assertEqual(report["counts"]["val"]["NC"], {"images": 2, "patients": 1})
        reversed_rows = copy.deepcopy(rows[::-1])
        assign_splits(reversed_rows, 42, 0.2)
        self.assertEqual({r["path"]: r["split"] for r in rows},
                         {r["path"]: r["split"] for r in reversed_rows})
        self.assertTrue(all(r["split"] == "test" for r in rows if r["source_split"] == "test"))
        output = Path(self.temp.name) / "splits"
        save_splits(output, rows, report)
        saved, _ = load_splits(output)
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse({r["patient_id"] for r in saved[a]} & {r["patient_id"] for r in saved[b]})
        dataset = ADNIDataset(self.root, saved["train"])
        for image, label in dataset:
            self.assertEqual(image.shape, (3, 224, 224))
            self.assertEqual(image.dtype, torch.float32)
            self.assertTrue(torch.equal(image[0], image[1]) and torch.equal(image[1], image[2]))
            self.assertTrue(torch.isfinite(image).all())
            self.assertGreaterEqual(image.min().item(), 0)
            self.assertLessEqual(image.max().item(), 1)
            self.assertIn(label, (0, 1))
        with self.assertRaises(FileExistsError):
            save_splits(output, rows, report)

    def test_patient_overlap_blocks_saved_splits(self):
        (self.root / "test/AD/150_1.jpeg").rename(self.root / "test/AD/100_3.jpeg")
        rows, report = audit_and_split(self.root)
        self.assertFalse(report["passed"])
        self.assertIn("100", report["source_patient_overlap"])
        output = Path(self.temp.name) / "failed"
        save_splits(output, rows, report)
        self.assertFalse((output / "train.csv").exists())
        with self.assertRaises(ValueError):
            load_splits(output)

    def test_conflicting_patient_labels(self):
        (self.root / "train/NC/200_1.jpeg").rename(self.root / "train/NC/100_3.jpeg")
        _, report = audit_and_split(self.root)
        self.assertFalse(report["passed"])
        self.assertTrue(any("patient_id" in issue and "conflicting" in issue for issue in report["issues"]))

    def test_file_and_pixel_duplicates_across_test_boundary(self):
        raw = (self.root / "train/AD/100_1.jpeg").read_bytes()
        (self.root / "test/AD/150_1.jpeg").write_bytes(raw)
        # Distinct bytes after JPEG end marker leave decoded pixels identical.
        (self.root / "test/AD/150_2.jpeg").write_bytes(raw + b"different trailing metadata")
        _, report = audit_and_split(self.root)
        self.assertFalse(report["passed"])
        for key in ("file_sha256", "pixel_sha256"):
            self.assertEqual(report["duplicates"][key]["cross_split_groups"], 1)
        self.assertEqual(len(report["duplicates"]["pixel_sha256"]["paths"][0]), 3)

    def test_duplicate_across_training_validation_boundary(self):
        rows, _ = audit_and_split(self.root)
        train = next(r for r in rows if r["split"] == "train" and r["class_name"] == "AD")
        val = next(r for r in rows if r["split"] == "val" and r["class_name"] == "AD")
        (self.root / val["path"]).write_bytes((self.root / train["path"]).read_bytes())
        _, report = audit_and_split(self.root)
        self.assertFalse(report["passed"])
        self.assertEqual(report["duplicates"]["pixel_sha256"]["cross_split_groups"], 1)

    def test_corrupt_and_misnamed_images_block_publication(self):
        (self.root / "train/AD/100_1.jpeg").write_bytes(b"corrupt")
        (self.root / "train/AD/100_2.jpeg").rename(self.root / "train/AD/wrong.jpeg")
        _, report = audit_and_split(self.root)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["issues"]), 2)

    def test_changed_manifest_or_image_is_rejected(self):
        rows, report = audit_and_split(self.root)
        output = Path(self.temp.name) / "splits"
        save_splits(output, rows, report)
        splits, _ = load_splits(output)
        dataset = ADNIDataset(self.root, splits["train"])
        image_path = self.root / splits["train"][0]["path"]
        image_path.write_bytes(image_path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed since"):
            dataset[0]
        with (output / "train.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed since"):
            load_splits(output)

    def test_insufficient_patients_and_invalid_fraction(self):
        rows, _ = audit_and_split(self.root)
        limited = [r for r in rows if r["source_split"] == "test" or r["patient_id"] in {"100", "200"}]
        with self.assertRaises(ValueError):
            assign_splits(limited, 42, 0.2)
        for fraction in (0, 1, -0.2, float("nan")):
            with self.assertRaises(ValueError):
                assign_splits(rows, 42, fraction)


class TrainingTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    def test_bfloat16_without_scaler_rejects_nonfinite_gradients(self):
        model = torch.nn.Linear(4, 2).cuda()
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        loader = DataLoader(TensorDataset(torch.ones(2, 4), torch.tensor([0, 1])), batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        try:
            with self.assertRaises(RuntimeError):
                run_epoch(model, loader, torch.device("cuda"), optimizer, scaler, amp=True, amp_dtype=torch.bfloat16)
        finally:
            hook.remove()
        self.assertTrue(all(torch.equal(initial[name], p) for name, p in model.named_parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the AMP overflow test")
    def test_amp_overflow_skips_update_without_corrupting_weights(self):
        seed_everything(42)
        model = torch.nn.Linear(4, 2).cuda()
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        loader = DataLoader(TensorDataset(torch.ones(2, 4), torch.tensor([0, 1])), batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", init_scale=8.0)
        hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        try:
            result = run_epoch(model, loader, torch.device("cuda"), optimizer, scaler, amp=True)
        finally:
            hook.remove()
        self.assertEqual(result["amp_skipped_updates"], 1)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertTrue(all(torch.equal(initial[name], p) for name, p in model.named_parameters()))
        recovered = run_epoch(model, loader, torch.device("cuda"), optimizer, scaler, amp=True)
        self.assertEqual(recovered["optimizer_steps"], 1)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertTrue(any(not torch.equal(initial[name], p) for name, p in model.named_parameters()))

    def test_full_epoch_updates_weights_and_validation_is_read_only(self):
        seed_everything(42)
        torch.set_num_threads(2)
        model = build_model("small_cnn")
        images = torch.rand(3, 3, 224, 224)
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        before = model.features[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        result = run_epoch(model, loader, torch.device("cpu"), optimizer, scaler)
        self.assertEqual(result["samples"], 3)  # Includes the last partial batch.
        self.assertFalse(torch.equal(before, model.features[0].weight))
        state = {name: value.clone() for name, value in model.state_dict().items()}
        validation = run_epoch(model, loader, torch.device("cpu"))
        self.assertTrue(all(torch.equal(state[name], value) for name, value in model.state_dict().items()))
        with torch.no_grad():
            logits = model(images)
            expected = torch.nn.functional.cross_entropy(logits, labels).item()
        self.assertEqual(logits.shape, (3, 2))
        self.assertAlmostEqual(validation["loss"], expected, places=5)
        self.assertAlmostEqual(validation["slice_accuracy"], (logits.argmax(1) == labels).float().mean().item(), places=6)


if __name__ == "__main__":
    unittest.main()
