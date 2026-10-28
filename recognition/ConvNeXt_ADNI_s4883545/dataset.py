"""Audit ADNI JPEGs, save patient-disjoint splits and load grayscale tensors.

Staff guidance: 392586_92.jpeg identifies patient 392586 and slice 92.
No raw JSON lookup is required. AI-assisted implementation, validated with
synthetic leakage fixtures and the local ADNI dataset.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import random
import re

from PIL import Image, __version__ as pillow_version
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomAffine
from torchvision.transforms import functional as TF

CLASS_TO_INDEX = {"NC": 0, "AD": 1}
SEED = 42
IMAGE_SIZE = 224
AUGMENTATION_SETTINGS = {
    "none": {"name": "none"},
    "mild_affine": {"name": "mild_affine", "degrees": [-7.0, 7.0],
                    "translate": [0.05, 0.05], "scale": [0.95, 1.05],
                    "shear": None, "interpolation": "bilinear", "fill": 0,
                    "applied_to": "training grayscale PIL image after resize, before scaling and channel repetition"},
}
PARTITIONS = ("train", "val", "test")
PATTERN = re.compile(r"([0-9]+)_([0-9]+)\.(?:jpeg|jpg)", re.IGNORECASE)
FIELDS = ("path", "patient_id", "slice_id", "class_name", "label", "source_split",
          "split", "width", "height", "file_sha256", "pixel_sha256")


def patient_and_slice(filename: str) -> tuple[str, int]:
    """Require the complete numeric filename and canonicalise leading zeroes."""
    match = PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Expected patient_slice.jpeg, got {filename!r}")
    return str(int(match[1])), int(match[2])


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def assign_splits(rows: list[dict], seed: int, val_fraction: float) -> None:
    """Stratify training patients by class and preserve the supplied test set."""
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be strictly between zero and one")
    rng = random.Random(seed)
    val_patients = set()
    for label in sorted(CLASS_TO_INDEX):
        patients = sorted({r["patient_id"] for r in rows
                           if r["source_split"] == "train" and r["class_name"] == label})
        if len(patients) < 2:
            raise ValueError(f"At least two training patients are required in {label}")
        rng.shuffle(patients)
        # Nearest whole patient, retaining both classes in train and validation.
        count = max(1, min(len(patients) - 1, int(len(patients) * val_fraction + 0.5)))
        val_patients.update(patients[:count])
    for row in rows:
        row["split"] = ("test" if row["source_split"] == "test" else
                        "val" if row["patient_id"] in val_patients else "train")


def duplicate_groups(rows: list[dict], key: str) -> list[list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return [group for group in groups.values() if len(group) > 1]


def partition_issues(rows: list[dict]) -> list[str]:
    """Find patient/content overlap and conflicting labels in all partitions."""
    issues = []
    for key in ("patient_id", "file_sha256", "pixel_sha256"):
        groups = duplicate_groups(rows, key)
        crossing = sum(len({r["split"] for r in g}) > 1 for g in groups)
        conflicting = sum(len({r["label"] for r in g}) > 1 for g in groups)
        if crossing:
            issues.append(f"{key}: {crossing} groups overlap final partitions")
        if conflicting:
            issues.append(f"{key}: {conflicting} groups have conflicting labels")
    for split in PARTITIONS:
        for label in CLASS_TO_INDEX:
            if not any(r["split"] == split and r["class_name"] == label for r in rows):
                issues.append(f"No valid {label} images in {split}")
    return issues


def split_counts(rows: list[dict]) -> dict:
    counts = {}
    for split in PARTITIONS:
        counts[split] = {}
        for label in sorted(CLASS_TO_INDEX):
            group = [r for r in rows if r["split"] == split and r["class_name"] == label]
            counts[split][label] = {"images": len(group),
                                    "patients": len({r["patient_id"] for r in group})}
    return counts


def audit_and_split(root: Path, seed: int = SEED, val_fraction: float = 0.2) -> tuple[list[dict], dict]:
    """Decode every JPEG and hash file bytes and grayscale pixels with dimensions.

    Same-class duplicates confined to one partition are reported, not deleted.
    Cross-partition duplicates and invalid data block split publication. Exact
    hashes do not detect near duplicates or unknown aliases for the same patient.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be strictly between zero and one")
    root = root.resolve(strict=True)
    rows, issues, ignored, modes = [], [], [], Counter()
    identities = set()
    for split in ("train", "test"):
        for label in CLASS_TO_INDEX:
            if not (root / split / label).is_dir():
                issues.append(f"Missing folder: {split}/{label}")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            issues.append(f"Unexpected symbolic link: {relative}")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpeg", ".jpg"}:
            if path.suffix.lower() in {".png", ".bmp", ".tif", ".tiff", ".webp"}:
                issues.append(f"Unexpected image format: {relative}")
            else:
                ignored.append(relative)
            continue
        parts = PurePosixPath(relative).parts
        if len(parts) != 3 or parts[0] not in {"train", "test"} or parts[1] not in CLASS_TO_INDEX:
            issues.append(f"Unexpected JPEG location: {relative}")
            continue
        try:
            patient, slice_id = patient_and_slice(path.name)
            raw = path.read_bytes()
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "JPEG":
                    raise ValueError("Not a JPEG-encoded image")
                image.load()
                gray = image.convert("L")
                width, height = gray.size
                mode = image.mode
                pixel_hash = sha256(f"L:{width}:{height}:".encode("ascii") + gray.tobytes())
            identity = (patient, slice_id)
            if identity in identities:
                issues.append(f"Repeated patient/slice identity: {relative}")
            identities.add(identity)
            modes[mode] += 1
            rows.append({"path": relative, "patient_id": patient, "slice_id": slice_id,
                         "class_name": parts[1], "label": CLASS_TO_INDEX[parts[1]],
                         "source_split": parts[0], "split": parts[0],
                         "width": width, "height": height,
                         "file_sha256": sha256(raw), "pixel_sha256": pixel_hash})
            if len(rows) % 5000 == 0:
                print(f"Audited {len(rows):,} images", flush=True)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            issues.append(f"{relative}: {error}")
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    overlap = sorted(p for p, group in by_patient.items()
                     if len({r["source_split"] for r in group}) > 1)
    if overlap:
        issues.append(f"{len(overlap)} patients overlap supplied train and test")
    try:
        assign_splits(rows, seed, val_fraction)
    except ValueError as error:
        issues.append(str(error))
    issues.extend(partition_issues(rows))
    duplicates = {}
    for key in ("file_sha256", "pixel_sha256"):
        groups = duplicate_groups(rows, key)
        duplicates[key] = {"group_count": len(groups),
                           "cross_split_groups": sum(len({r["split"] for r in g}) > 1 for g in groups),
                           "paths": [[r["path"] for r in g] for g in groups]}
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "seed": seed, "val_fraction": val_fraction, "class_to_index": CLASS_TO_INDEX,
              "python": platform.python_version(), "pillow": pillow_version,
              "valid_images": len(rows), "counts": split_counts(rows),
              "source_patient_overlap": overlap, "duplicates": duplicates,
              "shapes_width_height": dict(Counter(f'{r["width"]}x{r["height"]}' for r in rows)),
              "modes": dict(modes), "ignored_non_image_files": ignored,
              "issues": issues, "passed": not issues,
              "limitations": ["Patient identity follows the staff-supplied filename rule.",
                              "Exact hashes do not detect near duplicates.",
                              "Test images are read for hygiene, not model evaluation."]}
    return rows, report


def save_splits(output: Path, rows: list[dict], report: dict) -> None:
    """Use a fresh directory so failed audits cannot leave stale split lists."""
    output.mkdir(parents=True, exist_ok=False)
    report = dict(report)
    if report["passed"]:
        report["split_sha256"] = {}
        for split in PARTITIONS:
            path = output / f"{split}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(r for r in rows if r["split"] == split)
            report["split_sha256"][split] = sha256(path.read_bytes())
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def load_splits(directory: Path) -> tuple[dict[str, list[dict]], dict]:
    """Verify saved CSV digests, identities and partition separation."""
    report = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or not report.get("passed") or report.get("class_to_index") != CLASS_TO_INDEX:
        raise ValueError("Missing, failed or incompatible split audit")
    splits, all_rows = {}, []
    for split in PARTITIONS:
        path = directory / f"{split}.csv"
        raw = path.read_bytes()
        if sha256(raw) != report["split_sha256"][split]:
            raise ValueError(f"Split file has changed since audit: {path}")
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected manifest columns: {path}")
        rows = list(reader)
        for row in rows:
            parts = PurePosixPath(row["path"]).parts
            if (len(parts) != 3 or parts[0] not in {"train", "test"}
                    or parts[1] not in CLASS_TO_INDEX or "\\" in row["path"]):
                raise ValueError(f"Invalid dataset-relative path: {row['path']}")
            patient, slice_id = patient_and_slice(parts[2])
            row["label"] = int(row["label"])
            row["slice_id"] = int(row["slice_id"])
            if (row["split"] != split or row["source_split"] != parts[0]
                    or row["class_name"] != parts[1] or row["label"] != CLASS_TO_INDEX[parts[1]]
                    or row["patient_id"] != patient or row["slice_id"] != slice_id
                    or (split == "test") != (parts[0] == "test")):
                raise ValueError(f"Inconsistent manifest identity: {row['path']}")
        splits[split] = rows
        all_rows.extend(rows)
    issues = partition_issues(all_rows)
    if issues:
        raise ValueError("Invalid saved partitions: " + "; ".join(issues))
    return splits, report


def preprocess_image(raw: bytes, input_scaling: str = "unit", transform=None) -> torch.Tensor:
    """Shared training/inference preprocessing from encoded image bytes.

    Transform, when supplied by the training dataset, acts on the resized
    grayscale image. Prediction uses no transform. All three output channels
    contain identical grayscale data.
    """
    if input_scaling not in {"unit", "centered"}:
        raise ValueError("input_scaling must be unit or centered")
    with Image.open(io.BytesIO(raw)) as image:
        gray = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        if transform is not None:
            gray = transform(gray)
        tensor = TF.pil_to_tensor(gray).to(dtype=torch.float32).div_(255.0)
    if input_scaling == "centered":
        tensor = tensor.mul(2).sub(1)
    return tensor.repeat(3, 1, 1)


class ADNIDataset(Dataset):
    """Float32 CHW tensors, bilinear 224x224 resize and three identical channels.

    Fixed unit [0, 1] or centered [-1, 1] scaling uses no dataset statistics.
    Optional mild affine augmentation is allowed only for audited train rows.
    Recheck hashes to reject data changed since the audit.
    """

    def __init__(self, root: Path, rows: list[dict], input_scaling: str = "unit",
                 augmentation: str = "none"):
        if input_scaling not in {"unit", "centered"}:
            raise ValueError("input_scaling must be unit or centered")
        self.root = root.resolve(strict=True)
        self.rows = rows
        self.input_scaling = input_scaling
        if augmentation not in AUGMENTATION_SETTINGS:
            raise ValueError("Unknown augmentation")
        if augmentation != "none" and any(row.get("split") != "train" for row in rows):
            raise ValueError("Augmentation requires exclusively audited training rows")
        self.augmentation = augmentation
        self.transform = None
        if augmentation == "mild_affine":
            settings = AUGMENTATION_SETTINGS[augmentation]
            # torchvision 0.24 API: https://docs.pytorch.org/vision/0.24/generated/torchvision.transforms.RandomAffine.html
            # Local experimental bounds, not a clinically validated invariance.
            # Transform one grayscale image so copied channels remain identical.
            self.transform = RandomAffine(settings["degrees"], translate=settings["translate"],
                                          scale=settings["scale"], interpolation=InterpolationMode.BILINEAR,
                                          fill=settings["fill"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = (self.root / row["path"]).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("Image path escapes dataset root")
        raw = path.read_bytes()
        if sha256(raw) != row["file_sha256"]:
            raise ValueError(f"Image changed since split audit: {row['path']}")
        return preprocess_image(raw, self.input_scaling, self.transform), int(row["label"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts/splits_seed42")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Reuse its saved splits or choose a new directory.")
    if args.output.resolve().is_relative_to(args.data_root.resolve()):
        parser.error("Split output must be outside the dataset")
    rows, report = audit_and_split(args.data_root, args.seed, args.val_fraction)
    save_splits(args.output, rows, report)
    print(json.dumps({k: report[k] for k in ("passed", "counts", "issues")}, indent=2))
    print(f"Audit saved to {args.output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
