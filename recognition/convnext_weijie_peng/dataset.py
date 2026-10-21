"""Inspect patient grouping using the filename rule.

Example: 392586_92.jpeg identifies patient 392586 and slice 92. Raw JSON metadata
is not required. This setup check does not yet decode images or check duplicates.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


CLASS_TO_INDEX = {"NC": 0, "AD": 1}
SEED = 42
PATTERN = re.compile(r"([0-9]+)_([0-9]+)\.(?:jpeg|jpg)", re.IGNORECASE)


def patient_and_slice(filename: str) -> tuple[str, int]:
    """Parse the complete filename, treating leading zeroes consistently."""
    match = PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Expected patient_slice.jpeg, got {filename!r}")
    return str(int(match[1])), int(match[2])


def inspect_patient_groups(root: Path) -> dict:
    """Count patients and reject ambiguous identities before future splitting."""
    root = root.resolve(strict=True)
    groups = defaultdict(list)
    counts = {}
    identities = set()
    for split in ("train", "test"):
        counts[split] = {}
        for label in ("AD", "NC"):
            folder = root / split / label
            if not folder.is_dir():
                raise ValueError(f"Required folder is missing: {folder}")
            files = sorted(p for p in folder.iterdir()
                           if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg"})
            if not files:
                raise ValueError(f"No JPEG images found in {folder}")
            patients = set()
            for path in files:
                patient, slice_id = patient_and_slice(path.name)
                identity = (patient, slice_id)
                if identity in identities:
                    raise ValueError(f"Repeated patient/slice identity at {path}")
                identities.add(identity)
                patients.add(patient)
                groups[patient].append((split, label))
            counts[split][label] = {"images": len(files), "patients": len(patients)}
    overlap = sorted(p for p, records in groups.items() if len({s for s, _ in records}) > 1)
    conflicting = sorted(p for p, records in groups.items() if len({c for _, c in records}) > 1)
    return {"counts": counts, "total_patients": len(groups),
            "train_test_patient_overlap": overlap, "conflicting_patient_labels": conflicting,
            "patient_checks_passed": not overlap and not conflicting,
            "scope": "Direct JPEG files in train/AD, train/NC, test/AD and test/NC only",
            "image_integrity_and_duplicate_checks": "Not performed by this setup script"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    report = inspect_patient_groups(args.data_root)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["patient_checks_passed"] else 1)


if __name__ == "__main__":
    main()
