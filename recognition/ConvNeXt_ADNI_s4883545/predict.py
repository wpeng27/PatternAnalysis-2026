"""Prediction entry-point scaffold for the next implementation milestone.

Future implementation will load trained weights, use the fixed data manifest,
and report predictions and evaluation metrics. No trained checkpoint exists yet.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error("Prediction is not implemented yet. Complete the model and training milestones first.")


if __name__ == "__main__":
    main()
