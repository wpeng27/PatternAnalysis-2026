"""Verify that runtime projections use measured full epochs and remain estimates."""

from pathlib import Path
import unittest
from unittest.mock import patch

from report_run import budget_report, completed_run


class ReportTests(unittest.TestCase):
    def test_budget_includes_warm_epochs_and_two_process_startups(self):
        history = [{"epoch_seconds": seconds, "peak_training_allocated_bytes": 2**30,
                    "peak_training_reserved_bytes": 2 * 2**30} for seconds in (10, 6, 8)]
        config = {"device_name": "fixture", "batch_size": 32, "precision": "fixture",
                  "workers": 0, "early_stopping_patience": 5}
        status = {"start_epoch": 1, "process_seconds": 30, "process_startup_seconds": 2}
        with patch("report_run.completed_run", return_value=(history, config, status)), \
             patch("report_run.write_json"):
            report = budget_report(Path("fixture"), max_epochs=5)
        self.assertEqual(report["warm_epoch_mean_seconds"], 7)
        self.assertEqual(report["estimated_total_seconds_including_trial"], 48)
        self.assertEqual(report["conservative_estimated_total_seconds_including_trial"], 54)
        self.assertEqual(report["peak_allocated_gib_measured"], 1)
        self.assertEqual(report["peak_reserved_gib_measured"], 2)
        self.assertIn("Estimates", report["projection_caveat"])

    def test_running_job_cannot_be_reported_as_complete(self):
        with patch("report_run.read_json", return_value={"state": "running"}):
            with self.assertRaisesRegex(ValueError, "not complete"):
                completed_run(Path("fixture"))


if __name__ == "__main__":
    unittest.main()
