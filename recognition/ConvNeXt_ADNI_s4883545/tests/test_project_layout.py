"""Check relocated source provenance and standalone frozen-code imports."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts._paths import PROJECT_ROOT, source_path
from scripts.final_evaluation import frozen_evaluator, snapshot_code


class ProjectLayoutTests(unittest.TestCase):
    def test_source_paths_resolve_core_and_helper_files(self):
        self.assertEqual(source_path('metrics.py'), PROJECT_ROOT / 'metrics.py')
        self.assertEqual(source_path('compare_validation.py'), PROJECT_ROOT / 'scripts/compare_validation.py')
        with self.assertRaises(ValueError):
            source_path('../train.py')
        with self.assertRaises(FileNotFoundError):
            source_path('missing.py')

    def test_snapshot_runs_without_original_project_on_import_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(frozen_evaluator(root), root / 'code/final_evaluation.py')
            snapshot_code(root)
            script = frozen_evaluator(root)
            self.assertEqual(script, root / 'code/scripts/final_evaluation.py')
            self.assertEqual((root / 'code/metrics.py').read_bytes(), source_path('metrics.py').read_bytes())
            process = subprocess.run([sys.executable, '-I', str(script), '--help'],
                                     cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn('freeze', process.stdout)


if __name__ == '__main__':
    unittest.main()
