"""Guard the pre-test freeze and prevent filtered accuracy replacing full accuracy."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.final_evaluation import verify_freeze, summarize_test, write_new_json
from scripts.compare_validation import file_hash
from metrics import classification_metrics, aggregate_patients


class FinalEvaluationTests(unittest.TestCase):
    def test_freeze_detects_modified_artifact_and_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'artifact').write_text('original')
            record = {'state': 'frozen_before_test', 'test_evaluated_at_freeze': False,
                      'files_sha256': {'artifact': file_hash(root/'artifact')}}
            write_new_json(root/'freeze.json', record)
            (root/'freeze.sha256').write_text(file_hash(root/'freeze.json'))
            self.assertEqual(verify_freeze(root)[0], record)
            (root/'artifact').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                verify_freeze(root)
            (root/'artifact').write_text('original')
            (root/'freeze.json').write_text(json.dumps(dict(record, test_evaluated_at_freeze=True)))
            with self.assertRaisesRegex(ValueError, 'specification changed'):
                verify_freeze(root)

    def test_existing_records_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'record.json'
            write_new_json(path, {'a': 1})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {'a': 2})

    def test_filtered_success_does_not_pass_unfiltered_requirement(self):
        labels, probabilities = [0, 0, 1, 1], [.1, .6, .4, .9]
        rows = [{'path': str(i), 'patient_id': str(i), 'label': label, 'class_name': 'AD' if label else 'NC'}
                for i, label in enumerate(labels)]
        metrics = classification_metrics(labels, probabilities)
        metrics['patient'] = aggregate_patients(rows, labels, probabilities)
        result = summarize_test(rows, probabilities, metrics, .8)
        self.assertEqual(result['unfiltered_slice']['accuracy'], .5)
        self.assertEqual(result['accepted_slice']['accuracy'], 1.)
        self.assertEqual(result['review']['coverage'], .5)
        self.assertFalse(result['meets_minimum_unfiltered_slice_accuracy'])

    def test_no_accepted_cases_are_explicit(self):
        labels, probabilities = [0, 1], [.5, .5]
        rows = [{'path': str(i), 'patient_id': str(i), 'label': label, 'class_name': 'AD' if label else 'NC'}
                for i, label in enumerate(labels)]
        metrics = classification_metrics(labels, probabilities)
        metrics['patient'] = aggregate_patients(rows, labels, probabilities)
        result = summarize_test(rows, probabilities, metrics, .9)
        self.assertIsNone(result['accepted_slice'])
        self.assertIsNone(result['review']['accepted_accuracy'])


if __name__ == '__main__':
    unittest.main()
