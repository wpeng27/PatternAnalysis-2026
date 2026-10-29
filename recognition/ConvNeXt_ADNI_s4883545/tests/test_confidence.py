"""Known-label, tie and abstention tests for validation confidence analysis."""

import unittest
import numpy as np

from scripts.analyze_confidence import (coverage_curve, select_threshold, threshold_summary,
                                calibration_summary, clustered_intervals, ranked_errors)


class ConfidenceTests(unittest.TestCase):
    def test_known_curve_and_maximum_coverage_rule(self):
        # Highest two correct, next two wrong, lowest correct: 2/2, 2/4, 3/5.
        y, p = [1, 0, 0, 1, 0], [1., 0., 0.75, 0.25, 0.5]
        curve = coverage_curve(y, p)
        self.assertEqual([r['accepted'] for r in curve], [2, 4, 5])
        self.assertEqual([r['accuracy'] for r in curve], [1., 0.5, 0.6])
        self.assertIsNone(select_threshold(curve, 0.9, 0.5))
        self.assertEqual(select_threshold(curve, 0.9, 0.4)['threshold'], 1.)
        self.assertEqual(select_threshold(curve, 0.6, 0.5)['threshold'], 0.5)

    def test_equal_confidence_group_is_never_split_to_hit_target(self):
        curve = coverage_curve([1, 0, 0, 1], [1., 0., 1., 0.])
        self.assertTrue(all(point['accepted'] == 4 for point in curve))
        self.assertIsNone(select_threshold(curve, 0.9, 0.5))

    def test_all_correct_selects_half_even_without_observed_half(self):
        choice = select_threshold(coverage_curve([0, 1], [0., 1.]))
        self.assertEqual(choice['threshold'], 0.5)
        self.assertEqual(choice['coverage'], 1.)

    def test_nonmonotonic_accuracy_searches_every_candidate(self):
        # Accuracy 1, 1/2, 2/3, 3/4, 3/5: a greedy stop at failure is wrong.
        curve = coverage_curve([1, 0, 1, 1, 0], [1., .9, .8, .7, .6])
        choice = select_threshold(curve, 0.75, 0.5)
        self.assertEqual(choice['accepted'], 4)
        self.assertEqual(choice['threshold'], .7)

    def test_boundary_prediction_referral_and_class_denominators(self):
        y, p = [0, 1, 1, 0], [.5, .5, 1., 1.]
        s = threshold_summary(y, p, 1.)
        self.assertEqual((s['accepted'], s['referred']), (2, 2))
        self.assertEqual(s['accepted_accuracy'], .5)
        self.assertEqual(s['referred_accuracy'], .5)
        self.assertEqual(s['fraction_of_all_errors_referred'], .5)
        self.assertEqual(s['per_class']['AD']['accepted_precision'], .5)
        self.assertEqual(s['per_class']['AD']['accepted_recall'], 1.)
        self.assertIsNone(s['per_class']['NC']['accepted_precision'])
        self.assertEqual(s['per_class']['NC']['coverage'], .5)
        self.assertIsNone(threshold_summary(y, [.5]*4, 1.)['accepted_accuracy'])

    def test_hand_computed_calibration_brier_and_correctness_auc(self):
        s = calibration_summary([0, 1, 1, 0], [.5, .5, 1., 1.])
        self.assertEqual(s['accuracy'], .5)
        self.assertEqual(s['mean_confidence'], .75)
        self.assertEqual(s['binary_brier'], .375)
        self.assertEqual(s['ece'], .25)
        self.assertEqual(s['correctness_auroc'], .5)
        self.assertEqual(s['bins'][0]['count'], 2)
        self.assertEqual(s['bins'][-1]['count'], 2)
        self.assertIsNone(s['bins'][1]['accuracy'])
        self.assertIsNone(calibration_summary([0, 1], [0., 1.])['correctness_auroc'])

    def test_cluster_bootstrap_preserves_whole_patients_and_is_reproducible(self):
        rows = [{'patient_id': 'A'}]*2 + [{'patient_id': 'B'}]*2
        a = clustered_intervals(rows, [0, 0, 1, 1], [0., 0., 0., 0.], .5, repeats=100)
        b = clustered_intervals(rows, [0, 0, 1, 1], [0., 0., 0., 0.], .5, repeats=100)
        self.assertEqual(a, b)
        self.assertEqual(a['accepted_accuracy'], [0., 1.])
        self.assertEqual(a['coverage'], [1., 1.])
        self.assertEqual(a['patients'], 2)

    def test_error_ranking_deduplicates_patients_and_includes_threshold_ties(self):
        rows = [{'path': str(i), 'patient_id': str(i//2)} for i in range(4)]
        errors, distinct = ranked_errors(rows, [0]*4, [.9, 1., .9, .75], .9)
        self.assertEqual([e['path'] for e in distinct], ['1', '2'])
        self.assertEqual([e['accepted_by_selected_rule'] for e in errors], [True, True, True, False])
        self.assertTrue(all(e['error_type'] == 'false_positive_AD' for e in errors))

    def test_bad_inputs_and_invalid_selection_constraints(self):
        for y, p in (([], []), ([0], [np.nan]), ([2], [.5]), ([0], [1.1]), ([0], [.2, .3])):
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                coverage_curve(y, p)
        for target, coverage in ((0, .5), (1.1, .5), (.9, 0), (.9, 1.1)):
            with self.assertRaises(ValueError):
                select_threshold([], target, coverage)
        for threshold in (np.nan, .4, 1.1):
            with self.assertRaises(ValueError):
                threshold_summary([0], [.2], threshold)

    def test_all_wrong_and_empty_acceptance_do_not_create_false_success(self):
        self.assertIsNone(select_threshold(coverage_curve([1, 0], [0., 1.])))
        summary = threshold_summary([0, 1], [.5, .5], .9)
        self.assertEqual(summary['accepted'], 0)
        self.assertIsNone(summary['accepted_accuracy'])

    def test_curve_matches_direct_masks_and_is_permutation_invariant(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=60)
        p = rng.choice([0., .125, .25, .5, .75, .875, 1.], size=60)
        curve = coverage_curve(y, p)
        confidence = np.maximum(p, 1-p)
        for point in curve:
            mask = confidence >= point['threshold']
            self.assertEqual(point['accepted'], int(mask.sum()))
            self.assertEqual(point['correct'], int(((p[mask] > .5) == y[mask]).sum()))
        permutation = rng.permutation(len(y))
        self.assertEqual(curve, coverage_curve(y[permutation], p[permutation]))


if __name__ == '__main__':
    unittest.main()
