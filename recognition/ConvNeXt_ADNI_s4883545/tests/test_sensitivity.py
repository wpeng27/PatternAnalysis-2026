"""Known-input checks for perturbation semantics and diagnostic counts."""
import unittest
import torch
from scripts.analyze_sensitivity import perturb, compare_probabilities, routing_results


class SensitivityTests(unittest.TestCase):
    def test_point_transforms_and_background(self):
        x = torch.tensor([[[[0., .2, .4, .8, 1.]]]])
        before = x.clone()
        torch.testing.assert_close(perturb(x, "brightness_1.1"), torch.tensor([[[[0., .22, .44, .88, 1.]]]]))
        # Nonzero mean is 0.6, zero background must not be lifted.
        torch.testing.assert_close(perturb(x, "contrast_0.9"), torch.tensor([[[[0., .24, .42, .78, .96]]]]))
        torch.testing.assert_close(perturb(x, "gamma_0.9"), x ** .9)
        torch.testing.assert_close(perturb(x, "identity"), x, rtol=0, atol=0)
        self.assertTrue(torch.equal(x, before))
        self.assertTrue(torch.equal(perturb(torch.zeros_like(x), "contrast_1.1"), torch.zeros_like(x)))

    def test_shifts_no_wrap_or_mutation(self):
        x = torch.zeros(1, 1, 16, 16)
        x[0, 0, 8, 8] = 1
        for name, row, col in [("left_4", 8, 4), ("right_4", 8, 12), ("up_4", 4, 8), ("down_4", 12, 8)]:
            y = perturb(x, name)
            self.assertEqual(y.sum().item(), 1)
            self.assertEqual(y[0, 0, row, col].item(), 1)
        edge = torch.zeros_like(x)
        edge[..., 0] = 1
        self.assertEqual(perturb(edge, "left_4").sum().item(), 0)
        self.assertEqual(x[0, 0, 8, 8].item(), 1)

    def test_crop_removes_border(self):
        x = torch.zeros(1, 1, 16, 16)
        x[..., :4, :] = 1
        y = perturb(x, "crop_4")
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.sum().item(), 0)
        self.assertTrue(torch.equal(perturb(torch.ones_like(x), "crop_4"), torch.ones_like(x)))

    def test_transition_counts_and_exact_tie(self):
        r = compare_probabilities([0, 1, 0, 1], [.1, .9, .8, .2], [.6, .8, .5, .7], .75)
        self.assertEqual(r["class_flips"], 3)
        self.assertEqual(r["correct_to_wrong"], 1)
        self.assertEqual(r["wrong_to_correct"], 2)
        self.assertEqual(r["metrics"]["accuracy"], .75)
        self.assertEqual(r["accepted"], 1)
        self.assertEqual(r["accepted_errors"], 0)
        self.assertEqual(r["referral_action_changes"], 3)

    def test_bad_input_rejected(self):
        for x in [torch.ones(1, 3, 16, 16), torch.full((1, 1, 16, 16), float('nan')),
                  torch.full((1, 1, 16, 16), 1.1)]:
            with self.assertRaises(ValueError):
                perturb(x, "identity")
        with self.assertRaises(ValueError):
            compare_probabilities([0, 1], [.1], [.1, .9], .5)

    def test_review_candidates_and_shared_error(self):
        rows = [{"patient_id": p, "label": y} for p, y in [("A", 0), ("A", 0), ("B", 1), ("B", 1)]]
        r = routing_results(rows, [.9, .1, .2, .8], [.1, .1, .9, .8], .6)
        self.assertEqual(r["current"]["referred"], 0)
        self.assertEqual(r["or_model_disagreement"]["referred_errors"], 2)
        self.assertEqual(r["or_model_disagreement"]["accepted_accuracy"], 1)
        self.assertEqual(r["or_patient_disagreement"]["referred_errors"], 1)
        self.assertEqual(r["or_patient_disagreement"]["referred_correct"], 1)
        self.assertEqual(r["or_either_disagreement"]["referred"], 3)
        self.assertEqual(r["or_either_disagreement"]["coverage"], .25)
        shared = routing_results(rows, [.99, .01, .01, .99], [.99, .01, .01, .99], .6)
        self.assertEqual(shared["both_wrong_confidence_at_least_095"]["slices"], 2)
        self.assertEqual(shared["both_wrong_confidence_at_least_095"]["false_negatives"], 1)
        self.assertEqual(shared["or_model_disagreement"]["referred_errors"], 0)


if __name__ == '__main__':
    unittest.main()
