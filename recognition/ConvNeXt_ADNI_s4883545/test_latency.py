"""Verify timing statistics and deterministic batch traversal, without a GPU."""
import unittest
import numpy as np
from benchmark_latency import summarize, cyclic_indices


class LatencyTests(unittest.TestCase):
    def test_throughput_uses_total_time_and_batch_size(self):
        result = summarize([1, 3], 2)
        self.assertEqual(result['mean_batch_ms'], 2.)
        self.assertEqual(result['median_batch_ms'], 2.)
        self.assertEqual(result['amortized_ms_per_image'], 1.)
        self.assertEqual(result['images_per_second'], 1000.)
        self.assertAlmostEqual(result['p95_batch_ms'], 2.9)
        self.assertEqual(result['std_batch_ms'], 1.)

    def test_invalid_timing_samples_rejected(self):
        for times in ([], [0], [-1], [np.nan], [np.inf], [[1, 2]]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                summarize(times, 1)
        with self.assertRaises(ValueError):
            summarize([1], 0)

    def test_cyclic_batches_stay_aligned(self):
        self.assertEqual(cyclic_indices(0, 2, 4), [0, 1])
        self.assertEqual(cyclic_indices(1, 2, 4), [2, 3])
        self.assertEqual(cyclic_indices(2, 2, 4), [0, 1])
        self.assertEqual(cyclic_indices(4, 1, 4), [0])
        for args in ((-1, 1, 4), (0, 0, 4), (0, 8, 4)):
            with self.assertRaises(ValueError):
                cyclic_indices(*args)


if __name__ == '__main__':
    unittest.main()
