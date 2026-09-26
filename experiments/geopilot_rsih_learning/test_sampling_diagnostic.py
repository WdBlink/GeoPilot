"""Keep the diagnostic's forward reduction equal to the bound evaluator."""
import unittest

import numpy as np

from sampling_diagnostic import METRICS, evaluator, summarize


class SamplingDiagnosticTest(unittest.TestCase):
    def test_reduction_matches_evaluator_on_real_geometry_operations(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        triangles = np.array([[0, 1, 2]])
        reference = np.array([[0., 0., .1], [1., 0., .1], [0., 1., .1]])
        for seed in (20260916, 20260925):
            expected, _ = evaluator.metrics_from_arrays(
                vertices, triangles, reference, reference, sample_count=1000, seed=seed)
            samples = evaluator.sample_surface(vertices, triangles, 1000, seed)
            actual = summarize(evaluator.KDTree(reference).query(samples, workers=1)[0])
            self.assertEqual(actual, {key: expected[key] for key in METRICS})

    def test_threshold_and_nonfinite(self):
        self.assertEqual(summarize(np.array([.2, .20000001]))['precision_0_20'], .5)
        with self.assertRaises(AssertionError):
            summarize(np.array([np.nan]))


if __name__ == '__main__':
    unittest.main()
