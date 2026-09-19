import unittest
import warnings

import numpy as np

from xcebra_ibl.experiment.evaluation import interval


class BootstrapSupportTests(unittest.TestCase):
    def test_missing_classes_are_counted_without_changing_point_score(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 2, 1, 1, 2, 0])
        trials = np.repeat(np.arange(3), 2)
        with warnings.catch_warnings(record=True) as captured:
            result = interval(y, pred, trials, True, 200, 42, baseline=pred)
        self.assertEqual(captured, [])
        self.assertAlmostEqual(result['score'], 2/3)
        self.assertGreater(result['missing_class_bootstraps'], 0)
        self.assertEqual(result['valid_bootstraps'] + result['missing_class_bootstraps'], 200)
        self.assertEqual(result['improvement_ci95'], [0., 0.])

    def test_no_supported_draw_has_no_interval(self):
        result = interval(np.array([0, 1]), np.array([0, 1]),
                          np.array([0, 0]), True, 20, 42)
        self.assertIsNone(result['ci95'])
        self.assertEqual(result['valid_bootstraps'], 0)

    def test_continuous_bootstrap_unchanged(self):
        y = np.arange(10, dtype=float)
        result = interval(y, y, np.arange(10), False, 20, 42)
        self.assertEqual(result['valid_bootstraps'], 20)
        self.assertEqual(result['ci95'], [1., 1.])
        self.assertEqual(result['missing_class_bootstraps'], 0)
