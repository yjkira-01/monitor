import unittest

import numpy as np

from audit_fusion_hip_recenter import prediction_hip_recenter, supplementary_metrics


class SupplementaryHipRecenterChecks(unittest.TestCase):
    def test_pure_translation_is_removed_and_primary_is_preserved(self):
        target = np.zeros((2, 51))
        prediction = np.tile([3., 4., 0.], (2, 17))
        original = prediction.copy()
        metrics = supplementary_metrics(prediction, target, np.array([5, 6]))
        self.assertEqual(metrics["primary_metrics"]["MPJPE_mm"], 5.)
        self.assertEqual(metrics["prediction_hip_recentered_metrics"]["MPJPE_mm"], 0.)
        self.assertEqual(metrics["signed_delta_recentered_minus_primary_mm"]["MPJPE_mm"], -5.)
        np.testing.assert_array_equal(prediction, original)

    def test_recentring_can_increase_error_and_positive_delta_is_retained(self):
        prediction, target = np.zeros((1, 51)), np.zeros((1, 51))
        prediction.reshape(1, 17, 3)[:, 11:13, 0] = 2.
        metrics = supplementary_metrics(prediction, target, np.array([5]))
        self.assertGreater(metrics["signed_delta_recentered_minus_primary_mm"]["MPJPE_mm"], 0.)
        self.assertAlmostEqual(metrics["primary_metrics"]["MPJPE_mm"], 4 / 17)
        self.assertAlmostEqual(metrics["prediction_hip_recentered_metrics"]["MPJPE_mm"], 30 / 17)

    def test_recentered_prediction_is_translation_invariant_without_target(self):
        prediction = np.arange(102).reshape(2, 51).astype(float)
        shifted = prediction + np.tile([10., -3., 4.], (2, 17))
        first, hip = prediction_hip_recenter(prediction)
        second, _ = prediction_hip_recenter(shifted)
        np.testing.assert_array_equal(first, second)
        pose = first.reshape(2, 17, 3)
        np.testing.assert_array_equal((pose[:, 11] + pose[:, 12]) / 2, 0.)
        self.assertEqual(hip.shape, (2, 3))

    def test_non_hip_centred_target_and_nonfinite_prediction_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "already hip-centred"):
            supplementary_metrics(np.zeros((1, 51)), np.ones((1, 51)), np.array([5]))
        with self.assertRaisesRegex(ValueError, "finite"):
            prediction_hip_recenter(np.full((1, 51), np.nan))


if __name__ == "__main__":
    unittest.main()
