import unittest

import numpy as np
import torch

from validate_mri_point_baselines import MatchedRGBRegressor, error_metrics, lag_candidates
from train_mri_point_baselines import PoseRegressor


class ValidationProtocolChecks(unittest.TestCase):
    def test_time_shift_uses_timestamps_and_preserves_zero_pair(self):
        points = np.ones((21, 14, 14, 5), dtype=np.float32)
        points[:, :, :, 0] = np.arange(21)[:, None, None]
        timestamps = 1651100000000 + np.arange(21) * 100
        candidates, common = lag_candidates(points, timestamps, [0, 20], {"action": [0, 21]}, np.array([6, 10, 14]))
        np.testing.assert_array_equal(common, [True] * 3)
        np.testing.assert_array_equal(candidates[500]["matched_frame_ids"], [11, 15, 19])
        np.testing.assert_array_equal(candidates[-500]["matched_frame_ids"], [1, 5, 9])
        np.testing.assert_array_equal(candidates[0]["points"], points[[6, 10, 14]].reshape(3, 196, 5))

    def test_boundary_crossing_and_empty_cloud_are_excluded_from_common_subset(self):
        points = np.ones((21, 14, 14, 5), dtype=np.float32)
        timestamps = np.arange(21) * 100
        intervals = {"first": [0, 9], "second": [9, 21]}
        _, common = lag_candidates(points, timestamps, [0, 20], intervals, np.array([4, 14]))
        np.testing.assert_array_equal(common, [False, True])
        points[19] = 0
        _, common = lag_candidates(points, timestamps, [0, 20], intervals, np.array([4, 14]))
        self.assertFalse(common.any())

    def test_capacity_control_has_no_radar_input_dependence(self):
        model = MatchedRGBRegressor()
        fusion = PoseRegressor("fusion")
        ratio = sum(p.numel() for p in model.parameters()) / sum(p.numel() for p in fusion.parameters())
        self.assertLess(abs(ratio - 1), .01)
        rgb = torch.randn(2, 51)
        torch.testing.assert_close(model(rgb, None, None), model(rgb, torch.full((2, 196, 5), torch.nan), None))

    def test_metric_is_euclidean_joint_error_in_millimetres(self):
        target = np.zeros((2, 51))
        prediction = np.tile([3., 4., 0.], (2, 17))
        metric = error_metrics(prediction, target, np.array([5, 6]))
        self.assertEqual(metric["MPJPE_mm"], 5.)
        self.assertEqual(metric["per_validation_subject_MPJPE_mm"], {"5": 5., "6": 5.})
        with self.assertRaises(ValueError):
            error_metrics(prediction * np.nan, target, np.array([5, 6]))


if __name__ == "__main__":
    unittest.main()
