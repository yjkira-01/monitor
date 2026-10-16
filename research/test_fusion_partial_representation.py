import inspect
import unittest

import numpy as np

from fusion_partial_representation import hip_preserving_fallback_features


class HipPreservingFallbackChecks(unittest.TestCase):
    def setUp(self):
        self.poses = np.zeros((4, 17, 3))
        self.poses[0, :, 0] = np.arange(17) * 10
        self.poses[0, :, 1] = np.arange(17) * 3
        self.poses[0, :, 2] = .8
        self.poses[1, 5] = [100, 200, .8]
        self.poses[1, 9] = [300, 200, .7]
        self.poses[2, 5] = [99, 123, .9]
        self.hip = np.zeros((4, 51))
        self.hip[0, :34] = np.linspace(-.51, .87, 34)
        self.hip[0, 34:] = .8
        self.focal = np.array([1000, 1100])

    def test_hip_valid_features_are_preserved_exactly_without_mutating_sources(self):
        original_poses, original_hip = self.poses.copy(), self.hip.copy()
        for mode in ("centroid", "confidence_only"):
            result = hip_preserving_fallback_features(self.poses, self.focal, self.hip, mode)
            np.testing.assert_array_equal(result[0], self.hip[0])
            self.assertEqual(result.dtype, self.hip.dtype)
        np.testing.assert_array_equal(self.poses, original_poses)
        np.testing.assert_array_equal(self.hip, original_hip)

    def test_both_modes_have_identical_confidence_and_availability(self):
        geometric = hip_preserving_fallback_features(self.poses, self.focal, self.hip, "centroid")
        confidence = hip_preserving_fallback_features(self.poses, self.focal, self.hip, "confidence_only")
        np.testing.assert_array_equal(geometric[:, 34:], confidence[:, 34:])
        np.testing.assert_array_equal((geometric[:, 34:] >= .3).any(1), [True, True, True, False])
        self.assertGreater(np.abs(geometric[1, :34]).max(), .05)
        np.testing.assert_array_equal(confidence[1, :34], np.zeros(34))
        np.testing.assert_array_equal(geometric[3], np.zeros(51))
        np.testing.assert_array_equal(confidence[3], np.zeros(51))

    def test_one_visible_joint_keeps_confidence_but_has_no_relative_geometry(self):
        for mode in ("centroid", "confidence_only"):
            result = hip_preserving_fallback_features(self.poses, self.focal, self.hip, mode)
            np.testing.assert_allclose(result[2, :34], np.zeros(34), atol=1e-15)
            self.assertEqual(result[2, 34 + 5], .9)

    def test_all_hip_valid_batch_does_not_call_empty_centroid(self):
        poses, hip = np.repeat(self.poses[:1], 3, axis=0), np.repeat(self.hip[:1], 3, axis=0)
        for mode in ("centroid", "confidence_only"):
            result = hip_preserving_fallback_features(poses, self.focal, hip, mode)
            np.testing.assert_array_equal(result, hip)

    def test_all_raw_missing_batch_remains_zero_without_empty_centroid(self):
        poses, hip = np.zeros((3, 17, 3)), np.zeros((3, 51))
        for mode in ("centroid", "confidence_only"):
            result = hip_preserving_fallback_features(poses, self.focal, hip, mode)
            np.testing.assert_array_equal(result, hip)

    def test_raw_zero_xy_becomes_fixed_observed_values_after_common_normalization(self):
        result = hip_preserving_fallback_features(self.poses, self.focal, self.hip, "confidence_only")
        visible = result[:, 34:] >= .3
        observed = np.concatenate((np.repeat(visible, 2, axis=1), visible), axis=1)
        mean, std = np.full(51, .1), np.full(51, .2)
        normalized = np.where(observed, (result - mean) / std, 0.)
        np.testing.assert_array_equal(normalized[1, [10, 11, 18, 19]], np.full(4, -.5))
        self.assertEqual(normalized[1, 0], 0.)

    def test_target_is_not_an_input_and_inconsistent_hip_cache_is_rejected(self):
        self.assertEqual(list(inspect.signature(hip_preserving_fallback_features).parameters),
                         ["poses", "focal", "hip_features", "mode"])
        with self.assertRaises(TypeError):
            hip_preserving_fallback_features(self.poses, self.focal, self.hip, target=np.zeros((4, 51)))
        invalid = self.hip.copy()
        invalid[1, 0] = 1.
        with self.assertRaises(ValueError):
            hip_preserving_fallback_features(self.poses, self.focal, invalid)


if __name__ == "__main__":
    unittest.main()
