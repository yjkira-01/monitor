import unittest

import numpy as np

from fusion_pose_representation import visible_centroid_features


class PartialVisualRepresentationChecks(unittest.TestCase):
    def test_missing_hips_do_not_remove_other_observed_joints(self):
        poses = np.zeros((2, 17, 3))
        poses[0, 5] = [100, 200, .8]
        poses[0, 9] = [300, 200, .8]
        features = visible_centroid_features(poses, [1000, 1000])
        self.assertEqual(features[0, 34 + 5], .8)
        self.assertEqual(features[0, 34 + 11], 0)
        self.assertAlmostEqual(features[0, 10], -.1)
        self.assertAlmostEqual(features[0, 18], .1)
        np.testing.assert_array_equal(features[1], np.zeros(51))

    def test_translation_invariance_and_no_unobserved_coordinate_leakage(self):
        poses = np.zeros((1, 17, 3))
        poses[0, 5:11, :2] = np.arange(12).reshape(6, 2) * 10
        poses[0, 5:11, 2] = .7
        expected = visible_centroid_features(poses, [500, 700])
        poses[..., :2] += [123, -321]
        np.testing.assert_allclose(visible_centroid_features(poses, [500, 700]), expected, atol=1e-15)
        poses[0, 0, :2] = [100000, -100000]
        np.testing.assert_allclose(visible_centroid_features(poses, [500, 700]), expected, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
