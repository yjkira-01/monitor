import unittest

import numpy as np
import torch

from train_mri_point_baselines import PointEncoder, point_normalization


class PointProbeChecks(unittest.TestCase):
    def test_padding_and_order_do_not_change_point_representation(self):
        torch.manual_seed(7)
        model = PointEncoder()
        points = torch.randn(2, 4, 5)
        mask = torch.tensor([[True, True, False, False], [False] * 4])
        original = model(points, mask)
        points[~mask] = torch.nan
        torch.testing.assert_close(original, model(points, mask))
        order = torch.tensor([3, 1, 0, 2])
        torch.testing.assert_close(original, model(points[:, order], mask[:, order]))
        torch.testing.assert_close(original[1], torch.zeros(128))
        original.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_normalizer_ignores_missing_padding(self):
        points = np.array([[[1., 2., 3., 4., 5.], [3., 4., 5., 6., 7.], [np.nan] * 5]])
        mean, std = point_normalization(points, np.array([[True, True, False]]))
        np.testing.assert_array_equal(mean, [2., 3., 4., 5., 6.])
        np.testing.assert_array_equal(std, np.ones(5))

    def test_invalid_observed_clouds_and_masks_are_rejected(self):
        model = PointEncoder()
        with self.assertRaises(ValueError):
            model(torch.full((1, 2, 5), torch.nan), torch.ones(1, 2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 2, 5), torch.ones(1, 2))
        with self.assertRaises(ValueError):
            point_normalization(np.zeros((1, 2, 5)), np.zeros((1, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()
