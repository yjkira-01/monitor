import unittest

import numpy as np
import torch

from run_fusion_experiments import augmented_batch, normalize_inputs, sparse_mask


class FusionDataChecks(unittest.TestCase):
    def test_raw_confidence_threshold_and_normalized_missing_fields(self):
        features = np.ones((2, 51))
        features[0, 34:] = 0
        features[1, 34:] = .8
        features[1, 34 + 3] = .1
        points = np.ones((2, 196, 5), dtype=np.float32)
        mask = np.ones((2, 196), dtype=bool)
        stats = dict(rgb_mean=np.ones(51) * .9, rgb_std=np.ones(51), point_mean=np.zeros(5), point_std=np.ones(5))
        rgb, _, _, confidence = normalize_inputs(features, points, mask, stats)
        torch.testing.assert_close(rgb[0], torch.zeros(51))
        self.assertEqual(rgb[1, 6].item(), 0)
        self.assertEqual(rgb[1, 7].item(), 0)
        self.assertEqual(rgb[1, 37].item(), 0)
        self.assertGreater(confidence[1, 0], .3)
        self.assertLess(rgb[1, 34], 0)  # scaled score differs from the raw threshold

    def test_augmentation_draws_reproducible_and_no_empty_two_sensor_rows(self):
        raw = np.ones((256, 51))
        points = np.ones((256, 196, 5), dtype=np.float32)
        mask = np.zeros((256, 196), dtype=bool)
        mask[:, 0] = True
        stats = dict(rgb_mean=np.zeros(51), rgb_std=np.ones(51), point_mean=np.zeros(5), point_std=np.ones(5))
        training = [normalize_inputs(raw, points, mask, stats)]
        indices = torch.arange(256)
        a = augmented_batch(training, indices, torch.Generator().manual_seed(17))
        b = augmented_batch(training, indices, torch.Generator().manual_seed(17))
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second)
        available = (a[3] >= .3).any(1) | a[2].any(1)
        self.assertTrue(available.all())
        self.assertTrue((~a[2].any(1)).any())
        self.assertTrue((~(a[3] >= .3).any(1)).any())

    def test_sparse_validation_does_not_drop_the_last_observed_point(self):
        mask = np.zeros((20, 196), dtype=bool)
        mask[:, 0] = True
        np.testing.assert_array_equal(sparse_mask(mask), mask)


if __name__ == '__main__':
    unittest.main()
