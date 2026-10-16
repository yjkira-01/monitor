"""Scientific invariants for padding and root-relative objective controls."""

import unittest
import numpy as np
import torch

from xfi_accuracy_models import native_radar_batch, recenter_prediction, training_loss


class AccuracyControlTests(unittest.TestCase):
    def test_native_collate_preserves_real_rows_and_origin_measurements(self):
        points = np.zeros((2, 8, 5), dtype=np.float32)
        points[0, 1] = [0, 0, 0, 0, 3]  # Actual origin measurement, not padding.
        points[0, 4] = [1, 2, 3, 4, 5]
        points[1, 3] = [-1, 0, 0, 2, 1]
        batch, count = native_radar_batch(points)
        np.testing.assert_array_equal(count, [2, 1])
        np.testing.assert_array_equal(batch[0], points[0, [1, 4]])
        np.testing.assert_array_equal(batch[1, 0], points[1, 3])
        np.testing.assert_array_equal(batch[1, 1], np.zeros(5))
        padded, _ = native_radar_batch(np.pad(points, ((0, 0), (0, 12), (0, 0))))
        np.testing.assert_array_equal(batch, padded)

    def test_absent_cloud_is_safe_and_nonfinite_is_rejected(self):
        batch, count = native_radar_batch(np.zeros((2, 196, 5)))
        self.assertEqual(batch.shape, (2, 1, 5))
        np.testing.assert_array_equal(count, [0, 0])
        with self.assertRaises(ValueError):
            native_radar_batch(np.full((2, 3, 5), np.nan))

    def test_centering_is_translation_invariant_and_gradient_coupled(self):
        torch.manual_seed(7)
        pose = torch.randn(3, 17, 3, dtype=torch.float64, requires_grad=True)
        offset = torch.tensor([[[3., -7., 2.]]], dtype=torch.float64)
        target = recenter_prediction(torch.randn_like(pose))
        a = recenter_prediction(pose)
        b = recenter_prediction(pose + offset)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close((a[:, 11] + a[:, 12]) / 2, torch.zeros((3, 3), dtype=torch.float64), rtol=0, atol=1e-12)
        training_loss(a, target, "root_mpjpe").backward()
        self.assertTrue(torch.isfinite(pose.grad).all())
        torch.testing.assert_close(pose.grad.sum(1), torch.zeros((3, 3), dtype=torch.float64), rtol=0, atol=1e-12)

    def test_actual_distance_units_and_zero_error_gradients(self):
        target = torch.zeros((2, 17, 3), dtype=torch.float64)
        prediction = torch.zeros_like(target)
        prediction[:, :, 0] = .003
        prediction[:, :, 1] = .004
        self.assertAlmostEqual(training_loss(prediction, target, "native_mpjpe").item(), .005)
        self.assertAlmostEqual(training_loss(prediction, target, "native_mse").item(), .000025 / 3)
        exact = target.clone().requires_grad_()
        training_loss(exact, target, "native_mpjpe").backward()
        torch.testing.assert_close(exact.grad, torch.zeros_like(exact.grad), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
