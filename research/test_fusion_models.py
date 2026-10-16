import unittest

import numpy as np
import torch

from fusion_models import METHODS, FUSION_METHODS, FusionModel, fusion_loss, utility_targets


class FusionMechanismChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.rgb = torch.randn(3, 51)
        self.points = torch.randn(3, 196, 5)
        self.mask = torch.ones(3, 196, dtype=torch.bool)
        self.confidence = torch.ones(3, 17)
        self.target = torch.randn(3, 51) * .1

    def test_all_methods_exactly_fall_back_to_available_branch(self):
        for method in FUSION_METHODS:
            model = FusionModel(method)
            radar_missing = model(self.rgb, self.points, torch.zeros_like(self.mask), self.confidence)
            torch.testing.assert_close(radar_missing['fused'], radar_missing['video'])
            video_missing = model(torch.zeros_like(self.rgb), self.points, self.mask, torch.zeros_like(self.confidence))
            torch.testing.assert_close(video_missing['fused'], video_missing['radar'])

    def test_masked_nan_and_point_order_do_not_change_predictions(self):
        self.mask[:, 10:] = False
        for method in ('concat', 'joint_attention', 'utility_gate'):
            model = FusionModel(method)
            expected = model(self.rgb, self.points, self.mask, self.confidence)['fused']
            self.points[~self.mask] = torch.nan
            order = torch.randperm(196)
            actual = model(self.rgb, self.points[:, order], self.mask[:, order], self.confidence)['fused']
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_utility_targets_are_detached_and_compare_ungated_candidate(self):
        video = torch.ones(2, 17, 3, requires_grad=True) * .1
        candidate = torch.zeros(2, 17, 3, requires_grad=True)
        target = torch.zeros_like(video)
        label = utility_targets(video, candidate, target)
        self.assertFalse(label.requires_grad)
        self.assertTrue((label > .99).all())
        reverse = utility_targets(candidate, video, target)
        self.assertTrue((reverse < .01).all())

    def test_finite_gradients_with_missing_cloud_and_confidence(self):
        self.mask[0] = False
        self.confidence[1] = 0
        for method in METHODS:
            model = FusionModel(method)
            out = model(self.rgb, self.points, self.mask, self.confidence)
            loss, _ = fusion_loss(out, self.target, method, torch.Generator().manual_seed(10))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_risk_auxiliary_does_not_send_gradient_to_pose_parameters(self):
        model = FusionModel('utility_gate')
        out = model(self.rgb, self.points, self.mask, self.confidence)
        labels = utility_targets(out['video'], out['candidate'], self.target.reshape(-1, 17, 3))
        torch.nn.functional.binary_cross_entropy_with_logits(out['gate_logits'], labels).backward()
        self.assertTrue(all(p.grad is None for p in model.video_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.residual.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.gate.parameters()))

    def test_independent_single_modality_has_no_other_encoder(self):
        rgb = FusionModel('rgb_matched')
        radar = FusionModel('radar_single')
        self.assertFalse(hasattr(rgb, 'point_encoder'))
        self.assertFalse(hasattr(radar, 'rgb_encoder'))
        ratio = sum(p.numel() for p in rgb.parameters()) / sum(p.numel() for p in FusionModel('concat').parameters())
        self.assertLess(abs(ratio - 1), .01)


if __name__ == '__main__':
    unittest.main()
