import unittest

import torch

from fusion_models import FusionModel
from fusion_scaled_gate_models import SCALED_METHODS, ScaledGateFusionModel
from run_scaled_gate_experiments import mapped_loss


class GateUnitChecks(unittest.TestCase):
    def test_scale_is_fixed_and_preserves_parameter_count(self):
        for mode in SCALED_METHODS:
            plain = FusionModel(mode.removeprefix('scaled_'))
            scaled = ScaledGateFusionModel(mode)
            self.assertEqual(sum(p.numel() for p in plain.parameters()), sum(p.numel() for p in scaled.parameters()))
            self.assertFalse(scaled.gate.scale.requires_grad)
            self.assertEqual(scaled.gate.scale.shape, (12,))

    def test_wrapper_keeps_utility_loss_and_missing_fallback(self):
        torch.set_num_threads(2)
        rgb, points = torch.randn(2, 51), torch.randn(2, 196, 5)
        mask, confidence = torch.ones(2, 196, dtype=torch.bool), torch.ones(2, 17)
        mask[0] = False
        for mode in SCALED_METHODS:
            model = ScaledGateFusionModel(mode)
            output = model(rgb, points, mask, confidence)
            torch.testing.assert_close(output['fused'][0], output['video'][0])
            loss, components = mapped_loss(output, torch.zeros(2, 51), mode, torch.Generator().manual_seed(7))
            self.assertEqual('utility_BCE' in components, 'utility' in mode)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
