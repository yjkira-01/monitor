import unittest

import torch

from fusion_graph_models import GRAPH_METHODS, GraphFusionModel, adjacency
from fusion_models import fusion_loss


class GraphControlChecks(unittest.TestCase):
    def test_adjacencies_stochastic_and_same_parameter_budget(self):
        counts = []
        for mode in GRAPH_METHODS:
            matrix = adjacency(mode)
            torch.testing.assert_close(matrix.sum(-1), torch.ones(17))
            counts.append(sum(p.numel() for p in GraphFusionModel(mode).parameters()))
        self.assertEqual(len(set(counts)), 1)
        self.assertFalse(torch.equal(adjacency('graph_anatomical'), adjacency('graph_permuted')))

    def test_graph_corrective_branch_finite_and_exact_missing_fallback(self):
        torch.set_num_threads(2)
        rgb, points = torch.randn(2, 51), torch.randn(2, 196, 5)
        mask, confidence = torch.ones(2, 196, dtype=torch.bool), torch.ones(2, 17)
        mask[0] = False
        for mode in GRAPH_METHODS:
            model = GraphFusionModel(mode)
            out = model(rgb, points, mask, confidence)
            torch.testing.assert_close(out['fused'][0], out['video'][0])
            loss, _ = fusion_loss(out, torch.zeros(2, 51), mode)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
