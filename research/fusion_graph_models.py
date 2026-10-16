"""Controlled skeletal-neighbour hypotheses, all components are known tools.

Same residual gate, same dimensions, same initialization draw count; only the
adjacency differs. This tests anatomy against extra depth and arbitrary mixing.
"""

import torch
from torch import nn

from fusion_models import BONES, FusionModel

GRAPH_METHODS = ("graph_self", "graph_anatomical", "graph_permuted")


def adjacency(mode):
    matrix = torch.eye(17)
    if mode != "graph_self":
        for a, b in BONES + ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)):
            matrix[a, b] = matrix[b, a] = 1.
        if mode == "graph_permuted":
            permutation = torch.randperm(17, generator=torch.Generator().manual_seed(20261013))
            matrix = matrix[permutation][:, permutation]
    return matrix / matrix.sum(-1, keepdim=True)


class GraphResidual(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.register_buffer("adjacency", adjacency(mode))
        self.input = nn.Linear(144, 64)
        self.message = nn.Linear(64, 64)
        self.output = nn.Linear(64, 3)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens):
        features = self.input(tokens).relu()
        mixed = features + self.message(self.adjacency @ features)
        return self.output(mixed.relu())


class GraphFusionModel(FusionModel):
    def __init__(self, method):
        if method not in GRAPH_METHODS:
            raise ValueError("Unknown graph control")
        # Parent keeps its joint_gate computation and common training losses.
        super().__init__("joint_gate")
        self.residual = GraphResidual(method)

