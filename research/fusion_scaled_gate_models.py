"""Unit-scaling controls: conditioning of gate evidence, not a novel gate."""

import torch
from torch import nn

from fusion_models import FusionModel

SCALED_METHODS = ("scaled_joint_gate", "scaled_utility_gate", "scaled_utility_random")
SEMANTIC_METHODS = {m: m.removeprefix("scaled_") for m in SCALED_METHODS}


class ScaledEvidenceGate(nn.Module):
    def __init__(self, gate):
        super().__init__()
        self.network = gate
        # Metre-scale pose -> decimetre-scale; mm-scale differences -> 40mm units.
        # Confidence and point fraction already dimensionless. Fixed, not val-fit.
        self.register_buffer("scale", torch.tensor([10.] * 6 + [25.] * 3 + [1., 1., 25.]))

    def forward(self, evidence):
        return self.network(evidence * self.scale)


class ScaledGateFusionModel(FusionModel):
    def __init__(self, method):
        if method not in SCALED_METHODS:
            raise ValueError("Unknown fixed-unit gate control")
        super().__init__(SEMANTIC_METHODS[method])
        self.gate = ScaledEvidenceGate(self.gate)

