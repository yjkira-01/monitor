"""Controlled routing to fixed-unit gate models and unchanged semantic losses."""

import hashlib
import json
import sys
from pathlib import Path

import run_fusion_experiments as runner
from fusion_models import fusion_loss
from fusion_scaled_gate_models import SCALED_METHODS, SEMANTIC_METHODS, ScaledGateFusionModel


def mapped_loss(output, target, method, generator=None):
    return fusion_loss(output, target, SEMANTIC_METHODS[method], generator)


def main():
    runner.FusionModel, runner.METHODS, runner.fusion_loss = ScaledGateFusionModel, SCALED_METHODS, mapped_loss
    original_writer = runner.write_json

    def snapshot_writer(path, value):
        if path.name == "run_summary.json":
            value["model_factory"] = "fusion_scaled_gate_models.ScaledGateFusionModel"
            value["source_sha256"].update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                           (Path(__file__), Path(__file__).with_name("fusion_scaled_gate_models.py"))})
            snapshot = path.parent / "source_snapshot"
            snapshot.mkdir(exist_ok=True)
            for name in ("fusion_models.py", "run_fusion_experiments.py", "fusion_scaled_gate_models.py", "run_scaled_gate_experiments.py"):
                source = Path(__file__).with_name(name)
                destination = snapshot / name
                if not destination.exists():
                    destination.write_bytes(source.read_bytes())
        original_writer(path, value)

    runner.write_json = snapshot_writer
    runner.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    report["model_factory"] = "fusion_scaled_gate_models.ScaledGateFusionModel"
    report["semantic_methods"] = SEMANTIC_METHODS
    report["gate_unit_scaling"] = "pose*10, absolute disagreement/bone differences*25, confidence/count unchanged; fixed before this run"
    report["source_sha256"].update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   (Path(__file__), Path(__file__).with_name("fusion_scaled_gate_models.py"))})
    runner.write_json(output / "run_summary.json", report)


if __name__ == "__main__":
    main()
