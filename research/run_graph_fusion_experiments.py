"""Route the unchanged controlled runner to three declared graph alternatives."""

import hashlib
import json
import sys
from pathlib import Path

import run_fusion_experiments as runner
from fusion_graph_models import GRAPH_METHODS, GraphFusionModel


def main():
    runner.FusionModel = GraphFusionModel
    runner.METHODS = GRAPH_METHODS
    original_writer = runner.write_json

    def snapshot_writer(path, value):
        if path.name == "run_summary.json":
            value["model_factory"] = "fusion_graph_models.GraphFusionModel"
            value["source_sha256"].update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                           (Path(__file__), Path(__file__).with_name("fusion_graph_models.py"))})
            snapshot = path.parent / "source_snapshot"
            snapshot.mkdir(exist_ok=True)
            for name in ("fusion_models.py", "run_fusion_experiments.py", "fusion_graph_models.py", "run_graph_fusion_experiments.py"):
                source = Path(__file__).with_name(name)
                destination = snapshot / name
                if not destination.exists():
                    destination.write_bytes(source.read_bytes())
        original_writer(path, value)

    runner.write_json = snapshot_writer
    # Explicitly route the runner's default list, which was read from METHODS.
    runner.main()
    position = sys.argv.index("--output")
    output = Path(sys.argv[position + 1])
    summary = output / "run_summary.json"
    report = json.loads(summary.read_text(encoding="utf-8"))
    report["model_factory"] = "fusion_graph_models.GraphFusionModel"
    report["graph_control"] = "Same trainable graph residual architecture; self-only vs anatomical vs fixed permuted adjacency"
    report["source_sha256"].update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   (Path(__file__), Path(__file__).with_name("fusion_graph_models.py"))})
    runner.write_json(summary, report)


if __name__ == "__main__":
    main()
