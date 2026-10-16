"""Create a static optimizer control while preserving the original runner."""

import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
original = ROOT / "research/run_xfi_accuracy_experiments.py"
output = ROOT / "research/run_xfi_stability_experiments.py"
receipt = ROOT / "research/xfi_stability_runner_generation.json"
if output.exists() or receipt.exists():
    raise ValueError("Refusing to overwrite a generated control")
before = original.read_text(encoding="utf-8")
after = before
changes = (
    ('SOURCES = ("run_xfi_accuracy_experiments.py",', 'SOURCES = ("run_xfi_stability_experiments.py", "run_xfi_accuracy_experiments.py",'),
    ('choices=METHODS, default=list(METHODS)', 'choices=("native_mse",), default=["native_mse"]'),
    ('lr=.001, weight_decay=.01)', 'lr=.0001, weight_decay=.01)'),
    ('"lr": .001, "weight_decay": .01', '"lr": .0001, "weight_decay": .01'),
    ('"X-Fi source-preserving mRI adaptation and ordinary objective controls; development, no official reproduction"',
     '"X-Fi source-preserving mRI adaptation; fixed low-learning-rate MSE control; development, no novelty claim"'),
    ('    snapshot = args.output / "source_snapshot"',
     '    report["control_protocol_sha256"] = sha(ROOT / "research/xfi_stability_protocol_20261013.md")\n'
     '    snapshot = args.output / "source_snapshot"'),
    ('    for name in SOURCE_FILES:\n',
     '    (snapshot / "xfi_stability_protocol_20261013.md").write_bytes((ROOT / "research/xfi_stability_protocol_20261013.md").read_bytes())\n'
     '    for name in SOURCE_FILES:\n'),
)
for old, new in changes:
    if after.count(old) != 1:
        raise ValueError("Original source no longer matches fixed optimizer-control generation: " + old)
    after = after.replace(old, new, 1)
compile(after, str(output), "exec")
output.write_text(after, encoding="utf-8")
record = {"original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
          "control_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
          "control_protocol_sha256": hashlib.sha256((ROOT / "research/xfi_stability_protocol_20261013.md").read_bytes()).hexdigest(),
          "diff": list(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=original.name, tofile=output.name, lineterm="")),
          "scope": "Single optimizer learning-rate change, static source preserved; no training executed"}
receipt.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps({"status": "generated", "changes": len(changes), "control_sha256": record["control_sha256"]}))
