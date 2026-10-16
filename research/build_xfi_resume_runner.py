"""Generate a static continuation preserving the complete training loop."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "research/run_xfi_accuracy_experiments.py"
output = ROOT / "research/run_xfi_accuracy_resume_20261016.py"
receipt = ROOT / "research/xfi_resume_generation_20261016.json"
if output.exists() or receipt.exists():
    raise ValueError("Refusing to overwrite continuation source")
before = source.read_text(encoding="utf-8")
after = before
changes = (
    ('from validate_mri_point_baselines import error_metrics', 'from xfi_resume_support import load_prior, stage_completed\nfrom validate_mri_point_baselines import error_metrics'),
    ('SOURCES = ("run_xfi_accuracy_experiments.py",', 'SOURCES = ("run_xfi_accuracy_resume_20261016.py", "xfi_resume_support.py", "run_xfi_accuracy_experiments.py",'),
    ('    parser.add_argument("--output", type=Path, required=True)', '    parser.add_argument("--output", type=Path, required=True)\n    parser.add_argument("--resume-from", type=Path, required=True)'),
    ('    args = parser.parse_args()', '    args = parser.parse_args()\n    if args.smoke or args.methods != list(METHODS):\n        raise ValueError("Continuation requires the complete original formal design")\n    prior, provenance = load_prior(args.resume_from, args.cache)'),
    ('    snapshot = args.output / "source_snapshot"', '    report["resume_provenance"] = provenance\n    snapshot = args.output / "source_snapshot"'),
    ('        run = {"seed": seed, "models": {}}\n        initial_reference, schedule_reference = None, None',
     '        reused, initial_reference, schedule_reference = stage_completed(seed, args.resume_from, args.output, prior, provenance)\n        run = {"seed": seed, "models": reused}\n        write_json(seed_dir / "training.json", run)'),
    ('        for method in args.methods:\n            started = time.perf_counter()',
     '        for method in args.methods:\n            if method in run["models"]:\n                print(json.dumps({"seed": seed, "method": method, "status": "reused_complete_no_retraining"}), flush=True)\n                continue\n            started = time.perf_counter()'),
)
for old, new in changes:
    if after.count(old) != 1:
        raise ValueError("Original source differs at continuation anchor: " + old)
    after = after.replace(old, new, 1)


def training_loop(text):
    nodes = [node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch"]
    if len(nodes) != 1:
        raise ValueError("Unique original training loop required")
    return ast.dump(nodes[0], include_attributes=False)


if training_loop(before) != training_loop(after):
    raise ValueError("Continuation changed original training")
compile(after, str(output), "exec")
output.write_text(after, encoding="utf-8")
record = {"original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "continuation_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
          "support_sha256": hashlib.sha256((ROOT / "research/xfi_resume_support.py").read_bytes()).hexdigest(),
          "training_loop_AST_identical": True,
          "diff": list(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=source.name, tofile=output.name, lineterm=""))}
receipt.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps({"status": "generated", "training_loop_AST_identical": True, "continuation_sha256": record["continuation_sha256"]}))
