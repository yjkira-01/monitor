"""Generate a resumer while proving the original epoch loop is unchanged."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'research/run_xfi_strong_inputs.py'
destination=ROOT/'research/run_xfi_strong_inputs_resume_20261016.py'
original=source.read_text(encoding='utf-8')
updated=original.replace('from analyze_fusion_trials import independent_metrics',
    'from analyze_fusion_trials import independent_metrics\nfrom xfi_strong_resume_support import prepare_reuse,is_reused')
updated=updated.replace("'run_fusion_experiments_gpu.py']},'runs':[]}",
    "'run_fusion_experiments_gpu.py','xfi_strong_resume_support.py','run_xfi_strong_inputs.py']},'runs':[]}")
updated=updated.replace("    save(args.output/'summary.json',summary)\n    for seed in args.seeds:",
    "    summary['resume']=prepare_reuse(args,summary)\n    save(args.output/'summary.json',summary)\n    for seed in args.seeds:")
updated=updated.replace("            runid=f'seed{seed}_lr{lr:g}';dest=args.output/runid;dest.mkdir()",
    "            runid=f'seed{seed}_lr{lr:g}'\n            if is_reused(runid,summary):\n                print(json.dumps({'event':'reused_complete_no_retraining','run':runid}),flush=True)\n                continue\n            dest=args.output/runid;dest.mkdir()")
if updated==original or 'summary[\'resume\']=prepare_reuse' not in updated:raise ValueError('Resumer transformation failed')
def epoch_ast(text):
    return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch']
assert epoch_ast(updated)==epoch_ast(original)
destination.write_text(updated,encoding='utf-8')
receipt={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'resumer_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
         'epoch_training_loop_AST_unchanged':True}
(ROOT/'research/xfi_strong_resume_generation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))
