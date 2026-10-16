import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=(ROOT/'research/audit_action_window_research.py').read_text(encoding='utf-8')
prior=src.replace("else FusionModel('concat' if method=='concat_mse' else method)",'else FusionModel(method)')
old=ROOT/'artifacts/action_window_runtime_pilot_20261016'
receipt=json.loads((old/'audit.json').read_text())
assert hashlib.sha256(prior.encode()).hexdigest()==receipt['source_sha256']
(old/'auditor_source.py').write_text(prior,encoding='utf-8')
new=ROOT/'artifacts/action_window_robust_runtime_pilot_20261016'
(new/'auditor_source.py').write_text(src,encoding='utf-8')
assert hashlib.sha256(src.encode()).hexdigest()==json.loads((new/'audit.json').read_text())['source_sha256']
path=ROOT/'research/audit_xfi_strong_inputs.py'
(ROOT/'artifacts/xfi_strong_inputs_resumed_20261016/auditor_source.py').write_bytes(path.read_bytes())
print('auditor_sources_archived')
