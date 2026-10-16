"""Adapt the independently verified small-protocol audit to large data paths."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'research/audit_ray_oof_research.py';dst=ROOT/'research/audit_ray_oof_action_windows_20261016.py'
s=src.read_text(encoding='utf-8')
s=s.replace('from run_ray_oof_research import payload,features,PIXEL,CONDS,TRAIN,VAL,FOLDS',
    'from run_ray_oof_action_windows_20261016 import payload,features,PIXEL,CONDS,TRAIN,VAL,FOLDS\nfrom ray_action_window_data import load_action_window_rows')
s=s.replace("'artifacts/ray_oof_research_20261016'","'artifacts/ray_oof_action_windows_20261016'")
s=s.replace("'artifacts/fusion_method_research_20261013_expanded_centroid'","'artifacts/action_window_main_20261016'")
s=s.replace("OLD/'run_summary.json'","OLD/'summary.json'")
s=s.replace("cache=ROOT/'artifacts/fusion_rgb_expanded_20261013'","cache=ROOT/'artifacts/mri_action_windows_resumed_20261016'")
s=s.replace("for s,h in old['cache_sha256'].items():", "for s,h in json.loads((OLD/'fold0/data.json').read_text())['cache_sha256'].items():")
s=s.replace("_,rows=load_data(cache,PIXEL,'visible_centroid')","_,rows=load_action_window_rows(cache)")
s=s.replace("OLD/f'seed{seed}/concat_weights.npz'","OLD/f'fold0/seed{seed}/concat/weights.npz'")
s=s.replace("OLD/'normalization.npz'","OLD/'fold0/normalization.npz'")
s=s.replace("source='constant_candidate' if method=='constant_ray_gate' else 'radar_candidate'",
    "source='constant_candidate' if method in ('constant_ray_gate','free_constant_residual') else 'radar_candidate'\n            if method=='free_no_ray_residual':\n                validation=dict(validation);validation['radar_candidate']=validation['base'].copy()")
s=s.replace("RaySelector(method=='free_radar_residual')","RaySelector(method.startswith('free_'))")
s=s.replace("if method=='free_radar_residual':pred=", "if method.startswith('free_'):pred=")
s=s.replace("'models':9", "'models':15").replace('all nine learned selectors','all fifteen learned selectors')
assert 'load_action_window_rows(cache)' in s and 'fold0/seed{seed}/concat/weights.npz' in s
compile(s,str(dst),'exec');dst.write_text(s,encoding='utf-8')
print(str(dst))
