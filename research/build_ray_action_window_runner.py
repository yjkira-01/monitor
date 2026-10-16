"""Apply the tested OOF algorithm to the large, robust all-population protocol."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'research/run_ray_oof_research.py';dst=ROOT/'research/run_ray_oof_action_windows_20261016.py'
s=src.read_text(encoding='utf-8')
s=s.replace('from analyze_fusion_trials import strict_load_weights,independent_metrics',
    'from analyze_fusion_trials import strict_load_weights,independent_metrics\nfrom ray_action_window_data import load_action_window_rows\nfrom robust_action_losses import robust_pose_loss')
s=s.replace("CONDS=['clean','dark','noisy_dark','occlusion','video_missing','radar_missing']\nPIXEL=CONDS[:4]",
    "CONDS=['clean','noisy_dark','occlusion','video_missing','radar_missing']\nPIXEL=CONDS[:3]")
s=s.replace("'artifacts/ray_oof_research_20261016'","'artifacts/ray_oof_action_windows_20261016'")
s=s.replace("'artifacts/fusion_method_research_20261013_expanded_centroid'","'artifacts/action_window_main_20261016'")
s=s.replace("cache=ROOT/'artifacts/fusion_rgb_expanded_20261013'\n    split,rows=load_data(cache,PIXEL,'visible_centroid')",
    "cache=ROOT/'artifacts/mri_action_windows_resumed_20261016'\n    assert json.loads((OLD/'audit.json').read_text())['status']=='complete'\n    split,rows=load_action_window_rows(cache)")
s=s.replace("'fusion_models.py','run_fusion_experiments_gpu.py']}",
    "'fusion_models.py','run_fusion_experiments_gpu.py','ray_action_window_data.py','robust_action_losses.py','run_ray_oof_research.py']}")
s=s.replace("'methods':['radar_ray_gate','constant_ray_gate','free_radar_residual']",
    "'methods':['radar_ray_gate','constant_ray_gate','free_radar_residual','free_constant_residual','free_no_ray_residual']")
s=s.replace("'old_summary_sha256':sha(OLD/'run_summary.json')",
    "'old_summary_sha256':sha(OLD/'summary.json'),'cache_protocol_sha256':sha(cache/'protocol.json'),'population':'all finite targets, same large action windows as all primary baselines','OOF_objective':'MPJPE plus .25 valid branch MPJPE, train coordinate median template, same as large robust concat'")
s=s.replace("np.maximum(stack(train,'clean').std(0),1e-8)","np.maximum(stack(train,'clean').std(0),1e-6)")
s=s.replace("'target_mean_mm':stack(train,'target').mean(0)","'target_mean_mm':np.median(stack(train,'target'),axis=0)")
s=s.replace("loss,_=fusion_loss(model(*batch),target[ix],'concat');assert torch.isfinite(loss)",
    "loss=robust_pose_loss(model(*batch),target[ix]);assert torch.isfinite(loss)")
s=s.replace("OLD/f'seed{seed}/concat_weights.npz'","OLD/f'fold0/seed{seed}/concat/weights.npz'")
s=s.replace("OLD/'normalization.npz'","OLD/'fold0/normalization.npz'")
s=s.replace("source='constant_candidate' if method=='constant_ray_gate' else 'radar_candidate'",
    "source='constant_candidate' if method in ('constant_ray_gate','free_constant_residual') else 'radar_candidate'\n            if method=='free_no_ray_residual':\n                data=dict(data);validation=dict(validation)\n                data['radar_candidate']=data['base'].copy();validation['radar_candidate']=validation['base'].copy()")
s=s.replace("RaySelector(method=='free_radar_residual')","RaySelector(method.startswith('free_'))")
assert "load_action_window_rows(cache)" in s and 'robust_pose_loss(model(*batch)' in s and 'free_constant_residual' in s
compile(s,str(dst),'exec');dst.write_text(s,encoding='utf-8')
receipt={'base_source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),
 'changes':'Large audited cache, robust common OOF objective/template, three actual pixel conditions, source base from audited large fold0; add same-capacity free constant-ray control; no final-test data'}
(ROOT/'research/ray_action_window_generation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))
