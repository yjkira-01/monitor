"""Freeze the all-population robust-loss amendment before large-data training.

The prior two-epoch runtime pilot remains bound to its unchanged original source.
This version responds only to a data preflight, not to comparative model scores.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'research/run_action_window_research.py'
dst=ROOT/'research/run_action_window_formal_20261016.py'
text=src.read_text(encoding='utf-8')
text=text.replace('from awc_training_rule import AWCConcat,',
                  'from robust_action_losses import robust_pose_loss,displacement_loss\nfrom awc_training_rule import AWCConcat,')
text=text.replace("'run_fusion_experiments_gpu.py','awc_training_rule.py']}",
                  "'run_fusion_experiments_gpu.py','awc_training_rule.py','robust_action_losses.py','run_action_window_research.py']}")
text=text.replace("['rgb_single','radar_single','concat','concat_mpjpe','awc_concat']",
                  "['rgb_single','radar_single','concat','concat_mse','concat_mpjpe','awc_concat']")
text=text.replace("'base_loss':'original fused MSE + .25 valid branch MSE',",
                  "'base_loss':'RGB/radar/concat: MPJPE + .25 valid branch MPJPE; concat_mse retains original MSE control; concat_mpjpe and AWC have MPJPE task only',\n        'target_template':'train-only coordinate median; stored under legacy numeric target_mean_mm key',\n        'data_preflight_sha256':sha(ROOT/'artifacts/mri_action_window_preflight_20261016.json'),\n        'environment':{'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'fp32':'IEEE; no mixed precision'},")
text=text.replace("'temporal_loss':'pose MSE; anchor_transport adds .25 adjacent displacement MSE over radar-valid pairs',",
                  "'temporal_loss':'MPJPE; anchor_transport adds .25 adjacent displacement Huber(beta=.05m) over radar-valid pairs',")
text=text.replace("'target_mean_mm':tr['target_mm'].mean(0)","'target_mean_mm':np.median(tr['target_mm'],axis=0)")
text=text.replace("else FusionModel(method)","else FusionModel('concat' if method=='concat_mse' else method)")
text=text.replace("else:loss,_=fusion_loss(out,y[ix],method)",
                  "elif method=='concat_mse':loss,_=fusion_loss(out,y[ix],'concat')\n                            else:loss=robust_pose_loss(out,y[ix])")
text=text.replace("loss=(out['prediction']-y[ix]).square().mean();ml=loss*0",
                  "loss=(out['prediction']-y[ix]).norm(dim=-1).mean();ml=loss*0")
text=text.replace("valid=out['pair_valid'];d=(out['displacement']-delta).square().mean((-1,-2))\n                            ml=(d*valid).sum()/valid.sum().clamp_min(1);loss=loss+.25*ml",
                  "valid=out['pair_valid']\n                            ml=displacement_loss(out['displacement'],delta,valid);loss=loss+.25*ml")
text=text.replace("'motion_MSE_m2':float(np.mean(motion_losses))","'motion_Huber_m':float(np.mean(motion_losses))")
assert text!=src.read_text(encoding='utf-8')
assert "robust_pose_loss(out,y[ix])" in text and 'np.median(tr' in text and 'displacement_loss(out' in text
compile(text,str(dst),'exec')
dst.write_text(text,encoding='utf-8')
receipt={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'formal_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),
 'preflight_sha256':hashlib.sha256((ROOT/'artifacts/mri_action_window_preflight_20261016.json').read_bytes()).hexdigest(),
 'amendment_reason':'Physically impossible released finite targets found before formal training; same robust loss and median template for all primary methods; original MSE retained as control',
 'comparative_scores_used':False}
(ROOT/'research/action_window_formal_generation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))
