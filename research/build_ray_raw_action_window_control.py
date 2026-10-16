"""Freeze the stronger same-width raw control for the 12288-frame protocol.

The small-data result does not replace this large-population comparison. No
new hyperparameter choice: hidden32/embedding16/80ep/AdamW .001 are carried
over. Reuse the audited large OOF predictions, never retrain those experts.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(source_name, target_name, replacements):
    source = ROOT / 'research' / source_name
    target = ROOT / 'research' / target_name
    assert not target.exists()
    text = source.read_text(encoding='utf-8')
    for before, after in replacements:
        assert text.count(before) == 1, (before, text.count(before))
        text = text.replace(before, after)
    compile(text, str(target), 'exec')
    target.write_text(text, encoding='utf-8')
    return {'source': source_name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'generated': target_name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}


records = []
records.append(generate('run_ray_raw_width32_control_fixed_20261016.py',
    'run_ray_raw_action_window_control_20261016.py', [
    ('ray_raw_width32_control_fixed_20261016', 'ray_raw_action_window_control_20261016'),
    ('ray_oof_research_20261016', 'ray_oof_action_windows_20261016'),
    ("CONDS=['clean','dark','noisy_dark','occlusion','video_missing','radar_missing']",
     "CONDS=['clean','noisy_dark','occlusion','video_missing','radar_missing']"),
    ('artifacts/fusion_rgb_expanded_20261013/subject{s}.npz', 'artifacts/mri_action_windows_resumed_20261016/subject{s}.npz'),
    ('for c in CONDS[:4]', 'for c in CONDS[:3]'),
]))
records.append(generate('audit_report_ray_raw_control.py',
    'audit_report_ray_raw_action_window_control_20261016.py', [
    ('ray_raw_observation_control_20261016', 'ray_raw_action_window_control_20261016'),
    ('ray_oof_research_20261016', 'ray_oof_action_windows_20261016'),
    ("CONDS = ['clean', 'dark', 'noisy_dark', 'occlusion', 'video_missing', 'radar_missing']",
     "CONDS = ['clean', 'noisy_dark', 'occlusion', 'video_missing', 'radar_missing']"),
    ('artifacts/fusion_rgb_expanded_20261013/subject{subject}.npz', 'artifacts/mri_action_windows_resumed_20261016/subject{subject}.npz'),
    ("ROOT / f'artifacts/ray_oof_strict_controls_20261016/seed{seed}/{name}/predictions.npz'",
     "PARENT / f'seed{seed}/{name}/predictions.npz'"),
    ('1536帧开发协议', '12288帧开发协议（9216训练、3072验证）'),
    ('；dark不计入S', ''),
]))
receipt = {'status': 'registered_before_large_control_training',
    'question': 'Does explicit geometry improve the larger action-window population against the stronger same-observation same-width neural control?',
    'population': 'Exact audited 12288 development frames, same fold0 and 3 initialization seeds as main experiment.',
    'control': 'hidden32, embedding16, 1843 parameters, 80ep, same OOF payloads/targets/optimizer/cap/mask; no hyperparameter search.',
    'prerequisite': 'Parent main and geometric pipeline complete, including independent audits.',
    'test_subjects_used': [], 'generated': records}
(ROOT / 'research/ray_raw_action_window_control_generation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt))
