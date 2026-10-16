"""Reload raw-observation controls and independently reconstruct inference.

No training module is imported. Rebuild rays from the frozen observations and
static camera calibration, then use only NumPy to replay all saved heads.
Subject is the analysis unit; seeds describe initialization sensitivity.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps/research_cuda'))
import numpy as np
from audit_mri import load_labels

PARENT = ROOT / 'artifacts/ray_oof_action_windows_20261016'
CONDS = ['clean', 'noisy_dark', 'occlusion', 'video_missing', 'radar_missing']
SCORE = ['clean', 'noisy_dark', 'occlusion', 'video_missing', 'radar_missing']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def reconstruct_features(data):
    """A separate implementation: construct each row in payload order."""
    result = np.empty((len(data['base']), 17, 29), np.float32)
    with zipfile.ZipFile(ROOT / 'dataset_release.zip') as archive:
        for subject in np.unique(data['subject_ids']):
            subject = int(subject)
            assert subject not in [3, 10, 12, 18]
            camera = load_labels(archive.read(
                f'dataset_release/aligned_data/pose_labels/subject{subject}_all_labels.cpl'))['camera_matrix']
            K, rotation = camera['cam1_intrinsic'], camera['cam1_R']
            observed = arrays(ROOT / f'artifacts/mri_action_windows_resumed_20261016/subject{subject}.npz')
            lookup = {int(frame): i for i, frame in enumerate(observed['frame_ids'])}
            for index in np.flatnonzero(data['subject_ids'] == subject):
                condition = CONDS[int(data['condition_ids'][index])]
                if condition == 'video_missing':
                    pose = np.zeros((17, 3), dtype=float)
                    ray = np.zeros((17, 2), dtype=float)
                else:
                    key = 'clean' if condition == 'radar_missing' else condition
                    pose = observed[key + '_poses'][lookup[int(data['frame_ids'][index])]].astype(float)
                    ray = np.column_stack(((pose[:, 0] - K[0, 2]) / K[0, 0],
                                           (pose[:, 1] - K[1, 2]) / K[1, 1]))
                base = data['base'][index]
                for joint in range(17):
                    result[index, joint] = np.r_[
                        base[joint], base[11], base[12], ray[joint], ray[11], ray[12],
                        rotation.ravel(), pose[joint, 2], pose[11, 2], pose[12, 2],
                        data['depth'][index], data['radar_present'][index]]
    return result


def describe(prediction, data):
    # No metric helper from the training code; explicit Euclidean errors in mm.
    error = np.linalg.norm(prediction.reshape(-1, 17, 3) - data['target'] * 1000, axis=2)
    assert np.isfinite(error).all()
    by_condition, by_subject = {}, {}
    for ci, condition in enumerate(CONDS):
        mask = data['condition_ids'] == ci
        by_condition[condition] = float(error[mask].mean())
    for subject in np.unique(data['subject_ids']):
        mask = data['subject_ids'] == subject
        cm = {c: float(error[mask & (data['condition_ids'] == CONDS.index(c))].mean()) for c in CONDS}
        by_subject[str(int(subject))] = {'clean_mm': cm['clean'], 'S_mm': float(np.mean([cm[c] for c in SCORE]))}
    return {'clean_mm': by_condition['clean'],
            'S_mm': float(np.mean([v['S_mm'] for v in by_subject.values()])),
            'conditions_mm': by_condition, 'subjects': by_subject}


def main():
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'artifacts/ray_raw_action_window_control_20261016'
    summary = json.loads((dest / 'summary.json').read_text())
    assert summary['status'] == 'complete' and not summary['test_subjects_used']
    assert summary['source_sha256'] == sha(dest / 'source.py')
    assert summary['parent_audit_sha256'] == sha(PARENT / 'audit.json')
    assert json.loads((PARENT / 'audit.json').read_text())['status'] == 'complete'
    assert {r['seed'] for r in summary['runs']} == {7, 17, 27}
    records, checks = [], []
    for run in summary['runs']:
        seed = run['seed']
        data = arrays(PARENT / f'seed{seed}/validation_payload.npz')
        assert sorted(np.unique(data['subject_ids']).tolist()) == [5, 6, 19, 20]
        sd = dest / f'seed{seed}'
        assert sha(sd / 'weights.npz') == run['weights_sha256']
        assert sha(sd / 'predictions.npz') == run['predictions_sha256']
        weights = arrays(sd / 'weights.npz')
        assert sum(w.size for w in weights.values()) == run['parameters']
        features = reconstruct_features(data).astype(float)
        identities = np.broadcast_to(weights['joint'], (len(features), *weights['joint'].shape))
        inputs = np.concatenate((features, identities), axis=2)
        hidden = np.maximum(inputs @ weights['network.0.weight'].T + weights['network.0.bias'], 0)
        residual = .1 * np.tanh(hidden @ weights['network.2.weight'].T + weights['network.2.bias'])
        replay = (data['base'] + residual * data['eligible'][..., None]).reshape(-1, 51) * 1000
        pred = arrays(sd / 'predictions.npz')['prediction_mm']
        difference = float(np.max(np.abs(replay - pred)))
        assert difference <= .01, difference
        fallback = ~data['eligible']
        fallback_difference = float(abs(pred.reshape(-1, 17, 3)[fallback] - data['base'][fallback] * 1000).max())
        assert fallback_difference <= .001
        raw = describe(pred, data)
        metric_difference = max(abs(raw['conditions_mm'][c] - run['metrics'][c]['MPJPE_mm']) for c in CONDS)
        assert metric_difference < .0001
        methods = {'raw_observations': pred,
                   'recentered_base': data['base'].reshape(-1, 51) * 1000}
        for name in ['free_radar_residual', 'radar_ray_gate', 'constant_ray_gate']:
            methods[name] = arrays(PARENT / f'seed{seed}/{name}/predictions.npz')['prediction_mm']
        for name in ['free_constant_residual', 'free_no_ray_residual']:
            methods[name] = arrays(PARENT / f'seed{seed}/{name}/predictions.npz')['prediction_mm']
        for name, prediction in methods.items():
            records.append({'seed': seed, 'method': name, **describe(prediction, data)})
        checks.append({'seed': seed, 'parameters': run['parameters'], 'NumPy_replay_max_mm': difference,
                       'missing_joint_fallback_max_mm': fallback_difference, 'metric_difference_mm': metric_difference,
                       'payload_sha256': sha(PARENT / f'seed{seed}/validation_payload.npz')})
    aggregate = {}
    for name in methods:
        selected = [r for r in records if r['method'] == name]
        aggregate[name] = {key: {'mean': float(np.mean([r[key] for r in selected])),
                                 'initialization_sd': float(np.std([r[key] for r in selected], ddof=1))}
                           for key in ['clean_mm', 'S_mm']}
    gains = {}
    for control in ['raw_observations', 'recentered_base', 'free_constant_residual']:
        deltas = []
        for seed in [7, 17, 27]:
            a = next(r for r in records if r['method'] == control and r['seed'] == seed)
            b = next(r for r in records if r['method'] == 'free_radar_residual' and r['seed'] == seed)
            deltas.append({'seed': seed, 'clean_mm': a['clean_mm'] - b['clean_mm'], 'S_mm': a['S_mm'] - b['S_mm'],
                           'subjects': {s: {k: a['subjects'][s][k] - b['subjects'][s][k] for k in ['clean_mm', 'S_mm']} for s in a['subjects']}})
        gains[control] = {'per_initialization': deltas,
            'per_subject_mean': {s: {k: float(np.mean([d['subjects'][s][k] for d in deltas])) for k in ['clean_mm', 'S_mm']} for s in ['5', '6', '19', '20']}}
    audit = {'status': 'complete', 'runs': checks, 'test_subjects_used': [],
             'scope': 'External NumPy reload; independently rebuild raw features from archived observations and K/R; parent OOF/depth audit inherited.',
             'source_sha256': sha(Path(__file__)), 'summary_sha256': sha(dest / 'summary.json')}
    (dest / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (dest / 'auditor_source.py').write_bytes(Path(__file__).read_bytes())
    out = dest / 'reporting'
    out.mkdir(exist_ok=True)
    report = {'status': 'complete', 'aggregate': aggregate, 'records': records, 'geometry_gains': gains,
              'audit_sha256': sha(dest / 'audit.json'), 'test_subjects_used': []}
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# 相同观测信息下的几何表示对照', '',
        '12288帧开发协议（9216训练、3072验证），一个主体划分、三个初始化。单位mm，越低越好。S为五条件、四验证主体等权平均。', '',
        '| 方法 | 正常光照MPJPE | 五条件S |', '|---|---:|---:|']
    for name, a in aggregate.items():
        lines.append(f"| {name} | {a['clean_mm']['mean']:.6f} ± {a['clean_mm']['initialization_sd']:.6f} | {a['S_mm']['mean']:.6f} ± {a['S_mm']['initialization_sd']:.6f} |")
    lines += ['', '±表示三次初始化的样本标准差，不代表独立主体划分或置信区间。每人/每初始化的配对收益完整保存在results.json。', '',
        'raw_observations提供原始归一化2D及双髋、完整静态相机R、预测3D及双髋、置信度、估计雷达距离和可用性；几何模型显式计算三维候选。二者共享OOF预测、监督、轮数、batch、优化器和输出修正上限。', '',
        f"本次raw网络参数{checks[0]['parameters']}，隐藏层{weights['network.0.bias'].size}维、关节嵌入{weights['joint'].shape[1]}维；几何网络1395参数，隐藏层32维、嵌入16维。参数匹配的raw隐藏22对照和更大预算的raw隐藏32对照分别控制参数量与隐藏宽度，不能据此断言所有神经表示都不及几何。", '',
        '全部raw预测经独立重载、从观测重建特征、NumPy前向、缺失关节回退和指标复算。父实验OOF主体分离及训练内雷达距离拟合审计复用，最终测试主体没有参与。此表证明开发误差与配对差异，不证明方法新颖性。']
    (out / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'audit': audit, 'aggregate': aggregate, 'geometry_gains': gains}, ensure_ascii=False))


if __name__ == '__main__':
    main()
