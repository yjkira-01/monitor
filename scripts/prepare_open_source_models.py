"""Fetch official, versioned real-time model assets. Never starts training."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/open_source_research_20261009'
BYTE_REVISION = 'cc5965d8a14e06ca4f1a63f87423a09df06255ce'
POSE_URL = ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
            'rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip')
RTMO_URL = ('https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
            'rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip')
RTMO_SHA256 = '750d872151ff4652f02c738efc3a547e112ce9b688fe920bc17f948e8c3afdac'


def fetch(url: str) -> bytes:
    error = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'CareMonitorResearch/1.0'})
            with urllib.request.urlopen(req, timeout=40) as response:
                return response.read()
        except Exception as exc:
            error = exc
            time.sleep(min(3, attempt + .5))
    raise RuntimeError(f'Download failed: {url}: {error}')


def github_file(repo: str, path: str, revision: str) -> bytes:
    return fetch(f'https://raw.githubusercontent.com/{repo}/{revision}/{path}')


def prepare_tracker() -> None:
    target = ROOT / 'desktop/ByteTrack'
    target.mkdir(exist_ok=True)
    provenance = target / 'provenance.json'
    if provenance.exists():
        record = json.loads(provenance.read_text(encoding='utf-8'))
        changes = {r['file']: r['sha256'] for r in record.get('local_modifications', [])}
        if record.get('revision') == BYTE_REVISION and all(
                (target / r['file']).is_file() and hashlib.sha256((target / r['file']).read_bytes()).hexdigest()
                == changes.get(r['file'], r['sha256']) for r in record['files']):
            print('Reusing verified ByteTrack sources, including recorded local modifications.', flush=True)
            return
        raise ValueError('ByteTrack sources have unrecorded changes; refusing to overwrite them.')
    archive = OUT / 'bytetrack.zip'
    if not archive.exists():
        archive.write_bytes(fetch(f'https://codeload.github.com/arturbuszka/ByteTrack.NET/zip/{BYTE_REVISION}'))
    records = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith('/'):
                continue
            path = '/'.join(name.split('/')[1:])
            if not (path.startswith('src/ByteTrack/') or path in ('LICENSE', 'NOTICE', 'README.md')):
                continue
            relative = path.removeprefix('src/ByteTrack/')
            dest = target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(name)
            dest.write_bytes(data)
            records.append({'file': relative, 'sha256': hashlib.sha256(data).hexdigest()})
    (target / 'provenance.json').write_text(json.dumps({
        'source': 'https://github.com/arturbuszka/ByteTrack.NET', 'revision': BYTE_REVISION,
        'license': 'MIT', 'files': records,
    }, indent=2), encoding='utf-8')
    print('ByteTrack C# sources fetched and hashed.', flush=True)


def prepare_pose() -> None:
    archive = OUT / 'rtmpose_l_body7.zip'
    if not archive.exists():
        data = fetch(POSE_URL)
        tmp = archive.with_suffix('.download')
        tmp.write_bytes(data)
        with zipfile.ZipFile(tmp) as check:
            if check.testzip() is not None:
                raise ValueError('Corrupt RTMPose archive')
        tmp.replace(archive)
    assets = ROOT / 'desktop/CareMonitor/Assets/Realtime'
    assets.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith('.onnx')]
        if len(names) != 1:
            raise ValueError(f'Ambiguous ONNX models: {names}')
        model = z.read(names[0])
        (assets / 'rtmpose.onnx').write_bytes(model)
        for name in z.namelist():
            if name.endswith('.json'):
                (OUT / Path(name).name).write_bytes(z.read(name))
    provenance = {'name': 'RTMPose-l body7 256x192', 'source': POSE_URL,
                  'sha256': hashlib.sha256(model).hexdigest(), 'bytes': len(model),
                  'input': [1, 3, 256, 192], 'keypoints': 17, 'split_ratio': 2.0,
                  'license': 'Apache-2.0', 'benchmark': 'COCO pose AP; not fall accuracy'}
    (assets / 'rtmpose.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    shutil.copyfile(ROOT / 'artifacts/pretrained/people_rf_detr_medium.onnx', assets / 'people.onnx')
    baseline = json.loads((assets.parent / 'models.json').read_text(encoding='utf-8'))
    baseline['people'] = {'sha256': hashlib.sha256((assets / 'people.onnx').read_bytes()).hexdigest(),
                          'input': [1, 3, 576, 576], 'source': 'RF-DETR Medium', 'box_scale': 320}
    baseline['rtmpose'] = provenance
    baseline['note'] = 'Realtime profile: RF-DETR and RTMPose; existing R3D checkpoint retained as a comparison.'
    (assets / 'models.json').write_text(json.dumps(baseline, indent=2), encoding='utf-8')
    print(json.dumps(provenance), flush=True)


def research_action() -> None:
    repo = 'open-mmlab/mmaction2'
    revision_file = OUT / 'mmaction_revision.json'
    revision = (json.loads(revision_file.read_text(encoding='utf-8'))['revision'] if revision_file.exists()
                else 'v1.2.0')
    paths = {
        'stgcn_readme': 'configs/skeleton/stgcnpp/README.md',
        'stgcn_config': 'configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py',
        'ntu_labels': 'tools/data/skeleton/label_map_ntu60.txt',
        'mmaction_license': 'LICENSE',
    }
    for name, path in paths.items():
        if (OUT / (name + '.txt')).exists():
            continue
        (OUT / (name + '.txt')).write_bytes(github_file(repo, path, revision))
    (OUT / 'mmaction_revision.json').write_text(json.dumps({'source': repo, 'revision': revision}), encoding='utf-8')
    print('Official STGCN++ configuration and class labels fetched.', flush=True)


def prepare_rtmo() -> None:
    assets = ROOT / 'desktop/CareMonitor/Assets/Realtime'
    model_path = assets / 'rtmo.onnx'
    if model_path.exists() and hashlib.sha256(model_path.read_bytes()).hexdigest() == RTMO_SHA256:
        print('Reusing verified RTMO-m.', flush=True)
        return
    archive = ROOT / 'artifacts/action_upgrade_20261010/rtmo-m.zip'
    if not archive.exists():
        archive = OUT / 'rtmo-m.zip'
        if not archive.exists():
            archive.write_bytes(fetch(RTMO_URL))
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith('.onnx')]
        if len(names) != 1:
            raise ValueError('Ambiguous RTMO model archive')
        model = z.read(names[0])
    if hashlib.sha256(model).hexdigest() != RTMO_SHA256:
        raise ValueError('RTMO checksum mismatch')
    model_path.write_bytes(model)
    (assets / 'rtmo.json').write_text(json.dumps(dict(
        name='RTMO-m body7 640x640', source=RTMO_URL, sha256=RTMO_SHA256,
        bytes=len(model), license='Apache-2.0', input=[1,3,640,640], keypoints=17,
        channel_order='BGR', input_range='0..255', resize='preserve aspect, top-left pad 114',
        note='Person detection and pose estimation, not a trained action classifier'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--research-actions', action='store_true', help='Also fetch optional, unaccepted action research assets')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    prepare_tracker()
    prepare_pose()
    prepare_rtmo()
    if args.research_actions:
        research_action()
