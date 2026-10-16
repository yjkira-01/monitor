"""Wait for the registered parent, then train/audit its fair raw control.

Local foreground process; no scheduler or automatic restart. Parent failures or
stops prevent launch. Interrupt terminates the active child before recording stop.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/continuous_research_pipeline_20261016/status.json'
DEST = ROOT / 'artifacts/ray_raw_action_window_followthrough_20261016'
STAGES = [('training', 'run_ray_raw_action_window_control_20261016.py'),
          ('audit_report', 'audit_report_ray_raw_action_window_control_20261016.py')]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(value):
    temp = DEST / 'status.tmp'
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, DEST / 'status.json')


def main():
    assert not DEST.exists()
    DEST.mkdir()
    sources = {script: sha(ROOT / 'research' / script) for _, script in STAGES}
    snapshot = DEST / 'source_snapshot'
    snapshot.mkdir()
    for name in sources:
        (snapshot / name).write_bytes((ROOT / 'research' / name).read_bytes())
    status = {'status': 'waiting_for_audited_parent', 'sources': sources, 'stages': [],
              'parent': str(PARENT), 'test_subjects_used': [], 'automatic_restart': False}
    save(status)
    active = None
    try:
        while True:
            parent = json.loads(PARENT.read_text())
            if parent['status'] == 'complete':
                status['parent_complete_status_sha256'] = sha(PARENT)
                break
            if parent['status'] in ['failed', 'stopped_by_user']:
                status['status'] = 'not_launched_parent_' + parent['status']
                save(status)
                return
            time.sleep(10)
        for stage, script in STAGES:
            assert sha(ROOT / 'research' / script) == sources[script]
            status.update(status='running', current_stage=stage)
            save(status)
            print(json.dumps({'event': 'stage_start', 'stage': stage}), flush=True)
            with (DEST / (stage + '.console.log')).open('wb') as stream:
                active = subprocess.Popen([sys.executable, '-u', str(ROOT / 'research' / script)],
                                          cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                status['child_pid'] = active.pid
                save(status)
                code = active.wait()
                active = None
            assert code == 0, f'{stage} failed, exit{code}; see stage log'
            status['stages'].append({'name': stage, 'status': 'complete'})
            save(status)
        status['status'] = 'complete'
        status.pop('child_pid', None)
        save(status)
    except KeyboardInterrupt:
        if active is not None and active.poll() is None:
            active.terminate()
            try:
                active.wait(timeout=15)
            except subprocess.TimeoutExpired:
                active.kill()
                active.wait()
        status['status'] = 'stopped_by_user'
        save(status)
        raise
    except Exception as error:
        status.update(status='failed', error=str(error))
        save(status)
        raise


if __name__ == '__main__':
    main()
