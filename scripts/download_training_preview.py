"""Authenticated, resumable parallel SFTP transfer of the frozen preview checkpoint."""
import concurrent.futures
import getpass
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
from project_archive import ensure_dependency
sys.path.insert(0, str(ensure_dependency('transfer_dependencies')))
import paramiko

REMOTE = '/root/autodl-fs/yanglao-monitor/artifacts/activity_stage_f_internvideo2_base_20261009/preview_20261009'
LOCAL = ROOT/'artifacts/training_preview_20261009'


def main():
    password = getpass.getpass('SSH password (not saved): ')
    LOCAL.mkdir(parents=True, exist_ok=True)

    def connect():
        client = paramiko.SSHClient()
        client.load_host_keys(str(ROOT/'artifacts/cloud_known_hosts'))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect('your-cloud-host', port=49588, username='root', password=password,
                       allow_agent=False, look_for_keys=False, timeout=25, auth_timeout=25, banner_timeout=25)
        transport = client.get_transport()
        transport.set_keepalive(15)
        return client, paramiko.SFTPClient.from_transport(transport, window_size=64*1024**2, max_packet_size=1024**2)

    client, sftp = connect()
    try:
        metadata = json.loads(sftp.open(REMOTE+'/checkpoint.json').read().decode())
        (LOCAL/'checkpoint.json').write_text(json.dumps(metadata, indent=2))
        size = sftp.stat(REMOTE+'/internvideo2_preview.pt').st_size
        assert size == metadata['bytes']
    finally:
        sftp.close(); client.close()
    folder = LOCAL/('parts-'+metadata['checkpoint_sha256'][:12])
    folder.mkdir(exist_ok=True)
    block = 8*1024**2
    chunks = [(i, offset, min(block, size-offset)) for i, offset in enumerate(range(0, size, block))]
    pending = [chunk for chunk in chunks if not (folder/str(chunk[0])).exists() or (folder/str(chunk[0])).stat().st_size != chunk[2]]
    lock, completed = threading.Lock(), [size-sum(chunk[2] for chunk in pending)]
    started = time.monotonic()

    def worker(assigned):
        client, sftp = connect()
        try:
            for index, offset, count in assigned:
                temporary = folder/(str(index)+'.tmp')
                with sftp.open(REMOTE+'/internvideo2_preview.pt', 'rb') as remote, temporary.open('wb') as local:
                    remote.seek(offset)
                    remote.prefetch(offset+count, max_concurrent_requests=64)
                    remaining = count
                    while remaining:
                        data = remote.read(min(1024**2, remaining))
                        if not data:
                            raise EOFError('Incomplete immutable checkpoint')
                        local.write(data); remaining -= len(data)
                        with lock: completed[0] += len(data)
                temporary.replace(folder/str(index))
        finally:
            sftp.close(); client.close()

    resumed = completed[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, pending[i::4]) for i in range(4) if pending[i::4]]
        while any(not future.done() for future in futures):
            elapsed = max(time.monotonic()-started, .001)
            print(f'{completed[0]/size:6.1%} {completed[0]/1024**2:.1f}/{size/1024**2:.1f} MiB '
                  f'{(completed[0]-resumed)/1024**2/elapsed:.2f} MiB/s', flush=True)
            time.sleep(5)
        for future in futures: future.result()
    temporary = LOCAL/'internvideo2_preview.pt.parallel.tmp'
    digest = hashlib.sha256()
    with temporary.open('wb') as output:
        for index, _, _ in chunks:
            with (folder/str(index)).open('rb') as part:
                while data := part.read(1024**2):
                    output.write(data); digest.update(data)
    assert digest.hexdigest() == metadata['checkpoint_sha256'], 'Checkpoint SHA256 mismatch; candidate not published'
    temporary.replace(LOCAL/'internvideo2_preview.pt')
    report = {'bytes': size, 'sha256': digest.hexdigest(), 'seconds': time.monotonic()-started,
              'streams': 4, 'epoch': metadata['epoch'], 'passed': True}
    (LOCAL/'download.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
