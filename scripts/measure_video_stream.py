"""Measure distinct received MJPEG frames on the actual local video API."""
import hashlib
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:5173"


def request(path, token=None, body=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer "+token
    with urlopen(Request(BASE+path, data=json.dumps(body).encode() if body is not None else None, headers=headers), timeout=30) as response:
        return json.load(response)


def main(camera=False):
    token = request('/api/auth/login', body={"username": "admin", "password": "admin123"})['access_token']
    for _ in range(120):
        if request('/api/video/status', token)['analysis']['model_state'] == 'ready':
            break
        time.sleep(.25)
    if camera:
        request('/api/video/camera', token, {})
        for _ in range(60):
            if request('/api/video/status', token)['connected']:
                break
            time.sleep(.2)
        else:
            raise RuntimeError('C60E did not produce frames')
    else:
        request('/api/video/source', token, {"sample_id": "adl"})
    time.sleep(1)
    buffer = b''
    hashes, times = [], []
    with urlopen(Request(BASE+'/api/video/feed?annotated=true',headers={'Authorization':'Bearer '+token}), timeout=15) as response:
        started = time.monotonic()
        while time.monotonic()-started < 4:
            buffer += response.read(4096)
            while b'\r\n\r\n' in buffer:
                header, payload = buffer.split(b'\r\n\r\n',1)
                length = next((int(line.split(b':',1)[1]) for line in header.split(b'\r\n') if line.lower().startswith(b'content-length:')),None)
                if length is None:
                    raise ValueError('Missing MJPEG content length')
                if len(payload)<length+2:
                    break
                jpeg, buffer = payload[:length], payload[length+2:]
                digest = hashlib.sha256(jpeg).hexdigest()
                if not hashes or digest != hashes[-1]:
                    hashes.append(digest)
                    times.append(time.monotonic())
    status = request('/api/video/status', token)
    elapsed = times[-1]-times[0] if len(times)>1 else 0
    result = {'distinct_received_frames':len(hashes),'seconds':elapsed,'received_fps':(len(hashes)-1)/elapsed if elapsed else 0,
              'capture_fps':status.get('display_fps'),'source_fps':status['fps'],'annotated':True,
              'source':status['source'],'source_mode':status['source_mode'],'resolution':[status['width'],status['height']],
              'capture_backend':status.get('capture_backend'),'read_ms':status.get('read_ms'),'processing_ms':status.get('processing_ms'),
              'note':'Measured on localhost; box detection and classification retain independent slower update rates.'}
    if camera:
        assert result['source_mode'] == 'live' and status['analysis']['windows_analyzed'] >= 1, status
    assert result['received_fps']>(25 if camera else 15), result
    (ROOT/('artifacts/c60e_stream_fps.json' if camera else 'artifacts/video_stream_fps.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', action='store_true')
    main(parser.parse_args().camera)
