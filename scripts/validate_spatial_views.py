"""Real Chrome spatial rendering / pose cadence check (GPU is not disabled)."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

import websockets
from validate_video_workbench import request, ROOT


async def main(label):
    login = await asyncio.to_thread(request, '/api/auth/login', {'username': 'admin', 'password': 'admin123'})
    token = login['access_token']
    process = subprocess.Popen([
        'C:/Program Files/Google/Chrome/Application/chrome.exe', '--headless=new',
        '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=9226',
        '--window-size=1600,1100', '--user-data-dir='+str(ROOT/f'artifacts/browser-spatial-{label}-profile'),
        'http://127.0.0.1:5173'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(100):
            try:
                with urlopen('http://127.0.0.1:9226/json', timeout=2) as r:
                    target = next(t for t in json.load(r) if t['type'] == 'page')
                break
            except (OSError, StopIteration):
                await asyncio.sleep(.1)
        else:
            raise RuntimeError('Chrome did not start')
        async with websockets.connect(target['webSocketDebuggerUrl'], max_size=16*1024*1024) as ws:
            counter = 0
            async def cdp(method, params=None):
                nonlocal counter
                counter += 1
                await ws.send(json.dumps({'id': counter, 'method': method, 'params': params or {}}))
                while True:
                    data = json.loads(await asyncio.wait_for(ws.recv(), 20))
                    if data.get('id') == counter:
                        assert 'error' not in data, data
                        return data.get('result', {})
            async def evaluate(expression):
                data = await cdp('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True})
                assert 'exceptionDetails' not in data, data
                return data['result'].get('value')
            await cdp('Page.enable')
            await cdp('Page.addScriptToEvaluateOnNewDocument', {'source': '''
              window.spatialRenders = new Map();
              for (const cls of [WebGLRenderingContext, WebGL2RenderingContext]) {
                const clear = cls.prototype.clear;
                cls.prototype.clear = function(mask) {
                  if (!(mask & this.COLOR_BUFFER_BIT) || this.getParameter(this.FRAMEBUFFER_BINDING)) return clear.call(this,mask);
                  const item = window.spatialRenders.get(this.canvas) || {frames:0,gpu:''};
                  item.frames++;
                  if (!item.gpu) { const ext=this.getExtension('WEBGL_debug_renderer_info');
                    item.gpu=ext ? this.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'unknown'; }
                  window.spatialRenders.set(this.canvas,item);
                  return clear.call(this,mask);
                };
              }
            '''})
            await evaluate('localStorage.setItem("access_token",'+json.dumps(token)+'); true')
            await cdp('Page.navigate', {'url': 'http://127.0.0.1:5173'})
            await asyncio.sleep(.5)
            for _ in range(100):
                if await evaluate('typeof spatialRenders!=="undefined" && document.querySelectorAll(".spatial-scene canvas").length===2'):
                    break
                await asyncio.sleep(.2)
            else:
                raise RuntimeError('Spatial canvases missing')
            await evaluate('document.querySelector(".spatial-views").scrollIntoView({block:"center"}); true')
            await asyncio.to_thread(request, '/api/video/source', {'sample_id': 'adl'}, token)
            await asyncio.sleep(2)
            before = await evaluate('[...spatialRenders.values()].map(v=>v.frames)')
            initial = await asyncio.to_thread(request, '/api/video/status', None, token)
            initial_frames = initial['analysis'].get('people_detection', {}).get('frames_analyzed', 0)
            started = time.monotonic()
            poses, pose_latencies, capture_fps = set(), [], []
            while time.monotonic()-started < 8:
                status = await asyncio.to_thread(request, '/api/video/status', None, token)
                detection = status['analysis'].get('people_detection', {})
                if detection.get('state') == 'ready':
                    poses.add(detection['frame_s'])
                    if 'inference_ms' in detection:
                        pose_latencies.append(detection['inference_ms'])
                capture_fps.append(status.get('display_fps', 0))
                await asyncio.sleep(.06)
            elapsed = time.monotonic()-started
            assert status['analysis']['generation'] == initial['analysis']['generation'], 'Video source changed during benchmark'
            after = await evaluate('[...spatialRenders.values()].map(v=>({...v}))')
            assert len(after) == 2, after
            report = {'label': label, 'seconds': elapsed, 'gpu_disabled': False,
                      'source': status['source'], 'generation': status['analysis']['generation'],
                      'render_fps': [(item['frames']-before[i])/elapsed for i, item in enumerate(after)],
                      'gpu': [item['gpu'] for item in after], 'observed_pose_updates': len(poses),
                      'pose_update_hz': max(0, len(poses)-1)/elapsed,
                      'pose_processed_hz': (detection.get('frames_analyzed', initial_frames)-initial_frames)/elapsed,
                      'pose_backend_update_hz': detection.get('update_hz'),
                      'pose_inference_ms_mean': sum(pose_latencies)/len(pose_latencies) if pose_latencies else None,
                      'video_capture_fps_mean': sum(capture_fps)/len(capture_fps),
                      'canvases': await evaluate('[...document.querySelectorAll(".spatial-scene canvas")].map(c=>({...c.dataset,width:c.width,height:c.height}))')}
            assert len(poses) >= 2, report
            if label != 'baseline':
                assert all(fps >= 20 for fps in report['render_fps']), report
                assert report['pose_update_hz'] >= 5, report
                assert report['pose_processed_hz'] >= 5, report
                assert report['canvases'][0].get('avatar') == 'ready', report
                async def click(text):
                    await evaluate('(()=>{const b=[...document.querySelectorAll("button")].find(b=>b.textContent.trim()==='+json.dumps(text)+');if(!b)throw Error("missing button");b.click();return true})()')
                await click('暂停播放')
                await asyncio.sleep(.6)
                paused = await asyncio.to_thread(request, '/api/video/status', None, token)
                await asyncio.sleep(.5)
                held = await asyncio.to_thread(request, '/api/video/status', None, token)
                assert paused['position_s'] == held['position_s']
                assert paused['analysis']['people_detection']['frame_s'] == held['analysis']['people_detection']['frame_s']
                await click('雷达'); await asyncio.sleep(.2)
                assert 'OFFLINE / 无有效实测帧' in await evaluate('document.body.innerText')
                await click('模拟'); await click('侧面'); await click('复位')
                await cdp('Emulation.setDeviceMetricsOverride', {'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
                await asyncio.sleep(.5)
                dims = await evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
                assert dims['scroll'] <= dims['width'], dims
                report['pause_resume_and_radar_switch'] = True
                report['mobile_dimensions'] = dims
                await cdp('Emulation.clearDeviceMetricsOverride')
                await asyncio.sleep(.3)
                await evaluate('document.querySelector(".spatial-views").scrollIntoView({block:"center"});true')
            screen = await cdp('Page.captureScreenshot', {'format':'png'})
            (ROOT/f'artifacts/spatial_{label}.png').write_bytes(base64.b64decode(screen['data']))
            (ROOT/f'artifacts/spatial_{label}_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='optimized')
    asyncio.run(main(parser.parse_args().label))
