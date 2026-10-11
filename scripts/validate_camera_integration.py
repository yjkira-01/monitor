"""C60E browser checks; save numeric metadata only, never camera images."""
import asyncio
import json
import re
import subprocess
import time
from urllib.request import urlopen

import websockets
from validate_video_workbench import ROOT, request


async def main():
    token = (await asyncio.to_thread(request, '/api/auth/login', {'username': 'admin', 'password': 'admin123'}))['access_token']
    process = subprocess.Popen([
        'C:/Program Files/Google/Chrome/Application/chrome.exe', '--headless=new',
        '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=9227',
        '--window-size=1600,1100', '--user-data-dir='+str(ROOT/'artifacts/browser-c60e-profile'),
        'http://127.0.0.1:5173'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(100):
            try:
                with urlopen('http://127.0.0.1:9227/json', timeout=2) as response:
                    target = next(t for t in json.load(response) if t['type'] == 'page')
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
            async def wait_for(expression, seconds=30):
                deadline = time.monotonic()+seconds
                while time.monotonic() < deadline:
                    if await evaluate(expression):
                        return
                    await asyncio.sleep(.2)
                raise AssertionError('Browser check timed out: '+expression)
            async def select(value):
                await evaluate('(()=>{const s=document.querySelector(".source-selector select");s.value='+json.dumps(value)+';s.dispatchEvent(new Event("change",{bubbles:true}));return true})()')
            async def status():
                return await asyncio.to_thread(request, '/api/video/status', None, token)
            await cdp('Page.enable')
            # Mimic an already-open page from an old backend with a higher source generation.
            await cdp('Page.addScriptToEvaluateOnNewDocument', {'source': '''
              window.cameraOldSession=false;
              const poison=data=>{
                if(!window.cameraOldSession)return data;
                const edit=v=>{v.session_id='previous-backend';v.generation=(v.generation||0)+1000;v.updated_at_ms=Date.now()+60000;};
                const a=data.analysis||data;edit(a);
                if(a.people_detection){edit(a.people_detection);a.people_detection.frame_s=9999;}
                if(a.people){a.frame_s=9999;}
                return data;
              };
              const nativeFetch=window.fetch;
              window.fetch=async(...args)=>{
                const r=await nativeFetch(...args);
                if(window.cameraOldSession && /\\/api\\/video\\/(status|pose)$/.test(String(args[0])) && r.ok)
                  return new Response(JSON.stringify(poison(await r.json())),{status:r.status,headers:{'Content-Type':'application/json'}});
                return r;
              };
              const NativeWebSocket=window.WebSocket;
              window.WebSocket=class extends NativeWebSocket {
                constructor(...args){super(...args);this.addEventListener('message',event=>{
                  const message=JSON.parse(event.data);
                  if(window.cameraOldSession && message.type==='pose'){
                    message.data=poison(message.data);Object.defineProperty(event,'data',{value:JSON.stringify(message)});
                  }
                });}
              };
            '''})
            await evaluate('localStorage.setItem("access_token",'+json.dumps(token)+');true')
            await cdp('Page.navigate', {'url': 'http://127.0.0.1:5173'})
            await wait_for('document.querySelector(".source-selector option[value=camera]") && document.querySelectorAll(".spatial-scene canvas").length===2')
            await select('adl')
            await wait_for('!!document.querySelector(".playback-progress")')
            replay = await status()
            assert replay['source_mode'] == 'file_replay'
            await select('camera')
            await wait_for('document.querySelector(".footage-heading").innerText.includes("C60E") && document.querySelector(".footage-frame img").naturalWidth===640 && !document.querySelector(".playback-progress")')
            await wait_for('document.querySelector(".window-chart") && !document.querySelector(".workbench-error")')
            live = await status()
            assert live['connected'] and live['source_mode'] == 'live' and live['width'] == 640 and live['height'] in (360,480)
            assert live['analysis']['generation'] > replay['analysis']['generation']
            assert live['analysis']['people_detection']['session_id'] == live['analysis']['session_id']
            await wait_for(f'document.querySelector(".footage-frame img").naturalHeight==={live["height"]}')
            assert await evaluate('![...document.querySelectorAll(".footage-controls button")].some(b=>/暂停|重播/.test(b.textContent))')
            await evaluate('(()=>{[...document.querySelectorAll("button")].find(b=>b.textContent.trim()==="重新连接 C60E").click();return true})()')
            await wait_for('!document.querySelector(".workbench-error") && !document.querySelector(".source-selector select").disabled && document.querySelector(".footage-frame img").naturalWidth===640')
            reconnected = await status()
            assert reconnected['analysis']['generation'] > live['analysis']['generation']
            await evaluate('window.cameraOldSession=true;true')
            await wait_for('document.querySelector(".footage-frame img").src.includes("previous-backend")')
            await wait_for('document.querySelector(".pose-view .spatial-source").innerText.includes("9999")')
            await evaluate('window.cameraOldSession=false;true')
            await wait_for('!document.querySelector(".footage-frame img").src.includes("previous-backend") && !document.querySelector(".pose-view .spatial-source").innerText.includes("9999")')
            await evaluate('document.querySelector(".spatial-views").scrollIntoView({block:"center"});true')
            await asyncio.sleep(1)
            before = await status()
            started = time.monotonic()
            frontend_frames, capture_rates = set(), []
            while time.monotonic()-started < 6:
                value = await evaluate('document.querySelector(".pose-view .spatial-source").innerText')
                frame = re.search(r'· ([\d.]+)s', value)
                if frame:
                    frontend_frames.add(frame.group(1))
                current = await status()
                capture_rates.append(current.get('display_fps', 0))
                await asyncio.sleep(.15)
            elapsed = time.monotonic()-started
            after = await status()
            a, b = before['analysis'], after['analysis']
            assert b['generation'] == a['generation'] and b['session_id'] == a['session_id']
            assert b['windows_analyzed'] > a['windows_analyzed'] and b['people_detection']['frames_analyzed'] > a['people_detection']['frames_analyzed']
            assert len(frontend_frames) >= 5, frontend_frames
            canvases = await evaluate('[...document.querySelectorAll(".spatial-scene canvas")].map(c=>({...c.dataset,width:c.width,height:c.height}))')
            assert len(canvases) == 2 and canvases[0].get('avatar') == 'ready', canvases
            assert await evaluate('![...document.querySelectorAll(".spatial-empty")].some(e=>/不可用|失败/.test(e.innerText))')
            await cdp('Emulation.setDeviceMetricsOverride', {'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
            await asyncio.sleep(.5)
            dims = await evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
            assert dims['scroll'] <= dims['width'], dims
            report = {'source': after['source'], 'resolution': [after['width'], after['height']],
                      'capture_backend': after.get('capture_backend'), 'source_switch': True, 'reconnect': True,
                      'old_session_recovery': True, 'live_controls': True, 'mobile_dimensions': dims,
                      'seconds': elapsed, 'capture_fps_mean': sum(capture_rates)/len(capture_rates),
                      'model_windows_added': b['windows_analyzed']-a['windows_analyzed'], 'model_inference_ms': b.get('inference_ms'),
                      'pose_processed_hz': (b['people_detection']['frames_analyzed']-a['people_detection']['frames_analyzed'])/elapsed,
                      'pose_reported_hz': b['people_detection'].get('update_hz'), 'people_detected': len(b['people_detection']['people']),
                      'frontend_pose_samples': len(frontend_frames), 'canvases': canvases, 'camera_images_saved': False,
                      'model_revision': b.get('model_revision'), 'partial_body_trained': b.get('partial_body_trained'), 'session_id': b.get('session_id')}
            (ROOT/'artifacts/c60e_browser_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        if (await asyncio.to_thread(request, '/api/video/status', None, token))['source_mode'] != 'live':
            await asyncio.to_thread(request, '/api/video/camera', {}, token)


if __name__ == '__main__':
    asyncio.run(main())
