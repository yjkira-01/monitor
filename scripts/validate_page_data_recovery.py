"""Real UI operations after session/media failures; only radar screenshots, no camera images."""
import asyncio
import base64
import json
import subprocess
import time
from urllib.request import urlopen

import websockets
from validate_video_workbench import ROOT, request


async def main():
    token = (await asyncio.to_thread(request, '/api/auth/login', {'username':'admin','password':'admin123'}))['access_token']
    original_source = (await asyncio.to_thread(request, '/api/data-source'))['source']
    before = await asyncio.to_thread(request, '/api/video/status', None, token)
    assert before['connected'] and before['width'] > 0, 'The existing source must have a decoded frame'
    process = subprocess.Popen([
        'C:/Program Files/Google/Chrome/Application/chrome.exe', '--headless=new', '--no-first-run',
        '--no-default-browser-check', '--remote-debugging-port=9230', '--window-size=1500,1050',
        '--user-data-dir='+str(ROOT/'artifacts/browser-data-recovery'), 'http://127.0.0.1:5173',
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    report = {'camera_images_saved':False, 'radar_fixture_injected':False, 'video_source_changed':False}
    try:
        for _ in range(100):
            try:
                with urlopen('http://127.0.0.1:9230/json', timeout=2) as r:
                    target = next(t for t in json.load(r) if t['type']=='page')
                break
            except (OSError, StopIteration):
                await asyncio.sleep(.1)
        else:
            raise RuntimeError('Chrome did not start')
        async with websockets.connect(target['webSocketDebuggerUrl'], max_size=32*1024*1024) as ws:
            counter, errors = 0, []
            async def cdp(method, params=None):
                nonlocal counter
                counter += 1
                await ws.send(json.dumps({'id':counter,'method':method,'params':params or {}}))
                while True:
                    data = json.loads(await asyncio.wait_for(ws.recv(),25))
                    if data.get('method')=='Runtime.exceptionThrown':
                        errors.append(data['params']['exceptionDetails'])
                    if data.get('id')==counter:
                        assert 'error' not in data, data
                        return data.get('result',{})
            async def evaluate(expression):
                data = await cdp('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':True})
                assert 'exceptionDetails' not in data, data
                return data['result'].get('value')
            async def wait_for(expression, seconds=30):
                deadline = time.monotonic()+seconds
                while time.monotonic()<deadline:
                    if await evaluate(expression):
                        return
                    await asyncio.sleep(.15)
                state = await evaluate('({login:!!document.querySelector(".login-form"),video:document.querySelector(".footage-frame img")?.naturalWidth,errors:[...document.querySelectorAll(".workbench-error")].map(e=>e.innerText)})')
                raise AssertionError('Timed out: '+expression+'; state='+json.dumps(state,ensure_ascii=False))
            async def click(selector):
                hit = await evaluate('(()=>{const b=document.querySelector('+json.dumps(selector)+');if(!b)throw Error("missing control");b.scrollIntoView({block:"center",inline:"center"});const r=b.getBoundingClientRect(),x=r.x+r.width/2,y=r.y+r.height/2;return {x,y,hit:b.contains(document.elementFromPoint(x,y)),disabled:b.disabled}})()')
                assert hit['hit'] and not hit.get('disabled'), hit
                for kind in ['mousePressed','mouseReleased']:
                    await cdp('Input.dispatchMouseEvent',{'type':kind,'button':'left','clickCount':1,'x':hit['x'],'y':hit['y']})
            async def nav(key):
                await click(f'.nav-item:nth-child({key})')
            async def login():
                await click('.login-form button')
                await wait_for('document.querySelector(".footage-frame img")?.naturalWidth>0 && !document.querySelector(".workbench-error")')
            await cdp('Runtime.enable')
            await cdp('Network.enable')
            await wait_for('location.origin==="http://127.0.0.1:5173" && document.readyState!=="loading"')
            await evaluate('localStorage.setItem("access_token","invalid-session-recovery");true')
            await cdp('Page.navigate', {'url':'http://127.0.0.1:5173'})
            await wait_for('!!document.querySelector(".login-form .session-notice")')
            assert await evaluate('!document.querySelector(".care-app") && !localStorage.getItem("access_token")')
            report['invalid_session_returns_to_login'] = True
            await login()
            image_size = await evaluate('({width:document.querySelector(".footage-frame img").naturalWidth,height:document.querySelector(".footage-frame img").naturalHeight,ratio:document.querySelector(".footage-frame").getBoundingClientRect().width/document.querySelector(".footage-frame").getBoundingClientRect().height})')
            assert image_size['width']==before['width'] and image_size['height']==before['height']
            assert abs(image_size['ratio']-before['width']/before['height'])<.02, 'The preview must display the complete capture aspect ratio'
            report['video_frame_dimensions'] = image_size
            report['pointer_login_restores_real_video'] = True
            await wait_for('document.querySelector(".header-status").innerText.includes("实时链路已连接")')
            report['websocket_connected'] = True
            await nav(3)
            await wait_for('!!document.querySelector(".radar-mode-button")')
            if not await evaluate('!!document.querySelector(".radar-demo-notice")'):
                await click('.radar-mode-button')
            await wait_for('document.querySelector(".radar-demo-notice") && document.querySelector(".radar-cloud canvas")?.dataset.points==="192" && document.querySelectorAll(".radar-wave-chart").length===2')
            first = await evaluate('({heart:document.querySelector("[data-testid=radar-heart-rate]").textContent,line:document.querySelector(".radar-wave-chart .recharts-line-curve").getAttribute("d"),updates:document.querySelector(".radar-trends .radar-section-head > span").textContent})')
            await asyncio.sleep(1.2)
            second = await evaluate('({line:document.querySelector(".radar-wave-chart .recharts-line-curve").getAttribute("d"),updates:document.querySelector(".radar-trends .radar-section-head > span").textContent})')
            assert first['line'] != second['line'] and first['updates'] != second['updates']
            report['interactive_demo'] = {'points':192,'waveforms':2,'changing_waveform':True,'changing_trend':True}
            assert await evaluate('!document.querySelector(".footage-frame")')
            metrics = await cdp('Page.getLayoutMetrics')
            size = metrics['cssContentSize']
            shot = await cdp('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True,'clip':{'x':0,'y':0,'width':size['width'],'height':size['height'],'scale':1}})
            (ROOT/'artifacts/radar_demo_usable.png').write_bytes(base64.b64decode(shot['data']))
            await click('.radar-mode-button')
            await wait_for('!document.querySelector(".radar-demo-notice")')
            real = await asyncio.to_thread(request, '/api/edge/radar', None, token)
            if real['stale'] or real['status']=='inactive':
                await wait_for('document.querySelector(".radar-cloud canvas").dataset.points==="0" && document.querySelector("[data-testid=radar-heart-rate]").textContent==="—"')
            else:
                await wait_for('Number(document.querySelector(".radar-cloud canvas").dataset.points)>0 && document.querySelectorAll(".radar-wave-chart").length===2')
                report['real_radar'] = await evaluate('({points:Number(document.querySelector(".radar-cloud canvas").dataset.points),heart:document.querySelector("[data-testid=radar-heart-rate]").textContent,waveforms:document.querySelectorAll(".radar-wave-chart").length,frame:document.querySelector(".radar-acquisition").innerText})')
                real_line = await evaluate('document.querySelector(".radar-wave-chart .recharts-line-curve").getAttribute("d")')
                await wait_for('document.querySelector(".radar-wave-chart .recharts-line-curve")?.getAttribute("d") !== '+json.dumps(real_line))
                report['real_radar']['waveform_updated'] = True
                await click('.radar-cloud-controls button:last-child')
                await wait_for('document.querySelector(".radar-cloud-controls button:last-child").getAttribute("aria-pressed")==="true" && document.querySelector(".radar-cloud canvas").getAttribute("aria-label").includes("原始")')
                report['real_radar']['raw_mode_points'] = await evaluate('Number(document.querySelector(".radar-cloud canvas").dataset.points)')
                await click('.radar-cloud-controls button:first-child')
                await wait_for('document.querySelector(".radar-cloud canvas").getAttribute("aria-label").includes("500") && Number(document.querySelector(".radar-cloud canvas").dataset.points)>0')
                report['real_radar']['processing_toggle'] = True
                report['real_radar']['track_stroke'] = await evaluate('document.querySelector("[data-track-path]")?.getAttribute("stroke-width")')
                assert float(report['real_radar']['track_stroke']) >= 3
                metrics = await cdp('Page.getLayoutMetrics')
                size = metrics['cssContentSize']
                shot = await cdp('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True,'clip':{'x':0,'y':0,'width':size['width'],'height':size['height'],'scale':1}})
                (ROOT/'artifacts/radar_real_usable.png').write_bytes(base64.b64decode(shot['data']))
            await click('.radar-mode-button')
            await wait_for('document.querySelector(".radar-cloud canvas").dataset.points==="192"')
            assert real['source'] != 'mock' and all(v.get('source') != 'mock' for v in real['vitals'])
            vitals = await asyncio.to_thread(request, '/api/vitals?limit=20', None, token)
            assert all(v['device_id'] != 'UI_DEMO' for v in vitals)
            report['demo_never_becomes_real_or_persisted'] = True
            await cdp('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
            await asyncio.sleep(.4)
            dims = await evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
            assert dims['width'] >= dims['scroll'], dims
            report['mobile_dimensions'] = dims
            await cdp('Emulation.clearDeviceMetricsOverride')
            await nav(1)
            await wait_for('document.querySelector(".footage-frame img")?.naturalWidth>0')
            await evaluate('localStorage.setItem("access_token","expired-while-page-open");true')
            await wait_for('!!document.querySelector(".login-form .session-notice")')
            report['expiry_while_open_recovers'] = True
            await login()

            await cdp('Network.setBlockedURLs',{'urls':['*/api/video/feed*']})
            await click('.view-switch button:last-child')
            await wait_for('!!document.querySelector(".video-stream-error")')
            assert await evaluate('!!document.querySelector(".care-app")'), 'Media failure must not expire a valid login'
            await cdp('Network.setBlockedURLs',{'urls':[]})
            await click('.video-stream-error button')
            await wait_for('document.querySelector(".footage-frame img")?.naturalWidth>0 && !document.querySelector(".video-stream-error")')
            report['media_failure_and_retry'] = True
            report['current_spatial_data'] = await evaluate('({people:Number(document.querySelector(".pose-view canvas")?.dataset.activePeople||0),points:Number(document.querySelector(".point-view canvas")?.dataset.points||0),avatar:document.querySelector(".pose-view canvas")?.dataset.avatar})')
            if real['active_source']=='real' and not real['stale']:
                await click('.point-view .view-switch button:last-child')
                await wait_for('Number(document.querySelector(".point-view canvas")?.dataset.points)>0')
                report['overview_real_cloud_points'] = await evaluate('Number(document.querySelector(".point-view canvas").dataset.points)')
                await click('.point-view .radar-cloud-controls button:last-child')
                await wait_for('document.querySelector(".point-view canvas").getAttribute("aria-label").includes("原始")')
                await click('.point-view .radar-cloud-controls button:first-child')
                await wait_for('document.querySelector(".point-view canvas").getAttribute("aria-label").includes("500")')
                report['overview_processing_toggle'] = True
            after = await asyncio.to_thread(request, '/api/video/status', None, token)
            assert all(after.get(k)==before.get(k) for k in ['source','source_mode','paused','ended','width','height'])
            assert after['analysis']['session_id']==before['analysis']['session_id']
            if before.get('paused') or before.get('ended'):
                assert after['position_s']==before['position_s']
            assert (await asyncio.to_thread(request, '/api/data-source'))['source']==original_source
            report['final_video'] = {k:after.get(k) for k in ['source','source_mode','width','height','display_fps']}
            report['existing_playback_preserved'] = True
            assert not errors, errors
            report['uncaught_browser_errors'] = errors
        (ROOT/'artifacts/page_data_recovery_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__=='__main__':
    asyncio.run(main())
