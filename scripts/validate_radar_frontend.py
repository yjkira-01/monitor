"""Radar-only screenshots and isolated synthetic browser fixtures; no camera images saved."""
import asyncio
import base64
import json
import subprocess
import time
from urllib.request import urlopen

import websockets
from validate_video_workbench import ROOT, request


async def main():
    token = (await asyncio.to_thread(request, '/api/auth/login', {'username': 'admin', 'password': 'admin123'}))['access_token']
    original_source = (await asyncio.to_thread(request, '/api/data-source'))['source']
    before_video = await asyncio.to_thread(request, '/api/video/status', None, token)
    process = subprocess.Popen([
        'C:/Program Files/Google/Chrome/Application/chrome.exe', '--headless=new', '--no-first-run',
        '--no-default-browser-check', '--remote-debugging-port=9228', '--window-size=1600,1100',
        '--user-data-dir='+str(ROOT/'artifacts/browser-radar-profile'), 'http://127.0.0.1:5173',
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    report = {'synthetic_fixture_is_browser_only': True, 'camera_images_saved': False}
    try:
        for _ in range(100):
            try:
                with urlopen('http://127.0.0.1:9228/json', timeout=2) as response:
                    target = next(t for t in json.load(response) if t['type'] == 'page')
                break
            except (OSError, StopIteration):
                await asyncio.sleep(.1)
        else:
            raise RuntimeError('Chrome did not start')
        async with websockets.connect(target['webSocketDebuggerUrl'], max_size=32*1024*1024) as ws:
            counter = 0
            errors = []
            async def cdp(method, params=None):
                nonlocal counter
                counter += 1
                await ws.send(json.dumps({'id': counter, 'method': method, 'params': params or {}}))
                while True:
                    data = json.loads(await asyncio.wait_for(ws.recv(), 25))
                    if data.get('method') == 'Runtime.exceptionThrown':
                        errors.append(data['params']['exceptionDetails'])
                    if data.get('id') == counter:
                        assert 'error' not in data, data
                        return data.get('result', {})
            async def evaluate(expression):
                data = await cdp('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True})
                assert 'exceptionDetails' not in data, data
                return data['result'].get('value')
            async def wait_for(expression, seconds=25):
                deadline = time.monotonic()+seconds
                while time.monotonic() < deadline:
                    if await evaluate(expression):
                        return
                    await asyncio.sleep(.15)
                raise AssertionError('Browser check timed out: '+expression)
            async def nav(label):
                await evaluate('(()=>{[...document.querySelectorAll(".nav-item")].find(b=>b.querySelector("strong").textContent==='+json.dumps(label)+').click();return true})()')
            async def select(selector, value):
                await evaluate('(()=>{const s=document.querySelector('+json.dumps(selector)+');s.value='+json.dumps(str(value))+';s.dispatchEvent(new Event("change",{bubbles:true}));return true})()')
            async def screenshot(name):
                assert await evaluate('!document.querySelector(".footage-frame")'), 'Never capture camera UI'
                metrics = await cdp('Page.getLayoutMetrics')
                size = metrics['cssContentSize']
                data = await cdp('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': True,
                                  'clip': {'x': 0, 'y': 0, 'width': size['width'], 'height': size['height'], 'scale': 1}})
                (ROOT/'artifacts'/name).write_bytes(base64.b64decode(data['data']))
            await cdp('Page.enable')
            await cdp('Runtime.enable')
            await cdp('Page.addScriptToEvaluateOnNewDocument', {'source': '''
              window.radarFixture=null; window.radarFetchMode='normal';
              const nativeFetch=window.fetch;
              window.fetch=async(...args)=>{
                if(/\\/api\\/edge\\/radar$/.test(String(args[0]))){
                  if(window.radarFetchMode==='error')return new Response('{}',{status:503});
                  if(window.radarFetchMode==='hold')await new Promise(r=>setTimeout(r,6500));
                  if(window.radarFixture){
                    const data=structuredClone(window.radarFixture),now=Date.now();
                    if(data.status==='no_target')data.timestamp_ms=now;
                    data.server_time_ms=now;data.age_ms=now-data.timestamp_ms;data.stale=data.age_ms>3000;
                    if(data.stale)data.status='stale';
                    for(const v of data.vitals)v.stale=now-v.timestamp_ms>3000;
                    return new Response(JSON.stringify(data),{headers:{'Content-Type':'application/json'}});
                  }
                }
                return nativeFetch(...args);
              };
            '''})
            await wait_for('location.origin==="http://127.0.0.1:5173" && document.readyState!=="loading"')
            await evaluate('localStorage.setItem("access_token",'+json.dumps(token)+');true')
            await cdp('Page.navigate', {'url': 'http://127.0.0.1:5173'})
            await wait_for('!!document.querySelector(".nav-item")')
            await nav('设备与数据')
            await wait_for('!!document.querySelector("#radar-source")')
            await select('#radar-source', 'real')
            await wait_for('document.querySelector("#radar-source").value==="real"')
            real = await asyncio.to_thread(request, '/api/edge/radar', None, token)
            await nav('雷达与体征')
            await wait_for('!!document.querySelector(".radar-cloud-controls button")')
            await evaluate('document.querySelector(".radar-cloud-controls button:last-child").click();true')
            if real['status']=='disconnected':
                assert not real['points'] and not real['vitals'], real
                report['actual_offline'] = {k: real[k] for k in ['status', 'source', 'active_source', 'stale']}
                await wait_for('document.querySelector(".radar-cloud canvas")?.dataset.points==="0"')
                assert await evaluate('document.querySelector("[data-testid=radar-heart-rate]").textContent==="—" && document.querySelector(".radar-notice").textContent.includes("尚未连接")')
                await screenshot('radar_frontend_offline.png')
            else:
                report['actual_hardware'] = {k: real[k] for k in ['status', 'source', 'active_source', 'stale']}

            # Fixtures do not touch serial devices, the database, or production source messages.
            await asyncio.to_thread(request, '/api/data-source', {'source': original_source}, token)
            await evaluate('''(()=>{
              const now=Date.now(),wave=(phase)=>Array.from({length:15},(_,i)=>Math.sin(i*.65+phase)*.2);
              window.radarFixture={active_source:'real',source:'real',session_id:'synthetic-browser-fixture',status:'tracking',
                timestamp_ms:now,frame_number:100,tlv_types:[1020,1010,1040],quality:null,calibrated:false,
                coordinate_system:'radar_xyz_m',waveform_units:'firmware_unverified',
                hardware:{configured:true,connected:true,port:'SYNTHETIC / NO DEVICE',baud:921600,serial_available:true,error:null},
                points:Array.from({length:160},(_,i)=>[Math.sin(i*1.3)*.35,1.5+Math.cos(i*.7)*.3,.3+(i%30)/30*1.3]),
                tracks:[0,1].map(id=>({target_id:id,position:[id ? .8 : -.3,1.5,1],velocity:[id ? 0 : .2,0,0],acceleration:[0,0,0],confidence:.9})),
                vitals:[0,1].map(id=>({target_id:id,timestamp_ms:now,device_id:'SYNTHETIC',heart_rate:id ? 82 : 66,
                  breath_rate:id ? 18 : 15,breathing_deviation:.1,range_bin:21,heart_waveform:wave(id),breath_waveform:wave(id+1),
                  source:'real',firmware_frame_number:100,needs_review:true,watch_reason:'合成浏览器验证数据，不是实测',
                  quality:null,chest_displacement_mm:null,ahi_index:null,motion_flag:false}))};
              const badge=document.createElement('div');badge.id='radar-fixture-label';badge.textContent='界面验收 / 合成数据 / 非实测';
              badge.style.cssText='position:fixed;bottom:16px;right:16px;padding:12px;background:#fff3cf;color:#624711;border:2px solid #987329;font:14px sans-serif;z-index:100';document.body.append(badge);
              return true;
            })()''')
            await wait_for('document.querySelector("[data-testid=radar-heart-rate]").textContent==="66" && document.querySelectorAll(".radar-wave-chart .recharts-line").length===2')
            await asyncio.sleep(.8)
            assert await evaluate('document.querySelector(".radar-trends .radar-section-head > span").textContent==="1 次更新"'), 'Repeated snapshots must not duplicate samples'
            report['deduplication'] = True
            for _ in range(4):
                await evaluate('(()=>{const d=window.radarFixture;d.timestamp_ms=Date.now();d.frame_number++;d.tracks[0].position[0]+=.08;d.vitals.forEach(v=>{v.timestamp_ms=d.timestamp_ms;v.firmware_frame_number=d.frame_number;v.heart_rate+=1});return true})()')
                await asyncio.sleep(.35)
            await wait_for('document.querySelector("[data-testid=radar-heart-rate]").textContent==="70"')
            assert await evaluate('document.querySelector("[data-track-path=\\"0\\"]").getAttribute("points").trim().split(" ").length>=4')
            assert await evaluate('document.querySelector(".radar-cloud canvas").dataset.points==="160"')
            report['point_cloud_and_trajectory'] = True
            await select('#radar-target', 1)
            await wait_for('document.querySelector("[data-testid=radar-heart-rate]").textContent==="86" && document.querySelector("[data-testid=radar-breath-rate]").textContent==="18.0"')
            report['target_switching'] = True
            await evaluate('window.radarFixture.timestamp_ms=Date.now();window.radarFixture.vitals.forEach(v=>v.timestamp_ms=Date.now());true')
            await screenshot('radar_frontend_fixture.png')
            await cdp('Emulation.setDeviceMetricsOverride', {'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
            await asyncio.sleep(.5)
            dims = await evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
            assert dims['scroll'] <= dims['width'], dims
            await screenshot('radar_frontend_mobile_fixture.png')
            report['mobile_dimensions'] = dims
            await cdp('Emulation.clearDeviceMetricsOverride')

            await evaluate('window.radarFixture.timestamp_ms=Date.now();window.radarFixture.vitals.forEach(v=>v.timestamp_ms=Date.now());true')
            await wait_for('document.querySelector("[data-testid=radar-heart-rate]").textContent==="86"')
            await evaluate('window.radarFetchMode="hold";true')
            await wait_for('document.querySelector("[data-testid=radar-heart-rate]").textContent==="—" && !document.querySelector(".radar-wave-chart") && document.querySelector(".radar-cloud canvas").dataset.points==="0"')
            report['client_staleness_during_hung_request'] = True
            await evaluate('window.radarFetchMode="normal";true')

            await evaluate('(()=>{const d=window.radarFixture;d.timestamp_ms=Date.now();d.tracks=[];d.vitals=[];d.points=[];d.status="no_target";return true})()')
            await wait_for('document.querySelector(".radar-connection").textContent.includes("未发现跟踪目标") && document.querySelector("[data-testid=radar-heart-rate]").textContent==="—" && document.querySelector("#radar-target").disabled && !document.querySelector(".radar-wave-chart")')
            report['target_loss_clears_readings'] = True
            await evaluate('(()=>{const d=window.radarFixture;d.timestamp_ms=Date.now()-4000;d.status="stale";return true})()')
            await wait_for('document.querySelector(".radar-connection").textContent.includes("数据已过期")')
            report['server_staleness'] = True
            await evaluate('window.radarFetchMode="error";true')
            await wait_for('document.querySelector(".radar-connection").textContent.includes("服务不可达")')
            report['request_failure'] = True
            await evaluate('window.radarFetchMode="normal";window.radarFixture=null;document.querySelector("#radar-fixture-label").remove();true')
            await nav('设备与数据')
            await wait_for('document.querySelector("#radar-source").value==='+json.dumps(original_source))
            await select('#radar-source', 'real')
            await wait_for('document.querySelector("#radar-source").value==="real"')
            await select('#radar-source', original_source)
            await wait_for('document.querySelector("#radar-source").value==='+json.dumps(original_source))
            report['actual_source_switching'] = True
            await nav('系统设置')
            await wait_for('document.body.innerText.includes("API Key")')
            report['llm_settings_entry'] = True
            await nav('监测台')
            await wait_for('document.querySelector(".footage-frame img")?.naturalWidth===640')
            after_video = await asyncio.to_thread(request, '/api/video/status', None, token)
            assert after_video['source_mode'] == before_video['source_mode'] == 'live'
            assert after_video['connected'] and [after_video['width'], after_video['height']] == [before_video['width'], before_video['height']]
            added = after_video['analysis']['windows_analyzed']-before_video['analysis']['windows_analyzed']
            assert added > 0, 'Video inference must keep running during radar checks'
            report['video'] = {'connected':True,'resolution':[after_video['width'],after_video['height']], 'capture_fps':after_video['display_fps'], 'model_windows_added':added}
            assert not errors, errors
            report['uncaught_browser_errors'] = errors
        (ROOT/'artifacts/radar_frontend_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        await asyncio.to_thread(request, '/api/data-source', {'source':original_source}, token)


if __name__ == '__main__':
    asyncio.run(main())
