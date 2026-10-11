"""真实权重/API/页面的本地端到端验收；只使用隔离的无头 Chrome。"""
import asyncio
import base64
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen
import websockets

ROOT = Path(__file__).resolve().parents[1]


def request(path, body=None, token=None):
    headers = {'Content-Type':'application/json'}
    if token:
        headers['Authorization'] = 'Bearer '+token
    req = Request('http://127.0.0.1:5173'+path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    with urlopen(req,timeout=90) as response:
        return json.load(response)


async def main():
    login = await asyncio.to_thread(request, '/api/auth/login', {'username':'admin','password':'admin123'})
    token = login['access_token']
    print('API login passed',flush=True)
    chrome = Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
    process = subprocess.Popen([str(chrome),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
                                '--remote-debugging-port=9223','--window-size=1600,1100',
                                '--user-data-dir='+str(ROOT/'artifacts/browser-live-profile'),'http://127.0.0.1:5173'],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        targets = []
        for _ in range(100):
            try:
                with urlopen('http://127.0.0.1:9223/json',timeout=2) as response:
                    targets = json.load(response)
                if any(t['type']=='page' for t in targets):
                    break
            except OSError:
                pass
            await asyncio.sleep(.1)
        target = next(t for t in targets if t['type']=='page')
        print('Chrome debugging connected',flush=True)
        async with websockets.connect(target['webSocketDebuggerUrl'],max_size=16*1024*1024) as ws:
            counter = 0
            async def cdp(method, params=None):
                nonlocal counter
                counter += 1
                current = counter
                await ws.send(json.dumps({'id':current,'method':method,'params':params or {}}))
                while True:
                    result = json.loads(await asyncio.wait_for(ws.recv(),timeout=15))
                    if result.get('id') == current:
                        if 'error' in result:
                            raise RuntimeError(result['error'])
                        return result.get('result',{})
            async def evaluate(expression):
                value = await cdp('Runtime.evaluate',{'expression':expression,'awaitPromise':True,'returnByValue':True})
                if 'exceptionDetails' in value:
                    raise RuntimeError(value['exceptionDetails'])
                return value['result'].get('value')
            async def text():
                return await evaluate('document.body.innerText')
            async def click(label):
                return await evaluate('(()=>{const b=[...document.querySelectorAll("button")].find(b=>b.textContent.includes('+json.dumps(label)+')); if(!b)throw Error("missing button");b.click();return true})()')
            await evaluate('localStorage.setItem("access_token",'+json.dumps(token)+'); location.reload(); true')
            for _ in range(80):
                if '视频观察工作台' in (await text()):
                    break
                await asyncio.sleep(.25)
            else:
                raise RuntimeError('workbench did not render')
            print('Workbench rendered',flush=True)
            await asyncio.to_thread(request, '/api/video/source', {'sample_id':'fall'}, token)
            observations = []
            for sample in ('fall','adl'):
                if sample == 'adl':
                    await evaluate('(()=>{const s=document.querySelector("select[aria-label=选择示例视频]");s.value="adl";s.dispatchEvent(new Event("change",{bubbles:true}));return true})()')
                deadline = time.monotonic()+35
                positions = set()
                while time.monotonic()<deadline:
                    status = await asyncio.to_thread(request, '/api/video/status', None, token)
                    positions.add(status['position_s'])
                    a = status['analysis']
                    if a['model_state']=='error':
                        raise RuntimeError(a['model_error'])
                    if ('跌倒' if sample=='fall' else '正常') in status['source'] and status['ended'] and a.get('window_end_s', -1)>=status['position_s']-.01:
                        break
                    await asyncio.sleep(.4)
                else:
                    raise RuntimeError('video did not complete')
                assert len(positions)>=3, positions
                assert a['predicted']==sample, a['predicted']
                assert a['windows_analyzed']>=3, a['windows_analyzed']
                detection = a['people_detection']
                assert detection['state'] == 'ready', detection
                assert abs(detection['frame_s']-status['position_s']) <= .02
                assert len(detection['people']) >= 1, detection
                assert any(len(p['world_landmarks']) == 33 for p in detection['people'])
                assert all(len(p['world_landmarks']) in (0, 33) for p in detection['people'])
                observations.append({'sample':sample,'status':status,'distinct_positions':len(positions)})
                print('Validated '+sample+' windows: '+str(a['windows_analyzed']),flush=True)
                await asyncio.sleep(1)
                images = await evaluate('[...document.images].map(i=>({alt:i.alt,width:i.naturalWidth,complete:i.complete}))')
                assert len([i for i in images if i['width']>0])>=2, images
                canvases = await evaluate('[...document.querySelectorAll("canvas")].map(c=>({width:c.width,height:c.height,label:c.getAttribute("aria-label")}))')
                assert len(canvases) == 2 and all(c['width'] > 0 for c in canvases), canvases
                assert await evaluate('document.querySelector(".pose-view canvas").dataset.avatar === "ready"')
                assert 'SIMULATED / 非雷达实测' in (await text())
                assert '三维人物' in (await text()) and '三维渲染不可用' not in (await text())
                screen = await cdp('Page.captureScreenshot', {'format':'png','captureBeyondViewport':False})
                (ROOT/f'artifacts/workbench_{sample}_desktop.png').write_bytes(base64.b64decode(screen['data']))
                metrics = await cdp('Page.getLayoutMetrics')
                content = metrics['cssContentSize']
                full = await cdp('Page.captureScreenshot', {'format':'png','captureBeyondViewport':True,'clip':{'x':0,'y':0,'width':content['width'],'height':content['height'],'scale':1}})
                (ROOT/f'artifacts/workbench_{sample}_full.png').write_bytes(base64.b64decode(full['data']))
            await click('雷达')
            await asyncio.sleep(.3)
            assert 'OFFLINE / 无有效实测帧' in (await text())
            await click('模拟')
            await click('侧面')
            await click('复位')
            await click('原始视频')
            await asyncio.sleep(.4)
            assert any(i['width'] > 0 and '原始视频' in i['alt'] for i in await evaluate('[...document.images].map(i=>({alt:i.alt,width:i.naturalWidth}))'))
            await click('人物标注')
            await click('从头重播')
            await asyncio.sleep(1.8)
            await click('暂停播放')
            await asyncio.sleep(.8)
            first = await asyncio.to_thread(request, '/api/video/status', None, token)
            await asyncio.sleep(1.1)
            second = await asyncio.to_thread(request, '/api/video/status', None, token)
            assert first['paused'] and first['position_s']==second['position_s']
            await click('继续播放')
            await asyncio.sleep(.8)
            third = await asyncio.to_thread(request, '/api/video/status', None, token)
            assert third['position_s']>second['position_s']
            await cdp('Emulation.setDeviceMetricsOverride', {'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
            await asyncio.sleep(1)
            dims = await evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
            assert dims['scroll']<=dims['width'], dims
            shot = await cdp('Page.captureScreenshot',{'format':'png'})
            (ROOT/'artifacts/workbench_mobile.png').write_bytes(base64.b64decode(shot['data']))
            await cdp('Emulation.clearDeviceMetricsOverride')
            document = await cdp('DOM.getDocument')
            input_node = await cdp('DOM.querySelector', {'nodeId':document['root']['nodeId'],'selector':'input[type=file]'})
            await cdp('DOM.setFileInputFiles', {'nodeId':input_node['nodeId'],'files':[str(ROOT/'data/raw/GMDCSA24/Subject 4/ADL/01.mp4')]})
            upload_deadline = time.monotonic()+30
            while time.monotonic()<upload_deadline:
                uploaded = await asyncio.to_thread(request, '/api/video/status', None, token)
                if uploaded['source']=='01.mp4' and uploaded['analysis']['generation']>third['analysis']['generation']:
                    break
                await asyncio.sleep(.25)
            else:
                raise RuntimeError('upload did not switch video')
            assert uploaded['source_mode']=='file_replay'
            print('Uploaded local video accepted',flush=True)
            await click('系统设置')
            await asyncio.sleep(.8)
            assert '大模型 API' in (await text())
            summary = await asyncio.to_thread(request, '/api/insights/summary', {'hours':24,'question':'总结最新视频窗口'}, token)
            assert summary['context']['video']['model']=='r3d18'
            assert summary['context']['video']['source_mode']=='file_replay'
            assert summary['context']['video']['people_detection'] is not None
            (ROOT/'artifacts/workbench_browser_validation.json').write_text(json.dumps({'observations':observations,'pause_resume':True,'upload':True,'person_boxes':True,'spatial_canvases':canvases,'radar_offline':True,'mobile_dimensions':dims,'summary':summary},ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'samples':[{'sample':o['sample'],'windows':o['status']['analysis']['windows_analyzed'],'score':o['status']['analysis']['fall_score']} for o in observations], 'pause_resume':True,'mobile_dimensions':dims,'images':images,'summary_provider':summary['provider']},ensure_ascii=False))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__=='__main__':
    asyncio.run(main())
