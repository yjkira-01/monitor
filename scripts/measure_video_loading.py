"""Measure cold page readiness without switching the user's source or saving video images."""
import argparse
import asyncio
import json
import subprocess
import time
from urllib.request import urlopen

import websockets
from validate_video_workbench import ROOT, request


async def main(label, url):
    token=await asyncio.to_thread(request,'/api/auth/login',{'username':'admin','password':'admin123'})
    process=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new',
        '--no-first-run','--no-default-browser-check','--remote-debugging-port=9233','--window-size=1440,900',
        '--user-data-dir='+str(ROOT/f'artifacts/browser-loading-{label}'),'about:blank'],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(80):
            try:
                with urlopen('http://127.0.0.1:9233/json',timeout=2) as response:
                    target=next(t for t in json.load(response) if t['type']=='page')
                break
            except (OSError,StopIteration):await asyncio.sleep(.1)
        else:raise RuntimeError('Chrome did not start')
        async with websockets.connect(target['webSocketDebuggerUrl'],max_size=16*1024**2) as socket:
            counter=0;errors=[]
            async def cdp(method,params=None):
                nonlocal counter
                counter+=1;await socket.send(json.dumps({'id':counter,'method':method,'params':params or {}}))
                while True:
                    data=json.loads(await asyncio.wait_for(socket.recv(),20))
                    if data.get('method')=='Runtime.exceptionThrown':errors.append(data['params']['exceptionDetails'].get('text'))
                    if data.get('id')==counter:
                        assert 'error' not in data,data
                        return data.get('result',{})
            async def evaluate(code):
                result=await cdp('Runtime.evaluate',{'expression':code,'returnByValue':True})
                assert 'exceptionDetails' not in result,result
                return result['result'].get('value')
            await cdp('Page.enable');await cdp('Runtime.enable');await cdp('Network.enable')
            await cdp('Network.setCacheDisabled',{'cacheDisabled':True})
            script='localStorage.setItem("access_token",'+json.dumps(token['access_token'])+');'+'''
              window.firstVideoAt=0;window.firstInterfaceAt=0;window.longTaskMs=0;
              new PerformanceObserver(list=>{for(const e of list.getEntries())window.longTaskMs+=e.duration;}).observe({type:'longtask',buffered:true});
              const check=()=>{if(!firstInterfaceAt&&document.querySelector('.footage-frame'))firstInterfaceAt=performance.now();
                if(!firstVideoAt&&document.querySelector('.footage-frame img')?.naturalWidth)firstVideoAt=performance.now();
                if(!firstVideoAt)setTimeout(check,20);};check();
            '''
            await cdp('Page.addScriptToEvaluateOnNewDocument',{'source':script})
            await cdp('Page.navigate',{'url':url})
            start=time.monotonic()
            while time.monotonic()-start<15:
                if await evaluate('Boolean(window.firstVideoAt)'):break
                await asyncio.sleep(.1)
            await asyncio.sleep(2)
            initial=await evaluate(r'''(()=>{const n=performance.getEntriesByType('navigation')[0], image=document.querySelector('.footage-frame img');
              const resources=performance.getEntriesByType('resource');return {interface_ms:firstInterfaceAt,video_ms:firstVideoAt,dom_ready_ms:n.domContentLoadedEventEnd,
                load_event_ms:n.loadEventEnd,fcp_ms:performance.getEntriesByName('first-contentful-paint')[0]?.startTime,long_task_ms:longTaskMs,
                requests:resources.length,script_requests:resources.filter(r=>r.initiatorType==='script').length,
                image_size:image?[image.naturalWidth,image.naturalHeight]:null,spatial_loaded:Boolean(document.querySelector('.spatial-scene canvas')),
                decision_title:document.querySelector('.interpretation-column h2')?.textContent,
                pose_readout:[...document.querySelectorAll('.interpretation-column .assessment-copy')].map(p=>p.textContent).find(t=>t.includes('TCN')),
                resources:resources.filter(r=>/three|recharts|Spatial|PersonScene|models\//.test(r.name)).map(r=>({name:r.name.split('/').pop(),ms:r.duration,bytes:r.encodedBodySize}))};})()''')
            assert initial['video_ms'] and initial['image_size'],initial
            await evaluate('(document.querySelector(".spatial-views") || document.querySelector(".spatial-deferred") || document.querySelector(".spatial-scene"))?.scrollIntoView({block:"center"})')
            spatial_started=time.monotonic()
            while time.monotonic()-spatial_started<15:
                if await evaluate('document.querySelector(".pose-view canvas")?.dataset.avatar==="ready"'):break
                await asyncio.sleep(.1)
            spatial=await evaluate('[...document.querySelectorAll(".spatial-scene canvas")].map(c=>({...c.dataset}))')
            person_readout=await evaluate('document.querySelector(".person-status-list")?.innerText')
            report={'url':url,'label':label,'initial':initial,'spatial_ready_after_scroll_ms':round((time.monotonic()-spatial_started)*1000),
                    'spatial':spatial,'person_readout':person_readout,'uncaught_errors':errors,'source_switched':False,'camera_images_saved':False}
            (ROOT/f'artifacts/video_loading_{label}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps({**report,'initial':{k:v for k,v in initial.items() if k!='resources'}},ensure_ascii=False),flush=True)
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--url',default='http://127.0.0.1:5173')
    args=parser.parse_args();asyncio.run(main(args.label,args.url))
