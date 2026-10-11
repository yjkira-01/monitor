"""Real dataset composite detection and numeric browser stability checks; no camera images saved."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

import cv2
import numpy as np
import websockets

from validate_video_workbench import ROOT, request
sys.path.insert(0,str(ROOT/'backend'))
from app.video_people import VideoPersonModel


def dataset_people():
    model = VideoPersonModel()
    crops = []
    single_counts = []
    single_conditions = []
    try:
        for subject in (1,2):
            path = next((ROOT/f'data/raw/OmniFall/Cauca_fall/Cauca_fall/clips/Subject.{subject}').glob('Walk*_0_*.avi'))
            cap = cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_MSEC,1500);ok,frame = cap.read();cap.release()
            assert ok,path
            model.reset();result,_ = model.predict(frame,1)
            for ms in (101,201):
                result,_=model.predict(frame,ms);single_counts.append(len(result['people']))
                assert len(result['people'])==1,'single-subject data must remain one target'
            people = [p for p in result['people'] if p['observation']['spatial_valid']]
            assert people,[(p['bbox'],p['observation']) for p in result['people']]
            p = max(people,key=lambda p:p['visibility'])
            h,w = frame.shape[:2];x1,y1,x2,y2 = p['bbox'];dx,dy=(x2-x1)*.15,(y2-y1)*.12
            a,b,c,d=(np.clip([x1-dx,y1-dy,x2+dx,y2+dy],0,1)*[w,h,w,h]).astype(int)
            crops.append(frame[b:d,a:c])
            for condition in ('normal','lower_frame_truncated'):
                model.reset();counts=[];cap=cv2.VideoCapture(str(path))
                try:
                    for sample in range(12):
                        cap.set(cv2.CAP_PROP_POS_MSEC,1000+sample*150);ok,current=cap.read()
                        assert ok,path
                        if condition=='lower_frame_truncated':current=current[:round(current.shape[0]*.7)]
                        measured,_=model.predict(current,1+sample*150);counts.append(len(measured['people']))
                    assert max(counts)<=1,(subject,condition,counts)
                    assert any(counts),(subject,condition,'all people missed')
                    single_conditions.append({'subject':subject,'condition':condition,'counts':counts})
                finally:cap.release()
        composite = np.full((540,960,3),230,np.uint8)
        for i,crop in enumerate(crops):
            h,w=crop.shape[:2];scale=min(400/w,460/h);width,height=round(w*scale),round(h*scale)
            left,top=i*480+(480-width)//2,(540-height)//2
            composite[top:top+height,left:left+width]=cv2.resize(crop,(width,height),interpolation=cv2.INTER_AREA)
        model.reset()
        results=[]
        for ms in (1,101,201):
            result,_=model.predict(composite,ms);results.append(result)
        people=result['people']
        assert len(people)==2,[(p['bbox'],p['observation']) for p in people]
        assert all(p['observation']['spatial_valid'] for p in people),[p['observation'] for p in people]
        assert {p['track_id'] for p in results[0]['people']}=={p['track_id'] for p in people}
        report={'dataset':'CAUCAFall walking crops, two distinct subjects, synthetic composition',
                'single_subject_counts':single_counts,
                'single_subject_motion_checks':single_conditions,
                'people':[{'track_id':p['track_id'],'observation':p['observation']} for p in people]}
        (ROOT/'artifacts/multipeople_detection_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return people,report
    finally:model.close()


async def main():
    people,detector_report = await asyncio.to_thread(dataset_people)
    # Browser fixture uses actual measured poses; independent scores test wiring, not classification accuracy.
    names = ('walk','fall','fallen','sit_down','sitting','lie_down','lying','stand_up','standing','other')
    for i,p in enumerate(people):
        label = 'standing' if i==0 else 'sitting'
        p['analysis']={'state':'adl','fall_score':.03+i*.03,'windows_analyzed':5,
                       'activity':{'label':label,'confidence':.82,'scores':{name:.82 if name==label else .02 for name in names},'basis':'browser_fixture'}}
    token=(await asyncio.to_thread(request,'/api/auth/login',{'username':'admin','password':'admin123'}))['access_token']
    process=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--no-first-run',
                              '--no-default-browser-check','--remote-debugging-port=9228','--window-size=1600,1100',
                              '--user-data-dir='+str(ROOT/'artifacts/browser-people-profile'),'http://127.0.0.1:5173'],
                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(100):
            try:
                with urlopen('http://127.0.0.1:9228/json',timeout=2) as response:target=next(t for t in json.load(response) if t['type']=='page')
                break
            except (OSError,StopIteration):await asyncio.sleep(.1)
        else:raise RuntimeError('Chrome did not start')
        async with websockets.connect(target['webSocketDebuggerUrl'],max_size=16*1024*1024) as ws:
            counter=0
            async def cdp(method,params=None):
                nonlocal counter
                counter+=1;await ws.send(json.dumps({'id':counter,'method':method,'params':params or {}}))
                while True:
                    data=json.loads(await asyncio.wait_for(ws.recv(),20))
                    if data.get('id')==counter:
                        assert 'error' not in data,data
                        return data.get('result',{})
            async def evaluate(expression):
                data=await cdp('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':True})
                assert 'exceptionDetails' not in data,data
                return data['result'].get('value')
            async def wait_for(expression,seconds=30):
                deadline=time.monotonic()+seconds
                while time.monotonic()<deadline:
                    if await evaluate(expression):return
                    await asyncio.sleep(.2)
                print(await evaluate('({body:document.body.innerText.slice(0,500),fixture:typeof fixturePeople,intercepts:window.fixtureIntercepts,lastFixture:window.lastFixture,canvases:[...document.querySelectorAll(".spatial-scene canvas")].map(c=>({...c.dataset})),people:document.querySelectorAll(".person-state").length})'),flush=True)
                raise AssertionError('Timed out: '+expression)
            script='window.fixturePeople='+json.dumps(people)+';'+'''
              window.partialFixture=false;window.poseGapFixture=false;window.absentFixture=false;window.fixtureIntercepts=0;window.contextStats={created:0,lost:0};
              const getContext=HTMLCanvasElement.prototype.getContext;
              HTMLCanvasElement.prototype.getContext=function(...a){const c=getContext.apply(this,a);if(c&&a[0].includes('webgl')&&!this.counted){this.counted=true;contextStats.created++;this.addEventListener('webglcontextlost',()=>contextStats.lost++);}return c;};
              const nativeFetch=window.fetch;
              window.fetch=async(...args)=>{const r=await nativeFetch(...args);if(!r.ok||!String(args[0]).endsWith('/api/video/pose'))return r;
                const data=await r.json();window.fixtureIntercepts++;data.people=structuredClone(fixturePeople);data.state='ready';data.stale=false;data.paused=false;data.ended=false;data.frame_s=performance.now()/1000;data.updated_at_ms=Date.now();window.lastFixture={session:data.session_id,generation:data.generation,people:data.people.length};
                if(partialFixture){const p=data.people[0];p.observation.spatial_valid=false;p.observation.complete=false;p.observation.reason='fixture: clipped body';p.analysis.state='partial_adl';for(const i of [23,24,25,26,27,28,29,30,31,32]){p.world_landmarks[i].visibility=0;p.world_landmarks[i].x=NaN;}}
                if(poseGapFixture)data.people[0].world_landmarks=[];
                if(absentFixture)data.people.shift();
                return new Response(JSON.stringify(data),{status:200,headers:{'Content-Type':'application/json'}});};
            '''
            await cdp('Page.enable')
            await cdp('Page.addScriptToEvaluateOnNewDocument',{'source':script})
            await evaluate('localStorage.setItem("access_token",'+json.dumps(token)+');true')
            await cdp('Page.navigate',{'url':'http://127.0.0.1:5173'})
            await wait_for('Boolean(document.querySelector(".spatial-deferred"))')
            await evaluate('document.querySelector(".spatial-deferred").scrollIntoView({block:"center"});true')
            await wait_for('document.querySelectorAll(".person-state").length===2 && [...document.querySelectorAll(".spatial-scene canvas")].every(c=>c.dataset.activePeople==="2")')
            await evaluate('document.querySelector(".spatial-views").scrollIntoView({block:"center"});true')
            await wait_for('document.querySelector(".pose-view canvas").dataset.avatar==="ready"')
            assert await evaluate('document.querySelectorAll(".activity-ranking").length===2 && document.querySelector(".person-status-list").innerText.includes("站姿") && document.querySelector(".person-status-list").innerText.includes("坐姿")')
            before=await asyncio.to_thread(request,'/api/video/status',None,token)
            await cdp('HeapProfiler.collectGarbage');heap_before=(await cdp('Runtime.getHeapUsage'))['usedSize']
            started=time.monotonic();samples=[]
            while time.monotonic()-started<60:
                samples.append(await evaluate('[...document.querySelectorAll(".spatial-scene canvas")].map(c=>({...c.dataset}))'))
                await asyncio.sleep(1)
            assert all(len(s)==2 and all(c['activePeople']=='2' for c in s) for s in samples)
            assert min(int(c.get('renderFps',0)) for s in samples[5:] for c in s)>=15,samples[-3:]
            await evaluate('window.partialFixture=true;true')
            await wait_for('[...document.querySelectorAll(".spatial-scene canvas")].every(c=>c.dataset.activePeople==="2" && Number(c.dataset.estimatedPeople)>=1)')
            assert await evaluate('document.querySelectorAll(".person-state").length===2 && document.querySelectorAll(".activity-ranking").length===2')
            assert await evaluate('document.querySelector(".spatial-person-legend").innerText.includes("补全")')
            await asyncio.sleep(1)
            assert await evaluate('document.querySelector(".pose-view canvas").dataset.activePeople==="2"'),'sustained partial visibility must retain both bodies'
            await evaluate('window.partialFixture=false;true')
            await wait_for('document.querySelector(".pose-view canvas").dataset.activePeople==="2"')
            await evaluate('window.poseGapFixture=true;true')
            await asyncio.sleep(.25)
            assert await evaluate('document.querySelector(".pose-view canvas").dataset.activePeople==="2"'),'brief pose loss must retain temporal display'
            await asyncio.sleep(.8)
            if people[0].get('detector_score',0) >= .7:
                await wait_for('document.querySelector(".pose-view canvas").dataset.activePeople==="2" && document.querySelector(".spatial-person-legend").innerText.includes("0/33")')
            else:
                await wait_for('document.querySelector(".pose-view canvas").dataset.activePeople==="1"')
            await evaluate('window.poseGapFixture=false;true')
            await wait_for('document.querySelector(".pose-view canvas").dataset.activePeople==="2"')
            await evaluate('window.absentFixture=true;true')
            await wait_for('document.querySelector(".pose-view canvas").dataset.activePeople==="1" && document.querySelectorAll(".person-state").length===1')
            await evaluate('window.absentFixture=false;window.partialFixture=true;true')
            await wait_for('document.querySelector(".pose-view canvas").dataset.activePeople==="2" && document.querySelector(".spatial-person-legend").innerText.includes("参考结构")')
            await evaluate('window.partialFixture=false;true')
            await evaluate('document.querySelector(".pose-view canvas").getContext("webgl2").getExtension("WEBGL_lose_context").loseContext();true')
            await wait_for('[...document.querySelectorAll(".spatial-empty button")].some(b=>b.textContent.includes("重试"))')
            await evaluate('[...document.querySelectorAll(".spatial-empty button")].find(b=>b.textContent.includes("重试")).click();true')
            await wait_for('document.querySelector(".pose-view canvas").dataset.avatar==="ready" && document.querySelector(".pose-view canvas").dataset.activePeople==="2"')
            for _ in range(5):
                await evaluate('[...document.querySelectorAll(".nav-item")].find(b=>b.innerText.includes("系统设置")).click();true')
                await wait_for('!document.querySelector(".spatial-scene canvas")')
                await evaluate('document.querySelector(".nav-item").click();true')
                await wait_for('Boolean(document.querySelector(".spatial-deferred"))')
                await evaluate('document.querySelector(".spatial-deferred").scrollIntoView({block:"center"});true')
                await wait_for('document.querySelector(".pose-view canvas")?.dataset.avatar==="ready" && document.querySelector(".pose-view canvas").dataset.activePeople==="2"')
            await cdp('HeapProfiler.collectGarbage');heap_after=(await cdp('Runtime.getHeapUsage'))['usedSize']
            contexts=await evaluate('contextStats')
            assert contexts['created']-contexts['lost']<=2,contexts
            assert heap_after-heap_before<16*1024**2,(heap_before,heap_after)
            after=await asyncio.to_thread(request,'/api/video/status',None,token)
            same_source = after['analysis']['session_id']==before['analysis']['session_id'] and after['analysis']['generation']==before['analysis']['generation']
            windows_added = after['analysis']['windows_analyzed']-before['analysis']['windows_analyzed'] if same_source else None
            if same_source and before['source_mode']=='live' and after['source_mode']=='live':assert windows_added>0
            await cdp('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
            await asyncio.sleep(.5)
            dimensions=await evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
            assert dimensions['scroll']<=dimensions['width'],dimensions
            report={'detector':detector_report,'seconds':time.monotonic()-started,'fixture_states':'wiring only, not prediction accuracy',
                    'partial_person_isolation':True,'partial_complete_display':True,'pose_gap_retention_and_expiry':True,
                    'absent_person_cleared':True,'partial_bootstrap':True,'webgl_retry':True,'navigation_cycles':5,'contexts':contexts,
                    'heap_before':heap_before,'heap_after':heap_after,'samples':samples,'mobile_dimensions':dimensions,
                    'model_windows_added':windows_added,'source_unchanged':same_source,'backend_source_mode':after['source_mode'],
                    'backend_paused':after.get('paused'),'backend_ended':after.get('ended'),'camera_images_saved':False}
            (ROOT/'artifacts/people_browser_stability.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill()


if __name__=='__main__':asyncio.run(main())
