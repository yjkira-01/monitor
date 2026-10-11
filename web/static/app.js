const $ = (id) => document.getElementById(id);
function timeNow(){ $('clock').textContent = new Date().toLocaleTimeString('zh-CN',{hour12:false}); }
function renderEvents(items){
  const box=$('event-list');
  if(!items.length){box.innerHTML='<div class="empty">尚无告警事件，系统持续观察中。</div>';return;}
  box.innerHTML=items.map(e=>`<div class="event"><i class="event-dot"></i><time>${new Date(e.time).toLocaleTimeString('zh-CN',{hour12:false})}</time><span>疑似跌倒姿态</span><strong>${(e.score||0).toFixed(2)}</strong></div>`).join('');
}
async function refresh(){
  try{
    const s=await (await fetch('/api/status')).json();
    $('connection').textContent=s.connected?'已连接':'未连接'; $('connection').className='status-badge '+(s.connected?'on':'off');
    $('detector').textContent=s.detector==='off'?'检测关闭':s.detector; $('score').textContent=Number(s.fall_score||0).toFixed(2);
    $('meter').style.width=`${Math.min(100,Number(s.fall_score||0)*100)}%`; $('fps').textContent=`${Number(s.fps||0).toFixed(1)} FPS`;
    $('alerts').textContent=s.alerts||0; $('nav-alerts').textContent=s.alerts||0; $('video-frames').textContent=`${s.frames||0} frames`;
    $('score-note').textContent=s.error?s.error:(s.connected?'正在分析视频姿态':'等待视频源…');
    renderEvents(await (await fetch('/api/events')).json());
  }catch(e){$('score-note').textContent='页面正在等待本地服务…';}
}
setInterval(timeNow,1000); setInterval(refresh,1000); timeNow(); refresh();
