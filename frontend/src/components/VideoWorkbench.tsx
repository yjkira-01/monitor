import { lazy, memo, Suspense, useEffect, useRef, useState } from 'react'
import { authFetch, getToken } from '../api/client'
import { type PeopleDetection } from './PeopleStatus'
import { activityNames, type Activity } from './activityLabels'
const PeopleStatus = lazy(() => import('./PeopleStatus'))
const WindowScoreChart = lazy(() => import('./WindowScoreChart'))

type Quality = { brightness: number; sharpness: number; motion_ratio: number; activity: string; usable: boolean; note: string }
export type WindowResult = { window_start_s: number; window_end_s: number; fall_score: number; state: string; updated_at_ms: number; feature_change: number | null; inference_ms: number }
export type VideoAnalysis = {
  fall_score: number; confidence: number; action?: string; is_fall?: boolean; watch?: boolean;
  analysis_backend?: string; model_state?: string; model_error?: string | null; source_mode?: string; source_label?: string;
  model?: string; model_revision?: string; scope?: string; state?: string; assessment?: string; generation?: number; session_id?: string; updated_at_ms?: number;
  window_start_s?: number; window_end_s?: number; result_age_ms?: number; windows_analyzed?: number;
  inference_ms?: number; feature_change?: number | null; consecutive_fall_windows?: number; device?: string;
  short_fall_score?: number | null; short_consecutive_fall_windows?: number; decision_basis?: string;
  quality?: Quality; current_quality?: Quality; stale?: boolean; attention_class?: string;
  history?: WindowResult[]; events?: (WindowResult & { id: string; message: string })[];
  people_detection?: PeopleDetection;
  activity?: Activity | null; activity_classes?: string[];
}
export type VideoStatus = {
  source: string; connected: boolean; frames: number; fps: number; display_fps?: number; capture_backend?: string; width: number; height: number; error: string | null;
  paused?: boolean; ended?: boolean; position_s?: number; duration_s?: number; source_mode?: string; quality?: Quality;
  analysis?: VideoAnalysis;
}

const stateNames: Record<string, string> = { adl: '正常活动特征', model_disagreement: '疑似跌倒 · 模型分歧待复核', fall_candidate: '疑似跌倒 · 观察中', fall_sustained: '连续跌倒特征 · 待复核', quality_limited: '画面质量受限', observation_limited: '观测受限 · 查看人员状态', warming_up: '收集视频片段', unavailable: '模型不可用' }
const seconds = (value = 0) => `${value.toFixed(1)}s`

export default memo(function VideoWorkbench({ status, onStatus }: { status: VideoStatus; onStatus: (next: VideoStatus) => void }) {
  const [samples, setSamples] = useState<{ id: string; name: string }[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [attention, setAttention] = useState(true)
  const [annotated, setAnnotated] = useState(true)
  const [streamKey, setStreamKey] = useState(0)
  const [streamFailed, setStreamFailed] = useState(false)
  const videoImage = useRef<HTMLImageElement>(null)
  const spatialHost = useRef<HTMLDivElement>(null)
  const [spatialReady, setSpatialReady] = useState(false)
  const uploadInput = useRef<HTMLInputElement>(null)
  const analysis = status.analysis
  const history = analysis?.history || []
  const events = analysis?.events || []
  const detection = analysis?.people_detection
  const poseResults = (detection?.people || []).filter(p => p.pose_analysis?.fall_score != null)
  const poseScore = poseResults.length ? Math.max(...poseResults.map(p => p.pose_analysis!.fall_score!)) : null
  const showDetection = annotated && detection?.state === 'ready' && !analysis?.stale
  const quality = analysis?.quality
  const token = getToken() || ''
  const mediaQuery = `token=${encodeURIComponent(token)}&session=${analysis?.session_id || ''}`
  const replay = status.source_mode === 'file_replay'
  const complete = Boolean(analysis?.windows_analyzed)
  const unavailable = !status.connected || analysis?.model_state === 'error' || analysis?.stale
  const tone = unavailable ? 'muted' : analysis?.is_fall ? 'danger' : analysis?.watch || !quality?.usable ? 'warn' : 'ok'
  const title = !status.connected ? '等待视频源' : analysis?.model_state === 'error' ? '模型暂不可用' : analysis?.stale ? '结果更新延迟' : !complete ? '正在收集视频片段' : stateNames[analysis?.state || ''] || '等待模型判断'

  useEffect(() => {
    setStreamFailed(false)
    const timer = window.setInterval(() => { if (!videoImage.current?.naturalWidth) setStreamFailed(true) }, 4000)
    return () => window.clearInterval(timer)
  }, [streamKey, annotated, analysis?.session_id])

  useEffect(() => {
    const node = spatialHost.current
    if (!node) return
    let timer: number
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { timer = window.setTimeout(() => setSpatialReady(true), 80); observer.disconnect() }
    }, { rootMargin: '80px' })
    observer.observe(node)
    return () => { observer.disconnect(); window.clearTimeout(timer) }
  }, [])

  useEffect(() => {
    authFetch('/api/video/samples', { headers: { Authorization: `Bearer ${getToken()}` } })
      .then(response => { if (!response.ok) throw new Error('无法读取示例视频'); return response.json() })
      .then(setSamples).catch(cause => setError(cause.message))
  }, [])

  async function command(path: string, body?: unknown, file?: File) {
    setBusy(true); setError('')
    try {
      const headers: Record<string, string> = { Authorization: `Bearer ${token}` }
      let payload: BodyInit | undefined
      if (file) {
        if (file.size > 200 * 1024 * 1024) throw new Error('单个视频最大 200MB')
        const form = new FormData(); form.append('file', file); payload = form
      } else if (body) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body) }
      const response = await authFetch(`/api/video/${path}`, { method: 'POST', headers, body: payload })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '操作失败，请重试')
      onStatus(data)
      if (path !== 'playback') setStreamKey(key => key + 1)
    } catch (cause) { setError(cause instanceof Error ? cause.message : '视频操作失败') }
    finally { setBusy(false); if (uploadInput.current) uploadInput.current.value = '' }
  }

  function exportSession() {
    const blob = new Blob([JSON.stringify({ source: status.source, mode: status.source_mode, exported_at: new Date().toISOString(), analysis }, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a'); link.href = url; link.download = 'caresense-video-report.json'; link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  return <section className="video-workbench" aria-label="视频模型实时工作台">
    <div className="workbench-toolbar">
      <div className="source-selector"><span className="field-caption">视频来源</span><select aria-label="选择示例视频" value="" disabled={busy} onChange={event => event.target.value === 'camera' ? command('camera') : command('source', { sample_id: event.target.value })}><option value="" disabled>{status.source}</option><option value="camera">C60E 摄像头 · 实时</option>{samples.map(sample => <option key={sample.id} value={sample.id}>{sample.name}</option>)}</select></div>
      <input ref={uploadInput} hidden type="file" accept=".mp4,.avi,.mov,.mkv,.webm" aria-label="上传测试视频" onChange={event => event.target.files?.[0] && command('upload', undefined, event.target.files[0])} />
      <button className="workbench-button primary" disabled={busy} onClick={() => uploadInput.current?.click()}>{busy ? '正在处理…' : '上传视频'}</button>
      <div className="model-stamp"><span className={`status-dot status-${analysis?.model_state === 'ready' ? 'ok' : 'muted'}`} />视频分析 · 5秒 / 2秒 <small>{analysis?.model_state === 'ready' ? `${analysis.device?.toUpperCase()} 推理` : analysis?.model_state === 'error' ? '加载失败' : '加载中'}</small></div>
    </div>
    {error && <p className="workbench-error" role="alert">{error}</p>}
    <div className="observation-grid">
      <div className="footage-column">
        <div className="footage-heading"><span>{replay ? '视频回放 / 测试环境' : `${status.source} / 实时监测`}</span><div className="view-switch"><button className={annotated ? 'active' : ''} onClick={() => setAnnotated(true)}>人物标注</button><button className={!annotated ? 'active' : ''} onClick={() => setAnnotated(false)}>原始视频</button></div></div>
        <div className="footage-frame" style={status.width && status.height ? { aspectRatio: `${status.width} / ${status.height}` } : undefined}><img loading="lazy" decoding="async" ref={videoImage} onLoad={() => setStreamFailed(false)} onError={() => { setStreamFailed(true); authFetch('/api/auth/me').catch(() => undefined) }} key={`${streamKey}:${annotated}`} src={`/api/video/feed?${mediaQuery}&stream=${streamKey}&annotated=${annotated}`} alt={annotated ? '连续视频与人物跟踪框，无骨架线条' : '当前监测原始视频'} /><span className="footage-label"><span className={`status-dot status-${status.connected ? 'ok' : 'muted'}`} />{streamFailed ? '画面未加载' : status.ended ? '回放结束 · 保留结果' : status.paused ? '回放暂停' : status.connected ? '连续视频' : '源未连接'}</span>{showDetection && <span className="frame-timestamp">最近检测 {seconds(detection?.frame_s)} · 帧间跟踪</span>}</div>
        {streamFailed && <div className="video-stream-error" role="alert"><span>视频画面未加载。可重新打开视频流，或直接用测试视频验证识别。</span><button onClick={() => setStreamKey(key => key+1)}>重试画面</button><button disabled={busy || !samples.some(s => s.id === 'fall')} onClick={() => command('source', { sample_id: 'fall' })}>播放测试视频</button></div>}
        <div className="person-readout"><span><strong>{detection?.state === 'ready' && !analysis?.stale ? detection.people.length : '—'}</strong> 个检测目标</span><p>{detection?.state === 'error' ? '人物检测暂不可用，原始视频与跌倒分类继续运行。' : detection?.state !== 'ready' ? '正在检测人物；检测框与画面时间同步。' : detection.people.length > 1 ? '各人物独立跟踪与分析，结果在下方人员状态中更新。' : detection.people.length === 0 ? '当前帧未检测到人物；这不代表无人或安全。' : '检测框对应当前画面，人员状态与分析依据在下方更新。'}</p></div>
        <div className="playback-line">{replay && <div className="playback-progress" role="progressbar" aria-label="视频回放进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round((status.position_s || 0) / (status.duration_s || 1) * 100)}><span style={{ width: `${Math.min(100, (status.position_s || 0) / (status.duration_s || 1) * 100)}%` }} /></div>}<span>{seconds(status.position_s)} / {replay ? seconds(status.duration_s) : 'LIVE · 持续采集'}</span></div>
        <div className="footage-controls">{replay ? <><button disabled={busy || status.ended} onClick={() => command('playback', { paused: !status.paused })}>{status.paused ? '继续播放' : '暂停播放'}</button><button disabled={busy} onClick={() => command('restart')}>从头重播</button></> : <button disabled={busy} onClick={() => command('camera')}>重新连接 C60E</button>}<span>{status.width} × {status.height} · 画面 {(status.display_fps || 0).toFixed(1)} / 源 {status.fps.toFixed(1)} FPS{!replay && status.capture_backend ? ` · ${status.capture_backend}` : ''}</span></div>
        {status.error && <p className="workbench-error">{status.error}。可选择示例视频或上传本地视频。</p>}
      </div>
      <aside className={`interpretation-column interpretation-${tone}`}>
        <div className="readout-caption">识别结果 / MODEL OUTPUT</div>
        <h2 aria-live="polite">{title}</h2>
        {analysis?.activity && <p className="model-activity">整幅动作候选：{activityNames[analysis.activity.label] || analysis.activity.label} · {(analysis.activity.confidence*100).toFixed(1)}%</p>}
        <p className="assessment-copy">{analysis?.model_state === 'error' ? analysis.model_error : unavailable ? '请检查视频源和模型更新状态，恢复后再参考当前判断。' : analysis?.assessment || '加载模型后，将对最近最多 5 秒的画面持续分析。'}</p>
        <p className="assessment-copy">姿态时序 TCN · {poseScore == null ? '正在收集有效关节片段' : `逐人跌倒特征最高 ${(poseScore*100).toFixed(1)}%`}{analysis?.state === 'model_disagreement' ? ' · 已触发复核提示' : ''}</p>
        {analysis?.short_fall_score != null && <p className="assessment-copy">最近 2 秒视频分值 {analysis.short_fall_score.toFixed(3)} · 连续 {analysis.short_consecutive_fall_windows || 0} 窗{analysis.decision_basis === 'supported_short_window' ? ' · 姿态与短窗口共同确认异常' : ' · 继续核查'}</p>}
        <div className="class-comparison"><div><span>RGB 正常活动特征</span><strong>{complete ? `${((1-(analysis?.fall_score || 0))*100).toFixed(1)}%` : '—'}</strong></div><div><span>RGB 跌倒特征</span><strong>{complete ? `${((analysis?.fall_score || 0)*100).toFixed(1)}%` : '—'}</strong></div><div className="class-bar"><i style={{ width: `${(analysis?.fall_score || 0)*100}%` }} /></div><small>{analysis?.state === 'model_disagreement' ? 'RGB与姿态时序结果分歧，不能据RGB低分判断安全' : analysis?.state === 'observation_limited' ? '整幅画面参考分数 · 人员状态见下方' : analysis?.activity_classes?.length ? '跌倒过程与倒地类合计分数 · 非实际风险概率' : 'RGB两类模型输出 · 非实际风险概率'}</small></div>
        <dl className="model-evidence"><div><dt>分析窗口</dt><dd>{complete ? `${seconds(analysis?.window_start_s)} — ${seconds(analysis?.window_end_s)}` : '至少收集 1 秒视频'}</dd></div><div><dt>判断范围</dt><dd>整个视频窗口 <small>{analysis?.activity_classes?.length ? `${analysis.activity_classes.length} 类训练动作` : 'ADL / Fall'}</small></dd></div><div><dt>连续跌倒窗口</dt><dd>{analysis?.consecutive_fall_windows || 0} <small>窗口</small></dd></div><div><dt>画面活动变化</dt><dd>{quality?.activity || '等待采样'}</dd></div><div><dt>模型特征变化</dt><dd>{analysis?.feature_change == null ? '等待下一窗口' : analysis.feature_change.toFixed(4)}</dd></div><div><dt>图像质量</dt><dd>{quality?.note || '等待采样'}</dd></div><div><dt>推理耗时 / 结果年龄</dt><dd>{complete ? `${analysis?.inference_ms?.toFixed(0)}ms / ${((analysis?.result_age_ms || 0)/1000).toFixed(1)}s` : '模型预热中'}</dd></div></dl>
        <p className="decision-note">{replay ? '当前为回放测试，疑似事件仅进入本次分析记录。' : '连续两个有效窗口出现跌倒特征时，生成待复核事件。'}{status.ended || status.paused ? '此处保留暂停位置的分析结果。' : '判断对应最近完成窗口，视频播放会继续。'}</p>
      </aside>
    </div>
    <div ref={spatialHost} className="spatial-deferred">{spatialReady ? <Suspense fallback={<div className="evidence-empty">正在读取人员状态…</div>}><PeopleStatus detection={detection} generation={analysis?.generation} sessionId={analysis?.session_id} /></Suspense> : <div className="evidence-empty">人员状态</div>}</div>
    <div className="window-analysis-grid">
      <div className="window-chart-panel"><div className="section-title-line"><div><span className="readout-caption">时序证据</span><h3>跌倒特征趋势</h3></div><span>{analysis?.windows_analyzed || 0} 个完成窗口</span></div>{history.length ? <div className="window-chart" aria-label="滑动窗口跌倒特征分数趋势">{spatialReady ? <Suspense fallback={<div className="evidence-empty">加载趋势图…</div>}><WindowScoreChart history={history} /></Suspense> : <div className="evidence-empty">窗口趋势</div>}</div> : <div className="evidence-empty">选择或上传视频后，真实窗口记录会出现在这里。</div>}<p className="evidence-caption">虚线为分类分界 0.5。连续窗口存在重叠，不能视为独立事件计数。</p></div>
      <div className="attention-evidence"><div className="section-title-line"><div><span className="readout-caption">空间证据</span><h3>模型响应区域</h3></div><label><input type="checkbox" checked={attention} onChange={event => setAttention(event.target.checked)} />显示</label></div>{attention && complete && analysis?.model_state === 'ready' ? <img loading="lazy" decoding="async" src={`/api/video/attention?${mediaQuery}&version=${analysis.updated_at_ms}`} alt={`最近完成窗口对${activityNames[analysis.attention_class || 'adl'] || '活动'}分类的空间响应图`} /> : <div className="evidence-empty">{attention ? '等待模型完成首个窗口' : '响应图已隐藏'}</div>}<p className="evidence-caption">{complete ? `${seconds(analysis?.window_start_s)}–${seconds(analysis?.window_end_s)} · ${analysis?.attention_class === 'fall' ? '跌倒' : '正常活动'}响应。` : ''}CAM 是整窗弱空间解释；白框为模型裁剪区域。</p></div>
    </div>
    <div className="session-ledger"><div className="section-title-line"><div><span className="readout-caption">本次分析记录</span><h3>状态与事件记录</h3></div><button className="text-button" onClick={exportSession} disabled={!complete}>导出分析 JSON</button></div><div className="session-events">{events.length ? [...events].reverse().map(event => <div className={`session-event event-${event.state}`} key={event.id}><time>{seconds(event.window_end_s)}</time><span className="ledger-indicator" /><div><strong>{stateNames[event.state]}</strong><p>{event.message}</p></div><span className="event-score">{event.fall_score.toFixed(3)}</span></div>) : <p className="evidence-empty">还没有完成的分析记录。</p>}</div></div>
  </section>
})

