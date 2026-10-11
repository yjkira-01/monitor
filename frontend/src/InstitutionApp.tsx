import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { authFetch, getAlerts, getToken, login, logout, setToken } from './api/client'
import { createWs } from './api/ws'
import type { AlertData, RadarSnapshot, VitalData } from './types'

type Screen = 'overview' | 'alerts' | 'vitals' | 'devices' | 'elders' | 'settings'
import VideoWorkbench, { type VideoAnalysis, type VideoStatus } from './components/VideoWorkbench'
import { radarIsFresh, radarSourceNames, radarStatus } from './components/radarStatus'
const RadarWorkbench = lazy(() => import('./components/RadarWorkbench'))
type Insight = { summary: string; provider: string; model: string; error?: string | null; generated_at_ms: number }
type LlmConfig = { configured: boolean; model: string; base_url: string; api_key_set: boolean; api_key_masked?: string | null; runtime: boolean }

const neuShadow = 'neu-shadow'
const neuButton = `neu-button ${neuShadow}`

function timeText(timestamp?: number) {
  return timestamp ? new Date(timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }) : '--:--'
}

function dateText() {
  return new Intl.DateTimeFormat('zh-CN', { month: 'long', day: 'numeric', weekday: 'long' }).format(new Date())
}

function NeuButton({ children, primary = false, onClick, disabled = false, className = '' }: { children: React.ReactNode; primary?: boolean; onClick?: () => void; disabled?: boolean; className?: string }) {
  return <button className={`${neuButton} ${primary ? 'neu-button-primary' : ''} ${className}`} onClick={onClick} disabled={disabled}>{children}</button>
}

function StatusDot({ tone = 'ok' }: { tone?: 'ok' | 'warn' | 'danger' | 'muted' }) {
  return <span className={`status-dot status-${tone}`} aria-hidden="true" />
}

function SectionHeading({ index, title, detail, action }: { index: string; title: string; detail?: string; action?: React.ReactNode }) {
  return <div className="section-heading"><div className="section-index">{index}</div><div><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{action && <div className="section-action">{action}</div>}</div>
}

function AttentionPanel({ alerts, onOpenAlerts, onHandle }: { alerts: AlertData[]; onOpenAlerts: () => void; onHandle: (id: number) => void }) {
  const urgent = alerts.filter((item) => !item.is_handled).slice(0, 4)
  return <section className="neu-panel attention-panel"><div className="panel-topline"><div><div className="panel-kicker">机构事件 · 包含模拟数据</div><h2>待处理预警</h2></div><button className="text-button" onClick={onOpenAlerts}>全部预警</button></div><div className="attention-summary"><strong>{alerts.filter((a) => !a.is_handled).length}</strong><span>条待处理<br />近期预警</span></div><div className="attention-list">{urgent.length === 0 ? <div className="empty-state">当前没有未处理预警<br /><span>仅统计当前列表</span></div> : urgent.map((alert) => <div className="attention-item" key={alert.id}><StatusDot tone={alert.level === 'red' || alert.level === 'critical' ? 'danger' : 'warn'} /><div className="attention-copy"><strong>{alert.type === 'fall' ? '疑似跌倒' : alert.type === 'device' ? '设备状态' : '生命体征'}</strong><p>{alert.message}</p></div><div className="attention-time">{timeText(alert.timestamp_ms)}<button onClick={() => onHandle(alert.id)}>处理</button></div></div>)}</div></section>
}

function InsightPanel({ insight, busy, configured, onGenerate, contextKey }: { insight: Insight | null; busy: boolean; configured: boolean; onGenerate: (question?: string) => void; contextKey: string }) {
  const [question, setQuestion] = useState('请总结视频最近窗口的活动变化、异常线索和建议，区分回放与现场')
  const [auto, setAuto] = useState(true)
  const generate = useRef(onGenerate)
  generate.current = onGenerate
  const lastContext = useRef('')
  const running = useRef(busy)
  running.current = busy
  useEffect(() => {
    if (!auto) return
    const timer = window.setInterval(() => { if (!running.current) generate.current() }, 30000)
    return () => window.clearInterval(timer)
  }, [auto])
  useEffect(() => {
    if (!auto || busy || lastContext.current === contextKey) return
    const timer = window.setTimeout(() => {
      if (!running.current) { lastContext.current = contextKey; generate.current() }
    }, 750)
    return () => window.clearTimeout(timer)
  }, [auto, busy, contextKey])
  return <section className="neu-panel insight-panel"><div className="panel-topline"><div><div className="panel-kicker">辅助研判</div><h2>监测摘要</h2></div><span className={`source-chip ${configured ? 'source-chip-live' : ''}`}>{configured ? '大模型已配置' : '本地规则摘要'}</span></div><p className="insight-copy" aria-live="polite">{(insight?.summary || '根据近期预警、生命体征和视频识别结果生成摘要。').replace(/\*\*([^*]+)\*\*/g, '$1').replace(/^#{1,6}\s+/gm, '')}</p><label className="auto-report"><input type="checkbox" checked={auto} onChange={event => setAuto(event.target.checked)} />状态变化时更新 · 每 30 秒刷新</label><div className="insight-controls"><input aria-label="摘要要求" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="输入摘要要求" /><NeuButton primary onClick={() => onGenerate(question)} disabled={busy}>{busy ? '生成中' : '生成摘要'}</NeuButton></div><div className="panel-footline"><span>{insight ? `${insight.provider} · ${timeText(insight.generated_at_ms)}` : '读取模型窗口、质量与事件记录，不上传视频'}</span>{insight?.error && <span className="muted-warning">模型不可用，已回退本地摘要</span>}</div></section>
}

function Overview({ alerts, vitals, radar, now, source, radarError, video, insight, insightBusy, llmConfigured, onNavigate, onHandleAlert, onGenerateInsight, onVideoStatus }: { alerts: AlertData[]; vitals: VitalData[]; radar: RadarSnapshot | null; now: number; source: string; radarError: string; video: VideoStatus; insight: Insight | null; insightBusy: boolean; llmConfigured: boolean; onNavigate: (screen: Screen) => void; onHandleAlert: (id: number) => void; onGenerateInsight: (question?: string) => void; onVideoStatus: (next: VideoStatus) => void }) {
  const radarLive = !radarError && radarIsFresh(radar, now) && radar?.active_source === source
  const latest = source === 'real' ? radarLive ? radar!.vitals.find(v => !v.stale && radar!.tracks.some(t => t.target_id === v.target_id) && radar!.server_time_ms-v.timestamp_ms+now-radar!.received_at_ms <= 3000) : undefined : vitals.find(v => v.source === source && now-v.timestamp_ms <= 3000)
  return <div className="neo-page observation-page">
    <div className="observation-intro"><div><div className="page-kicker">青禾康养 / 视频监测</div><h1>实时监测</h1><p>查看现场或回放中的人员状态、识别依据与事件记录。</p></div><div className="observation-status"><span className="status-dot status-ok" />{video.analysis?.model_state === 'ready' ? '视频模型已就绪' : '视频模型加载中'}<span>视频主窗口 / 短窗口复核</span></div></div>
    <VideoWorkbench status={video} onStatus={onVideoStatus} />
    <div className="report-care-grid"><InsightPanel insight={insight} busy={insightBusy} configured={llmConfigured} onGenerate={onGenerateInsight} contextKey={`${video.analysis?.session_id}:${video.analysis?.generation}:${video.analysis?.state}:${video.analysis?.model_state}:${video.connected}:${video.analysis?.stale}:${video.ended}`} /><AttentionPanel alerts={alerts} onOpenAlerts={() => onNavigate('alerts')} onHandle={onHandleAlert} /></div>
    <div className="radar-data-strip"><div><strong>毫米波 / 独立采集通道</strong><span>数据源：{radarSourceNames[source] || source} · {radarError ? '服务不可达' : radarStatus(radar, now)}</span></div><span>呼吸 <strong>{latest?.breath_rate == null ? '—' : latest.breath_rate.toFixed(1)}</strong> 次/分</span><span>心率 <strong>{latest?.heart_rate == null ? '—' : latest.heart_rate.toFixed(0)}</strong> bpm</span><button className="text-button" onClick={() => onNavigate('vitals')}>雷达与体征 →</button></div>
  </div>
}

function AlertsScreen({ alerts, onHandle }: { alerts: AlertData[]; onHandle: (id: number) => void }) {
  const [filter, setFilter] = useState<'all' | 'open' | 'closed'>('all'); const [selected, setSelected] = useState<number | null>(alerts[0]?.id ?? null)
  const list = alerts.filter((item) => filter === 'all' || (filter === 'open' ? !item.is_handled : item.is_handled)); const current = alerts.find((item) => item.id === selected) || list[0]
  return <div className="neo-page"><div className="page-intro compact"><div><div className="page-kicker">事件中心 · 预警记录</div><h1>预警管理</h1><p>查看预警详情，记录处理状态。</p></div></div><div className="filter-row">{[['all', '全部'], ['open', '未处理'], ['closed', '已处理']].map(([key, label]) => <button key={key} className={filter === key ? 'filter-pill filter-active' : 'filter-pill'} onClick={() => setFilter(key as typeof filter)}>{label}<strong>{key === 'all' ? alerts.length : key === 'open' ? alerts.filter((a) => !a.is_handled).length : alerts.filter((a) => a.is_handled).length}</strong></button>)}</div><div className="alerts-layout"><section className="neu-panel event-list">{list.length === 0 ? <div className="empty-state">没有匹配的预警<br /><span>可调整筛选条件</span></div> : list.map((alert) => <button key={alert.id} className={current?.id === alert.id ? 'event-row event-row-active' : 'event-row'} onClick={() => setSelected(alert.id)}><StatusDot tone={alert.level === 'red' || alert.level === 'critical' ? 'danger' : 'warn'} /><span><strong>{alert.type === 'fall' ? '跌倒风险' : alert.type === 'device' ? '设备事件' : '生命体征'}</strong><small>{alert.message}</small></span><time>{timeText(alert.timestamp_ms)}</time></button>)}</section><section className="neu-panel event-detail">{current ? <><div className="panel-kicker">事件详情 · #{current.id}</div><h2>{current.message}</h2><div className="detail-grid"><span>风险等级</span><strong className={current.level === 'red' ? 'text-danger' : 'text-warn'}>{current.level === 'red' ? '立即处理' : '需要确认'}</strong><span>发生时间</span><strong>{new Date(current.timestamp_ms).toLocaleString('zh-CN')}</strong><span>来源</span><strong>端侧视频与监测数据</strong></div><div className="detail-actions">{!current.is_handled && <NeuButton primary onClick={() => onHandle(current.id)}>标记为已处理</NeuButton>}</div></> : <div className="empty-state">选择左侧事件查看详情</div>}</section></div></div>
}

function DevicesScreen({ source, sourceOptions, onSource, video, online, radar, now, radarError, llmConfigured, onRadar }: { source: string; sourceOptions: string[]; onSource: (source: string) => void; video: VideoStatus; online: boolean; radar: RadarSnapshot | null; now: number; radarError: string; llmConfigured: boolean; onRadar: () => void }) {
  return <div className="neo-page"><div className="page-intro compact"><div><div className="page-kicker">设备与数据 · 实际采集状态</div><h1>设备与数据</h1><p>查看采集状态、数据来源与服务连接。</p></div></div>
    <div className="radar-toolbar"><label htmlFor="radar-source">毫米波数据源</label><select id="radar-source" value={source} onChange={e => onSource(e.target.value)}>{sourceOptions.map(item => <option key={item} value={item}>{radarSourceNames[item] || item}</option>)}</select><span>切换后清除旧通道的当前样本</span></div>
    <section className="device-live-list"><article><div className="device-live-heading"><span className="device-mark">VIS</span><div><h2>视频采集</h2><p>{video.source_mode === 'live' ? '实时摄像头' : '视频文件回放'} · {video.source}</p></div><strong>{video.connected ? '已连接' : '未连接'}</strong></div><dl><div><dt>实际分辨率</dt><dd>{video.width} × {video.height}</dd></div><div><dt>采集帧率</dt><dd>{(video.display_fps || video.fps || 0).toFixed(1)} FPS</dd></div><div><dt>识别模型</dt><dd>{video.analysis?.model_state || '等待状态'}</dd></div></dl>{video.error && <p className="radar-notice">{video.error}</p>}</article>
    <article><div className="device-live-heading"><span className="device-mark">RF</span><div><h2>SR112 毫米波</h2><p>实测点云 · 目标跟踪 · 固件体征与波形</p></div><strong>{radarError ? '服务不可达' : radarStatus(radar, now)}</strong></div><dl><div><dt>数据端口</dt><dd>{radar?.hardware.port || '未配置'}</dd></div><div><dt>串口速率</dt><dd>{radar?.hardware.baud || 921600} baud</dd></div><div><dt>最后接收帧</dt><dd>{radar?.frame_number ?? '暂无'}</dd></div></dl><div className="device-live-foot"><span>{radar?.hardware.error || '设备配置与实机标定尚待验证'}</span><button className="text-button" onClick={onRadar}>打开雷达与体征 →</button></div></article></section>
    <section className="radar-acquisition"><div className="radar-section-head"><h2>服务连接状态</h2></div><dl><div><dt>WebSocket</dt><dd>{online ? '已连接' : '未连接'}</dd></div><div><dt>雷达信号接口</dt><dd>{radarError ? radarError : radar ? '快照接口可用' : '读取中'}</dd></div><div><dt>大模型摘要</dt><dd>{llmConfigured ? '已配置外部 API' : '本地规则摘要'}</dd></div><div><dt>雷达训练状态</dt><dd>独立采集，暂不进入主模型训练</dd></div></dl></section>
  </div>
}

function EldersScreen() {
  return <div className="neo-page"><div className="page-intro compact"><div><div className="page-kicker">住户信息</div><h1>住户档案</h1><p>视频跟踪编号用于本次观测，不能自动对应住户身份。</p></div></div><section className="neu-panel"><div className="empty-state">尚未接入住户档案服务<br /><span>可在桌面应用中管理档案；网页版目前不读取桌面应用的个人资料。</span></div></section></div>
}

function SettingsScreen({ config, onSaved }: { config: LlmConfig; onSaved: (next: LlmConfig) => void }) {
  const [baseUrl, setBaseUrl] = useState(config.base_url)
  const [model, setModel] = useState(config.model)
  const [apiKey, setApiKey] = useState('')
  const [clearKey, setClearKey] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [testBusy, setTestBusy] = useState(false)

  async function save() {
    setBusy(true); setError(''); setMessage('')
    try {
      const response = await authFetch('/api/insights/config', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${getToken() || ''}` }, body: JSON.stringify({ base_url: baseUrl, model, api_key: apiKey || null, clear_api_key: clearKey }) })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || '保存失败')
      onSaved(data); setApiKey(''); setClearKey(false); setMessage('配置已生效，密钥只保存在当前后端进程内。')
    } catch (cause) { setError(cause instanceof Error ? cause.message : '保存失败') } finally { setBusy(false) }
  }

  async function testSummary() {
    setTestBusy(true); setError(''); setMessage('')
    try {
      const response = await authFetch('/api/insights/summary', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${getToken() || ''}` }, body: JSON.stringify({ hours: 24, question: '请返回一句配置连通性测试结果' }) })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || '测试失败')
      setMessage(`${data.provider === 'local' ? '当前使用本地摘要' : '大模型接口已返回'}：${data.summary.slice(0, 90)}`)
    } catch (cause) { setError(cause instanceof Error ? cause.message : '测试失败') } finally { setTestBusy(false) }
  }

  return <div className="neo-page"><div className="page-intro compact"><div><div className="page-kicker">系统设置 · 监测摘要</div><h1>大模型接口设置</h1><p>配置接口地址、模型名称与 API Key，用于生成监测摘要。</p></div><span className={`source-chip ${config.configured ? 'source-chip-live' : ''}`}>{config.configured ? '接口已配置' : '尚未配置密钥'}</span></div><div className="settings-layout"><section className="neu-panel settings-form"><SectionHeading index="01" title="大模型 API" detail="配置只作用于当前后端进程" /><label className="settings-field">接口地址<input className="neu-input" value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} placeholder="https://api.openai.com/v1" /><small>必须是 http 或 https 地址，末尾不需要加 /chat/completions。</small></label><label className="settings-field">模型名称<input className="neu-input" value={model} onChange={(event) => setModel(event.target.value)} placeholder="gpt-4o-mini" /></label><label className="settings-field">API Key<input className="neu-input" type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder={config.api_key_set ? `已保存 ${config.api_key_masked || '密钥'}` : 'sk-...'} autoComplete="new-password" /><small>留空会保留已保存密钥；勾选清除后再保存可以移除运行时密钥。</small></label><label className="settings-check"><input type="checkbox" checked={clearKey} onChange={(event) => setClearKey(event.target.checked)} />清除当前运行时 API Key</label><div className="settings-actions"><NeuButton primary onClick={save} disabled={busy}>{busy ? '保存中' : '保存配置'}</NeuButton><NeuButton onClick={testSummary} disabled={testBusy}>{testBusy ? '测试中' : '测试摘要'}</NeuButton></div>{message && <div className="settings-message">{message}</div>}{error && <div className="form-error">{error}</div>}</section><section className="neu-panel settings-notes"><SectionHeading index="02" title="安全和回退" detail="页面会明确显示当前来源" /><div className="settings-note-list"><div><strong>密钥</strong><span>不会回传到页面，也不会写入日志；当前仅保存在后端内存。</span></div><div><strong>无密钥</strong><span>自动使用本地规则摘要，监测页面仍可继续使用。</span></div><div><strong>原始视频</strong><span>摘要读取模型窗口、质量与事件记录，不上传视频画面。</span></div><div><strong>兼容服务</strong><span>支持 OpenAI-compatible Chat Completions 接口。</span></div></div></section></div></div>
}

export default function InstitutionApp() {
  const [token, setTokenState] = useState(getToken() || '')
  const [sessionMessage, setSessionMessage] = useState('')
  const [screen, setScreen] = useState<Screen>('overview')
  const [alerts, setAlerts] = useState<AlertData[]>([])
  const [vitals, setVitals] = useState<VitalData[]>([])
  const [radar, setRadar] = useState<RadarSnapshot | null>(null)
  const [radarError, setRadarError] = useState('')
  const [now, setNow] = useState(Date.now())
  const [video, setVideo] = useState<VideoStatus>({ source: '0', connected: false, frames: 0, fps: 0, width: 0, height: 0, error: null })
  const [online, setOnline] = useState(false)
  const [source, setSource] = useState('mock')
  const [sourceOptions, setSourceOptions] = useState<string[]>(['mock', 'mmfi', 'replay', 'real'])
  const [insight, setInsight] = useState<Insight | null>(null)
  const [insightBusy, setInsightBusy] = useState(false)
  const [llmConfigured, setLlmConfigured] = useState(false)
  const [llmConfig, setLlmConfig] = useState<LlmConfig>({ configured: false, model: 'gpt-4o-mini', base_url: 'https://api.openai.com/v1', api_key_set: false, runtime: false })

  useEffect(() => {
    const expired = () => {
      setTokenState(''); setScreen('overview'); setAlerts([]); setVitals([]); setRadar(null)
      setOnline(false)
      setVideo({ source: '0', connected: false, frames: 0, fps: 0, width: 0, height: 0, error: null }); setInsight(null)
      setSessionMessage('登录已失效，请重新登录以恢复视频与实时数据。')
    }
    window.addEventListener('auth-expired', expired)
    return () => window.removeEventListener('auth-expired', expired)
  }, [])

  useEffect(() => {
    if (!token) return
    let alive = true
    authFetch('/api/auth/me').catch(() => undefined)
    getAlerts().then(data => { if (alive) setAlerts(data) }).catch(() => undefined)
    authFetch('/api/data-source').then(response => response.ok ? response.json() : Promise.reject()).then(data => { if (alive) { setSource(data.source); setSourceOptions(data.options) } }).catch(() => undefined)
    authFetch('/api/insights/config').then(response => response.ok ? response.json() : Promise.reject()).then(data => { if (alive) { setLlmConfig(data); setLlmConfigured(Boolean(data.configured)) } }).catch(() => undefined)
    return () => { alive = false }
  }, [token])

  useEffect(() => {
    if (!token) return
    let alive = true
    let timer: number
    const refresh = async () => {
      try {
        const response = await authFetch('/api/video/status', { headers: { Authorization: `Bearer ${token}` } })
        if (!response.ok) throw new Error('视频状态请求失败')
        const data = await response.json()
        if (alive) setVideo(current => {
          const sameSession = data.analysis?.session_id === current.analysis?.session_id
          if (sameSession && (data.analysis?.generation || 0) < (current.analysis?.generation || 0)) return current
          const sameWindow = sameSession && ['generation', 'updated_at_ms', 'state', 'stale', 'model_state'].every(key => data.analysis?.[key] === current.analysis?.[key as keyof VideoAnalysis])
          return sameWindow ? { ...data, analysis: current.analysis } : data
        })
      } catch { if (alive) setVideo(current => ({ ...current, connected: false, error: '监测服务连接中断，请检查后端' })) }
      finally { if (alive) timer = window.setTimeout(refresh, document.hidden ? 2000 : 750) }
    }
    refresh()
    return () => { alive = false; window.clearTimeout(timer) }
  }, [token])

  useEffect(() => {
    if (!token) return
    let alive = true, timer: number
    const controller = new AbortController()
    const clock = window.setInterval(() => setNow(Date.now()), 500)
    async function refresh() {
      try {
        const response = await authFetch('/api/edge/radar', { headers: { Authorization: `Bearer ${token}` }, signal: controller.signal })
        if (!response.ok) throw new Error('雷达快照请求失败')
        const data: RadarSnapshot = { ...await response.json(), received_at_ms: Date.now() }
        if (alive) {
          setRadar(data); setRadarError('')
          if (data.active_source !== source) { setSource(data.active_source); setVitals([]) }
        }
      } catch { if (alive) setRadarError('雷达服务连接中断，请检查后端') }
      finally { if (alive) timer = window.setTimeout(refresh, document.hidden ? 1500 : screen === 'vitals' ? 100 : 1000) }
    }
    refresh()
    return () => { alive = false; controller.abort(); window.clearTimeout(timer); window.clearInterval(clock) }
  }, [token, screen, source])

  useEffect(() => {
    if (!token) return
    return createWs(token, (message) => {
      if (message.type === 'alert') setAlerts((items) => [message.data, ...items.filter((item) => item.id !== message.data.id)].slice(0, 80))
      if (message.type === 'vital' && message.data.source === source) setVitals((items) => [message.data, ...items].slice(0, 30))
      if (message.type === 'pose') {
        const next = message.data as VideoAnalysis
        setVideo(item => next.session_id === item.analysis?.session_id && ((next.generation || 0) < (item.analysis?.generation || 0) || next.generation === item.analysis?.generation && (next.updated_at_ms || 0) < (item.analysis?.updated_at_ms || 0)) ? item : { ...item, analysis: next })
      }
    }, setOnline)
  }, [token, source])

  const unhandled = useMemo(() => alerts.filter((item) => !item.is_handled).length, [alerts])

  async function generateInsight(question = '') {
    setInsightBusy(true)
    try {
      const response = await authFetch('/api/insights/summary', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }, body: JSON.stringify({ hours: 24, question }) })
      if (!response.ok) throw new Error('摘要服务请求失败')
      setInsight(await response.json())
    } catch { setInsight({ summary: '摘要暂不可用，请参考上方最新视频分析。', provider: 'error', model: '', generated_at_ms: Date.now(), error: '请求失败' }) } finally { setInsightBusy(false) }
  }

  async function changeSource(next: string) {
    try {
      const response = await authFetch('/api/data-source', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }, body: JSON.stringify({ source: next }) })
      if (!response.ok) throw new Error('切换失败')
      const data = await response.json()
      setSource(data.source); setVitals([]); setRadar(null)
    } catch { window.alert('数据源切换失败，请检查连接后重试。') }
  }

  async function markHandled(id: number) {
    try {
      const response = await authFetch(`/api/alerts/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }, body: JSON.stringify({ is_handled: true }) })
      if (!response.ok) throw new Error('预警处理保存失败')
      const updated = await response.json()
      setAlerts(items => items.map(item => item.id === id ? updated : item))
    } catch { window.alert('处理结果未能保存，请检查连接后重试。') }
  }

  if (!token) return <LoginScreen notice={sessionMessage} onSuccess={() => { setSessionMessage(''); setTokenState(getToken() || '') }} />

  const nav: { id: Screen; label: string; hint: string; key: string }[] = [
    { id: 'overview', label: '监测台', hint: '视频与识别结果', key: '01' }, { id: 'alerts', label: '预警管理', hint: `近期 ${unhandled} 条待处理`, key: '02' },
    { id: 'vitals', label: '雷达与体征', hint: '点云 · 轨迹 · 波形', key: '03' }, { id: 'devices', label: '设备与数据', hint: '采集 · 质量 · 来源', key: '04' }, { id: 'elders', label: '住户档案', hint: '人员 · 床位 · 联系人', key: '05' }, { id: 'settings', label: '系统设置', hint: '大模型 API', key: '06' },
  ]
  const titles: Record<Screen, [string, string]> = { overview: ['监测台', '实时视频与识别结果'], alerts: ['预警管理', '预警记录与处理状态'], vitals: ['雷达与体征', '点云、轨迹与生命体征'], devices: ['设备与数据', '采集设备与数据连接'], elders: ['老人档案', '人员信息与床位'], settings: ['系统设置', 'API 配置与监测摘要'] }
  const content = screen === 'overview' ? <Overview alerts={alerts} vitals={vitals} radar={radar} now={now} source={source} radarError={radarError} video={video} insight={insight} insightBusy={insightBusy} llmConfigured={llmConfigured} onNavigate={setScreen} onHandleAlert={markHandled} onGenerateInsight={generateInsight} onVideoStatus={setVideo} /> : screen === 'alerts' ? <AlertsScreen alerts={alerts} onHandle={markHandled} /> : screen === 'vitals' ? <RadarWorkbench radar={radar} now={now} vitals={vitals} source={source} error={radarError} onDevices={() => setScreen('devices')} /> : screen === 'devices' ? <DevicesScreen source={source} sourceOptions={sourceOptions} onSource={changeSource} video={video} online={online} radar={radar} now={now} radarError={radarError} llmConfigured={llmConfigured} onRadar={() => setScreen('vitals')} /> : screen === 'elders' ? <EldersScreen /> : <SettingsScreen config={llmConfig} onSaved={(next) => { setLlmConfig(next); setLlmConfigured(Boolean(next.configured)) }} />

  return <div className="care-app"><aside className="care-sidebar"><div className="brand-lockup"><div className="brand-mark">禾</div><div><strong>青禾康养</strong><span>智能康养监测</span></div></div><div className="sidebar-rule" /><div className="nav-caption">机构工作台</div><nav>{nav.map((item) => <button key={item.id} className={screen === item.id ? 'nav-item nav-item-active' : 'nav-item'} onClick={() => setScreen(item.id)}><span className="nav-key">{item.key}</span><span className="nav-copy"><strong>{item.label}</strong><small>{item.hint}</small></span>{item.id === 'alerts' && unhandled > 0 && <em>{unhandled}</em>}</button>)}</nav><div className="sidebar-spacer" /><div className="privacy-note"><StatusDot tone="ok" /><div><strong>端侧隐私模式</strong><span>原始视频不上云</span></div></div><button className={`${neuButton} sidebar-logout`} onClick={() => { logout(); setTokenState('') }}>退出当前账号</button></aside><main className="care-main"><header className="care-header"><div><span className="header-path">青禾康养 / {titles[screen][0]}</span><strong>{titles[screen][1]}</strong></div><div className="header-status"><span><StatusDot tone={online ? 'ok' : 'muted'} />{online ? '实时链路已连接' : '正在重连'}</span><span>{dateText()}</span><div className="user-chip">本地监测账号</div></div></header><Suspense fallback={<div className="evidence-empty">正在读取监测视图…</div>}>{content}</Suspense></main></div>
}

function LoginScreen({ onSuccess, notice = '' }: { onSuccess: () => void; notice?: string }) {
  const [username, setUsername] = useState('admin'); const [password, setPassword] = useState('admin123'); const [error, setError] = useState(''); const [busy, setBusy] = useState(false)
  async function submit() { setBusy(true); setError(''); try { const result = await login(username, password); setToken(result.access_token); onSuccess() } catch { setError('登录失败，请检查服务是否启动或账号密码。') } finally { setBusy(false) } }
  return <div className="login-stage"><div className="login-orbit orbit-one" /><div className="login-orbit orbit-two" /><section className="neu-panel login-panel"><div className="brand-lockup login-brand"><div className="brand-mark">禾</div><div><strong>青禾康养</strong><span>智能康养监测</span></div></div><div className="login-copy"><div className="page-kicker">机构监测端 · 本地部署</div><h1>智能康养监测系统</h1><p>视频识别、雷达监测与事件管理。</p></div><div className="login-form">{notice && <p className="session-notice" role="status">{notice}</p>}<label>账号<input className="neu-input" value={username} onChange={(e) => setUsername(e.target.value)} /></label><label>密码<input className="neu-input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && submit()} /></label>{error && <div className="form-error">{error}</div>}<NeuButton primary onClick={submit} disabled={busy}>{busy ? '正在进入' : '进入监测台'}</NeuButton></div><div className="login-foot">默认演示账号 admin / admin123 · 数据仅用于项目联调</div></section></div>
}
