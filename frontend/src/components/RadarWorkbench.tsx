import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { RadarSnapshot, RadarTrack, VitalData } from '../types'
import { radarDemoSnapshot } from '../data/demo'
import SpatialBoundary from './SpatialBoundary'
import { useRadarCloud } from './radarCloud'
import { radarIsFresh, radarSourceNames, radarStatus } from './radarStatus'
export { radarIsFresh, radarSourceNames, radarStatus } from './radarStatus'

const SpatialScene = lazy(() => import('./PersonScene'))
const time = (ms: number) => ms ? new Date(ms).toLocaleTimeString('zh-CN', { hour12: false }) : '暂无'
const number = (v: number | null | undefined, digits = 1) => v == null || !Number.isFinite(v) ? '—' : v.toFixed(digits)
const norm = (v: number[]) => Math.hypot(...v)

type Sample = { timestamp_ms: number; heart_rate: number | null; breath_rate: number | null; frame: number | undefined }
type History = Record<number, { path: { position: number[]; timestamp_ms: number; frame: number | null }[]; samples: Sample[] }>

function PlanView({ tracks, points, history, selected, onSelect }: { tracks: RadarTrack[]; points: number[][]; history: History; selected?: number; onSelect: (id: number) => void }) {
  const positions = [...points, ...tracks.map(t => t.position), ...Object.values(history).flatMap(h => h.path.map(p => p.position))]
  const extent = Math.max(3, Math.ceil(positions.reduce((v, p) => Math.max(v, Math.abs(p[0]), Math.abs(p[1])), 0)))
  const x = (v: number) => 240+v/extent*190, y = (v: number) => 240-v/extent*190
  return <svg className="radar-plan" viewBox="0 0 480 480" role="img" aria-label="雷达水平位置与目标轨迹，坐标单位米">
    <rect x="0" y="0" width="480" height="480" fill="#f4f8f5" />
    {[-1, -.5, 0, .5, 1].map(v => <g key={v}><line x1={x(v*extent)} x2={x(v*extent)} y1="50" y2="430" stroke="#d8e4dd" /><line x1="50" x2="430" y1={y(v*extent)} y2={y(v*extent)} stroke="#d8e4dd" /><text x={x(v*extent)} y="453" textAnchor="middle">{number(v*extent)}</text><text x="38" y={y(v*extent)+4} textAnchor="end">{number(v*extent)}</text></g>)}
    <text x="459" y="453">X</text><text x="30" y="25">Y / m</text>
    {points.map((p, i) => <circle key={i} cx={x(p[0])} cy={y(p[1])} r="2" fill="#527864" opacity=".75" />)}
    {tracks.map(t => <g key={t.target_id}>
      <polyline data-track-path={t.target_id} points={(history[t.target_id]?.path || []).map(p => `${x(p.position[0])},${y(p.position[1])}`).join(' ')} fill="none" stroke={selected === t.target_id ? '#17543c' : '#527864'} strokeWidth={selected === t.target_id ? 3.5 : 2.5} strokeLinecap="round" strokeLinejoin="round" />
      <g className="radar-plan-target" role="button" tabIndex={0} aria-label={`选择目标 T${t.target_id}`} aria-pressed={selected === t.target_id} onClick={() => onSelect(t.target_id)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(t.target_id) } }}>
        <circle cx={x(t.position[0])} cy={y(t.position[1])} r="16" fill="transparent" />
        <circle cx={x(t.position[0])} cy={y(t.position[1])} r={selected === t.target_id ? 7 : 5} fill="#28644f" stroke="white" strokeWidth="2" />
        <text x={x(t.position[0])+12} y={y(t.position[1])-12}>T{t.target_id}</text>
      </g>
    </g>)}
    <path d="M235 246 L240 234 L245 246 Z" fill="#526b5d" /><text x="253" y="259">雷达原点</text>
  </svg>
}

function Waveform({ values, label, frame, demo = false }: { values?: number[]; label: string; frame?: number; demo?: boolean }) {
  const data = (values || []).map((value, sample) => ({ sample, value }))
  return <section className="radar-wave"><div className="radar-section-head"><h3>{label}</h3><span>{data.length ? `${data.length} 个样本 · 帧 ${frame}` : '尚无有效波形'}</span></div>
    {data.length ? <div className="radar-wave-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ top: 12, right: 12, bottom: 5, left: 0 }}><CartesianGrid vertical={false} stroke="#e2ebe5" /><XAxis dataKey="sample" tick={{ fontSize: 10, fill: '#526b5d' }} tickLine={false} axisLine={false} /><YAxis width={44} tick={{ fontSize: 10, fill: '#526b5d' }} tickLine={false} axisLine={false} domain={['auto', 'auto']} /><Tooltip formatter={(value: number) => [number(value, 3), demo ? '合成幅值' : '固件幅值']} labelFormatter={v => `样本 ${v}`} /><Line type="linear" dataKey="value" stroke="#28644f" strokeWidth={1.6} dot={false} isAnimationActive={false} /></LineChart></ResponsiveContainer></div> : <div className="radar-chart-empty">目标存在且收到有效体征包后显示</div>}
    <p className="radar-caption">{demo ? '合成示意波形，横轴为示意样本序号；幅值没有生理测量意义。' : '横轴为当前包的样本序号，幅值单位待实机确认；逐包更新，不拼接为心电图。'}</p>
  </section>
}

export default function RadarWorkbench({ radar: actualRadar, now, vitals, source, error, onDevices }: { radar: RadarSnapshot | null; now: number; vitals: VitalData[]; source: string; error: string; onDevices: () => void }) {
  const [demo, setDemo] = useState(source === 'mock')
  const [optimizeCloud, setOptimizeCloud] = useState(true)
  useEffect(() => { setDemo(source === 'mock') }, [source])
  const demoData = useMemo(() => demo ? radarDemoSnapshot(now) : null, [demo, now])
  const radar = demo ? demoData : actualRadar
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [history, setHistory] = useState<History>({})
  const live = demo || !error && radarIsFresh(radar, now) && radar?.active_source === source
  const tracks = live ? radar!.tracks : []
  const selected = tracks.find(t => t.target_id === selectedId) || tracks[0]
  const vital = selected ? radar?.vitals.find(v => v.target_id === selected.target_id && !v.stale && radar.server_time_ms-v.timestamp_ms + Math.max(0, now-radar.received_at_ms) <= 3000) : undefined
  const cloud = useMemo(() => live ? radar!.points.map(([x, y, z]): [number, number, number] => [x, z, -y]) : [], [live, radar?.timestamp_ms, radar?.frame_number, radar?.session_id])
  const processed = useRadarCloud(cloud, radar?.timestamp_ms || 0, radar?.session_id || '', live && !demo)
  const shownCloud = optimizeCloud && !demo ? processed : cloud
  useEffect(() => { setHistory({}); setSelectedId(null) }, [radar?.session_id, source])
  useEffect(() => {
    if (!live || !radar) { setHistory(current => Object.keys(current).length ? {} : current); return }
    setHistory(current => {
      const next: History = {}
      let changed = Object.keys(current).length !== radar.tracks.length
      for (const track of radar.tracks) {
        const prev = current[track.target_id] || { path: [], samples: [] }
        const v = radar.vitals.find(v => v.target_id === track.target_id && !v.stale)
        const lastPath = prev.path[prev.path.length-1], lastSample = prev.samples[prev.samples.length-1]
        const pathNew = lastPath?.timestamp_ms !== radar.timestamp_ms || lastPath?.frame !== radar.frame_number
        const sampleNew = !!v && (lastSample?.timestamp_ms !== v.timestamp_ms || lastSample?.frame !== v.firmware_frame_number)
        changed ||= pathNew || sampleNew
        next[track.target_id] = {
          path: pathNew ? [...prev.path, { position: track.position, timestamp_ms: radar.timestamp_ms, frame: radar.frame_number }].slice(-120) : prev.path,
          samples: sampleNew ? [...prev.samples, { timestamp_ms: v.timestamp_ms, heart_rate: v.heart_rate, breath_rate: v.breath_rate, frame: v.firmware_frame_number }].slice(-120) : prev.samples,
        }
      }
      return changed ? next : current
    })
  }, [radar, live])
  const samples = selected ? history[selected.target_id]?.samples || [] : []
  const age = actualRadar?.age_ms == null ? null : actualRadar.age_ms+Math.max(0, now-actualRadar.received_at_ms)
  const stream = vitals.filter(v => v.source === source).slice(0, 8)
  const status = demo ? '合成演示运行中' : error ? '监测服务不可达' : radarStatus(radar, now)
  return <div className="neo-page radar-page">
    <div className="radar-intro"><div className="page-kicker">青禾康养 / 毫米波监测</div><h1>雷达与生命体征</h1><p>实时点云、目标轨迹及心率和呼吸波形。</p></div>
    <div className="radar-toolbar"><div className={live ? 'radar-connection live' : 'radar-connection'} role="status"><span className={`status-dot ${live ? 'status-ok' : 'status-muted'}`} />{status}</div><span>数据源：{demo ? '界面合成演示' : radarSourceNames[source] || source}</span><span>更新：{radar?.timestamp_ms ? time(radar.timestamp_ms) : '暂无'}</span><button className="radar-mode-button" aria-pressed={demo} onClick={() => setDemo(value => !value)}>{demo ? '查看实测' : '开启演示'}</button><button onClick={onDevices}>设备与数据 →</button></div>
    {demo && <div className="radar-notice radar-demo-notice" role="status">演示模式：点云、轨迹、心率与呼吸波形由本页面合成，仅用于交互验证，不是雷达实测。真实设备状态见下方采集信息。</div>}
    {!demo && (error || radar?.hardware.error || !live) && <div className="radar-notice">{error || radar?.hardware.error || (source !== 'real' ? '当前是样例或回放通道。切换到真实雷达后，空间和波形区域才显示实测数据。' : radar?.status === 'disconnected' ? '尚未连接雷达数据串口；点云、轨迹和生命体征保持空值。' : radar?.status === 'waiting' ? '串口已打开，等待可解析的雷达数据包。' : '未收到新鲜数据，当前读数和目标已清空。')}</div>}
    <section className="radar-spatial">
      <div className="radar-cloud"><div className="radar-section-head"><h2>三维点云</h2><span>{shownCloud.length} 显示点 / {cloud.length} 当前点 · {demo ? '合成示意' : optimizeCloud ? '优化显示' : '原始实测'}</span></div>{!demo && <div className="radar-cloud-controls view-switch"><button className={optimizeCloud ? 'active' : ''} aria-pressed={optimizeCloud} onClick={() => setOptimizeCloud(true)}>稳定显示</button><button className={!optimizeCloud ? 'active' : ''} aria-pressed={!optimizeCloud} onClick={() => setOptimizeCloud(false)}>原始点云</button></div>}<SpatialBoundary><Suspense fallback={<div className="radar-chart-empty">加载点云画布…</div>}><SpatialScene cloud points={shownCloud} label={demo ? '合成演示三维点云，非实测' : optimizeCloud ? '毫米波实测点云优化显示，保留500毫秒回波' : '毫米波原始实测三维点云'} resetKey={`radar:${radar?.session_id}`} sampleTime={radar?.timestamp_ms} /></Suspense></SpatialBoundary><p className="radar-caption">{!demo && optimizeCloud ? '500ms 实测回波累积、6cm 体素合并与孤立点抑制，仅用于显示。' : ''}X 横向 / Y 前向 / Z 高度，单位米。与视频尚未进行空间标定。</p></div>
      <div className="radar-position"><div className="radar-section-head"><h2>位置与轨迹</h2><span>最近 120 个接收位置</span></div><PlanView tracks={tracks} points={live ? radar!.points : []} history={live ? history : {}} selected={selected?.target_id} onSelect={setSelectedId} /><div className="radar-plan-legend"><span>· 回波点</span><span>● 跟踪目标</span><span>线段为接收轨迹</span></div></div>
    </section>
    <section className="radar-targets"><div className="radar-target-selector"><label htmlFor="radar-target">观察目标</label><select id="radar-target" value={selected?.target_id ?? ''} disabled={!tracks.length} onChange={e => setSelectedId(Number(e.target.value))}>{!tracks.length && <option value="">暂无跟踪目标</option>}{tracks.map(t => <option key={t.target_id} value={t.target_id}>目标 T{t.target_id}</option>)}</select><small>{demo ? '合成演示目标，不代表真实人物' : '设备轨迹编号，不代表老人身份'}</small></div><dl className="radar-target-details"><div><dt>当前位置 / m</dt><dd>{selected ? selected.position.map(v => number(v, 2)).join(' / ') : '—'}</dd></div><div><dt>距雷达 / m</dt><dd>{selected ? number(norm(selected.position), 2) : '—'}</dd></div><div><dt>速度 / m·s⁻¹</dt><dd>{selected ? number(norm(selected.velocity), 2) : '—'}</dd></div><div><dt>目标状态</dt><dd>{selected ? norm(selected.velocity) >= .1 ? '移动中¹' : '低速 / 静止¹' : '无有效目标'}</dd></div></dl></section>
    <section className="radar-signals"><div className="radar-waveforms"><Waveform demo={demo} label="心率相关波形" values={vital?.heart_waveform} frame={vital?.firmware_frame_number} /><Waveform demo={demo} label="呼吸波形" values={vital?.breath_waveform} frame={vital?.firmware_frame_number} /></div><aside className="radar-vital-readout"><div className="radar-section-head"><h2>{selected ? `${demo ? '演示目标' : '目标'} T${selected.target_id} 的体征` : '等待目标体征'}</h2></div><div className="radar-rate"><span>心率</span><strong data-testid="radar-heart-rate">{number(vital?.heart_rate, 0)}</strong><small>bpm</small></div><div className="radar-rate"><span>呼吸率</span><strong data-testid="radar-breath-rate">{number(vital?.breath_rate)}</strong><small>次 / 分</small></div><dl className="radar-signal-details"><div><dt>体征更新时间</dt><dd>{vital ? time(vital.timestamp_ms) : '暂无有效包'}</dd></div><div><dt>呼吸偏差 / 固件值</dt><dd>{number(vital?.breathing_deviation, 4)}</dd></div><div><dt>距离单元编号</dt><dd>{vital?.range_bin ?? '—'}</dd></div><div><dt>采集质量评分</dt><dd>尚未标定</dd></div></dl><div className="radar-review" role="status">{vital?.watch_reason || '体征需要目标匹配和有效波形。目标离场或数据超过 3 秒，当前读数自动清空。'}</div><p className="radar-caption">¹ 速度阈值 0.1 m/s 为待校准规则。静止不等同睡眠；未启用睡眠分期或呼吸暂停诊断。</p></aside></section>
    <section className="radar-trends"><div className="radar-section-head"><div><h2>{demo ? '合成体征趋势' : '接收期间的体征趋势'}</h2><p>{demo ? '合成样本用于验证趋势绘图，不是采集记录。' : '仅保留所选目标最近 120 次有效体征更新；切换数据源或目标离场后清空。'}</p></div><span>{samples.length} 次更新</span></div>{samples.length ? <div className="radar-trend-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={samples} margin={{ top: 15, right: 25, bottom: 5, left: 0 }}><CartesianGrid vertical={false} stroke="#e2ebe5" /><XAxis dataKey="timestamp_ms" type="number" domain={['dataMin', 'dataMax']} tickFormatter={time} tick={{ fontSize: 10, fill: '#526b5d' }} tickLine={false} axisLine={false} /><YAxis width={45} tick={{ fontSize: 10, fill: '#526b5d' }} tickLine={false} axisLine={false} /><Tooltip labelFormatter={v => time(Number(v))} formatter={(v: number, key: string) => [number(v), key === 'heart_rate' ? '心率 / bpm' : '呼吸 / 次/分']} /><Line name="heart_rate" dataKey="heart_rate" type="linear" stroke="#28644f" dot={{ r: 2 }} connectNulls={false} isAnimationActive={false} /><Line name="breath_rate" dataKey="breath_rate" type="linear" stroke="#8d633d" strokeDasharray="5 3" dot={{ r: 2 }} connectNulls={false} isAnimationActive={false} /></LineChart></ResponsiveContainer></div> : <div className="radar-chart-empty">暂无实收趋势，接入设备并收到有效体征后开始记录</div>}<div className="radar-trend-legend"><span>绿色：心率 / bpm</span><span>棕色虚线：呼吸率 / 次/分</span></div></section>
    <section className="radar-acquisition"><div className="radar-section-head"><h2>真实采集与信号状态</h2><span>只读设备快照</span></div><dl><div><dt>数据串口</dt><dd>{actualRadar?.hardware.port || '未配置'} · {actualRadar?.hardware.baud || 921600} baud</dd></div><div><dt>串口连接</dt><dd>{actualRadar?.hardware.connected ? '已打开' : '未连接'}</dd></div><div><dt>最后接收帧</dt><dd>{actualRadar?.frame_number ?? '暂无'}</dd></div><div><dt>距离最后接收</dt><dd>{age == null ? '暂无' : `${number(age/1000)} s`}</dd></div><div><dt>本帧数据块 / TLV</dt><dd>{actualRadar?.tlv_types.length ? actualRadar.tlv_types.join(' / ') : '暂无'}</dd></div><div><dt>本帧跟踪目标</dt><dd>{!error && radarIsFresh(actualRadar, now) ? actualRadar!.tracks.length : 0}</dd></div><div><dt>目标可信度 / 固件值</dt><dd>{!demo && selected ? number(selected.confidence, 3) : '—'} · 未校准概率</dd></div><div><dt>实机对照标定</dt><dd>未完成</dd></div></dl></section>
    {source !== 'real' && <section className="radar-stream"><div className="radar-section-head"><h2>{radarSourceNames[source] || source}记录</h2><span>用于链路验证，不是实时健康结论</span></div><div className="radar-table-wrap"><table><thead><tr><th>时间</th><th>设备</th><th>心率 / bpm</th><th>呼吸 / 次/分</th><th>来源</th></tr></thead><tbody>{stream.map((v, i) => <tr key={`${v.timestamp_ms}:${i}`}><td>{time(v.timestamp_ms)}</td><td>{v.device_id}</td><td>{number(v.heart_rate, 0)}</td><td>{number(v.breath_rate)}</td><td>{radarSourceNames[v.source || ''] || v.source}</td></tr>)}</tbody></table></div>{!stream.length && <div className="radar-chart-empty">等待所选数据源的样本</div>}</section>}
  </div>
}
