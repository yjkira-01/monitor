import { memo, useEffect, useState } from 'react'
import { authFetch, getToken } from '../api/client'
import { activityNames, type Activity } from './activityLabels'

type Joint = { x: number; y: number; z: number; visibility: number }
type Person = { index: number; track_id?: number; bbox: number[]; detector_score?: number; landmarks: Joint[]; world_landmarks: Joint[]; visibility: number;
  pose_analysis?: { state: string; fall_score: number | null; pose_coverage?: number; model_revision?: string };
  observation?: { usable: boolean; spatial_valid: boolean; reason: string; posture: string; visible_joints: number; low_light?: boolean };
  analysis?: { state: string; fall_score: number | null; activity?: Activity; reason?: string; scope?: string; validated?: boolean; window_end_s?: number; windows_analyzed?: number; short_fall_score?: number | null; short_consecutive_fall_windows?: number } }
export type PeopleDetection = { state: string; model?: string; error?: string | null; frame_s?: number; people: Person[]; generation?: number; session_id?: string; updated_at_ms?: number; update_hz?: number; inference_ms?: number; stale?: boolean; paused?: boolean; ended?: boolean }
const postureNames: Record<string, string> = { upright: '躯干直立', horizontal: '躯干接近水平', bent_legs: '下肢弯曲', unknown: '局部观测' }
const modelNames: Record<string, string> = { adl: '正常活动特征', model_disagreement: '疑似跌倒 · 模型分歧待复核', fall_candidate: '疑似跌倒特征', fall_sustained: '连续跌倒特征 · 需复核', partial_adl: '局部画面 · 正常活动特征', partial_fall_candidate: '局部画面 · 疑似跌倒特征 · 需复核', unobservable: '缺少有效人物图像', warming_up: '建立个人窗口' }
export default memo(function PeopleStatus({ detection, generation = 0, sessionId }: { detection?: PeopleDetection; generation?: number; sessionId?: string }) {
  const [live, setLive] = useState<PeopleDetection>()
  useEffect(() => {
    setLive(undefined)
    let alive = true, timer: number
    const controller = new AbortController()
    async function refresh() {
      try {
        if (document.hidden) return
        const response = await authFetch('/api/video/pose', { headers: { Authorization: `Bearer ${getToken()}` }, signal: controller.signal })
        if (!response.ok) throw new Error('姿态接口不可用')
        const data: PeopleDetection = await response.json()
        if (alive && (!sessionId || data.session_id === sessionId) && (data.generation ?? 0) >= generation) setLive(current => {
          if (data.session_id === current?.session_id && (data.generation ?? 0) < (current?.generation ?? 0)) return current
          return data.generation === current?.generation && data.state === current?.state && data.updated_at_ms === current?.updated_at_ms && data.stale === current?.stale && data.paused === current?.paused && data.ended === current?.ended ? current : data
        })
      } catch { if (alive) setLive(current => current ? { ...current, stale: true } : undefined) }
      finally { if (alive) timer = window.setTimeout(refresh, document.hidden ? 1000 : 120) }
    }
    refresh()
    return () => { alive = false; controller.abort(); window.clearTimeout(timer) }
  }, [generation, sessionId])
  const current = live && live.session_id === sessionId && (live.generation ?? 0) >= generation ? live : detection
  const people = current?.stale ? [] : current?.people || []
  return <section className="person-status-section"><div className="section-title-line"><div><span className="readout-caption">PERSON TRACKS / 人员跟踪</span><h3>人员状态</h3></div><span>{people.length} 个跟踪目标</span></div><div className="person-status-list">{people.length ? people.map(p => <article key={p.track_id ?? p.index} className={`person-state person-state-${p.analysis?.state || 'warming_up'}`}>
    <div><strong>人物 {String(p.track_id ?? p.index).padStart(2, '0')}</strong><span>{postureNames[p.observation?.posture || 'unknown']} · 姿态观测</span></div>
    <div>
      <b>{p.analysis?.state?.includes('fall') || p.analysis?.state === 'model_disagreement' ? modelNames[p.analysis.state] : p.analysis?.activity ? `${activityNames[p.analysis.activity.label] || p.analysis.activity.label} · 训练模型` : modelNames[p.analysis?.state || 'warming_up']}</b>
      <small>{p.analysis?.fall_score == null ? p.analysis?.reason || p.observation?.reason || '等待有效观测' : `长窗口视频分值 ${p.analysis.fall_score.toFixed(3)} · ${p.analysis.windows_analyzed || 0} 窗口`}</small>
      {p.analysis?.short_fall_score != null && <small>最近2秒视频分值 {p.analysis.short_fall_score.toFixed(3)} · 连续 {p.analysis.short_consecutive_fall_windows || 0} 窗</small>}
      <small>姿态时序 TCN · {p.pose_analysis?.fall_score == null ? p.pose_analysis?.state === 'error' ? '模型暂不可用' : '收集有效关节片段' : `跌倒特征 ${(p.pose_analysis.fall_score*100).toFixed(1)}%`}</small>
      {p.analysis?.state === 'model_disagreement' && p.analysis.fall_score != null && <small>{p.analysis.reason}</small>}
      {p.analysis?.activity && <div className="activity-ranking">{Object.entries(p.analysis.activity.scores).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([name,score])=><span key={name}><em>{activityNames[name] || name}</em><i><u style={{width:`${score*100}%`}} /></i><b>{(score*100).toFixed(0)}%</b></span>)}</div>}
    </div>
    <span>{p.observation?.visible_joints ?? 0}/33 可见关节{p.observation?.low_light ? ' · 暗部增强' : ''}</span>
  </article>) : <p className="evidence-empty">等待人物跟踪结果</p>}</div><p className="evidence-caption">每个人独立维护视频窗口与姿态时序。短窗口须有动作与位置依据并连续确认，分值用于复核。人物编号为本次视频的跟踪编号，遮挡或误检可能使编号变化；躺姿不能直接判断为跌倒或睡眠。</p></section>
})
