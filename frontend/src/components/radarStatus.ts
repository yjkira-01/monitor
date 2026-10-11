import type { RadarSnapshot } from '../types'

export const radarSourceNames: Record<string, string> = { real: '真实雷达', mock: '模拟样本', mmfi: 'MMFi 样例', replay: '录制回放' }
export function radarIsFresh(data: RadarSnapshot | null, now: number) {
  return !!data && data.active_source === 'real' && data.hardware.connected && !data.stale &&
    data.age_ms != null && data.age_ms + Math.max(0, now-data.received_at_ms) <= 3000
}
export function radarStatus(data: RadarSnapshot | null, now: number) {
  if (!data) return '状态读取中'
  if (data.status === 'inactive') return '真实采集未启用'
  if (data.status === 'disconnected') return '雷达未连接'
  if (data.status === 'waiting') return '等待有效数据包'
  if (!radarIsFresh(data, now)) return '数据已过期'
  return data.tracks.length ? '正在跟踪' : '未发现跟踪目标'
}
