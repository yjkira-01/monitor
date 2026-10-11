import { useEffect, useState } from 'react'
import { Card, CardHead } from './ui'
import { authFetch, getToken } from '../api/client'

type VideoStatus = {
  source: string
  connected: boolean
  frames: number
  fps: number
  width: number
  height: number
  error: string | null
}

const initial: VideoStatus = { source: '0', connected: false, frames: 0, fps: 0, width: 0, height: 0, error: null }

export default function VideoPreview() {
  const [status, setStatus] = useState<VideoStatus>(initial)
  const [streamKey, setStreamKey] = useState(Date.now())

  useEffect(() => {
    let alive = true
    const refresh = async () => {
      try {
        const response = await authFetch('/api/video/status', { headers: { Authorization: `Bearer ${getToken()}` } })
        if (response.ok && alive) setStatus(await response.json())
      } catch {
        if (alive) setStatus((prev) => ({ ...prev, connected: false, error: '后端视频接口未连接' }))
      }
    }
    refresh()
    const timer = window.setInterval(refresh, 1500)
    return () => { alive = false; window.clearInterval(timer) }
  }, [])

  return (
    <Card className="p-5">
      <CardHead
        title="实时视频"
        extra={
          <div className="flex items-center gap-3 text-xs">
            <span className={status.connected ? 'text-sage-deep' : 'text-ink-hint'}>
              <span className={`mr-1.5 inline-block h-2 w-2 rounded-full ${status.connected ? 'bg-sage' : 'bg-ink-hint'}`} />
              {status.connected ? '视频在线' : '等待视频源'}
            </span>
            <button className="text-sage hover:text-sage-deep" onClick={() => setStreamKey(Date.now())}>刷新</button>
          </div>
        }
      />
      <div className="relative overflow-hidden rounded-xl border border-line bg-[#151A16]">
        <img key={streamKey} src={`/api/video/feed?stream=${streamKey}&token=${encodeURIComponent(getToken() || '')}`} alt="实时视频流" className="block aspect-video w-full object-contain" />
        <div className="pointer-events-none absolute bottom-3 left-3 rounded bg-black/60 px-2 py-1 text-[11px] text-white/80">
          {status.connected ? `${status.width}×${status.height} · ${status.fps.toFixed(1)} FPS` : (status.error || '本地画面未连接')}
        </div>
      </div>
      <div className="mt-3 flex items-center justify-between text-xs text-ink-sub">
        <span>源：{status.source}</span>
        <span>边缘端处理 · 画面不上云</span>
      </div>
    </Card>
  )
}
