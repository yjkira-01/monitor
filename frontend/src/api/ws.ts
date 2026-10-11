import type { StreamMessage } from '../types'

export function createWs(
  token: string,
  onMessage: (msg: StreamMessage) => void,
  onStatus: (online: boolean) => void,
  source?: string,
) {
  let ws: WebSocket | null = null
  let closed = false
  let timer: ReturnType<typeof setTimeout> | undefined

  const connect = () => {
    if (closed) return
    const params = new URLSearchParams({ token })
    if (source) params.set('source', source)
    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/stream?${params}`)
    ws.onopen = () => { if (!closed) onStatus(true) }
    ws.onclose = () => {
      if (closed) return
      onStatus(false)
      if (!closed) timer = setTimeout(connect, 3000)
    }
    ws.onmessage = (event) => { if (!closed) onMessage(JSON.parse(event.data)) }
  }

  connect()

  return () => {
    closed = true
    clearTimeout(timer)
    ws?.close()
  }
}
