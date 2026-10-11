import { useMemo, useRef } from 'react'
import type { Point } from './poseMotion'

// Display only: 500ms measured returns, 6cm voxels; never interpolate unordered points.
export class RadarCloudBuffer {
  private frames: { time: number; points: Point[] }[] = []
  private key = ''
  private time = -1
  private result: Point[] = []

  clear() { this.frames = []; this.time = -1; this.result = []; return this.result }

  push(points: Point[], time: number, key: string): Point[] {
    if (!Number.isFinite(time)) return this.clear()
    if (key !== this.key || time < this.time || time-this.time > 500) this.clear()
    this.key = key
    if (time === this.time) return this.result
    this.time = time
    this.frames = [...this.frames.filter(f => time-f.time <= 500), {
      time, points: points.filter(p => p.length === 3 && p.every(Number.isFinite)).slice(0, 2048),
    }].slice(-6)
    const cells = new Map<string, { xyz: Point; sum: Point; count: number }>()
    for (const frame of this.frames) for (const point of frame.points) {
      const xyz = point.map(v => Math.floor(v/.06)) as Point
      const id = xyz.join(',')
      const cell = cells.get(id) || { xyz, sum: [0, 0, 0], count: 0 }
      for (let axis = 0; axis < 3; axis++) cell.sum[axis] += point[axis]
      cell.count++; cells.set(id, cell)
    }
    const result: Point[] = []
    for (const cell of cells.values()) {
      let supported = cell.count > 1
      for (let x = -1; x <= 1 && !supported; x++) for (let y = -1; y <= 1 && !supported; y++) for (let z = -1; z <= 1 && !supported; z++) {
        if ((x || y || z) && cells.has(`${cell.xyz[0]+x},${cell.xyz[1]+y},${cell.xyz[2]+z}`)) supported = true
      }
      if (supported) result.push(cell.sum.map(v => v/cell.count) as Point)
    }
    this.result = result.slice(-2048)
    return this.result
  }
}

export function useRadarCloud(points: Point[], time: number, key: string, active: boolean) {
  const buffer = useRef(new RadarCloudBuffer())
  return useMemo(() => active ? buffer.current.push(points, time, key) : buffer.current.clear(), [points, time, key, active])
}
