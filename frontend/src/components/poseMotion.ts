import { Matrix4, Quaternion, Vector3 } from 'three'

export type Point = [number, number, number]
export function validSpatialPose(points: Point[], confidence: number[]) {
  if (points.length !== 33 || confidence.length !== 33 || !points.every(p => p.length === 3 && p.every(v => Number.isFinite(v) && Math.abs(v) < 4))) return false
  if (![11, 12, 23, 24, 25, 26, 27, 28].every(i => confidence[i] >= .5) || confidence.filter(v => v >= .5).length < 18) return false
  const torso = new Vector3(...points[11]).add(new Vector3(...points[12])).sub(new Vector3(...points[23])).sub(new Vector3(...points[24])).multiplyScalar(.5).length()
  return torso > .08 && torso < 1.2
}

export type ReconstructedPose = { points: Point[]; confidence: number[]; estimated: boolean[]; observedCount: number; mode: 'observed' | 'temporal' | 'reference' }
const reference: Point[] = [
  [0,.72,.08], [.035,.75,.07], [.055,.75,.07], [.075,.75,.06], [-.035,.75,.07], [-.055,.75,.07], [-.075,.75,.06], [.09,.73,0], [-.09,.73,0], [.03,.69,.07], [-.03,.69,.07],
  [.2,.51,0], [-.2,.51,0], [.25,.24,0], [-.25,.24,0], [.27,-.02,.02], [-.27,-.02,.02], [.29,-.06,.02], [-.29,-.06,.02], [.28,-.07,.05], [-.28,-.07,.05], [.25,-.04,.06], [-.25,-.04,.06],
  [.11,0,0], [-.11,0,0], [.11,-.43,0], [-.11,-.43,0], [.11,-.86,0], [-.11,-.86,0], [.11,-.89,-.04], [-.11,-.89,-.04], [.11,-.91,.13], [-.11,-.91,.13],
]
const limbs = [[11,13],[13,15],[15,17],[15,19],[15,21],[12,14],[14,16],[16,18],[16,20],[16,22],[23,25],[25,27],[27,29],[27,31],[24,26],[26,28],[28,30],[28,32]]

// Display-only completion: retained joint directions are estimates, never classification evidence.
export class PoseHistory {
  private previous?: ReconstructedPose
  private body = new Quaternion()
  private center = new Vector3()
  private scale = 1
  private time = -1
  private lastObserved = -1
  private seen = Array(33).fill(-Infinity)
  private key = ''

  push(points: Point[], confidence: number[], time: number, key: string): ReconstructedPose | undefined {
    if (!Number.isFinite(time)) return undefined
    if (key !== this.key || time < this.time || time-this.time > .8) {
      this.previous = undefined; this.seen.fill(-Infinity); this.lastObserved = -1
    }
    if (key === this.key && time === this.time) return this.previous
    this.key = key; this.time = time
    const observed = reference.map((_, i) => confidence[i] >= .5 && points[i]?.length === 3 && points[i].every(v => Number.isFinite(v) && Math.abs(v) < 4))
    const input = reference.map((_, i) => observed[i] ? new Vector3(...points[i]) : new Vector3())
    const torso = [11,12,23,24].filter(i => observed[i])
    if (observed.filter(Boolean).length < 3 || (!this.previous && torso.length < 2)) {
      if (this.previous && time-this.lastObserved <= .6) {
        return { ...this.previous, estimated: Array(33).fill(true), observedCount: 0, mode: 'temporal' }
      }
      return undefined
    }
    const oldBody = this.body.clone(), body = this.previous ? oldBody.clone() : new Quaternion()
    const x = new Vector3(1,0,0).applyQuaternion(body), y = new Vector3(0,1,0).applyQuaternion(body), z = new Vector3()
    if (observed[23] && observed[24]) x.copy(input[23]).sub(input[24])
    else if (observed[11] && observed[12]) x.copy(input[11]).sub(input[12])
    if (x.lengthSq() < .002) return undefined
    x.normalize()
    const average = (ids: number[]) => ids.reduce((v,i) => v.add(input[i]),new Vector3()).divideScalar(ids.length)
    const shoulders = [11,12].filter(i => observed[i]), hips = [23,24].filter(i => observed[i])
    let scale = this.previous ? this.scale : 1
    if (shoulders.length && hips.length) {
      y.copy(average(shoulders)).sub(average(hips))
      if (shoulders.length === 1) y.addScaledVector(x, shoulders[0] === 11 ? -.2*scale : .2*scale)
      if (hips.length === 1) y.addScaledVector(x, hips[0] === 23 ? .11*scale : -.11*scale)
      if (y.length() < .08 || y.length() > 1.2) return undefined
      scale = Math.max(.65,Math.min(1.35,y.length()/.51))
    } else if (!this.previous && shoulders.length === 2 && observed[0]) {
      y.copy(input[0]).sub(average(shoulders))
    }
    z.crossVectors(x,y)
    if (z.lengthSq() < 1e-5) return undefined
    z.normalize(); y.crossVectors(z,x).normalize()
    body.setFromRotationMatrix(new Matrix4().makeBasis(x,y,z))
    const center = this.previous ? this.center.clone() : new Vector3()
    if (hips.length) {
      center.copy(average(hips))
      if (hips.length === 1) center.addScaledVector(x, hips[0] === 23 ? -.11*scale : .11*scale)
    } else if (shoulders.length) {
      center.copy(average(shoulders)).addScaledVector(y,-.51*scale)
      if (shoulders.length === 1) center.addScaledVector(x, shoulders[0] === 11 ? -.2*scale : .2*scale)
    }
    const delta = body.clone().multiply(oldBody.invert())
    const base = reference.map((p,i) => this.previous
      ? new Vector3(...this.previous.points[i]).sub(this.center).applyQuaternion(delta).multiplyScalar(scale/this.scale).add(center)
      : new Vector3(...p).multiplyScalar(scale).applyQuaternion(body).add(center))
    const result = base.map((p,i) => observed[i] ? input[i].clone() : p.clone())
    for (const [a,b] of limbs) {
      const length = new Vector3(...reference[b]).sub(new Vector3(...reference[a])).length()*scale
      if (observed[b] && (result[b].distanceTo(result[a]) < length*.3 || result[b].distanceTo(result[a]) > length*1.9)) observed[b] = false
      if (!observed[b]) result[b].copy(result[a]).add(base[b].clone().sub(base[a]))
    }
    observed.forEach((value,i) => { if (value) this.seen[i] = time })
    const estimated = observed.map(v => !v)
    const mode = !estimated.some(Boolean) ? 'observed' : estimated.some((v,i) => v && time-this.seen[i] > .8) ? 'reference' : 'temporal'
    const output: ReconstructedPose = { points: result.map(v => v.toArray() as Point), confidence: observed.map((v,i) => v ? confidence[i] : .55), estimated, observedCount: observed.filter(Boolean).length, mode }
    if (!validSpatialPose(output.points,output.confidence)) return undefined
    this.body.copy(body); this.center.copy(center); this.scale = scale; this.lastObserved = time; this.previous = output
    return output
  }

  reference(): ReconstructedPose {
    return { points: this.previous?.points || reference.map(p => [...p] as Point), confidence: Array(33).fill(.55),
      estimated: Array(33).fill(true), observedCount: 0, mode: 'reference' }
  }
}
export class PoseMotion {
  readonly joints = Array.from({ length: 33 }, () => new Vector3())
  readonly target = Array.from({ length: 33 }, () => new Vector3())
  private raw = new Float64Array(99)
  private derivative = new Float64Array(99)
  private time = -1
  private key = ''

  push(points: Point[], time: number, key: string) {
    const reset = key !== this.key || this.time < 0 || time <= this.time || time-this.time > .6
    const dt = Math.max(.001, time-this.time)
    const alpha = (cutoff: number) => 1/(1+1/(2*Math.PI*cutoff*dt))
    points.forEach((point, i) => point.forEach((value, axis) => {
      const index = i*3+axis
      if (reset) {
        this.target[i].setComponent(axis, value)
        this.joints[i].setComponent(axis, value)
        this.derivative[index] = 0
      } else {
        const velocity = (value-this.raw[index])/dt
        this.derivative[index] += alpha(1)*(velocity-this.derivative[index])
        const previous = this.target[i].getComponent(axis)
        this.target[i].setComponent(axis, previous+alpha(1.5+1.5*Math.abs(this.derivative[index]))*(value-previous))
      }
      this.raw[index] = value
    }))
    this.time = time; this.key = key
  }

  advance(dt: number) {
    let moving = false
    const alpha = 1-Math.exp(-Math.min(dt, .05)/.045)
    this.joints.forEach((joint, i) => {
      if (joint.distanceToSquared(this.target[i]) > 1e-8) { joint.lerp(this.target[i], alpha); moving = true }
      else joint.copy(this.target[i])
    })
    return moving
  }
}
