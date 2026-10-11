import { memo, useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { clone } from 'three/examples/jsm/utils/SkeletonUtils.js'
import { humanRig } from './humanRig'
import { PoseMotion, type Point, validSpatialPose } from './poseMotion'

export type SpatialSubject = { id: number; points: Point[]; confidence: number[]; estimated?: boolean[]; observedCount?: number; mode?: 'observed' | 'temporal' | 'reference' }
type Props = { points: Point[]; subjects?: SpatialSubject[]; cloud?: boolean; label: string; sampleTime?: number; resetKey?: string; paused?: boolean }
type Runtime = { update: (props: Props) => void; view: (side: boolean) => void; zoom: (factor: number) => void; rotate: (angle: number) => void }
const bones = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [29, 31], [28, 30], [30, 32]]

export default memo(function PersonScene(props: Props) {
  const { points, subjects = [], cloud = false, label } = props
  const canvas = useRef<HTMLCanvasElement>(null)
  const runtime = useRef<Runtime>()
  const latest = useRef(props); latest.current = props
  const [error, setError] = useState(''), [fps, setFps] = useState(0), [retry, setRetry] = useState(0)
  useEffect(() => {
    const node = canvas.current
    if (!node) return
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({ canvas: node, antialias: false, alpha: false, powerPreference: 'low-power' }) }
    catch { setError('三维渲染不可用'); return }
    setError(''); setFps(0); node.dataset.engine = `three.js r${THREE.REVISION}`
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25))
    renderer.setClearColor('#f2f6f3'); renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(34, 1, .05, 40)
    camera.position.set(.65, .35, 4.2)
    const controls = new OrbitControls(camera, node)
    controls.enablePan = false; controls.minDistance = 1.5; controls.maxDistance = 15
    scene.add(new THREE.HemisphereLight('#ffffff', '#afc1b5', 2.5))
    const light = new THREE.DirectionalLight('#fff6e9', 3); light.position.set(-3, 4, 5); scene.add(light)
    const grid = new THREE.GridHelper(8, 32, '#aec8b8', '#d7e3dc'); grid.position.y = -1; scene.add(grid)
    const geometry = new THREE.BufferGeometry()
    const positions = new THREE.BufferAttribute(new Float32Array(4096*3), 3).setUsage(THREE.DynamicDrawUsage)
    geometry.setAttribute('position', positions); geometry.setDrawRange(0, 0)
    const spriteData = new Uint8Array(16*16*4)
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
      const i = (y*16+x)*4
      spriteData.set([255, 255, 255, Math.hypot(x-7.5, y-7.5) <= 7 ? 255 : 0], i)
    }
    const sprite = new THREE.DataTexture(spriteData, 16, 16); sprite.needsUpdate = true
    const material = new THREE.PointsMaterial({ color: '#17644f', size: .05, sizeAttenuation: true, map: sprite, transparent: true, alphaTest: .5 })
    const dots = new THREE.Points(geometry, material); dots.frustumCulled = false; scene.add(dots)
    type Entry = { motion: PoseMotion; confidence: number[]; x: number; visible: boolean; dirty: boolean; sample: string; group: THREE.Group; avatar?: THREE.Group; drive?: ReturnType<typeof humanRig> }
    const entries = new Map<number, Entry>()
    let template: THREE.Group | undefined
    let alive = true, lost = false, inView = false, hasData = false, paused = false, simulated = false
    let rawCloud: Point[] = [], raf = 0, last = 0, frames = 0, measuredAt = performance.now(), visibleCount = 0, framedKey = ''
    function view(side: boolean) {
      const center = new THREE.Vector3()
      let distance = 4.2+Math.max(0, visibleCount-1)*1.2
      if (cloud && !simulated && rawCloud.length) {
        const sphere = new THREE.Box3().setFromPoints(rawCloud.map(p => new THREE.Vector3(...p))).getBoundingSphere(new THREE.Sphere())
        center.copy(sphere.center); distance = Math.max(1.5, Math.min(15, sphere.radius/Math.sin(17*Math.PI/180)*1.4))
      }
      camera.position.copy(center).add(new THREE.Vector3(side ? distance : .05, .35, side ? .1 : distance))
      controls.target.copy(center); controls.update(); wake()
    }
    function disposeSkeleton(root: THREE.Object3D) { root.traverse(o => { if (o instanceof THREE.SkinnedMesh) o.skeleton.dispose() }) }
    function disposeTemplate(root: THREE.Group) {
      const materials = new Set<THREE.Material>(), geometries = new Set<THREE.BufferGeometry>()
      root.traverse(o => { if (o instanceof THREE.Mesh) { geometries.add(o.geometry); (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => materials.add(m)) } })
      geometries.forEach(g => g.dispose()); materials.forEach(m => m.dispose()); disposeSkeleton(root)
    }
    function attach(entry: Entry) {
      if (!template || entry.avatar || cloud) return
      entry.avatar = clone(template) as THREE.Group; entry.drive = humanRig(entry.avatar)
      entry.group.add(entry.avatar); entry.dirty = true
    }
    function fillCloud() {
      let count = 0
      const write = (x: number, y: number, z: number) => { if (count < positions.count) positions.setXYZ(count++, x, y, z) }
      if (simulated) for (const entry of entries.values()) {
        if (!entry.visible) continue
        for (const [a, b] of bones) {
          if (Math.min(entry.confidence[a] ?? 0, entry.confidence[b] ?? 0) < .5) continue
          const start = entry.motion.joints[a], end = entry.motion.joints[b]
          for (let i = 0; i <= 10; i++) for (let j = 0; j < 4; j++) {
            const t = i/10, angle = j*Math.PI/2
            write(entry.x+start.x*(1-t)+end.x*t+.025*Math.cos(angle), start.y*(1-t)+end.y*t, start.z*(1-t)+end.z*t+.025*Math.sin(angle))
          }
        }
      } else for (const point of rawCloud.slice(0, 2048)) if (point.every(Number.isFinite)) write(...point)
      positions.needsUpdate = true; geometry.setDrawRange(0, count); node!.dataset.points = String(count)
    }
    function animate(now: number) {
      raf = 0
      if (!alive || lost || !inView || document.hidden) return
      if (last && now-last < 1000/30-.5) { raf = requestAnimationFrame(animate); return }
      const dt = last ? (now-last)/1000 : 1/30; last = now
      let moving = false, changed = false
      for (const entry of entries.values()) {
        if (!entry.visible) continue
        const interpolating = entry.motion.advance(dt)
        moving ||= interpolating; changed ||= interpolating || entry.dirty
        if (entry.drive && (interpolating || entry.dirty)) entry.drive(entry.motion.joints, entry.confidence)
        entry.dirty = false
      }
      if (cloud && simulated && changed) fillCloud()
      try { renderer.render(scene, camera) } catch { setError('三维画布暂不可用，视频识别仍在运行'); return }
      frames++
      if (now-measuredAt >= 1000) {
        const value = Math.round(frames*1000/(now-measuredAt))
        node!.dataset.renderFps = String(value); setFps(value); frames = 0; measuredAt = now
        node!.dataset.geometries = String(renderer.info.memory.geometries); node!.dataset.drawCalls = String(renderer.info.render.calls)
      }
      if (hasData && ((!paused && (!cloud || simulated)) || moving)) raf = requestAnimationFrame(animate)
    }
    function wake() { if (!raf && !lost && inView && !document.hidden) { last = 0; raf = requestAnimationFrame(animate) } }
    function update(next: Props) {
      paused = Boolean(next.paused); simulated = cloud && (next.resetKey || '').startsWith('simulation:')
      material.size = simulated ? .021 : .05
      const valid = (next.subjects || []).filter(s => validSpatialPose(s.points, s.confidence)).slice(0, 4)
      const ids = new Set(valid.map(s => s.id))
      for (const entry of entries.values()) { entry.visible = false; entry.group.visible = false }
      if (valid.length !== visibleCount) { visibleCount = valid.length; camera.position.z = 4.2+Math.max(0, visibleCount-1)*1.2; controls.update() }
      valid.forEach((subject, i) => {
        let entry = entries.get(subject.id)
        if (!entry) {
          for (const [id, old] of entries) if (entries.size >= 4 && !ids.has(id)) {
            scene.remove(old.group); if (old.avatar) disposeSkeleton(old.avatar); entries.delete(id)
          }
          entry = { motion: new PoseMotion(), confidence: [], x: 0, visible: true, dirty: true, sample: '', group: new THREE.Group() }
          entries.set(subject.id, entry); scene.add(entry.group); attach(entry)
        }
        entry.visible = true; entry.group.visible = !cloud; entry.confidence = subject.confidence
        entry.x = (i-(valid.length-1)/2)*1.3; entry.group.position.x = entry.x
        const sample = `${next.resetKey}:${next.sampleTime}`
        if (sample !== entry.sample) {
          entry.motion.push(subject.points, next.sampleTime ?? performance.now()/1000, `${next.resetKey}:${subject.id}`)
          entry.sample = sample; entry.dirty = true
        }
      })
      rawCloud = next.points
      hasData = cloud && !simulated ? rawCloud.some(p => p.every(Number.isFinite)) : valid.length > 0
      dots.visible = cloud && hasData; node!.dataset.activePeople = String(valid.length)
      node!.dataset.estimatedPeople = String(valid.filter(s => s.mode && s.mode !== 'observed').length)
      node!.dataset.subjectIds = valid.map(s => s.id).join(',')
      if (cloud && !simulated) fillCloud()
      if (cloud && !simulated && hasData && framedKey !== next.resetKey) { framedKey = next.resetKey || 'radar'; view(false) }
      if (!hasData) { geometry.setDrawRange(0, 0); node!.dataset.points = '0' }
      wake()
    }
    if (!cloud) new GLTFLoader().load('/models/care-human.glb', gltf => {
      if (!alive) { disposeTemplate(gltf.scene); return }
      template = gltf.scene
      template.traverse(o => {
        if (!(o instanceof THREE.Mesh)) return
        o.frustumCulled = false
        ;(Array.isArray(o.material) ? o.material : [o.material]).forEach(m => {
          if (m instanceof THREE.MeshStandardMaterial) { m.vertexColors = false; m.metalness = .08; m.roughness = .65; m.color.set('#99bbae') }
        })
      })
      entries.forEach(attach); node.dataset.avatar = 'ready'; wake()
    }, undefined, () => { if (alive) { setError('人物模型加载失败'); node.dataset.avatar = 'error' } })
    controls.addEventListener('change', wake)
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      if (!width || !height || lost) return
      renderer.setSize(width, height, false); camera.aspect = width/height; camera.updateProjectionMatrix(); wake()
    }); observer.observe(node)
    const intersection = new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting
      if (inView) { measuredAt = performance.now(); frames = 0; wake() }
      else { cancelAnimationFrame(raf); raf = 0; setFps(0) }
    }); intersection.observe(node)
    const onVisibility = () => { if (document.hidden) { cancelAnimationFrame(raf); raf = 0 } else { measuredAt = performance.now(); frames = 0; wake() } }
    const onLost = (event: Event) => { event.preventDefault(); lost = true; cancelAnimationFrame(raf); raf = 0; setError('三维画布连接中断，视频识别仍在运行') }
    const onRestored = () => setRetry(v => v+1)
    document.addEventListener('visibilitychange', onVisibility)
    node.addEventListener('webglcontextlost', onLost); node.addEventListener('webglcontextrestored', onRestored)
    runtime.current = {
      update,
      view,
      zoom: factor => { camera.position.sub(controls.target).multiplyScalar(factor).add(controls.target); controls.update(); wake() },
      rotate: angle => { camera.position.sub(controls.target).applyAxisAngle(new THREE.Vector3(0, 1, 0), angle).add(controls.target); controls.update(); wake() },
    }; update(latest.current)
    return () => {
      alive = false; cancelAnimationFrame(raf); runtime.current = undefined
      observer.disconnect(); intersection.disconnect(); document.removeEventListener('visibilitychange', onVisibility)
      node.removeEventListener('webglcontextlost', onLost); node.removeEventListener('webglcontextrestored', onRestored)
      controls.dispose(); entries.forEach(entry => { if (entry.avatar) disposeSkeleton(entry.avatar) })
      if (template) disposeTemplate(template)
      geometry.dispose(); material.dispose(); sprite.dispose(); grid.geometry.dispose()
      ;(Array.isArray(grid.material) ? grid.material : [grid.material]).forEach(m => m.dispose())
      renderer.dispose(); if (!node.isConnected) renderer.forceContextLoss()
    }
  }, [cloud, retry])
  useEffect(() => { runtime.current?.update(props) }, [points, props.subjects, props.sampleTime, props.resetKey, props.paused])
  const empty = cloud && !(props.resetKey || '').startsWith('simulation:') ? !points.length : !subjects.some(s => validSpatialPose(s.points, s.confidence))
  return <div className="spatial-scene"><canvas key={retry} ref={canvas} aria-label={label} tabIndex={0} onKeyDown={event => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); runtime.current?.rotate(event.key === 'ArrowLeft' ? -.15 : .15) }
    if (event.key === '+' || event.key === '-') runtime.current?.zoom(event.key === '+' ? .85 : 1.15)
  }} />{(empty || error) && <div className="spatial-empty">{error || (cloud ? '等待有效点云数据' : '等待人物姿态与可用历史')}<small>{error ? '视频和个人状态不受影响' : '取得可用观测后自动显示'}</small>{error && <button onClick={() => setRetry(v => v+1)}>重试三维画布</button>}</div>}
    {!empty && !error && <span className="spatial-fps">{props.paused ? '画面已停留' : `${fps || '—'} FPS 渲染`} · {cloud && !(props.resetKey || '').startsWith('simulation:') ? `${points.length} 点` : `${subjects.length} 人`}</span>}
    <div className="spatial-navigation"><span>拖动旋转 · 滚轮缩放</span><div><button onClick={() => runtime.current?.view(false)}>正面</button><button onClick={() => runtime.current?.view(true)}>侧面</button><button aria-label="放大三维视图" onClick={() => runtime.current?.zoom(.85)}>＋</button><button aria-label="缩小三维视图" onClick={() => runtime.current?.zoom(1.15)}>−</button><button onClick={() => runtime.current?.view(false)}>复位</button></div></div>
  </div>
})
