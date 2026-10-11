import assert from 'node:assert/strict'
import fs from 'node:fs'
import ts from 'typescript'

const target = new URL('./radarCloud.check.mjs', import.meta.url)
try {
  fs.writeFileSync(target, ts.transpileModule(fs.readFileSync(new URL('../src/components/radarCloud.ts', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
  }).outputText)
  const { RadarCloudBuffer } = await import(target.href)
  const buffer = new RadarCloudBuffer()
  const points = [[.01, .01, .01], [.02, .015, .01], [5, 5, 5], [NaN, 0, 0]]
  const original = structuredClone(points)
  const first = buffer.push(points, 1000, 'first')
  assert.equal(first.length, 1, 'an isolated distant return is suppressed')
  assert(first.every(p => p.every(Number.isFinite)))
  assert.deepEqual(points, original, 'raw measurement arrays must remain unchanged')
  assert.equal(buffer.push(points, 1000, 'first'), first, 'duplicate frames must not accumulate')
  const second = buffer.push([[.03, .01, .01], [.04, .015, .01]], 1100, 'first')
  assert(Math.abs(second[0][0]-first[0][0]) < .02, 'short jitter is reduced without interpolating point indices')
  assert(buffer.push([], 1200, 'first').length > 0, 'brief sparse packets share a bounded return window')
  assert.equal(buffer.push([], 1750, 'first').length, 0, 'old returns cannot persist after the time limit')
  buffer.push(points, 1800, 'first')
  assert.equal(buffer.push([[5, 5, 5]], 1850, 'new-session').length, 0, 'source changes discard old returns')
  assert.deepEqual(buffer.clear(), [])
  assert.deepEqual(buffer.push(points, NaN, 'invalid'), [])
  const many = Array.from({ length: 4000 }, (_, i) => [i*.01, .1, 1])
  for (let i = 0; i < 20; i++) assert(buffer.push(many, 2000+i*25, 'dense').length <= 2048)

  const recording = new URL('../../artifacts/radar_research/real_cloud_frames.json', import.meta.url)
  if (fs.existsSync(recording)) {
    const frames = JSON.parse(fs.readFileSync(recording, 'utf8'))
    const live = new RadarCloudBuffer(), raw = [], processed = []
    const started = performance.now()
    frames.forEach((frame, i) => {
      const output = live.push(frame.points, i*90, 'recorded-sr112')
      assert(output.every(p => p.length === 3 && p.every(Number.isFinite)))
      if (i > 10) { raw.push(frame.points.length); processed.push(output.length) }
    })
    const stats = values => {
      const mean = values.reduce((s, v) => s+v, 0)/values.length
      const sd = Math.sqrt(values.reduce((s, v) => s+(v-mean)**2, 0)/values.length)
      return { mean, sd, coefficient_of_variation: mean ? sd/mean : null, empty_frames: values.filter(v => !v).length }
    }
    const report = { kind: 'replay_of_measured_uart_points', frames: frames.length, processing_ms_per_frame: (performance.now()-started)/frames.length, raw: stats(raw), processed: stats(processed) }
    fs.writeFileSync(new URL('../../artifacts/radar_research/cloud_processing_audit.json', import.meta.url), JSON.stringify(report, null, 2))
    console.log(JSON.stringify(report))
  }
  console.log('Radar cloud checks passed: noise, finite values, raw preservation, duplicate frames, jitter, expiry, source reset, bounded size')
} finally {
  fs.rmSync(target, { force: true })
}
