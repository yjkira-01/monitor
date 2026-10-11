import assert from 'node:assert/strict'
import fs from 'node:fs'
import ts from 'typescript'

const created = []
const storage = new Map()
globalThis.localStorage = { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) }
globalThis.window = new EventTarget()
let expired = 0
window.addEventListener('auth-expired', () => expired++)
async function read(name, relative) {
  const target = new URL(`./${name}.check.mjs`, import.meta.url)
  const source = fs.readFileSync(new URL(relative, import.meta.url), 'utf8').replaceAll('import.meta.env', '({})')
  fs.writeFileSync(target, ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } }).outputText)
  created.push(target)
  return import(target.href)
}
try {
  const client = await read('client', '../src/api/client.ts')
  let captured
  globalThis.fetch = async (url, init) => { captured = init; return new Response('{}', { status: 200 }) }
  client.setToken('test-active-session')
  const form = new FormData(); form.append('file', new Blob(['test']))
  await client.authFetch('/api/video/upload', { method: 'POST', body: form })
  assert.equal(captured.headers.get('Authorization'), 'Bearer test-active-session')
  assert.equal(captured.headers.get('Content-Type'), null, 'multipart boundary belongs to the browser')
  assert.equal(captured.body, form)
  globalThis.fetch = async () => new Response('{}', { status: 401 })
  await client.authFetch('/api/auth/me')
  assert.equal(client.getToken(), null)
  assert.equal(expired, 1)
  client.setToken('old-session')
  globalThis.fetch = async () => { client.setToken('new-session'); return new Response('{}', { status: 401 }) }
  await client.authFetch('/api/video/status')
  assert.equal(client.getToken(), 'new-session', 'late response cannot log out a newer session')
  assert.equal(expired, 1)
  globalThis.fetch = async () => new Response('{}', { status: 503 })
  await client.authFetch('/api/edge/radar')
  assert.equal(client.getToken(), 'new-session', 'service outage is not login expiry')
  globalThis.fetch = async () => { throw new TypeError('network unavailable') }
  await assert.rejects(client.authFetch('/api/video/status'), TypeError)
  assert.equal(client.getToken(), 'new-session')
  globalThis.fetch = async () => new Response('{}', { status: 401 })
  await assert.rejects(client.login('invalid-user', 'invalid-password'))
  assert.equal(client.getToken(), 'new-session', 'wrong login must not expire an existing session')
  await client.logout()
  await client.authFetch('/api/auth/me')
  assert.equal(expired, 2, 'missing credentials must return an open workbench to login')

  const { radarDemoSnapshot } = await read('demo', '../src/data/demo.ts')
  const a = radarDemoSnapshot(50000), b = radarDemoSnapshot(51000)
  assert.equal(a.source, 'mock')
  assert.equal(a.hardware.connected, false, 'demo must not pretend hardware is connected')
  assert.equal(a.vitals[0].source, 'mock')
  assert.equal(a.vitals[0].chest_displacement_mm, null)
  assert.equal(a.vitals[0].ahi_index, null)
  assert(a.points.length === 192 && a.points.every(p => p.length === 3 && p.every(Number.isFinite)))
  assert(a.vitals[0].heart_waveform.length === 15 && a.vitals[0].breath_waveform.length === 15)
  assert.notDeepEqual(a.points, b.points)
  assert.notDeepEqual(a.vitals[0].heart_waveform, b.vitals[0].heart_waveform)
  console.log('Data recovery checks passed: auth expiry, stale responses, network failures, multipart upload, honest moving demo')
} finally {
  created.forEach(file => fs.rmSync(file, { force: true }))
}
