import assert from 'node:assert/strict'
import fs from 'node:fs'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { clone } from 'three/examples/jsm/utils/SkeletonUtils.js'

const files = ['poseMotion', 'humanRig'].map(name => new URL(`../src/components/${name}.check.mjs`, import.meta.url))
try {
  for (const [i, name] of ['poseMotion', 'humanRig'].entries()) {
    const source = fs.readFileSync(new URL(`../src/components/${name}.ts`, import.meta.url), 'utf8')
    fs.writeFileSync(files[i], ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } }).outputText)
  }
  const { PoseMotion, PoseHistory, validSpatialPose } = await import(files[0].href)
  const motion = new PoseMotion()
  const points = Array.from({ length: 33 }, () => [0, 0, 0])
  motion.push(points, 0, 'first'); motion.advance(1/60)
  points[0] = [1, 0, 0]
  motion.push(points, .1, 'first'); motion.advance(1/60)
  assert(motion.joints[0].x > 0 && motion.joints[0].x < motion.target[0].x && motion.target[0].x < 1)
  motion.push(points, .2, 'new-person')
  assert.equal(motion.joints[0].x, 1, 'source/person changes must discard smoothing history')
  points[0] = [2, 0, 0]; motion.push(points, 2, 'new-person')
  assert.equal(motion.joints[0].x, 2, 'long tracking gaps must snap to fresh data')

  const { humanRig } = await import(files[1].href)
  const bytes = fs.readFileSync(new URL('../public/models/care-human.glb', import.meta.url))
  const { scene } = await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset+bytes.byteLength), '')
  scene.updateMatrixWorld(true)
  const joints = Array.from({ length:33 }, () => new THREE.Vector3())
  const names = { 11:'upperarm_l', 12:'upperarm_r', 13:'lowerarm_l', 14:'lowerarm_r', 15:'hand_l', 16:'hand_r', 23:'thigh_l', 24:'thigh_r', 25:'calf_l', 26:'calf_r', 27:'foot_l', 28:'foot_r', 29:'foot_l', 30:'foot_r', 31:'ball_l', 32:'ball_r' }
  for (const [i, name] of Object.entries(names)) scene.getObjectByName(name).getWorldPosition(joints[Number(i)])
  joints[13].set(.35, 1.25, .05); joints[15].set(.42, 1.0, .1)
  const observed = joints.map(p => p.toArray())
  assert(validSpatialPose(observed, Array(33).fill(1)))
  const partial = Array(33).fill(1); partial[23] = .1
  assert(!validSpatialPose(observed, partial), 'missing hip must not drive a complete body')
  const malformed = observed.map(p => [...p]); malformed[25][1] = NaN
  assert(!validSpatialPose(malformed, Array(33).fill(1)), 'invalid joints must not enter rendering')
  const history = new PoseHistory()
  const seeded = history.push(observed,Array(33).fill(1),0,'camera:1')
  assert(seeded && validSpatialPose(seeded.points,seeded.confidence))
  const full = history.push(seeded.points,Array(33).fill(1),.1,'camera:1')
  assert.equal(full.mode,'observed')
  const incomplete = full.points.map(p => [...p]), visibility = Array(33).fill(1)
  for (const i of [23,24,25,26,27,28,29,30,31,32]) { visibility[i] = 0; incomplete[i] = [NaN,99,99] }
  for (let i=0;i<23;i++) incomplete[i][0] += .1
  const completed = history.push(incomplete,visibility,.2,'camera:1')
  assert(completed && validSpatialPose(completed.points,completed.confidence),'a cropped lower body must retain a complete display')
  assert.equal(completed.mode,'temporal')
  assert(completed.estimated[25] && !completed.estimated[11])
  assert(Math.abs(completed.points[25][0]-full.points[25][0]-.1)<1e-5,'missing legs follow observed torso movement')
  const independent = new PoseHistory()
  assert.equal(independent.push(full.points,Array(33).fill(1),.2,'camera:2').points[25][0],full.points[25][0],'each person owns its pose history')
  const upperOnly = new PoseHistory().push(incomplete,visibility,.2,'new-camera')
  assert(upperOnly && upperOnly.mode === 'reference' && validSpatialPose(upperOnly.points,upperOnly.confidence),'partial first observations use a labelled reference structure')
  assert(history.push([],[],.3,'camera:1')?.observedCount === 0,'short pose gaps retain the display')
  assert.equal(history.push([],[],.9,'camera:1'),undefined,'fully lost observations must expire')
  assert.equal(new PoseHistory().push([],[],0,'empty'),undefined,'no person means no invented avatar')
  const referenceOnly = new PoseHistory().reference()
  assert(referenceOnly.mode === 'reference' && referenceOnly.observedCount === 0 && validSpatialPose(referenceOnly.points,referenceOnly.confidence),'a separately confirmed person can display an explicit reference body without pretending to observe joints')
  const sourceReset = independent.push([],[],.3,'different-source')
  assert.equal(sourceReset,undefined,'source changes must discard another scene\'s body')
  const firstPerson = clone(scene), secondPerson = clone(scene)
  assert.notEqual(firstPerson.getObjectByName('pelvis'), secondPerson.getObjectByName('pelvis'))
  const firstMeshes = []; firstPerson.traverse(o => { if (o.isMesh) firstMeshes.push(o) })
  const secondMeshes = []; secondPerson.traverse(o => { if (o.isMesh) secondMeshes.push(o) })
  assert(firstMeshes.every((o, i) => o.geometry === secondMeshes[i].geometry), 'people share geometry, not bones')
  const before = new Map()
  scene.traverse(node => { if (node.isMesh) before.set(node.uuid, node.geometry) })
  const drive = humanRig(scene)
  const restHead = scene.getObjectByName('Head').getWorldQuaternion(new THREE.Quaternion())
  const unchanged = scene.getObjectByName('pelvis').quaternion.clone()
  drive(joints, partial)
  assert(scene.getObjectByName('pelvis').quaternion.equals(unchanged), 'partial torso must keep its previous transform')
  for (let i=0; i<120; i++) drive(joints, Array(33).fill(1))
  const arm = scene.getObjectByName('upperarm_l'), elbow = scene.getObjectByName('lowerarm_l')
  const actual = elbow.getWorldPosition(new THREE.Vector3()).sub(arm.getWorldPosition(new THREE.Vector3())).normalize()
  const expected = joints[13].clone().sub(joints[11]).normalize()
  assert(actual.dot(expected) > .999, 'arm must follow measured direction')
  joints[25].copy(joints[23]).add(new THREE.Vector3(0,-.28,.3))
  const shoulders = joints[11].clone().add(joints[12]).multiplyScalar(.5)
  joints[7].copy(shoulders).add(new THREE.Vector3(.08,.24,.08))
  joints[8].copy(shoulders).add(new THREE.Vector3(-.08,.24,.08))
  drive(joints,Array(33).fill(1))
  const thigh = scene.getObjectByName('thigh_l'), knee = scene.getObjectByName('calf_l')
  const legDirection = knee.getWorldPosition(new THREE.Vector3()).sub(thigh.getWorldPosition(new THREE.Vector3())).normalize()
  assert(legDirection.dot(joints[25].clone().sub(joints[23]).normalize()) > .999,'sitting/bent knees must drive the individual leg')
  const headDelta = scene.getObjectByName('Head').getWorldQuaternion(new THREE.Quaternion()).multiply(restHead.invert())
  const headUp = new THREE.Vector3(0,1,0).applyQuaternion(headDelta)
  assert(headUp.dot(new THREE.Vector3(0,.24,.08).normalize()) > .999,'head tilt needs a separate frame')
  const floor = () => Math.min(...['Head','upperarm_l','upperarm_r','pelvis','foot_l','foot_r'].map(name=>scene.getObjectByName(name).getWorldPosition(new THREE.Vector3()).y))
  assert(Math.abs(floor()+.95)<1e-5,'body support stays on the displayed floor')
  const height = scene.position.y
  drive(joints,Array(33).fill(1))
  assert(Math.abs(height-scene.position.y)<1e-5,'grounding must not accumulate across frames')
  scene.traverse(node => {
    assert(node.matrixWorld.elements.every(Number.isFinite), 'rig transforms must stay finite')
    if (node.isMesh) assert.equal(node.geometry, before.get(node.uuid), 'geometry must persist across pose updates')
  })
  console.log('Spatial checks passed: per-person temporal completion, partial bootstrap, expiry/reset, arm/leg/head retargeting, grounding, persistent geometry')
} finally {
  files.forEach(file => fs.rmSync(fileURLToPath(file), { force: true }))
}
