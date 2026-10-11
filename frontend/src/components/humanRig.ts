import * as THREE from 'three'

// Calibration is specific to the bundled Quaternius CC0 bind pose.
export function humanRig(root: THREE.Group) {
  root.updateMatrixWorld(true)
  const pelvis = root.getObjectByName('pelvis')!
  const origin = pelvis.getWorldPosition(new THREE.Vector3())
  root.position.sub(origin)
  root.updateMatrixWorld(true)
  const segments = [
    ['upperarm_l', 'lowerarm_l', 11, 13], ['lowerarm_l', 'hand_l', 13, 15],
    ['upperarm_r', 'lowerarm_r', 12, 14], ['lowerarm_r', 'hand_r', 14, 16],
    ['thigh_l', 'calf_l', 23, 25], ['calf_l', 'foot_l', 25, 27], ['foot_l', 'ball_l', 29, 31],
    ['thigh_r', 'calf_r', 24, 26], ['calf_r', 'foot_r', 26, 28], ['foot_r', 'ball_r', 30, 32],
  ] as const
  const rigs = segments.map(([name, child, a, b]) => {
    const bone = root.getObjectByName(name)!
    return { bone, a, b, rest: bone.getWorldQuaternion(new THREE.Quaternion()),
      direction: root.getObjectByName(child)!.getWorldPosition(new THREE.Vector3()).sub(bone.getWorldPosition(new THREE.Vector3())).normalize() }
  })
  const restPelvis = pelvis.getWorldQuaternion(new THREE.Quaternion())
  const spine = root.getObjectByName('spine_03')!, neck = root.getObjectByName('neck_01')!, head = root.getObjectByName('Head')!
  const restSpine = spine.getWorldQuaternion(new THREE.Quaternion()), restNeck = neck.getWorldQuaternion(new THREE.Quaternion()), restHead = head.getWorldQuaternion(new THREE.Quaternion())
  const support = ['Head','upperarm_l','upperarm_r','pelvis','foot_l','foot_r'].map(name => root.getObjectByName(name)!)
  const x = new THREE.Vector3(), y = new THREE.Vector3(), z = new THREE.Vector3()
  const direction = new THREE.Vector3(), reference = new THREE.Vector3(), shoulderCenter = new THREE.Vector3(), supportPoint = new THREE.Vector3()
  const basis = new THREE.Matrix4(), body = new THREE.Quaternion(), upperBody = new THREE.Quaternion(), face = new THREE.Quaternion(), world = new THREE.Quaternion(), parent = new THREE.Quaternion()
  function orient(bone: THREE.Object3D, rotation: THREE.Quaternion) {
    bone.parent!.getWorldQuaternion(parent).invert()
    bone.quaternion.copy(parent).multiply(rotation)
    bone.updateMatrixWorld(true)
  }
  return (joints: THREE.Vector3[], confidence: number[]) => {
    if (![11, 12, 23, 24].every(i => confidence[i] >= .5)) return
    y.copy(joints[11]).add(joints[12]).sub(joints[23]).sub(joints[24]).multiplyScalar(.5)
    x.copy(joints[23]).sub(joints[24])
    z.crossVectors(x, y)
    if (y.lengthSq() < .01 || z.lengthSq() < 1e-6) return
    const scale = THREE.MathUtils.clamp(y.length()/.51, .65, 1.35)
    root.scale.setScalar(scale)
    root.position.copy(origin).multiplyScalar(-scale)
    y.normalize(); z.normalize(); x.crossVectors(y, z).normalize()
    body.setFromRotationMatrix(basis.makeBasis(x, y, z))
    root.updateMatrixWorld(true)
    orient(pelvis, world.copy(body).multiply(restPelvis))
    // Shoulder twist and head tilt have their own frames; pelvis rotation alone cannot reproduce them.
    shoulderCenter.copy(joints[11]).add(joints[12]).multiplyScalar(.5)
    x.copy(joints[11]).sub(joints[12]).normalize(); z.crossVectors(x,y)
    upperBody.copy(body)
    if (z.lengthSq() > 1e-6) {
      z.normalize(); x.crossVectors(y,z).normalize()
      upperBody.setFromRotationMatrix(basis.makeBasis(x,y,z))
    }
    orient(spine, world.copy(upperBody).multiply(restSpine))
    face.copy(upperBody)
    if ([7,8].every(i => confidence[i] >= .5)) {
      x.copy(joints[7]).sub(joints[8]); y.copy(joints[7]).add(joints[8]).multiplyScalar(.5).sub(shoulderCenter)
      z.crossVectors(x,y)
      if (x.lengthSq() > .001 && y.lengthSq() > .005 && z.lengthSq() > 1e-6) {
        x.normalize(); z.normalize(); y.crossVectors(z,x).normalize()
        face.setFromRotationMatrix(basis.makeBasis(x,y,z))
      }
    }
    orient(neck, world.copy(upperBody).slerp(face,.5).multiply(restNeck))
    orient(head, world.copy(face).multiply(restHead))
    for (const rig of rigs) {
      if (Math.min(confidence[rig.a] ?? 0, confidence[rig.b] ?? 0) < .4) continue
      direction.copy(joints[rig.b]).sub(joints[rig.a])
      if (direction.lengthSq() < 1e-6) continue
      reference.copy(rig.direction).applyQuaternion(body)
      world.setFromUnitVectors(reference, direction.normalize()).multiply(body).multiply(rig.rest)
      orient(rig.bone, world)
    }
    // Monocular landmarks are hip-relative: anchor the displayed body to the floor, not a fixed hip height.
    let bottom = Infinity
    for (const bone of support) bottom = Math.min(bottom,bone.getWorldPosition(supportPoint).y)
    root.position.y += -.95-bottom
    root.updateMatrixWorld(true)
  }
}
