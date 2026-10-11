"""Bounded person tracks and observation quality; posture labels are geometric evidence."""
from collections import deque

import cv2
import numpy as np


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection, 1e-6)


def enhance_detection(frame):
    """Lift moderate shadows for the auxiliary detector, never invent light for classification."""
    brightness = float(cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).mean())
    if brightness >= 65:
        return frame, False
    gamma = max(.35, brightness/65)
    table = (255*(np.arange(256)/255)**gamma).astype(np.uint8)
    return cv2.LUT(frame, table), True


def enhance_clip(frames, trained=True):
    brightness = float(np.median([cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).mean() for f in frames]))
    if brightness >= 65 or brightness < 1:
        return frames
    if not trained:
        # The fixed-weight audit found false noise detection and worse recall after legacy brightening.
        return frames
    # Use one transform for the entire temporal clip so enhancement does not create artificial motion.
    gamma = float(np.clip(np.log(80/255)/np.log(brightness/255), .3, 1.))
    table = (255*(np.arange(256)/255)**gamma).astype(np.uint8)
    return [cv2.LUT(frame, table) for frame in frames]


def observation(person, frame, overlapping=False):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (np.clip(person['bbox'], 0, 1)*[width, height, width, height]).astype(int)
    region = frame[y1:y2, x1:x2]
    brightness = float(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).mean()) if region.size else 0.
    landmarks = person.get('landmarks', [])
    world = person.get('world_landmarks', [])
    visible = [bool(p['visibility'] >= .5 and 0 <= p['x'] <= 1 and 0 <= p['y'] <= 1) for p in landmarks]
    required = (11, 12, 23, 24, 25, 26, 27, 28)
    complete = len(visible) == 33 and all(visible[i] for i in required) and sum(visible) >= 18
    clipped = min(person['bbox'][:2]) < .005 or max(person['bbox'][2:]) > .995
    spatial = complete and len(world) == 33 and not clipped
    if spatial:
        points = np.array([[p['x'], p['y'], p['z']] for p in world])
        spatial = bool(np.isfinite(points).all() and np.abs(points).max() < 4 and
                       .08 < np.linalg.norm((points[11]+points[12]-points[23]-points[24])*.5) < 1.2)
    reason = '人物区域缺少有效图像' if brightness < 6 else '人物互相遮挡' if overlapping else '身体被画面边缘截断' if clipped else '躯干或下肢观测不完整' if not complete else '三维关节异常' if not spatial else '全身观测可用'
    usable = brightness >= 6 and not overlapping and spatial
    video_usable = brightness >= 6 and not overlapping and x2-x1 >= 24 and y2-y1 >= 32 and (sum(visible) >= 6 or person.get('detector_score',0) >= .35)
    posture = 'unknown'
    if usable:
        shoulders = np.mean([[landmarks[i]['x'], landmarks[i]['y']] for i in (11, 12)], axis=0)
        hips = np.mean([[landmarks[i]['x'], landmarks[i]['y']] for i in (23, 24)], axis=0)
        delta = hips-shoulders
        delta[0] *= width/height
        posture = 'horizontal' if abs(delta[0]) > abs(delta[1])*1.25 else 'upright'
        knee_angles = []
        for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
            p = np.array([[landmarks[i]['x']*width, landmarks[i]['y']*height] for i in (hip, knee, ankle)])
            a, b = p[0]-p[1], p[2]-p[1]
            knee_angles.append(np.degrees(np.arccos(np.clip(np.dot(a, b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-6), -1, 1))))
        if posture == 'upright' and max(knee_angles) < 140:
            posture = 'bent_legs'
    return {'usable': usable, 'video_usable': bool(video_usable), 'spatial_valid': usable, 'complete': complete, 'clipped': clipped,
            'overlapping': overlapping, 'brightness': round(brightness, 1), 'low_light': brightness < 65, 'visible_joints': sum(visible),
            'reason': reason, 'posture': posture, 'state_basis': 'geometry_not_trained_action'}


def scene_observation(people, fresh=True):
    usable = fresh and len(people) == 1 and people[0].get('observation', {}).get('usable', False)
    note = '人物观测尚未就绪' if not fresh else '未获得有效人物观测' if not people else '多人场景：请查看各人物独立记录，整幅分数不用于个人判定' if len(people) > 1 else people[0].get('observation', {}).get('reason', '身体观测不足')
    video_usable = fresh and len(people) == 1 and people[0].get('observation',{}).get('video_usable',False)
    pose_fall_people = [{'track_id': p['track_id'], **p['pose_analysis']} for p in people
                        if fresh and p.get('observation', {}).get('video_usable') and
                        p.get('pose_analysis', {}).get('state') == 'fall_candidate']
    return {'scene_usable': bool(usable), 'scene_video_usable': bool(video_usable), 'observation_note': note,
            'people_count': len(people), 'pose_fall_people': pose_fall_people}


class PersonTracks:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, people, frame, time_s):
        self.tracks = {key: value for key, value in self.tracks.items() if time_s-value[1] <= 1.}
        pairs = sorted(((box_iou(p['bbox'], box), i, key) for i, p in enumerate(people)
                        for key, (box, _, _, _) in self.tracks.items()), reverse=True)
        assignments, used = {}, set()
        for score, i, key in pairs:
            if score < .3 or i in assignments or key in used:
                continue
            # ponytail: crossing/occlusion can change track identity; no re-identification model is claimed.
            assignments[i] = key
            used.add(key)
        result = []
        for i, person in enumerate(people[:8]):
            key = assignments.get(i)
            if key is None:
                key, self.next_id = self.next_id, self.next_id+1
            quality = observation(person, frame)
            previous = self.tracks.get(key)
            hits, last_detection = (previous[2],previous[3]) if previous else (0,-float('inf'))
            has_pose = len(person.get('landmarks',[])) == 33 and quality['visible_joints'] >= 6
            if has_pose and person.get('box_source') != 'pose_extent':hits = 2
            elif has_pose and hits < 2 and time_s-last_detection >= .1:
                # A one-frame pose hallucination outside detector boxes is not a new person.
                hits = hits+1 if previous and time_s-previous[1] <= .3 else 1
                last_detection = time_s
            elif person.get('box_source') == 'ssdlite_coco_person' and time_s-last_detection >= .4:
                hits += 1;last_detection = time_s
            self.tracks[key] = (person['bbox'], time_s, hits, last_detection)
            # New pose-only targets need repeated poses; pose-free targets need detector refreshes.
            if hits < 2:continue
            result.append({**person, 'index': key, 'track_id': key, 'observation': quality})
        for person in result:
            overlap = any(box_iou(person['bbox'],other['bbox']) > .2 for other in result if other['track_id'] != person['track_id'])
            if overlap:person['observation'] = observation(person,frame,True)
        return sorted(result, key=lambda p: p['track_id'])


class PersonClips:
    """At most four ROI buffers; RGB evidence does not require valid 3D joints."""
    def __init__(self):
        self.buffers = {}

    def sample(self, frame, people, time_s):
        active = {p['track_id'] for p in people[:4] if p.get('observation', {}).get('video_usable')}
        self.buffers = {key: value for key, value in self.buffers.items() if key in active and time_s-value[-1][0] <= .6}
        height, width = frame.shape[:2]
        for person in people[:4]:
            key = person['track_id']
            if key not in active:
                continue
            x1, y1, x2, y2 = person['bbox']
            dx, dy = (x2-x1)*.1, (y2-y1)*.1
            x1, y1, x2, y2 = (np.clip([x1-dx, y1-dy, x2+dx, y2+dy], 0, 1)*[width, height, width, height]).astype(int)
            crop = frame[y1:y2, x1:x2]
            if not crop.size:
                continue
            clip = self.buffers.setdefault(key, deque(maxlen=48))
            quality = {**person['observation'], 'short_window_eligible': False}
            landmarks = person.get('landmarks', [])
            def valid(i):
                return len(landmarks) == 33 and landmarks[i]['visibility'] >= .6 and np.isfinite([landmarks[i]['x'], landmarks[i]['y']]).all() and 0 <= landmarks[i]['x'] <= 1 and 0 <= landmarks[i]['y'] <= 1
            sides = [(s, h) for s, h in ((11, 23), (12, 24)) if valid(s) and valid(h)]
            if sides:
                sx = np.mean([landmarks[s]['x']*width/height for s, h in sides])
                sy = np.mean([landmarks[s]['y'] for s, h in sides])
                hx = np.mean([landmarks[h]['x']*width/height for s, h in sides])
                hy = np.mean([landmarks[h]['y'] for s, h in sides])
                torso = float(np.hypot(hx-sx, hy-sy))
                tilt = float(np.degrees(np.arctan2(abs(hx-sx), abs(hy-sy))))
                feet = [landmarks[i]['y'] for i in (27, 28) if valid(i) and landmarks[i]['y'] >= hy+torso*.65]
                until = clip[-1][2].get('support_until', -1) if clip else -1
                prior = clip[-1][2] if clip else {}
                reference_foot, reference_torso = prior.get('reference_foot_y'), prior.get('reference_torso', 0)
                locked = prior.get('support_locked', False)
                upright_since = prior.get('upright_since', time_s) if quality.get('posture') == 'upright' else time_s
                if quality.get('posture') == 'upright' and time_s-upright_since >= .8 and feet and tilt < 35:
                    locked = False
                    reference_foot, reference_torso = None, 0
                if not locked and torso >= .025 and feet and (reference_foot is None or max(feet) > reference_foot):
                    reference_foot, reference_torso = max(feet), torso
                if quality.get('posture') == 'horizontal' and torso >= .025:
                    locked = True
                    for before_time, _, before in clip:
                        if not .2 <= time_s-before_time <= 2.5 or before.get('posture') not in ('upright', 'bent_legs'):
                            continue
                        length, foot = before.get('torso_length', 0), before.get('support_foot_y')
                        if length >= .025 and foot is not None and reference_foot is not None and (sy-before['shoulder_y'])/length > .4 and tilt-before['torso_tilt'] > 35 and person['bbox'][3] >= reference_foot-reference_torso*.35:
                            until = time_s+6
                            break
                elif quality.get('posture') == 'upright':
                    until = -1
                quality.update(shoulder_y=float(sy), torso_length=torso, torso_tilt=tilt, support_foot_y=max(feet) if feet else None, support_until=until, reference_foot_y=reference_foot, reference_torso=reference_torso, support_locked=locked, upright_since=upright_since)
                quality['short_window_eligible'] = bool(time_s <= until and person.get('pose_analysis', {}).get('state') == 'fall_candidate')
            clip.append((time_s, cv2.resize(crop, (171, 128), interpolation=cv2.INTER_AREA), quality))
            while clip and time_s-clip[0][0] > 5:
                clip.popleft()

    def ready(self):
        return [{'track_id': key, 'frames': [v[1] for v in clip], 'short_frames': [v[1] for v in clip if v[0] >= clip[-1][0]-2], 'start_s': clip[0][0], 'end_s': clip[-1][0], 'quality': clip[-1][2]}
                for key, clip in self.buffers.items() if len(clip) >= 6 and clip[-1][0]-clip[0][0] >= 1.]
