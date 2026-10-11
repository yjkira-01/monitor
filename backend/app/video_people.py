"""视频人物检测、短期跟踪和单目相对三维姿态。"""
from pathlib import Path
from collections import deque
from copy import deepcopy
import threading
import time

import cv2
import numpy as np
from .video_person_state import PersonTracks, enhance_detection, box_iou

def serialize_people(result):
    people = []
    for landmarks, world in zip(result.pose_landmarks, result.pose_world_landmarks):
        if len(landmarks) != 33 or len(world) != 33:
            continue
        values = [[p.x, p.y, p.z, p.visibility] for p in landmarks + world]
        if not np.isfinite(values).all():
            continue
        visible = [p for p in landmarks if p.visibility >= .5 and 0 <= p.x <= 1 and 0 <= p.y <= 1]
        if len(visible) < 6:
            continue
        x1, y1 = max(0., min(p.x for p in visible)-.035), max(0., min(p.y for p in visible)-.035)
        x2, y2 = min(1., max(p.x for p in visible)+.035), min(1., max(p.y for p in visible)+.035)
        def points(items):
            return [{"x": round(p.x, 5), "y": round(p.y, 5), "z": round(p.z, 5),
                     "visibility": round(float(p.visibility), 3)} for p in items]
        people.append({"index": len(people)+1, "bbox": [x1, y1, x2, y2], "landmarks": points(landmarks),
                       "world_landmarks": points(world), "visibility": round(float(np.mean([p.visibility for p in landmarks])), 3)})
        for image_point, world_point in zip(people[-1]['landmarks'], people[-1]['world_landmarks']):
            world_point['visibility'] = min(image_point['visibility'], world_point['visibility']) if 0 <= image_point['x'] <= 1 and 0 <= image_point['y'] <= 1 else 0.
    return people


def annotate_people(frame, people):
    annotated = frame.copy()
    height, width = frame.shape[:2]
    for person in people:
        x1, y1, x2, y2 = [int(v*s) for v, s in zip(person["bbox"], (width, height, width, height))]
        quality = person.get('observation', {})
        color = (116, 224, 157) if quality.get('usable', True) else (90, 182, 231)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        length = max(8, min(22, (x2-x1)//5))
        for x, dx in ((x1, 1), (x2, -1)):
            for y, dy in ((y1, 1), (y2, -1)):
                cv2.line(annotated, (x, y), (x+dx*length, y), color, 3, cv2.LINE_AA)
                cv2.line(annotated, (x, y), (x, y+dy*length), color, 3, cv2.LINE_AA)
        label_y = max(20, y1-7)
        cv2.rectangle(annotated, (x1, label_y-18), (min(width-1, x1+106), label_y+4), (240, 248, 241), -1)
        cv2.putText(annotated, f"P{person['index']:02d} {quality.get('posture', 'tracking').upper()}", (x1+6, label_y), cv2.FONT_HERSHEY_SIMPLEX, .35, (43, 94, 60), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


class VideoPersonModel:
    def __init__(self):
        import mediapipe as mp
        import torch
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        from mediapipe.tasks.python import BaseOptions, vision
        path = Path(__file__).resolve().parents[2] / "artifacts/pose_landmarker_full.task"
        if not path.is_file():
            raise FileNotFoundError("缺少官方 pose_landmarker_full.task 人体姿态权重")
        self.mp = mp
        self.torch = torch
        torch.set_num_threads(4)
        detector_path = path.parent / "ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
        with torch.device('meta'):
            self.detector = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        self.detector.load_state_dict(torch.load(detector_path, map_location="cpu", weights_only=True), assign=True)
        self.detector.eval()
        from .video_pose_model import PersonPoseModel
        self.pose_classifier = None
        self.pose_classifier_error = None
        try:
            self.pose_classifier = PersonPoseModel(path.parent / 'multifall_pose_tcn_stage_b_inference.pt')
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            self.pose_classifier_error = f'{type(exc).__name__}: {exc}'
        self.path = path
        self.model = None
        self.crop_model = None
        self.reset()

    def reset(self):
        from mediapipe.tasks.python import BaseOptions, vision
        from .video_tracking import PersonBoxTracker
        if self.model is not None:
            self.model.close()
        if self.crop_model is not None:
            self.crop_model.close()
            self.crop_model = None
        self.model = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.path)), num_poses=4,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=.5, min_pose_presence_confidence=.5))
        self.last_detection_ms = -1000
        self.reference = None
        self.tracker = PersonBoxTracker()
        self.person_tracks = PersonTracks()
        if self.pose_classifier is not None:
            self.pose_classifier.reset()

    def predict(self, frame, timestamp_ms):
        from .video_tracking import tracking_gray
        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, max(1, round(frame.shape[0]*640/frame.shape[1]))), interpolation=cv2.INTER_AREA)
        detection_frame, enhanced = enhance_detection(frame)
        rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        result = self.model.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        poses = serialize_people(result)
        boxes = []
        if timestamp_ms - self.last_detection_ms >= 500:
            with self.torch.inference_mode():
                image = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA)
                detections = self.detector([self.torch.from_numpy(image).permute(2, 0, 1).float()/255])[0]
            for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
                if int(label) != 1 or float(score) < .35:
                    continue
                bbox = np.clip(box.numpy()/320, 0, 1).tolist()
                if not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                boxes.append({"bbox": bbox, "detector_score": float(score), "box_source": "ssdlite_coco_person"})
            self.last_detection_ms = timestamp_ms
            fresh_detection = True
        else:
            boxes = [{**box, "box_source": "optical_flow"} for box in self.tracker.update(frame, self.reference, timestamp_ms/1000)]
            fresh_detection = False
        people = match_people(boxes[:8], poses)
        if any(not p['landmarks'] for p in people):
            from mediapipe.tasks.python import BaseOptions, vision
            missing = [p for p in people if not p['landmarks']][:2]
            if missing and self.crop_model is None:
                self.crop_model = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.path)), num_poses=1,
                    running_mode=vision.RunningMode.IMAGE, min_pose_detection_confidence=.5))
            for person in missing:
                x1, y1, x2, y2 = person['bbox']
                dx, dy = (x2-x1)*.12, (y2-y1)*.12
                x1, y1, x2, y2 = np.clip([x1-dx, y1-dy, x2+dx, y2+dy], 0, 1)
                height, width = frame.shape[:2]
                a, b, c, d = (np.array([x1,y1,x2,y2])*[width,height,width,height]).astype(int)
                if c <= a or d <= b:
                    continue
                crop,_ = enhance_detection(frame[b:d,a:c])
                local = serialize_people(self.crop_model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))))
                if local:
                    pose = local[0]
                    for point in pose['landmarks']:
                        point['x'], point['y'] = float(x1+point['x']*(x2-x1)), float(y1+point['y']*(y2-y1))
                    person.update({key: pose[key] for key in ('landmarks', 'world_landmarks', 'visibility')})
        people = self.person_tracks.update(people, frame, timestamp_ms/1000)
        if self.pose_classifier is not None:
            try:
                self.pose_classifier.update(people, timestamp_ms/1000, frame.shape[1]/frame.shape[0])
                self.pose_classifier_error = None
            except Exception as exc:
                self.pose_classifier.reset()
                self.pose_classifier_error = f'{type(exc).__name__}: {exc}'
        if self.pose_classifier_error:
            for person in people:
                person['pose_analysis'] = {'state': 'error', 'fall_score': None, 'error': self.pose_classifier_error}
        if fresh_detection:
            self.reference = (tracking_gray(frame), timestamp_ms/1000,
                              [{key:p[key] for key in ('bbox','index','detector_score') if key in p} for p in people])
        return {"state": "ready", "model": "ssdlite_coco_person + mediapipe_pose_full_video", "scope": "video_frame",
                "coordinate_system": "monocular_hip_relative", "people": people, "detection_enhanced": enhanced, "error": None}, None

    def close(self):
        self.model.close()
        if self.crop_model is not None:
            self.crop_model.close()


class VideoPoseWorker:
    """Latest-frame queue independent of the fall classifier; never accumulate lag."""
    def __init__(self, loader=VideoPersonModel):
        self.loader = loader
        self._condition = threading.Condition()
        self._thread = None
        self._stopped = False
        self._generation = 0
        self._pending = None
        self._clear()

    def _clear(self):
        self._result = {"state": "waiting", "people": [], "error": None, "generation": self._generation,
                        "updated_at_ms": 0, "frames_analyzed": 0, "update_hz": 0.}
        self._jpeg = self._reference = None
        self._frame = None
        self._times = deque(maxlen=30)

    def start(self):
        with self._condition:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="video-pose", daemon=True)
                self._thread.start()

    def stop(self):
        with self._condition:
            self._stopped = True
            self._pending = None
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=3)

    def reset(self):
        with self._condition:
            self._generation += 1
            self._pending = None
            error = self._result.get("error")
            self._clear()
            if error and self._thread and not self._thread.is_alive():
                self._result.update(state="error", error=error)

    def submit(self, frame, frame_s, generation, captured_at_ms):
        with self._condition:
            if self._stopped or generation != self._generation:
                return
            self._pending = (frame, frame_s, generation, captured_at_ms)
            self._condition.notify()

    def snapshot(self):
        with self._condition:
            return deepcopy(self._result)

    def tracking_detection(self):
        with self._condition:
            return self._reference

    def observation(self):
        with self._condition:
            jpeg, frame, people = self._jpeg, self._frame, self._result['people']
        if jpeg is not None or frame is None:
            return jpeg
        if frame.shape[1] > 960:
            preview = cv2.resize(frame, (960, max(1,round(frame.shape[0]*960/frame.shape[1]))), interpolation=cv2.INTER_AREA)
        else:preview = frame
        jpeg = annotate_people(preview, people)
        with self._condition:
            if self._frame is frame:
                self._jpeg = jpeg
                return jpeg
            return None

    def _run(self):
        model = None
        try:
            model = self.loader()
            model_generation = 0
            last_timestamp = -1
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopped or self._pending is not None)
                    if self._stopped:
                        return
                    job, self._pending = self._pending, None
                frame, frame_s, generation, captured = job
                started = time.monotonic()
                try:
                    if generation != model_generation:
                        model.reset()
                        model_generation = generation
                        last_timestamp = -1
                    timestamp = round(frame_s*1000)
                    if timestamp <= last_timestamp:
                        continue
                    last_timestamp = timestamp
                    result, jpeg = model.predict(frame, timestamp)
                    self._accept(result, jpeg, frame, frame_s, generation, captured, (time.monotonic()-started)*1000)
                except Exception as exc:
                    with self._condition:
                        if generation == self._generation and not self._stopped:
                            self._result.update(state="error", error=f"{type(exc).__name__}: {exc}", people=[])
                            self._jpeg = self._reference = self._frame = None
        except Exception as exc:
            with self._condition:
                self._result.update(state="error", error=f"{type(exc).__name__}: {exc}", people=[])
        finally:
            if model is not None:
                model.close()

    def _accept(self, result, jpeg, frame, frame_s, generation, captured, latency):
        from .video_tracking import tracking_gray
        gray = tracking_gray(frame)
        with self._condition:
            if self._stopped or generation != self._generation or frame_s <= self._result.get("frame_s", -1):
                return False
            self._times.append(time.monotonic())
            hz = (len(self._times)-1)/max(self._times[-1]-self._times[0], 1e-6) if len(self._times)>1 else 0.
            self._result = {**result, "frame_s": round(frame_s, 3), "generation": generation,
                            "captured_at_ms": captured, "updated_at_ms": int(time.time()*1000),
                            "inference_ms": round(latency, 1), "update_hz": round(hz, 1),
                            "frames_analyzed": self._result["frames_analyzed"]+1}
            self._jpeg = jpeg
            self._frame = frame
            self._reference = (gray, frame_s, [{key: p[key] for key in ('index', 'bbox', 'track_id', 'observation') if key in p} for p in result["people"]])
            return True


def match_people(boxes, poses):
    def duplicate_pose(a, b):
        first, second = a.get('landmarks',[]), b.get('landmarks',[])
        if len(first) != 33 or len(second) != 33:
            return False
        distances = [np.hypot(p['x']-q['x'],p['y']-q['y']) for p,q in zip(first,second)
                     if min(p['visibility'],q['visibility']) >= .5]
        scale = max(a['bbox'][2]-a['bbox'][0],a['bbox'][3]-a['bbox'][1],1e-6)
        return len(distances) >= 6 and float(np.median(distances))/scale < .035
    def contained(a, b):
        intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        return intersection/max(min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])),1e-6) >= .85
    unique = []
    for pose in sorted(poses,key=lambda p:p.get('visibility',0),reverse=True):
        if not any(duplicate_pose(pose,previous) for previous in unique):unique.append(pose)
    poses = unique
    if not boxes:
        return [{**p, "box_source": "pose_extent"} for p in poses]
    boxes = sorted(boxes,key=lambda b:b.get('detector_score',1),reverse=True)
    pairs = sorted(((box_iou(box['bbox'],pose['bbox']),i,j) for i,box in enumerate(boxes) for j,pose in enumerate(poses)),reverse=True)
    assignments, used = {}, set()
    for score,i,j in pairs:
        if score >= .25 and i not in assignments and j not in used:
            assignments[i] = j; used.add(j)
    distinct = []
    for i,box in enumerate(boxes):
        duplicate = next((previous for previous in distinct
                          if (box_iou(box['bbox'],boxes[previous]['bbox']) >= .45 or contained(box['bbox'],boxes[previous]['bbox']))
                          and not (i in assignments and previous in assignments)),None)
        if duplicate is None:distinct.append(i)
        elif i in assignments and duplicate not in assignments:assignments[duplicate] = assignments.pop(i)
    people, used = [], set()
    for i in distinct:
        pose_index = assignments.get(i)
        pose = poses[pose_index] if pose_index is not None else {"landmarks": [], "world_landmarks": [], "visibility": 0.}
        people.append({**pose,**boxes[i],"index":len(people)+1})
        if pose_index is not None:used.add(pose_index)
    for j,pose in enumerate(poses):
        if j not in used and not any(duplicate_pose(pose,p) for p in people):
            people.append({**pose,'index':len(people)+1,'box_source':'pose_extent'})
    return people
