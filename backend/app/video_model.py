"""持久 R3D 推理与有界滑窗队列；采集线程不等待模型。"""
from collections import deque
from copy import deepcopy
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np
from .video_person_state import enhance_clip
from yanglao_monitor.video import ACTIVITY_CLASSES


class TrainedVideoModel:
    def __init__(self, checkpoint: Path):
        import torch
        from torch import nn
        from torchvision.models.video import R3D_18_Weights, r3d_18

        torch.set_num_threads(4)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["classes"] not in (['adl','fall'],sorted(ACTIVITY_CLASSES)) or state["model_name"] != "r3d18" or state["clip_len"] != 16:
            raise ValueError("实时通道要求本项目16帧R3D-18二类或OmniFall十类权重")
        self.torch = torch
        self.read_size = tuple(state["read_size"])
        self.clip_len = state["clip_len"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enhance_low_light = True
        self.low_light_trained = state.get('low_light_trained', False)
        self.partial_body_trained = state.get('partial_body_trained', False)
        self.classes = state['classes']
        with torch.device('meta'):
            self.model = r3d_18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(state["model"], assign=True)
        self.model.to(self.device).eval()
        self.preprocess = R3D_18_Weights.DEFAULT.transforms()
        self.model.layer4.register_forward_hook(self._features)

    def _features(self, module, inputs, output):
        self.features = output

    def prepare_clip(self, frames):
        torch = self.torch
        indices = np.linspace(0, len(frames) - 1, self.clip_len).round().astype(int)
        selected = [cv2.resize(frames[i], self.read_size, interpolation=cv2.INTER_AREA) for i in indices]
        if getattr(self, 'enhance_low_light', True):
            selected = enhance_clip(selected, trained=getattr(self,'low_light_trained',False))
        rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in selected])
        clip = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous().unsqueeze(0).to(self.device)
        return self.preprocess(clip)

    def predict(self, frames):
        return self.predict_many([frames])[0]

    def predict_many(self, clips):
        torch = self.torch
        with torch.inference_mode():
            logits = self.model(torch.cat([self.prepare_clip(frames) for frames in clips]))
            probabilities = logits.softmax(1)
            risk_indices = [self.classes.index(name) for name in ('fall','fallen') if name in self.classes]
            scores = probabilities[:,risk_indices].sum(1).cpu().tolist()
            activities = [{ 'label':self.classes[int(row.argmax())], 'confidence':float(row.max()),
                            'scores':dict(zip(self.classes,row.tolist())), 'basis':'trained_multistate'}
                          for row in probabilities.cpu()] if len(self.classes) > 2 else [None]*len(clips)
            features = self.features[0]
            embedding = features.mean((1, 2, 3)).cpu().numpy()
            label = int(probabilities[0].argmax())
            cam = torch.relu((features * self.model.fc.weight[label, :, None, None, None]).sum(0).mean(0)).cpu().numpy()
        cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-6)
        # ponytail: CAM 是整窗弱空间解释；动作定位需事件标签和定位模型后升级。
        preview = cv2.resize(clips[0][-1], (171, 128))
        heat = cv2.applyColorMap((cv2.resize(cam, (112, 112)) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        preview[8:120, 29:141] = cv2.addWeighted(preview[8:120, 29:141], 0.6, heat, 0.4, 0)
        cv2.rectangle(preview, (29, 8), (140, 119), (235, 235, 235), 1)
        preview = cv2.resize(preview, (684, 512))
        jpeg = cv2.imencode(".jpg", preview)[1].tobytes()
        return [{"fall_score": scores[0], "activity":activities[0], "embedding": embedding, "attention_jpeg": jpeg,
                 "attention_class": self.classes[label], "device": self.device}] + [{"fall_score": score,'activity':activity} for score,activity in zip(scores[1:],activities[1:])]


class VideoModelWorker:
    def __init__(self, checkpoint: Path, loader=TrainedVideoModel):
        self.checkpoint = checkpoint
        self.loader = loader
        self._condition = threading.Condition()
        self._thread = None
        self._stopped = False
        self._pending = None
        self._generation = 0
        self.session_id = uuid.uuid4().hex
        self._event_id = 0
        self._jpeg = None
        self._reset_state()

    def _reset_state(self):
        self._person_results = {}
        self._person_cursor = 0
        self._history = deque(maxlen=90)
        self._events = deque(maxlen=30)
        self._previous_embedding = None
        self._consecutive = 0
        self._short_consecutive = 0
        self._last_state = None
        self._jpeg = None
        self._result = {"model": "r3d18", 'model_revision': self.checkpoint.stem, "scope": "sliding_window", "model_state": "loading",
                        "model_error": None, "windows_analyzed": 0, "fall_score": 0.0, "confidence": 0.0,
                        "action": "warming_up", "is_fall": False, "watch": False, "event_id": self._event_id,
                        "generation": self._generation, "session_id": self.session_id, "updated_at_ms": 0, "state": "warming_up",
                        "assessment": "正在加载训练模型，等待视频片段", "consecutive_fall_windows": 0,
                        'short_fall_score': None, 'short_consecutive_fall_windows': 0}

    def start(self):
        with self._condition:
            if self._thread is not None:
                return
            self._stopped = False
            self._thread = threading.Thread(target=self._run, name="r3d-inference", daemon=True)
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
            model_state = self._result["model_state"]
            model_error = self._result["model_error"]
            partial_body_trained = self._result.get('partial_body_trained',False)
            activity_classes = self._result.get('activity_classes',[])
            self._pending = None
            self._reset_state()
            self._result.update(model_state=model_state, model_error=model_error,partial_body_trained=partial_body_trained,activity_classes=activity_classes)
            return self._generation

    def submit(self, frames, start_s, end_s, generation, quality, captured_at_ms, person_clips=None, frame_times=None):
        with self._condition:
            if self._stopped or generation != self._generation:
                return
            self._pending = (frames, start_s, end_s, generation, quality, captured_at_ms, person_clips or [], frame_times)
            self._condition.notify()

    def snapshot(self):
        with self._condition:
            return {**deepcopy(self._result), "history": list(self._history), "events": list(self._events), "person_predictions": deepcopy(self._person_results)}

    def person_predictions(self):
        with self._condition:
            return deepcopy(self._person_results)

    def attention(self):
        with self._condition:
            return self._jpeg

    def _run(self):
        try:
            predictor = self.loader(self.checkpoint)
            with self._condition:
                self._result["model_state"] = "ready"
                self._result['partial_body_trained'] = getattr(predictor,'partial_body_trained',False)
                self._result['activity_classes'] = predictor.classes if len(getattr(predictor,'classes',[])) > 2 else []
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending is not None or self._stopped)
                    if self._stopped:
                        return
                    job, self._pending = self._pending, None
                frames, start_s, end_s, generation, quality, captured_at_ms, people, frame_times = job
                started = time.monotonic()
                selected = [people[(self._person_cursor+i) % len(people)] for i in range(min(2, len(people)))] if people else []
                self._person_cursor += len(selected)
                if selected:
                    predictions = predictor.predict_many([frames]+[p['frames'] for p in selected])
                else:
                    predictions = [predictor.predict(frames)]
                # Preserve the long-window result and CAM. Check recent motion only
                # when an independent pose observation supplies corroborating evidence.
                if quality.get('short_window_eligible') and quality.get('pose_fall_people') and frame_times is not None and len(frame_times) == len(frames):
                    short = [frame for time_s, frame in zip(frame_times, frames) if end_s-2 <= time_s <= end_s]
                    if 6 <= len(short) < len(frames):
                        predictions[0]['short_fall_score'] = predictor.predict(short)['fall_score']
                for person, prediction in zip(selected, predictions[1:]):
                    short = person.get('short_frames', [])
                    if person['quality'].get('short_window_eligible') and 6 <= len(short) < len(person['frames']):
                        prediction['short_fall_score'] = predictor.predict(short)['fall_score']
                latency_ms = round((time.monotonic() - started) * 1000, 1)
                if self._accept(predictions[0], start_s, end_s, generation, quality, captured_at_ms, latency_ms):
                    self._accept_people(people, selected, predictions[1:], generation)
        except Exception as exc:
            with self._condition:
                self._jpeg = None
                self._result.update(model_state="error", model_error=f"{type(exc).__name__}: {exc}",
                                    is_fall=False, watch=False, state="unavailable", assessment="训练模型不可用，请检查依赖和权重")

    def _accept(self, prediction, start_s, end_s, generation, quality, captured_at_ms, latency_ms):
        with self._condition:
            if generation != self._generation or self._stopped:
                return False
            if end_s <= self._result.get("window_end_s", -1):
                return False
            score = float(prediction["fall_score"])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("模型输出分数无效")
            embedding = prediction["embedding"]
            change = None
            if self._previous_embedding is not None:
                norm = np.linalg.norm(embedding) * np.linalg.norm(self._previous_embedding)
                change = round(float(np.clip(1 - np.dot(embedding, self._previous_embedding) / max(float(norm), 1e-9), 0, 2)), 4)
            self._previous_embedding = embedding
            usable = quality.get("usable", True) and (quality.get('scene_usable', True) or self._result.get('partial_body_trained',False) and quality.get('scene_video_usable',False))
            continuous = end_s-self._result.get('window_end_s', end_s) <= 3
            self._consecutive = (self._consecutive if continuous else 0) + 1 if score >= 0.5 and usable else 0
            short_score = prediction.get('short_fall_score')
            if short_score is not None and (not np.isfinite(short_score) or not 0 <= short_score <= 1):
                raise ValueError('短窗口模型分数无效')
            short_supported = usable and quality.get('short_window_eligible', False) and bool(quality.get('pose_fall_people')) and short_score is not None and short_score >= .5
            self._short_consecutive = (self._short_consecutive if continuous else 0)+1 if short_supported else 0
            short_confirmed = self._short_consecutive >= 2
            disagreement = score < .5 and bool(quality.get('pose_fall_people'))
            state = "quality_limited" if not quality.get('usable', True) else 'model_disagreement' if disagreement and not short_confirmed else 'observation_limited' if not usable else "fall_sustained" if self._consecutive >= 2 or short_confirmed else "fall_candidate" if score >= 0.5 else "adl"
            assessment = {"quality_limited": "画面过暗或过亮，当前判断需要复核", 'observation_limited': quality.get('observation_note', '人物观测不足，请复核'), "fall_sustained": "连续窗口呈现跌倒特征，请查看画面并人工确认",
                          "fall_candidate": "当前窗口出现疑似跌倒特征，继续观察", "adl": "当前窗口更接近正常活动特征",
                          'model_disagreement': '姿态时序模型出现跌倒特征，RGB模型未识别；请复核画面'}[state]
            if short_confirmed and self._consecutive < 2:
                assessment = '最近2秒视频连续呈现跌倒特征，并有姿态时序支持；请查看画面并人工确认'
            now = int(time.time() * 1000)
            if state == "fall_sustained" and self._last_state != state:
                self._event_id += 1
            self._result.update(model_state="ready", model_error=None, fall_score=score, confidence=max(score, 1-score),
                                activity=prediction.get('activity'),
                                predicted="fall" if score >= 0.5 else "adl", state=state, assessment=assessment,
                                action=('falling' if short_confirmed else (prediction.get('activity') or {}).get('label') or ("falling" if score >= 0.5 else "normal_activity")) if usable else 'unknown', is_fall=state == "fall_sustained",
                                watch=state in ("fall_candidate", 'model_disagreement'), event_id=self._event_id, updated_at_ms=now,
                                window_start_s=round(start_s, 2), window_end_s=round(end_s, 2), captured_at_ms=captured_at_ms,
                                inference_ms=latency_ms, feature_change=change, quality=quality, attention_class=prediction["attention_class"],
                                device=prediction["device"], consecutive_fall_windows=self._consecutive,
                                short_fall_score=short_score, short_consecutive_fall_windows=self._short_consecutive,
                                decision_basis='supported_short_window' if short_confirmed and self._consecutive < 2 else 'long_window',
                                windows_analyzed=self._result["windows_analyzed"] + 1)
            entry = {key: self._result[key] for key in ("window_start_s", "window_end_s", "fall_score", "state", "updated_at_ms", "feature_change", "inference_ms")}
            self._history.append(entry)
            if state != self._last_state:
                self._events.append({**entry, "message": assessment, "id": f"{generation}:{self._result['windows_analyzed']}"})
            self._last_state = state
            self._jpeg = prediction["attention_jpeg"]
            return True

    def _accept_people(self, active, selected, predictions, generation):
        with self._condition:
            if generation != self._generation or self._stopped:
                return
            ids = {p['track_id'] for p in active}
            self._person_results = {key: result for key, result in self._person_results.items() if key in ids}
            for person, prediction in zip(selected, predictions):
                score = float(prediction['fall_score'])
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('个人片段模型输出无效')
                key = person['track_id']
                previous = self._person_results.get(key, {})
                if person['end_s'] <= previous.get('window_end_s', -1):
                    continue
                continuous = person['end_s']-previous.get('window_end_s', person['end_s']) <= 3.
                consecutive = (previous.get('consecutive_fall_windows', 0) if continuous else 0)+1 if score >= .5 else 0
                short_score = prediction.get('short_fall_score')
                if short_score is not None and (not np.isfinite(short_score) or not 0 <= short_score <= 1):
                    raise ValueError('个人短窗口模型分数无效')
                short_supported = person['quality'].get('short_window_eligible') and short_score is not None and short_score >= .5
                short_count = (previous.get('short_consecutive_fall_windows', 0) if continuous else 0)+1 if short_supported else 0
                partial = not person['quality'].get('spatial_valid',True)
                self._person_results[key] = {'track_id': key, 'fall_score': score, 'scope': 'tracked_person_crop',
                    'activity':prediction.get('activity'),
                    'validated': False, 'partial_body':partial,
                    'state': 'fall_sustained' if short_count >= 2 else ('partial_fall_candidate' if score >= .5 else 'partial_adl') if partial else 'fall_sustained' if consecutive >= 2 else 'fall_candidate' if score >= .5 else 'adl',
                    'short_fall_score': short_score, 'short_consecutive_fall_windows': short_count,
                    'consecutive_fall_windows': consecutive, 'windows_analyzed': previous.get('windows_analyzed', 0)+1,
                    'window_start_s': round(person['start_s'], 2), 'window_end_s': round(person['end_s'], 2),
                    'updated_at_ms': int(time.time()*1000), 'quality': person['quality']}
