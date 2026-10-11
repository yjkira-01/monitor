"""Run the trained pose TCN on observed joints, independently for each tracked person."""
from collections import deque
from pathlib import Path

import numpy as np


class PersonPoseModel:
    def __init__(self, checkpoint: Path):
        import torch
        from yanglao_monitor.pose_temporal import PoseTemporalNet, pose_features

        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (state['model_name'] != 'pose_tcn' or state['classes'] != ['adl', 'fall'] or
                state['clip_len'] != 48 or state['preprocessing'] != 'pose_xy_visibility_body_relative_v1'):
            raise ValueError('姿态权重与48帧时序模型不匹配')
        self.torch, self.features = torch, pose_features
        self.model = PoseTemporalNet()
        self.model.load_state_dict(state['model'])
        self.model.eval()
        self.revision = checkpoint.stem
        self.reset()

    def reset(self):
        self.buffers, self.results = {}, {}

    def update(self, people, time_s, aspect):
        self.buffers = {key: clip for key, clip in self.buffers.items() if time_s-clip[-1][0] <= .6}
        self.results = {key: value for key, value in self.results.items() if key in self.buffers}
        for person in people[:4]:
            key = person['track_id']
            clip = self.buffers.setdefault(key, deque(maxlen=96))
            pose = np.zeros((33, 3), np.float32)
            joints = person.get('landmarks', [])
            if len(joints) == 33 and person.get('observation', {}).get('video_usable'):
                pose[:] = [[p['x']*aspect, p['y'], p['visibility']] for p in joints]
                pose[~np.isfinite(pose).all(1)] = 0
            if not clip or time_s > clip[-1][0]:
                clip.append((time_s, pose))
            while time_s-clip[0][0] > 5:
                clip.popleft()
            cached = self.results.get(key)
            if len(clip) < 6 or clip[-1][0]-clip[0][0] < 1:
                result = {'state': 'warming_up', 'fall_score': None}
            elif cached and time_s-cached['window_end_s'] < .5:
                result = cached
            else:
                sampled = np.stack([clip[i][1] for i in np.linspace(0, len(clip)-1, 48).round().astype(int)])
                coverage = float((sampled[:, [11, 12, 23, 24], 2] >= .3).all(1).mean())
                result = {'state': 'unobservable', 'fall_score': None, 'pose_coverage': coverage}
                if coverage >= .2:
                    # shortcut: this head was trained on single-person poses; crossing identities need re-identification.
                    with self.torch.inference_mode():
                        inputs = self.torch.from_numpy(sampled).permute(2, 0, 1).unsqueeze(0)
                        score = float(self.model(self.features(inputs)).softmax(1)[0, 1])
                    if not np.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError('姿态时序模型输出无效')
                    result.update(state='fall_candidate' if score >= .5 else 'adl', fall_score=score)
                result.update(window_start_s=round(clip[0][0], 3), window_end_s=round(time_s, 3))
                self.results[key] = result
            person['pose_analysis'] = {**result, 'model_revision': self.revision, 'scope': 'tracked_person_pose',
                                       'basis': 'trained_pose_tcn', 'validated': False}
