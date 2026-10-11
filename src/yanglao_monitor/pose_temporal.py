"""Visibility-masked, body-relative pose features and a small dilated temporal CNN."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def pose_features(poses, augment=False):
    if poses.ndim != 4 or poses.shape[1] != 3 or poses.shape[-1] != 33:
        raise ValueError("Pose shape must be B,3,T,33 (aspect-corrected x, y, visibility)")
    valid = torch.isfinite(poses).all(1, keepdim=True)
    poses = torch.nan_to_num(poses).clone()
    xy = poses[:, :2]
    visibility = poses[:, 2:3].clamp(0, 1) * valid
    if augment:
        flip = torch.rand(len(poses), 1, 1, 1, device=poses.device) < .5
        xy[:, :1] = torch.where(flip, -xy[:, :1], xy[:, :1])
        # Joint identities follow the mirrored body, including the face landmarks.
        permutation = [0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9] + [i + 1 if i % 2 else i - 1 for i in range(11, 33)]
        xy = torch.where(flip, xy[..., permutation], xy)
        visibility = torch.where(flip, visibility[..., permutation], visibility)
        visibility *= (torch.rand_like(visibility) >= .05)
    mask = visibility >= .3
    torso_valid = mask[..., [11, 12, 23, 24]].all(-1, keepdim=True)
    center = xy[..., [23, 24]].mean(-1, keepdim=True)
    shoulder = xy[..., [11, 12]].mean(-1, keepdim=True)
    scale = (shoulder-center).square().sum(1, keepdim=True).sqrt()
    body_scale = torch.where(torso_valid, scale, torch.nan).flatten(2).nanmedian(-1).values
    body_scale = torch.nan_to_num(body_scale, nan=1.).clamp_min(.02)[:, :, None, None]
    relative = ((xy-center) / body_scale).clamp(-6, 6) * mask * torso_valid
    velocity = torch.zeros_like(relative)
    velocity[:, :, 1:] = (relative[:, :, 1:]-relative[:, :, :-1]) * mask[:, :, 1:] * mask[:, :, :-1] * torso_valid[:, :, 1:] * torso_valid[:, :, :-1]
    center_velocity = torch.zeros_like(center)
    center_velocity[:, :, 1:] = ((center[:, :, 1:]-center[:, :, :-1]) / body_scale).clamp(-3, 3) * torso_valid[:, :, 1:] * torso_valid[:, :, :-1]
    flatten = lambda x: x.permute(0, 1, 3, 2).flatten(1, 2)
    return torch.cat([flatten(relative), flatten(velocity), flatten(visibility), flatten(center_velocity),
                      flatten((scale/body_scale).clamp(0, 6) * torso_valid), flatten(torso_valid.float())], 1)


class TemporalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dilation):
        super().__init__()
        self.padding = 4 * dilation
        self.conv = nn.Conv1d(input_channels, output_channels, 5, dilation=dilation)
        self.norm = nn.GroupNorm(8, output_channels)
        self.dropout = nn.Dropout(.2)
        self.skip = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()

    def forward(self, x):
        output = self.dropout(F.relu(self.norm(self.conv(F.pad(x, (self.padding, 0))))))
        return F.relu(output + self.skip(x))


class PoseTemporalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(TemporalBlock(169, 64, 1), TemporalBlock(64, 128, 2), TemporalBlock(128, 128, 4))
        self.classifier = nn.Linear(256, 2)

    def forward(self, features):
        x = self.backbone(features)
        return self.classifier(torch.cat([x.mean(-1), x.amax(-1)], 1))


class PoseManifestDataset(Dataset):
    def __init__(self, manifest, split, clip_len=32, size=None, classes=None):
        self.root = Path(manifest).resolve().parents[2]
        with Path(manifest).open(encoding="utf-8-sig") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        self.classes = list(classes) if classes is not None else sorted({r["label"] for r in self.rows})
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        if not self.rows or set(self.classes) != {"adl", "fall"}:
            raise ValueError("Pose split must be nonempty and use ADL/Fall classes")
        self.clip_len = clip_len
        self.samples = [(self.root/r["path"], self.class_to_idx[r["label"]]) for r in self.rows]
        self.ranges = [(float(r["start_s"]), float(r["end_s"])) for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.root / row["pose_path"], allow_pickle=False) as cache:
            times, poses = cache["times"], cache["poses"]
        start, end = self.ranges[index]
        selected = np.flatnonzero((times >= start-1e-6) & (times <= end+1e-6))
        if not len(selected):
            raise ValueError(f"No pose frames in window: {row['path']}")
        indices = selected[np.linspace(0, len(selected)-1, self.clip_len).round().astype(int)]
        return torch.from_numpy(poses[indices]).permute(2, 0, 1).float(), self.samples[index][1]
