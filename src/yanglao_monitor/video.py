"""视频片段数据集和小型时空分类器。

数据集格式：root/{split}/{class_name}/*.(mp4|avi|mov|mkv)，按视频而不是按帧划分。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ACTIVITY_CLASSES = ('walk','fall','fallen','sit_down','sitting','lie_down','lying','stand_up','standing','other')


def _read_frames(path: Path, clip_len: int, size: tuple[int, int], start_s: float | None = None, end_s: float | None = None, preserve_aspect: bool = False) -> torch.Tensor:
    """均匀采样整段或闭区间片段；不足长度时重复采样。"""
    if (start_s is None) != (end_s is None) or start_s is not None and (not np.isfinite(start_s) or not np.isfinite(end_s) or not 0 <= start_s < end_s):
        raise ValueError("视频时间范围必须为有限的 0 <= start_s < end_s")
    # 限制 FFmpeg 线程，避免与 DataLoader worker 叠加后抢满 CPU。
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]) if hasattr(cv2, "CAP_PROP_N_THREADS") else cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = max(length, 1)
    first, last_index = 0, length - 1
    if start_s is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0 or start_s >= length / fps:
            cap.release()
            raise ValueError("视频帧率或时间范围无效")
        first = int(np.ceil(start_s * fps - 1e-6))
        last_index = min(length-1, int(np.floor(end_s * fps + 1e-6)))
        if first > last_index:
            cap.release()
            raise ValueError("时间范围内没有视频帧")
    indices = np.linspace(first, last_index, clip_len).round().astype(int)
    sampled: dict[int, np.ndarray] = {}
    last: np.ndarray | None = None
    targets = set(int(index) for index in indices)
    for frame_index in range(int(indices[-1]) + 1):
        if not cap.grab():
            break
        if frame_index in targets:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if preserve_aspect:
                height, width = frame.shape[:2]
                scale = min(size[0] / width, size[1] / height)
                resized = cv2.resize(frame, (max(1, round(width*scale)), max(1, round(height*scale))), interpolation=cv2.INTER_AREA)
                canvas = np.full((size[1], size[0], 3), [124, 116, 104], dtype=np.uint8)
                top, left = (size[1]-resized.shape[0])//2, (size[0]-resized.shape[1])//2
                canvas[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
                frame = canvas
            else:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            sampled[frame_index] = frame
            last = frame
    cap.release()
    if last is None:
        raise RuntimeError(f"视频没有可解码的采样帧: {path}")
    fallback = last
    frames = [sampled.get(int(index), fallback) for index in indices]
    array = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(3, 0, 1, 2)


class VideoFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    """读取 `root/{class}/*` 或 `root/{split}/{class}/*`。"""

    def __init__(
        self,
        root: str | Path,
        clip_len: int = 16,
        size: tuple[int, int] = (160, 160),
        classes: Iterable[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.clip_len = clip_len
        self.size = size
        class_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if classes is None:
            self.classes = [p.name for p in class_dirs]
        else:
            self.classes = list(classes)
            class_dirs = [self.root / name for name in self.classes]
        if not self.classes:
            raise ValueError(f"没有找到类别目录: {self.root}")
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                    self.samples.append((path, self.class_to_idx[class_dir.name]))
        if not self.samples:
            raise ValueError(f"没有找到视频文件: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        return _read_frames(path, self.clip_len, self.size), label


class VideoManifestDataset(VideoFolderDataset):
    """按 CSV 清单读取视频，避免为大型数据集复制或链接文件。"""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        clip_len: int = 16,
        size: tuple[int, int] = (160, 160),
        classes: Iterable[str] | None = None,
        preserve_aspect: bool = False,
    ) -> None:
        manifest_path = Path(manifest).resolve()
        base_dir = manifest_path.parents[2]
        rows: list[dict[str, str]] = []
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
        discovered = sorted({row["label"] for row in rows if row.get("label")})
        self.root = base_dir
        self.clip_len = clip_len
        self.size = size
        self.preserve_aspect = preserve_aspect
        self.classes = list(classes) if classes is not None else discovered
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        rows = [row for row in rows if row.get("path") and row.get("label") in self.class_to_idx]
        self.samples = [(base_dir / row["path"], self.class_to_idx[row["label"]]) for row in rows]
        self.ranges = []
        for row in rows:
            start, end = row.get("start_s", ""), row.get("end_s", "")
            if bool(start) != bool(end):
                raise ValueError("片段清单必须同时提供 start_s 和 end_s")
            interval = (float(start), float(end)) if start else (None, None)
            if start and (not np.isfinite(interval).all() or not 0 <= interval[0] < interval[1]):
                raise ValueError("片段时间范围无效")
            self.ranges.append(interval)
        if not self.samples:
            raise ValueError(f"清单没有 {split} 视频: {manifest_path}")

    def __getitem__(self, index):
        path, label = self.samples[index]
        return _read_frames(path, self.clip_len, self.size, *self.ranges[index], preserve_aspect=self.preserve_aspect), label


class TinyVideoNet(nn.Module):
    """CPU 也能跑的 3D CNN，作为后续 YOLO/骨骼/Transformer 的可替换基线。"""

    def __init__(self, num_classes: int = 2, embedding_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(64, embedding_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.embedding(self.backbone(x))
        logits = self.classifier(features)
        return (logits, features) if return_embedding else logits
