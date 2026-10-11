"""不需要数据集的全链路自检。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yanglao_monitor.fusion import FusionNet
from yanglao_monitor.pose_fall import TemporalFallFilter, posture_fall_score
from yanglao_monitor.radar import estimate_vital_signs, radar_feature_vector, summarize_sleep_windows
from yanglao_monitor.video import TinyVideoNet


def main() -> None:
    fs = 20.0
    t = np.arange(0, 60, 1 / fs)
    x = np.sin(2 * np.pi * 0.25 * t) + 0.15 * np.sin(2 * np.pi * 1.2 * t)
    vital = estimate_vital_signs(x, fs)
    assert abs(vital["resp_rate_bpm"] - 15) < 2
    assert abs(vital["heart_rate_bpm"] - 72) < 4
    radar = torch.from_numpy(radar_feature_vector(x, fs)).unsqueeze(0)
    video = TinyVideoNet()(torch.rand(1, 3, 8, 64, 64), return_embedding=True)[1]
    result = FusionNet()(video, radar, video_mask=torch.ones(1, 1), radar_mask=torch.ones(1, 1))
    assert result["fall_logits"].shape == (1, 2)
    summary = summarize_sleep_windows([vital, vital])
    assert summary["window_count"] == 2
    score = posture_fall_score([0, 0, 120, 60])
    detector = TemporalFallFilter(threshold=0.6, patience=2)
    assert score > 0.6 and not detector.update(score) and detector.update(score)
    print("full smoke test passed", vital)


if __name__ == "__main__":
    main()
