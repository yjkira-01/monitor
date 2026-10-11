"""带模态缺失处理的雷达-视频融合网络。"""

from __future__ import annotations

import torch
from torch import nn


class FusionNet(nn.Module):
    """特征级融合；mask=0 时对应模态被置零，仍可单模态推理。"""

    def __init__(self, video_dim: int = 64, radar_dim: int = 8, hidden: int = 128, activity_classes: int = 4) -> None:
        super().__init__()
        self.video_dim = video_dim
        self.radar_dim = radar_dim
        self.encoder = nn.Sequential(
            nn.Linear(video_dim + radar_dim + 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.fall_head = nn.Linear(hidden, 2)
        self.activity_head = nn.Linear(hidden, activity_classes)
        self.sleep_head = nn.Linear(hidden, 4)
        self.vital_head = nn.Linear(hidden, 2)

    def forward(
        self,
        video: torch.Tensor,
        radar: torch.Tensor,
        video_mask: torch.Tensor | None = None,
        radar_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if video_mask is None:
            video_mask = torch.ones(video.shape[0], 1, device=video.device, dtype=video.dtype)
        if radar_mask is None:
            radar_mask = torch.ones(radar.shape[0], 1, device=radar.device, dtype=radar.dtype)
        video = torch.nan_to_num(video) * video_mask
        radar = torch.nan_to_num(radar) * radar_mask
        x = self.encoder(torch.cat((video, radar, video_mask, radar_mask), dim=-1))
        return {
            "fall_logits": self.fall_head(x),
            "activity_logits": self.activity_head(x),
            "sleep_logits": self.sleep_head(x),
            "vital_rates": self.vital_head(x),
            "embedding": x,
        }
