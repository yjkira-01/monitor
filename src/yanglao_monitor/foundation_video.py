"""Moderate video backbone from the author's K400 distillation checkpoint."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from ._vendor.internvideo2.internvideo2 import internvideo2_base_patch14_224

PRETRAINED_SHA256 = 'aa6ded7709ec41639577aec183f1baeddccda57fb3a7fdf6e76e5ebea2d8996f'
PREPROCESSING = 'internvideo2_rgb_imagenet_letterbox_v1'


def normalize_video(clips):
    """B,T,C,H,W RGB 0..1 from the shared training augmentation function."""
    if clips.ndim != 5 or tuple(clips.shape[1:]) != (8, 3, 224, 224):
        raise ValueError('InternVideo2-Base requires eight RGB frames at 224x224')
    clips = clips.transpose(1, 2)
    mean = clips.new_tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
    std = clips.new_tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)
    return (clips - mean) / std


def build_internvideo2(classes, weights=None, checkpointing=True, checkpoint_blocks=12):
    if classes < 2:
        raise ValueError('At least two classes are required')
    if not 0 <= checkpoint_blocks <= 12:
        raise ValueError('checkpoint_blocks must be 0..12')
    model = internvideo2_base_patch14_224(
        num_classes=400, num_frames=8, tubelet_size=1,
        use_flash_attn=False, use_fused_rmsnorm=False, use_fused_mlp=False,
        use_checkpoint=checkpointing, checkpoint_num=checkpoint_blocks,
        drop_path_rate=.1, fc_drop_rate=.1,
    )
    if weights:
        path = Path(weights)
        if hashlib.sha256(path.read_bytes()).hexdigest() != PRETRAINED_SHA256:
            raise ValueError('The official InternVideo2-Base checkpoint hash does not match')
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    model.head = nn.Linear(model.head.in_features, classes)
    nn.init.trunc_normal_(model.head.weight, std=.02)
    nn.init.zeros_(model.head.bias)
    model.amp_evaluation = True
    return model


def layer_parameters(model, lr):
    """All backbone layers learn; earlier layers use smaller learning rates."""
    depth = len(model.blocks)
    groups = {}
    for name, parameter in model.named_parameters():
        layer = int(name.split('.')[1]) + 1 if name.startswith('blocks.') else depth + 1 if name.startswith(('head.', 'fc_norm.', 'clip_projector.')) else 0
        groups.setdefault(layer, []).append(parameter)
    return [{'params': values, 'lr': lr * .75 ** (depth + 1 - layer)} for layer, values in sorted(groups.items())]
