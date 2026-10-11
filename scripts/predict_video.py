"""用训练导出的推理权重分类整个视频；不会把片段标签当作逐帧事件定位。"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yanglao_monitor.video import ACTIVITY_CLASSES, TinyVideoNet, _read_frames


@torch.no_grad()
def predict(video: str, checkpoint: str, device: str = "cpu") -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = state["classes"]
    if classes not in (['adl','fall'],sorted(ACTIVITY_CLASSES)) or state["clip_len"] < 1:
        raise ValueError("权重类别或片段长度不符合本项目")
    preprocess = None
    if state["model_name"] == "r3d18":
        from torchvision.models.video import R3D_18_Weights, r3d_18
        model = r3d_18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        preprocess = R3D_18_Weights.DEFAULT.transforms()
    elif state["model_name"] == "tiny":
        model = TinyVideoNet(len(classes))
    elif state['model_name'] == 'internvideo2_base':
        from yanglao_monitor.foundation_video import build_internvideo2, normalize_video, PREPROCESSING
        if state.get('preprocessing') != PREPROCESSING or state['clip_len'] != 8 or tuple(state['read_size']) != (224,224):
            raise ValueError('InternVideo2 preprocessing metadata is incompatible')
        model = build_internvideo2(len(classes), checkpointing=False)
        preprocess = normalize_video
    else:
        raise ValueError("不支持的模型架构")
    model.load_state_dict(state["model"])
    model.to(device).eval()
    clip = _read_frames(Path(video), state["clip_len"], tuple(state["read_size"]), preserve_aspect=state['model_name']=='internvideo2_base').unsqueeze(0).to(device)
    if preprocess:
        clip = preprocess(clip.permute(0, 2, 1, 3, 4).contiguous())
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=str(device).startswith('cuda') and state['model_name']=='internvideo2_base'):
        scores = model(clip).float().softmax(1)[0].cpu().tolist()
    score = sum(scores[classes.index(c)] for c in ('fall','fallen') if c in classes)
    label = classes[max(range(len(scores)),key=scores.__getitem__)]
    return {"video": str(Path(video).resolve()), "model": state["model_name"], "scope": "whole_video",
            "fall_probability": score, "predicted": label,
            **({'activity_scores':dict(zip(classes,scores))} if len(classes) > 2 else {}),
            "note": "整段视频分类，尚未定位跌倒时间；不是实时告警或校准风险概率。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--checkpoint", required=True, help="训练导出的 *_inference.pt")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(predict(args.video, args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu"),
                     ensure_ascii=False, indent=2))
