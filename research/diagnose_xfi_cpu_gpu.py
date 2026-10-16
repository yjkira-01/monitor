"""Locate cross-backend differences without changing the official model."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps/research_cuda"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from xfi_paper_baseline import XFiRGBRadar, official_modules


def main():
    directory = ROOT / "artifacts/xfi_mri_pretrained_probe_v4_20261013"
    destination = directory / "cross_backend_diagnosis.json"
    if destination.exists():
        raise ValueError("New diagnosis output required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    models = []
    with np.load(directory / "probe_weights.npz", allow_pickle=False) as saved:
        state = {key: torch.from_numpy(saved[key].copy()) for key in saved.files}
    for device in ("cpu", "cuda"):
        model = XFiRGBRadar().to(device)
        model.load_state_dict(state, strict=True)
        model.freeze_backbones()
        model.eval()
        models.append(model)
    with np.load(directory / "probe_observations_predictions.npz", allow_pickle=False) as saved:
        images = torch.tensor(saved["images"], dtype=torch.float32)
        points = torch.tensor(saved["points"], dtype=torch.float32)
    features, outputs = [], []
    flag = [True, False, True, False, False]
    with torch.no_grad():
        for model, device in zip(models, ("cpu", "cuda")):
            image, point = images.to(device), points.to(device)
            rgb = model.rgb_extractor(image).view(len(image), 512, -1).permute(0, 2, 1)
            radar = model.radar_extractor(point)[0]
            features.append((rgb, radar))
            projected = model.linear_projector([rgb, radar], None, flag)
            outputs.append(model.X_Fusion_block(projected, flag).cpu())
        common_rgb, common_radar = [value.to("cuda") for value in features[0]]
        common = models[1].X_Fusion_block(models[1].linear_projector([common_rgb, common_radar], None, flag), flag).cpu()
        radar_namespace = official_modules()[1]
        xyz = points[..., :3]
        cpu_indices = radar_namespace["square_distance"](xyz, xyz).argsort()[..., :16]
        gpu_xyz = xyz.to("cuda")
        gpu_indices = radar_namespace["square_distance"](gpu_xyz, gpu_xyz).argsort()[..., :16].cpu()
    result = {"status": "complete_diagnosis", "native_model_modified": False,
              "RGB_token_max_absolute_difference": float((features[0][0] - features[1][0].cpu()).abs().max()),
              "radar_token_max_absolute_difference": float((features[0][1] - features[1][1].cpu()).abs().max()),
              "full_prediction_max_difference_mm": float((outputs[0] - outputs[1]).abs().max() * 1000),
              "identical_CPU_tokens_fusion_CPU_vs_GPU_max_difference_mm": float((outputs[0] - common).abs().max() * 1000),
              "native_initial_knn_index_difference_count": int((cpu_indices != gpu_indices).sum()),
              "scope": "Four training frames, diagnosis only; retain failed .01mm replay gate"}
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
