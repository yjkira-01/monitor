"""固定权重的整段分类审计；扰动测试不能替代真实外部或事件级验证。"""
import csv
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
from app.video_model import TrainedVideoModel
from scripts.train_video import classification_metrics


def sampled_frames(path, count=16):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = sorted([n for n in archive.namelist() if n.lower().endswith(".png")],
                           key=lambda n: int(Path(n).stem.rsplit("-", 1)[1]))
            indices = np.linspace(0, len(names)-1, count).round().astype(int)
            frames = [cv2.imdecode(np.frombuffer(archive.read(names[i]), dtype=np.uint8), cv2.IMREAD_COLOR) for i in indices]
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"视频不可读：{path}")
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(length-1, 0), count).round().astype(int)
        targets, selected = set(indices), {}
        try:
            for index in range(int(indices[-1])+1):
                if not cap.grab():
                    break
                if index in targets:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    selected[index] = frame
        finally:
            cap.release()
        if not selected:
            raise ValueError(f"采样失败：{path}")
        frames = [selected.get(i, selected[max(selected)]) for i in indices]
    if len(frames) != count or any(f is None for f in frames):
        raise ValueError(f"帧解码失败：{path}")
    return frames


def variants(frames, size):
    area = [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames]
    linear = [cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR) for frame in frames]
    occluded, small, shifted = [], [], []
    width, height = size
    for frame in area:
        blocked = frame.copy()
        blocked[:, int(width*.4):int(width*.6)] = 128
        occluded.append(blocked)
        small.append(cv2.warpAffine(frame, np.array([[.65, 0, width*.175], [0, .65, height*.175]], dtype=np.float32), size, borderValue=(128,128,128)))
        shifted.append(cv2.warpAffine(frame, np.array([[1, 0, width*.15], [0, 1, 0]], dtype=np.float32), size, borderMode=cv2.BORDER_REPLICATE))
    return {"matched_area": area, "previous_linear": linear,
            "dim_50pct": [(frame.astype(np.float32)*.5).astype(np.uint8) for frame in area],
            "blur_5px": [cv2.GaussianBlur(frame, (5,5), 0) for frame in area],
            "occluded_center_20pct": occluded, "scaled_65pct": small, "shift_right_15pct": shifted}


def audit(model, manifest, output, stress):
    with manifest.open(encoding="utf-8-sig") as source:
        rows = [r for r in csv.DictReader(source) if r["split"] == ("val" if stress else "external")]
    if not rows:
        raise ValueError("审计清单没有验证/外部样本")
    predictions = []
    for index, row in enumerate(rows):
        frames = sampled_frames(ROOT / row["path"])
        clips = variants(frames, model.read_size) if stress else {"matched_area": frames}
        scores = {name: float(model.predict(clip)["fall_score"]) for name, clip in clips.items()}
        assert all(np.isfinite(s) and 0 <= s <= 1 for s in scores.values())
        predictions.append({"path": row["path"], "label": row["label"], "description": row.get("description", ""), "scores": scores})
        if (index+1) % 5 == 0 or index == len(rows)-1:
            print(f"{manifest.name}: {index+1}/{len(rows)}", flush=True)
    labels = [int(row["label"] == "fall") for row in predictions]
    metrics = {name: classification_metrics([int(row["scores"][name] >= .5) for row in predictions], labels, 1)
               for name in predictions[0]["scores"]}
    reference = [int(row["scores"]["matched_area"] >= .5) for row in predictions]
    flips = {name: sum(int(row["scores"][name] >= .5) != ref for row, ref in zip(predictions, reference)) for name in metrics}
    report = {"checkpoint": "gmdcsa24_r3d18_100ep_inference.pt", "scope": "whole_sequence",
              "note": "固定权重与0.5阈值；扰动是人工合成，原验证集参与过选模；外部结果仅用于泛化审计，不用于调阈值。",
              "count": len(rows), "metrics": metrics, "prediction_flips_vs_matched": flips, "predictions": predictions}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest.name, "metrics": metrics, "flips": flips}, ensure_ascii=False), flush=True)
    return report


def main():
    model = TrainedVideoModel(ROOT / "artifacts/gmdcsa24_r3d18_100ep_inference.pt")
    audit(model, ROOT / "data/manifests/gmdcsa24.csv", ROOT / "artifacts/gmdcsa24_robustness_audit.json", True)
    external = ROOT / "data/manifests/urfd_external.csv"
    if external.is_file():
        audit(model, external, ROOT / "artifacts/urfd_external_audit.json", False)


if __name__ == "__main__":
    main()
