"""把作者的粗时间区间转换为滑窗清单；缺失/畸形标签进入复核表。"""
import csv
from collections import Counter
import json
from pathlib import Path
import re

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fall_intervals(classes):
    sections = [s.strip() for s in classes.split(";") if re.search(r"\bFall(?:ing)?\b", s, re.I)]
    if not sections:
        raise ValueError("缺少跌倒时间区间")
    intervals = []
    for section in sections:
        match = re.search(r"\[\s*(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*\]", section)
        if not match:
            raise ValueError(f"跌倒区间格式不完整：{section}")
        start, end = map(float, match.groups())
        if not 0 <= start < end:
            raise ValueError("跌倒时间范围无效")
        intervals.append((start, end))
    return intervals


def window_label(start, end, intervals):
    if not intervals:
        return "adl"
    overlap = sum(max(0, min(end, stop) - max(start, begin)) for begin, stop in intervals)
    if overlap >= .5:
        return "fall"
    if end <= min(begin for begin, _ in intervals) - .2:
        return "adl"
    return None


def main():
    with (ROOT / "data/manifests/gmdcsa24.csv").open(encoding="utf-8-sig") as source:
        parents = list(csv.DictReader(source))
    rows, reviews, coverage = [], [], []
    for parent in parents:
        cap = cv2.VideoCapture(str(ROOT / parent["path"]))
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if not np.isfinite([fps, count]).all() or fps <= 0 or count < 2:
            reviews.append({"path": parent["path"], "reason": "无法读取视频时长", "classes": parent["classes"]})
            continue
        duration = (count-1) / fps
        try:
            intervals = fall_intervals(parent["classes"]) if parent["label"] == "fall" else []
            if any(begin >= duration or stop > duration+.25 for begin, stop in intervals):
                raise ValueError("作者时间区间超出实测视频长度，请人工对齐")
        except ValueError as exc:
            reviews.append({"path": parent["path"], "reason": str(exc), "classes": parent["classes"], "actual_duration_s": duration})
            continue
        ends = list(np.arange(1.0, duration, .8)) + [duration]
        used = set()
        for end in ends:
            end = round(float(end), 6)
            if end in used or end < .5:
                continue
            used.add(end)
            start = round(max(0, end-5), 6)
            label = window_label(start, end, intervals)
            if label is None:
                continue
            rows.append({"path": parent["path"], "split": parent["split"], "label": label, "subject": parent["subject"],
                         "start_s": start, "end_s": end, "parent_label": parent["label"], "dataset": "GMDCSA24",
                         "annotation_source": "author_coarse_interval_unreviewed", "description": parent["description"]})
        description = parent["description"].lower()
        coverage.append({"path": parent["path"], "bed": "bed" in description, "backward": "backward" in description,
                         "occlusion_mentioned": "not visible" in description or "visible after" in description,
                         "lying_negative": parent["label"] == "adl" and "sleep" in description})
    train_paths = {r["path"] for r in rows if r["split"] == "train"}
    val_paths = {r["path"] for r in rows if r["split"] == "val"}
    assert not train_paths & val_paths
    manifest = ROOT / "data/manifests/gmdcsa24_windows.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"counts": dict(Counter(f"{r['split']}/{r['label']}" for r in rows)), "windows": len(rows),
              "independent_parent_videos": len(train_paths | val_paths), "subjects": 4,
              "note": "粗标签准备清单；须复核视频与时间边界，窗口数增加不等于新增独立场景/受试者。不得把帧级Sleeping标签当成医学睡眠分期。",
              "needs_review": reviews, "scene_coverage": coverage}
    (ROOT / "artifacts/gmdcsa24_window_label_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "scene_coverage"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
