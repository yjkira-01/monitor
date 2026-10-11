"""Cache actual MediaPipe image poses once per video; retain missing-frame masks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_HASH = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"
model = None


def init_worker(model_path):
    global model
    from mediapipe.tasks.python import BaseOptions, vision
    cv2.setNumThreads(1)
    model = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU), num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5))


def extract(path_string):
    import mediapipe as mp
    path = ROOT/path_string
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    output = ROOT/"data/processed/poses"/(hashlib.sha256(path_string.encode()).hexdigest()+".npz")
    if output.exists():
        with np.load(output, allow_pickle=False) as cache:
            if str(cache["source_sha256"]) == source_hash and str(cache["model_sha256"]) == MODEL_HASH:
                return path_string, output.relative_to(ROOT).as_posix()
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]) if hasattr(cv2, "CAP_PROP_N_THREADS") else cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Unreadable video: {path_string}")
    times, poses = [], []
    index = 0
    next_time = 0.
    try:
        while cap.grab():
            timestamp = index/fps
            index += 1
            if timestamp+1e-6 < next_time:
                continue
            next_time += 1/15.
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f"Failed sampled frame: {path_string}/{index}")
            height, width = frame.shape[:2]
            if width > 640:
                frame = cv2.resize(frame, (640, round(height*640/width)), interpolation=cv2.INTER_AREA)
            result = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            pose = np.zeros((33, 3), dtype=np.float32)
            if result.pose_landmarks:
                pose[:] = [[p.x*width/height, p.y, p.visibility] for p in result.pose_landmarks[0]]
                pose[~np.isfinite(pose).all(1)] = 0
            times.append(timestamp)
            poses.append(pose)
    finally:
        cap.release()
    if not poses:
        raise ValueError(f"No decoded pose frames: {path_string}")
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, times=np.asarray(times), poses=np.asarray(poses), source_sha256=source_hash, model_sha256=MODEL_HASH)
    temporary.replace(output)
    return path_string, output.relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    model_path = ROOT/"artifacts/pose_landmarker_full.task"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_HASH:
        raise ValueError("Pose model checksum mismatch")
    (ROOT/"data/processed/poses").mkdir(parents=True, exist_ok=True)
    with (ROOT/args.manifest).open(encoding="utf-8-sig") as handle:
        # Final test frames are kept untouched, even during preprocessing.
        rows = [r for r in csv.DictReader(handle) if r["split"] in {"train", "val"}]
    paths = sorted({r["path"] for r in rows})
    cached = {}
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(str(model_path),)) as pool:
        for index, (path, cache) in enumerate(pool.map(extract, paths, chunksize=1), 1):
            cached[path] = cache
            if index % 20 == 0 or index == len(paths):
                print(f"pose_cached {index}/{len(paths)}", flush=True)
    accepted, excluded = [], []
    for row in rows:
        with np.load(ROOT/cached[row["path"]], allow_pickle=False) as cache:
            mask = (cache["times"] >= float(row["start_s"])-1e-6) & (cache["times"] <= float(row["end_s"])+1e-6)
            poses = cache["poses"][mask]
        torso_visible = (poses[:, [11, 12, 23, 24], 2] >= .3).all(1) if len(poses) else np.zeros(0)
        coverage = float(torso_visible.mean()) if len(poses) else 0.
        if coverage < .2:
            excluded.append({"path": row["path"], "split": row["split"], "label": row["label"], "start_s": row["start_s"], "coverage": coverage})
            continue
        accepted.append({**row, "pose_path": cached[row["path"]], "pose_coverage": coverage})
    if not accepted:
        raise ValueError("No windows with usable poses")
    output = ROOT/args.output
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, list(accepted[0]))
        writer.writeheader()
        writer.writerows(accepted)
    report = {"source_manifest_sha256": hashlib.sha256((ROOT/args.manifest).read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "model_sha256": MODEL_HASH,
              "accepted": len(accepted), "excluded": excluded,
              "note": "Only detected poses, no interpolation of missing joints. Selected single pose is not validated person tracking; coverage filtering can bias metrics."}
    (ROOT/"artifacts"/(output.stem+"_coverage.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"pose_manifest {args.output} accepted={len(accepted)} excluded={len(excluded)}", flush=True)


if __name__ == "__main__":
    main()
