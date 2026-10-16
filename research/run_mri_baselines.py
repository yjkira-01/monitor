"""Development-only ridge baselines on published 2D keypoints and radar data."""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from audit_mri import interval_bounds, load_labels


def radar_statistics(points):
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[1:] != (196, 5) or not np.isfinite(points).all():
        raise ValueError("Expected finite padded radar frames of shape (N,196,5)")
    mask = np.any(points != 0, axis=-1)
    count = mask.sum(-1)
    divisor = np.maximum(count, 1)[:, None]
    mean = (points * mask[..., None]).sum(1) / divisor
    variance = (((points - mean[:, None, :]) ** 2) * mask[..., None]).sum(1) / divisor
    low = np.where(mask[..., None], points, np.inf).min(1)
    high = np.where(mask[..., None], points, -np.inf).max(1)
    low[count == 0] = 0; high[count == 0] = 0
    return np.concatenate([mean, np.sqrt(variance), low, high, (count / 196.)[:, None]], axis=-1), count > 0


def subject_samples(archive, subject, seed):
    prefix = "dataset_release/"
    labels = load_labels(archive.read(prefix + f"aligned_data/pose_labels/subject{subject}_all_labels.cpl"))
    radar = np.load(io.BytesIO(archive.read(prefix + f"features/radar/subject{subject}_featuremap.npy")), allow_pickle=False)
    length = len(labels["refined_gt_kps"])
    rs, re = interval_bounds(labels["radar_avail_frames"], length)
    gs, ge = interval_bounds(labels["gt_avail_frames"], length)
    vs, ve = interval_bounds(labels["2d_l_avail_frames"], length)
    ids = np.arange(max(rs, gs, vs), min(re, ge, ve) + 1)
    gt = labels["refined_gt_kps"][ids].transpose(0, 2, 1).astype(np.float64)
    gt -= (gt[:, 11] + gt[:, 12])[:, None, :] / 2
    image = labels["pose_2d_l"][ids].transpose(0, 2, 1).astype(np.float64)
    points = radar[ids - rs].reshape(-1, 196, 5)
    point_mask = np.any(points != 0, axis=-1)
    bone_edges = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
    lengths = np.stack([np.linalg.norm(gt[:, a] - gt[:, b], axis=-1) for a, b in bone_edges], axis=-1)
    if not ((np.median(lengths, axis=0) > .1) & (np.median(lengths, axis=0) < .8)).all():
        raise ValueError("Typical label scale is not consistent with metres")
    valid = (np.isfinite(gt).all(axis=(1, 2)) & np.isfinite(image).all(axis=(1, 2)) &
             np.isfinite(points).all(axis=(1, 2)) & point_mask.any(-1) &
             (lengths.max(-1) <= 1.) & (np.linalg.norm(gt, axis=-1).max(-1) <= 3.))
    usable = np.flatnonzero(valid)
    rng = np.random.default_rng(np.random.SeedSequence([seed, subject]))
    chosen = np.sort(rng.choice(usable, size=min(256, len(usable)), replace=False))
    if not chosen.size:
        raise ValueError("Subject has no usable development observations")
    gt, image, points, point_mask = gt[chosen], image[chosen], points[chosen], point_mask[chosen]
    intrinsic = labels["camera_matrix"]["cam1_intrinsic"]
    focal = np.array([intrinsic[0, 0], intrinsic[1, 1]])
    if not np.isfinite(focal).all() or (focal <= 0).any():
        raise ValueError("Invalid camera focal length")
    image -= (image[:, 11] + image[:, 12])[:, None, :] / 2
    image /= focal
    radar_stats, _ = radar_statistics(points)
    return {"video": image.reshape(len(chosen), -1), "radar": radar_stats,
            "target_mm": gt.reshape(len(chosen), -1) * 1000,
            "frame_ids": ids[chosen], "usable_frames": len(usable), "excluded_frames": int((~valid).sum())}


def fit_ridge(train_x, train_y, validation_x):
    if not all(np.isfinite(x).all() for x in (train_x, train_y, validation_x)):
        raise ValueError("Nonfinite regression input")
    mean, std = train_x.mean(0), np.maximum(train_x.std(0), 1e-8)
    x = np.column_stack([(train_x - mean) / std, np.ones(len(train_x))])
    penalty = np.eye(x.shape[1]); penalty[-1, -1] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ train_y)
    val = np.column_stack([(validation_x - mean) / std, np.ones(len(validation_x))])
    return val @ weights, {"normalization_mean": mean, "normalization_std": std, "weights": weights}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    protocol = json.loads((base / "pose_fusion_protocol.json").read_text(encoding="utf-8"))
    split = protocol["splits"][0]
    train_ids = split["canonical_ids_train"]; val_ids = split["canonical_ids_validation"]
    if set(train_ids) & set(val_ids) or (set(train_ids) | set(val_ids)) & set(split["canonical_ids_test"]):
        raise ValueError("Subject leakage")
    audit = json.loads((base.parent / "artifacts/research_execution_20261012/mri_data_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "complete" or not audit["all_feature_mapping_matches"] or not audit["all_typical_pose_scales_consistent_with_metres"]:
        raise ValueError("Published data audit has not passed")
    from hashlib import sha256
    h = sha256()
    with args.archive.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != audit["archive_sha256"]:
        raise ValueError("Development data differ from audited archive")
    samples = {}
    with zipfile.ZipFile(args.archive) as z:
        for subject in train_ids + val_ids:
            samples[subject] = subject_samples(z, subject, split["seed"])
            print(json.dumps({"subject": subject, "samples": len(samples[subject]["target_mm"])}), flush=True)
    train_y = np.concatenate([samples[i]["target_mm"] for i in train_ids])
    val_y = np.concatenate([samples[i]["target_mm"] for i in val_ids])
    report = {"status": "complete", "kind": "Development validation only; no raw-RGB robustness or proposed-method evaluation",
              "train_subjects": train_ids, "validation_subjects": val_ids, "test_subjects_used": [],
              "train_frames": len(train_y), "validation_frames": len(val_y), "seed": split["seed"],
              "model": "frame-wise ridge, alpha=1, train-only feature normalization",
              "video_input": "author-provided 2D keypoints, not newly decoded RGB", "target": "COCO17 hip-centred pose, metre-scale labels converted to mm",
              "label_screen": "fixed limb <=1m and root-relative radius <=3m; filtered development result only",
              "subject_sample_counts": {str(i): {"usable": s["usable_frames"], "excluded": s["excluded_frames"], "sampled": len(s["target_mm"])} for i, s in samples.items()},
              "metrics": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, modalities in (("video_keypoints", ["video"]), ("radar_statistics", ["radar"]),
                             ("concatenation", ["video", "radar"])):
        def features(ids):
            return np.concatenate([np.concatenate([samples[i][m] for m in modalities], axis=-1) for i in ids])
        prediction, model = fit_ridge(features(train_ids), train_y, features(val_ids))
        errors = np.linalg.norm((prediction - val_y).reshape(-1, 17, 3), axis=-1)
        rows = errors.mean(-1)
        report["metrics"][name] = {"MPJPE_mm": float(rows.mean()), "frame_error_p95_mm": float(np.percentile(rows, 95)),
                                   "per_joint_mean_mm": errors.mean(0).tolist()}
        np.savez(args.output.with_name(name + "_ridge.npz"), **model)
        print(name, report["metrics"][name]["MPJPE_mm"], flush=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
