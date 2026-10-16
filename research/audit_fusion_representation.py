"""Audit visual information and representation semantics without models or GT."""

import hashlib
import json
from pathlib import Path

import numpy as np

from fusion_pose_representation import visible_centroid_features

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/fusion_rgb_expanded_20261013"
OUTPUT = ROOT / "artifacts/fusion_representation_audit_20261013"
CONDITIONS = ("clean", "dark", "noisy_dark", "occlusion")


def distribution(values):
    values = np.asarray(values)
    return None if not values.size else {"mean": float(values.mean()), "median": float(np.median(values)),
                                        "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def describe(poses, hip_features, confidence_mean, confidence_std):
    visible = poses[..., 2] >= .3
    hip_valid = (hip_features[:, 34:] >= .3).any(1)
    # Availability, geometry being zero, and confidence do not depend on focal.
    # Unit focal avoids rereading the original labels/GT for this input-only audit.
    centroid = visible_centroid_features(poses, np.ones(2))
    centroid_valid = (centroid[:, 34:] >= .3).any(1)
    raw_valid = visible.any(1)
    partial = ~hip_valid & raw_valid
    full_missing = ~raw_valid
    count = visible.sum(1)
    geometry = (np.abs(centroid[:, :34]) > 1e-9).any(1)
    assert np.array_equal(hip_valid, visible[:, 11] & visible[:, 12])
    assert np.array_equal(centroid_valid, raw_valid)
    assert np.array_equal(centroid[:, 34:], np.where(visible, poses[..., 2], 0.))
    assert np.array_equal(centroid[:, 34:][hip_valid], hip_features[:, 34:][hip_valid])
    weights = np.where(visible, np.clip(poses[..., 2], 0, 1), 0.)
    anchor = (poses[..., :2] * weights[..., None]).sum(1) / np.maximum(weights.sum(1), 1e-8)[:, None]
    hip_anchor = poses[:, [11, 12], :2].mean(1)
    shift = np.linalg.norm(anchor - hip_anchor, axis=-1)[hip_valid]
    z_conf = (centroid[:, 34:] - confidence_mean) / confidence_std
    observed_conf = centroid[:, 34:][visible]
    observed_z = z_conf[visible]
    return {"frames": len(poses), "hip_valid": int(hip_valid.sum()),
            "any_raw_visible": int(raw_valid.sum()), "centroid_valid": int(centroid_valid.sum()),
            "root_missing_but_partial_visible": int(partial.sum()), "all_raw_missing": int(full_missing.sum()),
            "centroid_geometry_nonzero_frames": int(geometry.sum()),
            "partial_geometry_nonzero_frames": int((partial & geometry).sum()),
            "partial_single_joint_frames": int((partial & (count == 1)).sum()),
            "partial_at_least_two_joints_frames": int((partial & (count >= 2)).sum()),
            "partial_at_least_four_joints_frames": int((partial & (count >= 4)).sum()),
            "partial_visible_joint_count": int(visible[partial].sum()),
            "raw_visible_joint_count": int(visible.sum()),
            "raw_joint_count_histogram": {str(k): int((count == k).sum()) for k in range(18) if (count == k).any()},
            "partial_joint_count_histogram": {str(k): int((count[partial] == k).sum()) for k in range(1, 18) if (count[partial] == k).any()},
            "common_hip_valid_anchor_shift_pixels": distribution(shift),
            "observed_raw_confidence": distribution(observed_conf),
            "observed_zscored_confidence_negative_count": int((observed_z < 0).sum()),
            "observed_zscored_confidence_count": int(observed_z.size),
            "raw_confidence_min": float(poses[..., 2].min()), "raw_confidence_max": float(poses[..., 2].max()),
            "raw_confidence_outside_0_1_count": int(((poses[..., 2] < 0) | (poses[..., 2] > 1)).sum())}


def main():
    protocol = json.loads((CACHE / "protocol.json").read_text(encoding="utf-8"))
    groups = {"train": protocol["train_subjects"], "validation": protocol["validation_subjects"]}
    rows = {}
    for subject in groups["train"] + groups["validation"]:
        with np.load(CACHE / f"subject{subject}.npz", allow_pickle=False) as cache:
            rows[subject] = {c: (cache[c + "_poses"].astype(np.float64), cache[c + "_features"].copy()) for c in CONDITIONS}
    clean_confidence = np.concatenate([rows[s]["clean"][1][:, 34:] for s in groups["train"]])
    centroid_clean_confidence = np.concatenate([visible_centroid_features(rows[s]["clean"][0], np.ones(2))[:, 34:] for s in groups["train"]])
    assert np.array_equal(clean_confidence, centroid_clean_confidence)
    confidence_mean, confidence_std = clean_confidence.mean(0), np.maximum(clean_confidence.std(0), 1e-8)
    report = {"status": "complete", "scope": "Input-only and current-source audit; no training or GT pose/error reads",
              "cache_directory": str(CACHE), "cache_integrity_recheck": False,
              "prior_integrity_record": str(CACHE / "cache_verification.json"),
              "test_subjects_used": [], "conditions": {}, "per_subject": {},
              "coordinate_focal_for_availability_audit": [1., 1.],
              "focal_note": "Only validity, centered geometry being nonzero, raw confidence and pixel anchor shift are computed. Actual runner uses published cam1 intrinsic focal; no original label archive is read here.",
              "confidence_normalization": {"clean_train_hip_centroid_confidence_exact_equal": True,
                  "mean": confidence_mean.tolist(), "std": confidence_std.tolist(),
                  "semantics": "Raw score side channel is copied before zscore. Observed RGB confidence slots are train-clean zscores; missing XY/conf slots explicitly zero. Model validity and gate evidence use raw score>=0.3, not the zscore.",
                  "calibration": "RTMO confidence scores are not calibrated 3D error probabilities. Anchor weights clip score to[0,1], returned confidence keeps raw score."},
              "source_sha256": {name: hashlib.sha256((ROOT / "research" / name).read_bytes()).hexdigest() for name in
                  ("fusion_pose_representation.py", "run_fusion_experiments_gpu.py", "fusion_models.py")},
              "fairness": {"same_feature_dimension": 51, "architecture_parameter_change": False,
                  "normalization": "Each representation fits its own clean-training XY mean/std. Confidence training statistics happen to be exactly identical on this cache.",
                  "GT_input": "visible_centroid_features accepts predicted poses and focal only. GPU loader uses labels.camera_matrix.cam1_intrinsic for focal; GT keypoints/targets do not determine the anchor or visibility.",
                  "radar_augmentation": "Shared original hip-cache availability governs radar dropout. Condition order, batch and augmentation generators are representation-independent.",
                  "loss_support_change": "Centroid expands video_valid and both masks on partial frames; video branch, candidate and utility auxiliary losses consequently train on more examples. Cross-representation gains include this change.",
                  "anchor_change": "Centroid changes the coordinate anchor on every observed frame, including hip-valid frames. This is not a hip-preserving fallback intervention.",
                  "execution_check_pending": "Paired formal runs must additionally match cache SHA, split, seeds, budgets and backend; radar_single weights/predictions/history must be identical."},
              "interpretation": {"full_missing": "For fusion models, both representations set video_valid false and choose the learned radar branch exactly. Between-run gains here can come from changed joint training gradients, not restored local visual inputs.",
                  "partial": "Hip suppresses the whole visual feature; centroid enables visual/confidence input and fusion. Single-joint centroid retains confidence/identity but has zero relative XY geometry.",
                  "common_hip_valid": "Both preserve the same visible confidence channels and pairwise joint geometry; centroid changes a frame-dependent common XY translation and the clean-training XY statistics.",
                  "statistics": "Frames/joints remain correlated within four validation subjects. Counts and subgroup metrics do not turn them into independent replicates."}}
    for group, subjects in groups.items():
        report["conditions"][group] = {}
        for condition in CONDITIONS:
            poses = np.concatenate([rows[s][condition][0] for s in subjects])
            features = np.concatenate([rows[s][condition][1] for s in subjects])
            report["conditions"][group][condition] = describe(poses, features, confidence_mean, confidence_std)
        for subject in subjects:
            report["per_subject"][str(subject)] = {"split": group, "conditions": {
                c: describe(*rows[subject][c], confidence_mean, confidence_std) for c in CONDITIONS}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "representation_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "conditions": {group: {c: {
        key: row[key] for key in ("frames", "hip_valid", "any_raw_visible", "centroid_valid", "root_missing_but_partial_visible", "all_raw_missing", "partial_geometry_nonzero_frames")}
        for c, row in records.items()} for group, records in report["conditions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
