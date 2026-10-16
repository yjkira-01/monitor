"""Finite input-only scale audit of both partial fallbacks; no models or GT."""

import json
from pathlib import Path

import numpy as np

from fusion_partial_representation import hip_preserving_fallback_features

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/fusion_rgb_expanded_20261013"
HIP_RUN = ROOT / "artifacts/fusion_method_research_20261013_expanded_hip"
OUTPUT = ROOT / "artifacts/fusion_representation_audit_20261013/partial_input_scale_audit.json"
CONDITIONS = ("clean", "dark", "noisy_dark", "occlusion")


def magnitude(values):
    values = np.asarray(values)
    absolute = np.abs(values).ravel()
    return None if not absolute.size else {"values": int(absolute.size), "finite": bool(np.isfinite(absolute).all()),
        "abs_max": float(absolute.max()), "abs_p50": float(np.percentile(absolute, 50)),
        "abs_p95": float(np.percentile(absolute, 95)), "abs_p99": float(np.percentile(absolute, 99)),
        "abs_p999": float(np.percentile(absolute, 99.9)),
        "abs_gt10": int((absolute > 10).sum()), "abs_gt100": int((absolute > 100).sum()),
        "abs_gt1000": int((absolute > 1000).sum())}


def infer_focal_from_inputs(poses, hip):
    visible = poses[..., 2] >= .3
    valid = (hip[:, 34:] >= .3).any(1)
    delta = poses[..., :2].astype(np.float64) - poses[:, [11, 12], :2].astype(np.float64).mean(1)[:, None]
    relative = hip[:, :34].reshape(-1, 17, 2)
    focal = []
    for axis in range(2):
        usable = valid[:, None] & visible & (np.abs(relative[..., axis]) > 1e-5)
        ratios = delta[..., axis][usable] / relative[..., axis][usable]
        if not ratios.size:
            raise ValueError("No observed hip-relative input to recover focal")
        focal.append(np.median(ratios))
    focal = np.asarray(focal)
    error = np.abs(delta / focal - relative)[valid[:, None] & visible]
    if not np.isfinite(focal).all() or (focal <= 0).any() or error.max() > 1e-10:
        raise ValueError("Cached predicted pixel/hip-feature scaling is not a constant focal")
    return focal, float(error.max())


def main():
    protocol = json.loads((CACHE / "protocol.json").read_text(encoding="utf-8"))
    with np.load(HIP_RUN / "normalization.npz", allow_pickle=False) as saved:
        mean, std = saved["rgb_mean"].copy(), saved["rgb_std"].copy()
    groups = {"train": protocol["train_subjects"], "validation": protocol["validation_subjects"]}
    rows, focal_record = {}, {}
    for subject in groups["train"] + groups["validation"]:
        with np.load(CACHE / f"subject{subject}.npz", allow_pickle=False) as cache:
            rows[subject] = {c: (cache[c + "_poses"].copy(), cache[c + "_features"].copy()) for c in CONDITIONS}
        focal, maximum_error = infer_focal_from_inputs(*rows[subject]["clean"])
        focal_record[str(subject)] = {"focal": focal.tolist(), "reconstruction_max_abs_error": maximum_error}
    clean = np.concatenate([rows[s]["clean"][1] for s in groups["train"]])
    np.testing.assert_array_equal(clean.mean(0), mean)
    np.testing.assert_array_equal(np.maximum(clean.std(0), 1e-8), std)
    clean_visible = clean[:, 34:] >= .3
    never_seen = ~clean_visible.any(0)
    report = {"status": "complete", "scope": "Input-only finite scale diagnostics; no training, target reads or ZIP hashing",
        "normalization_source": str(HIP_RUN / "normalization.npz"), "shared_hip_clean_stats_reconstruction_exact": True,
        "normalization_floor_unchanged": 1e-8, "test_subjects_used": [],
        "xy_std_min": float(std[:34].min()), "xy_std_max": float(std[:34].max()),
        "xy_std_at_floor_indices": np.flatnonzero(std[:34] <= 1.000001e-8).tolist(),
        "xy_std_below_1e_6_indices": np.flatnonzero(std[:34] < 1e-6).tolist(),
        "clean_training_never_seen_joint_ids": np.flatnonzero(never_seen).tolist(),
        "focal_recovery": "Recovered the existing scale from predicted pixel offsets / original cached hip-relative XY, using clean observed input fields only. No original labels or target archive read.",
        "per_subject_focal": focal_record, "conditions": {}}
    for group, subjects in groups.items():
        report["conditions"][group] = {}
        for c in CONDITIONS:
            by_mode = {mode: {"all": [], "partial": [], "partial_xy": [], "observed_partial_xy": []}
                       for mode in ("centroid", "confidence_only")}
            partial_frames, seen_after_never = 0, np.zeros(17, dtype=int)
            maximum_cases = []
            for subject in subjects:
                poses, hip = rows[subject][c]
                visible = poses[..., 2] >= .3
                partial = ~(hip[:, 34:] >= .3).any(1) & visible.any(1)
                partial_frames += int(partial.sum())
                seen_after_never += (visible[partial] & never_seen[None]).sum(0)
                focal = np.asarray(focal_record[str(subject)]["focal"])
                for mode in by_mode:
                    features = hip_preserving_fallback_features(poses, focal, hip, mode)
                    fields = np.concatenate((np.repeat(features[:, 34:] >= .3, 2, 1), features[:, 34:] >= .3), 1)
                    normalized = np.where(fields, (features - mean) / std, 0.).astype(np.float32)
                    by_mode[mode]["all"].append(normalized)
                    by_mode[mode]["partial"].append(normalized[partial])
                    by_mode[mode]["partial_xy"].append(normalized[partial, :34])
                    by_mode[mode]["observed_partial_xy"].append(normalized[partial, :34][fields[partial, :34]])
                    if partial.any():
                        maximum_cases.append({"subject": subject, "mode": mode, "partial_abs_max": float(np.abs(normalized[partial]).max())})
            report["conditions"][group][c] = {"partial_frames": partial_frames,
                "never_seen_clean_joint_becomes_partial_visible_counts": seen_after_never.tolist(),
                "modes": {mode: {kind: magnitude(np.concatenate(arrays)) for kind, arrays in kinds.items()}
                          for mode, kinds in by_mode.items()}, "per_subject_partial_maxima": maximum_cases}
    report["interpretation"] = ["A small clean-training std can amplify partial geometry; this audit does not tune floors or clamp inputs using validation.",
        "Confidence-only raw XY=0 produces fixed -hip_mean/hip_std in observed XY slots; unobserved slots remain zero.",
        "Input magnitude diagnoses optimization conditions, not actual geometric utility or 3D performance."]
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "xy_std_min": report["xy_std_min"],
        "xy_std_at_floor_indices": report["xy_std_at_floor_indices"], "never_seen_clean_joints": report["clean_training_never_seen_joint_ids"],
        "partial_occlusion": {group: {mode: row["modes"][mode]["partial"] for mode in row["modes"]}
            for group, records in report["conditions"].items() for row in [records["occlusion"]]}}, indent=2))


if __name__ == "__main__":
    main()
