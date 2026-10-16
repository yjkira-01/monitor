"""Audit the published mRI ZIP without extracting models or executing pickles."""

import argparse
import hashlib
import io
import json
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np

from pose_fusion import nearest_timestamp_pairs


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        for prefix in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[prefix, "_reconstruct"] = np._core.multiarray._reconstruct
            allowed[prefix, "scalar"] = np._core.multiarray.scalar
        if (module, name) not in allowed:
            raise ValueError(f"Unapproved pickle global: {module}.{name}")
        return allowed[module, name]


def load_labels(data):
    labels = NumpyOnlyUnpickler(io.BytesIO(data)).load()
    if not isinstance(labels, dict):
        raise ValueError("Expected a label dictionary")
    return labels


def interval_bounds(value, length):
    if not isinstance(value, list) or len(value) != 2 or any(type(x) is not int for x in value):
        raise ValueError("Availability must be an integer start/end interval")
    start, end = value
    if not 0 <= start <= end < length:
        raise ValueError("Availability is outside the recording")
    return start, end


def percentiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(zip(("min", "p50", "p95", "p99", "max"),
                    np.percentile(values, [0, 50, 95, 99, 100]).tolist())) if values.size else None


def audit_subject(archive, subject):
    prefix = "dataset_release/"
    names = archive.namelist()

    def read(name):
        info = archive.getinfo(prefix + name)
        if info.file_size > 128 * 1024 * 1024:
            raise ValueError("Audit member exceeds 128 MiB limit")
        return archive.read(info)

    labels = load_labels(read(f"aligned_data/pose_labels/subject{subject}_all_labels.cpl"))
    stamps = np.load(io.BytesIO(read(f"raw_data/unixtime/subject{subject}_unix.npy")), allow_pickle=False)
    features = np.load(io.BytesIO(read(f"features/radar/subject{subject}_featuremap.npy")), allow_pickle=False)
    if stamps.ndim != 1 or stamps.dtype.kind not in "iu" or (np.diff(stamps) <= 0).any():
        raise ValueError("Expected strictly increasing integer millisecond timestamps")
    length = len(stamps)
    for key, shape in (("refined_gt_kps", (length, 3, 17)), ("pose_2d_l", (length, 2, 17)),
                       ("pose_2d_r", (length, 2, 17))):
        if labels[key].shape != shape:
            raise ValueError(f"Unexpected {key} shape")
    start, end = interval_bounds(labels["radar_avail_frames"], length)
    if features.shape != (end - start + 1, 14, 14, 5):
        raise ValueError("Radar feature count does not match inclusive availability interval")
    gt_start, gt_end = interval_bounds(labels["gt_avail_frames"], length)
    raw_names = [n for n in names if re.fullmatch(prefix + f"raw_data/radar/subject{subject}_\\d+\\.csv", n)]
    if len(raw_names) != 1:
        raise ValueError("Expected exactly one raw radar recording per subject")
    raw = np.loadtxt(io.StringIO(read(raw_names[0][len(prefix):]).decode("utf-8")), delimiter=",", skiprows=1)
    aligned = np.loadtxt(io.StringIO(read(f"aligned_data/radar/singleframe/subject{subject}.csv").decode("utf-8")),
                         delimiter=",", skiprows=1)
    if raw.ndim != 2 or raw.shape[1] != 8 or aligned.ndim != 2 or aligned.shape[1] != 9:
        raise ValueError("Unexpected radar CSV schema")
    if not np.isfinite(raw).all() or not np.isfinite(aligned).all():
        raise ValueError("Nonfinite radar data")
    if not np.array_equal(raw[:, 0], raw[:, 0].astype(np.int64)):
        raise ValueError("Noninteger radar frame IDs")
    raw_ids, first, counts = np.unique(raw[:, 0].astype(np.int64), return_index=True, return_counts=True)
    raw_times = raw[first, 7]
    time_consistent = all(np.all(raw[i:i + n, 7] == raw[i, 7]) for i, n in zip(first, counts))
    raw_ids_ordered = not (np.diff(raw[:, 0]) < 0).any()
    if not raw_ids_ordered or not time_consistent:
        raise ValueError("Raw radar IDs or per-frame timestamps are ambiguous")
    pairs = np.unique(aligned[:, [0, 8]], axis=0)
    if not np.array_equal(pairs, pairs.astype(np.int64)):
        raise ValueError("Noninteger aligned radar or camera IDs")
    orig_ids = pairs[:, 0].astype(np.int64)
    positions = np.searchsorted(raw_ids, orig_ids)
    if (positions >= len(raw_ids)).any() or not np.array_equal(raw_ids[positions], orig_ids):
        raise ValueError("Aligned radar IDs not found in raw recording")
    camera_ids = pairs[:, 1].astype(np.int64)
    if (camera_ids < 0).any() or (camera_ids >= length).any():
        raise ValueError("Invalid camera frame mapping")
    if len(np.unique(camera_ids)) != len(camera_ids):
        raise ValueError("Camera mapping is not one-to-one over aligned radar frames")
    _, reuse_counts = np.unique(orig_ids, return_counts=True)
    native_gap_ms = (raw_times[positions] - stamps[camera_ids] / 1000.) * 1000.
    nonempty = np.any(features != 0, axis=(1, 2, 3))
    observed_camera = np.sort(np.unique(aligned[:, 8].astype(np.int64)))
    feature_camera = np.flatnonzero(nonempty) + start
    gt = labels["refined_gt_kps"].transpose(0, 2, 1).astype(np.float64)
    gt -= (gt[:, 11] + gt[:, 12])[:, None, :] / 2
    finite = np.isfinite(gt).all(axis=(1, 2))
    valid = finite & (np.arange(length) >= gt_start) & (np.arange(length) <= gt_end)
    bones = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
    lengths = np.stack([np.linalg.norm(gt[:, a] - gt[:, b], axis=-1) for a, b in bones], axis=-1)
    typical_bones = np.median(lengths[valid], axis=0)
    metre_scale = bool(((typical_bones > .1) & (typical_bones < .8)).all())
    # Fixed physical checks for descriptive auditing; preserve unfiltered results.
    implausible = valid & ((lengths.max(-1) > 1.) | (np.linalg.norm(gt, axis=-1).max(-1) > 3.))
    segments = labels["video_label"]
    actions = np.full(length, "", dtype="U32")
    for action, bounds in segments.items():
        a, b = bounds
        if not 0 <= a <= b <= length:
            raise ValueError("Action annotation is out of bounds")
        actions[a:b] = action
    sensitivity = []
    seconds = (stamps - stamps[0]).astype(np.float64) / 1000.
    for offset in [-.5, -.2, .2, .5]:
        match, gap, ok = nearest_timestamp_pairs(seconds, seconds + offset, .075)
        safe = np.maximum(match, 0)
        keep = ok & valid & valid[safe] & (actions != "") & (actions == actions[safe])
        errors = np.linalg.norm(gt[keep] - gt[safe[keep]], axis=-1).mean(-1)
        screened = keep & ~implausible & ~implausible[safe]
        screened_errors = np.linalg.norm(gt[screened] - gt[safe[screened]], axis=-1).mean(-1)
        sensitivity.append({"offset_ms": int(offset * 1000), "pairs": int(keep.sum()),
                            "label_pose_difference_native_units": percentiles(errors),
                            "mean_native_units": float(errors.mean()) if errors.size else None,
                            "screened_pairs": int(screened.sum()),
                            "screened_label_pose_difference_native_units": percentiles(screened_errors),
                            "screened_mean_native_units": float(screened_errors.mean()) if screened_errors.size else None})
    adjacent = valid[1:] & valid[:-1]
    jumps = np.linalg.norm(gt[1:] - gt[:-1], axis=-1).max(-1)[adjacent]
    return {"subject": subject, "camera_timestamp_frames": length, "camera_interval_ms": percentiles(np.diff(stamps)),
            "gt_availability_inclusive": [gt_start, gt_end], "radar_availability_inclusive": [start, end],
            "radar_feature_frames": len(features), "radar_feature_empty_frames": int((~nonempty).sum()),
            "aligned_camera_frame_count": len(observed_camera),
            "nonempty_features_match_aligned_camera_ids": bool(np.array_equal(feature_camera, observed_camera)),
            "raw_radar_frame_count": len(raw_ids), "raw_radar_interval_ms": percentiles(np.diff(raw_times) * 1000),
            "reused_radar_frames": int((reuse_counts > 1).sum()),
            "maximum_camera_frames_per_radar_frame": int(reuse_counts.max()),
            "signed_recorded_timestamp_gap_ms": percentiles(native_gap_ms),
            "absolute_recorded_timestamp_gap_ms": percentiles(abs(native_gap_ms)),
            "aligned_abs_time_distinct_values": int(len(np.unique(aligned[:, 7]))),
            "annotation_adjacent_max_joint_delta_native_units": percentiles(jumps),
            "adjacent_deltas_over_1_native_unit": int((jumps > 1).sum()),
            "annotation_nonfinite_frames": int((~finite).sum()),
            "typical_limb_bone_lengths_native_units": typical_bones.tolist(),
            "typical_pose_scale_consistent_with_metres": metre_scale,
            "implausible_valid_label_frames": int(implausible.sum()),
            "implausible_frame_ids": np.flatnonzero(implausible).tolist(),
            "gt_coordinate_absolute_p95": float(np.percentile(abs(labels["refined_gt_kps"]), 95)),
            "stored_rgb_prediction_absolute_p95": float(np.percentile(abs(labels["rgb_est_kps"]), 95)),
            "time_shift_label_sensitivity": sensitivity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.archive.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    expected = "4f603af77cd16c5da0eba30fbc33bfa19101d7fea7ba3bbba5ed6789a77f83c0"
    if digest.hexdigest() != expected:
        raise ValueError("Archive differs from the audited Dryad version")
    report = {"status": "running", "archive_sha256": digest.hexdigest(), "subjects": [],
              "scope": "Published timestamps, point-cloud mapping and label sensitivity; not physical synchronization truth or model accuracy",
              "raw_rgb_decoded": False,
              "label_unit_interpretation": "Numerical limb lengths checked for metre scale; not inferred from outlier extrema or calibration translation units",
              "descriptive_label_screen": "flag any limb over 1 native unit or root-relative radius over 3; only interpreted physically for metre-scale subjects; raw results preserved",
              "action_interval_convention": "half-open for sensitivity analysis; exact boundaries are excluded by same-action pairing"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        for subject in range(1, 21):
            row = audit_subject(z, subject)
            report["subjects"].append(row)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"subject": subject, "frames": row["camera_timestamp_frames"],
                              "gap_p50_ms": row["absolute_recorded_timestamp_gap_ms"]["p50"],
                              "feature_mapping_matches": row["nonempty_features_match_aligned_camera_ids"]}), flush=True)
    report["status"] = "complete"
    report["total_camera_timestamp_frames"] = sum(r["camera_timestamp_frames"] for r in report["subjects"])
    report["total_radar_feature_frames"] = sum(r["radar_feature_frames"] for r in report["subjects"])
    report["all_feature_mapping_matches"] = all(r["nonempty_features_match_aligned_camera_ids"] for r in report["subjects"])
    report["all_typical_pose_scales_consistent_with_metres"] = all(r["typical_pose_scale_consistent_with_metres"] for r in report["subjects"])
    report["total_implausible_valid_label_frames"] = sum(r["implausible_valid_label_frames"] for r in report["subjects"])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
