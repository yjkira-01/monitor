"""Matched development ridge probes using newly extracted RGB, never GT crops."""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from audit_mri import interval_bounds, load_labels
from cache_mri_rgb import MODEL_SHA, VIDEO_SHA, file_checks, pose_features
from pose_fusion import nearest_timestamp_pairs
from run_mri_baselines import fit_ridge, radar_statistics, subject_samples

ROOT = Path(__file__).resolve().parents[1]


def shifted_radar(archive, labels, subject, frame_ids):
    timestamps = np.load(io.BytesIO(archive.read(f"dataset_release/raw_data/unixtime/subject{subject}_unix.npy")), allow_pickle=False)
    seconds = (timestamps - timestamps[0]).astype(np.float64) / 1000
    points = np.load(io.BytesIO(archive.read(f"dataset_release/features/radar/subject{subject}_featuremap.npy")), allow_pickle=False)
    start, end = interval_bounds(labels["radar_avail_frames"], len(timestamps))
    statistics, observed = radar_statistics(points.reshape(-1, 196, 5))
    actions = np.full(len(timestamps), "", dtype="U32")
    for action, (a, b) in labels["video_label"].items():
        actions[a:b] = action
    candidates, common = {}, np.ones(len(frame_ids), dtype=bool)
    for offset_ms in [-500, -200, 0, 200, 500]:
        match, gap, valid = nearest_timestamp_pairs(seconds, seconds[frame_ids] + offset_ms / 1000, .075)
        safe = np.maximum(match, 0)
        positions = np.clip(safe - start, 0, end - start)
        valid &= ((safe >= start) & (safe <= end) & observed[positions] &
                  (actions[frame_ids] != "") & (actions[safe] == actions[frame_ids]))
        candidates[offset_ms] = {"statistics": statistics[positions], "valid": valid, "quantization_gap_ms": gap * 1000}
        common &= valid
    return candidates, common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/research_execution_20261012/rgb_features")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research_execution_20261012/rgb_development_baselines.json")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "research/pose_fusion_protocol.json").read_text(encoding="utf-8"))
    split = protocol["splits"][0]
    train, validation, test = (split["canonical_ids_" + key] for key in ("train", "validation", "test"))
    if set(train) & set(validation) or (set(train) | set(validation)) & set(test):
        raise ValueError("Subject leakage")
    audit = json.loads((args.cache / "rgb_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "complete" or audit["video_archive_sha256"] != VIDEO_SHA or set(s["subject"] for s in audit["subjects"]) != set(train + validation):
        raise ValueError("Complete matched development RGB audit required")
    data_audit = json.loads((ROOT / "artifacts/research_execution_20261012/mri_data_audit.json").read_text(encoding="utf-8"))
    if file_checks(ROOT / "dataset_release.zip")[0] != data_audit["archive_sha256"] or data_audit["status"] != "complete":
        raise ValueError("Development data changed")
    rows = {}
    with zipfile.ZipFile(ROOT / "dataset_release.zip") as archive:
        for subject in train + validation:
            samples = subject_samples(archive, subject, split["seed"])
            with np.load(args.cache / f"subject{subject}.npz", allow_pickle=False) as cache:
                if str(cache["model_sha256"]) != MODEL_SHA or str(cache["video_archive_sha256"]) != VIDEO_SHA:
                    raise ValueError("Feature cache provenance mismatch")
                frame_ids = cache["frame_ids"]
                selected = np.searchsorted(samples["frame_ids"], frame_ids)
                if (selected >= len(samples["frame_ids"])).any() or not np.array_equal(samples["frame_ids"][selected], frame_ids):
                    raise ValueError("Cached frames differ from the fixed development sample")
                rows[subject] = {"clean": cache["clean_features"].copy(), "dark": cache["dark_features"].copy(),
                                 "radar": samples["radar"][selected], "target": samples["target_mm"][selected]}
            labels = load_labels(archive.read(f"dataset_release/aligned_data/pose_labels/subject{subject}_all_labels.cpl"))
            points = labels["pose_2d_r"][frame_ids].transpose(0, 2, 1)
            points = np.concatenate([points, np.ones((*points.shape[:2], 1))], axis=-1)
            rows[subject]["annotation"] = pose_features(points, np.diag(labels["camera_matrix"]["cam1_intrinsic"])[:2])[0]
            if subject in validation:
                rows[subject]["shifted"], rows[subject]["lag_common"] = shifted_radar(archive, labels, subject, frame_ids)
    train_y = np.concatenate([rows[s]["target"] for s in train])
    validation_y = np.concatenate([rows[s]["target"] for s in validation])
    report = {"status": "complete", "scope": "Small development probes; no proposed temporal alignment or final test evaluation",
              "train_subjects": train, "validation_subjects": validation, "test_subjects_used": [],
              "train_frames": len(train_y), "validation_frames": len(validation_y), "seed": split["seed"],
              "model": "frame-wise ridge alpha=1; normalization fit only on clean training inputs",
              "video_input": "Frozen full-frame RTMO-m COCO17, hip-relative focal-normalized XY plus confidences; no GT crops or pose input",
              "radar_input": "21 point-cloud statistics; shared frames, identical in both illumination conditions",
              "target": "Published COCO17 hip-centred metre-scale labels converted to millimetres",
              "missingness": "Keep every sampled frame, zero unavailable image coordinates/confidences; no filtering by prediction",
              "label_screen": "Earlier fixed development limb<=1m/radius<=3m screen; not final evaluation filtering",
              "dark_condition": "Synthetic multiplication by 0.18 of decoded pixels before RTMO; train clean, evaluate dark without retraining",
              "metrics": {},
              "injected_lag_probe": {"scope": "Shift published radar feature correspondence, not measured physical synchronization; frozen clean-trained concatenation head",
                                     "offsets_ms": [-500, -200, 0, 200, 500], "max_pair_gap_ms": 75,
                                     "oracle": "Restore original known injected pairing; no GT pose used to pick candidates",
                                     "selection": "Same action, nonempty radar, common valid frame subset across all offsets and both illumination conditions",
                                     "common_validation_frames": int(sum(rows[s]["lag_common"].sum() for s in validation)), "conditions": {}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, columns in (("published_annotation_control", ["annotation"]), ("rgb", ["clean"]),
                          ("radar", ["radar"]), ("concatenation", ["clean", "radar"])):
        def features(ids, condition):
            return np.concatenate([np.concatenate([rows[s][condition if key == "clean" else key] for key in columns], axis=-1) for s in ids])
        training_x = features(train, "clean")
        report["metrics"][name] = {}
        for condition in (("clean",) if name == "published_annotation_control" else ("clean", "dark")):
            prediction, model = fit_ridge(training_x, train_y, features(validation, condition))
            errors = np.linalg.norm((prediction - validation_y).reshape(-1, 17, 3), axis=-1)
            report["metrics"][name][condition] = {"MPJPE_mm": float(errors.mean()),
                                                  "frame_error_p95_mm": float(np.percentile(errors.mean(-1), 95)),
                                                  "per_joint_mean_mm": errors.mean(0).tolist()}
            offset = 0
            per_subject = {}
            for subject in validation:
                length = len(rows[subject]["target"])
                per_subject[str(subject)] = float(errors[offset:offset + length].mean())
                offset += length
            report["metrics"][name][condition]["per_validation_subject_MPJPE_mm"] = per_subject
            if condition == "clean":
                np.savez_compressed(args.output.with_name(f"rgb_probe_{name}_ridge.npz"), **model)
            print(name, condition, report["metrics"][name][condition]["MPJPE_mm"], flush=True)
        if name == "concatenation":
            target = np.concatenate([rows[s]["target"][rows[s]["lag_common"]] for s in validation])
            if not len(target):
                raise ValueError("No common validation frames for lag probe")
            for condition in ("clean", "dark"):
                results = []
                for offset_ms in [-500, -200, 0, 200, 500]:
                    x = np.concatenate([np.concatenate([rows[s][condition][rows[s]["lag_common"]],
                                                       rows[s]["shifted"][offset_ms]["statistics"][rows[s]["lag_common"]]], axis=-1) for s in validation])
                    prediction = np.column_stack([(x - model["normalization_mean"]) / model["normalization_std"], np.ones(len(x))]) @ model["weights"]
                    error = np.linalg.norm((prediction - target).reshape(-1, 17, 3), axis=-1)
                    gaps = np.concatenate([rows[s]["shifted"][offset_ms]["quantization_gap_ms"][rows[s]["lag_common"]] for s in validation])
                    results.append({"offset_ms": offset_ms, "uncorrected_MPJPE_mm": float(error.mean()),
                                    "nearest_pair_gap_median_ms": float(np.median(gaps))})
                oracle = results[2]["uncorrected_MPJPE_mm"]
                for result in results:
                    result["oracle_restored_MPJPE_mm"] = oracle
                    result["oracle_gain_mm"] = result["uncorrected_MPJPE_mm"] - oracle
                report["injected_lag_probe"]["conditions"][condition] = results
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
