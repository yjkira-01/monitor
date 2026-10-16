"""Fixed-budget, three-seed point-fusion checks on the existing development split.

Keep the original single-seed experiment intact. Model initialization seeds vary;
the seed-7 subjects, sampled frames and 80-epoch budget stay fixed. The matched
RGB model and shuffled-pair training control test capacity and radar dependence.
"""

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit_mri import interval_bounds, load_labels
from pose_fusion import nearest_timestamp_pairs
from train_mri_point_baselines import ROOT, PoseRegressor, load_development, point_normalization

OFFSETS_MS = (-500, -200, 0, 200, 500)
MODELS = ("rgb", "rgb_matched", "radar", "fusion", "fusion_shuffled")


class MatchedRGBRegressor(nn.Module):
    """64,823 parameters versus the ordinary fusion model's 64,947."""

    def __init__(self):
        super().__init__()
        self.rgb = nn.Sequential(nn.Linear(51, 167), nn.ReLU(), nn.Linear(167, 167), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(167, 128), nn.ReLU(), nn.Linear(128, 51))
        nn.init.normal_(self.head[-1].weight, std=.001)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, rgb, points, mask):
        return self.head(self.rgb(rgb))


def lag_candidates(points, timestamps_ms, availability, video_labels, frame_ids):
    """Shift published camera-frame pairing using acquisition time, never GT pose.

    Availability is inclusive. Reject crossings of action intervals, record edges,
    empty radar and nearest matches farther than 75 ms. Keep signed gains later.
    """
    timestamps_ms, frame_ids = np.asarray(timestamps_ms), np.asarray(frame_ids)
    start, end = interval_bounds(availability, len(timestamps_ms))
    if points.shape != (end - start + 1, 14, 14, 5):
        raise ValueError("Radar availability does not match feature-map length")
    if frame_ids.ndim != 1 or not len(frame_ids) or (frame_ids < start).any() or (frame_ids > end).any():
        raise ValueError("Sample frame outside radar availability")
    seconds = (timestamps_ms - timestamps_ms[0]).astype(np.float64) / 1000
    # Interval IDs also reject a boundary between two occurrences of an action.
    intervals = np.full(len(timestamps_ms), -1, dtype=np.int32)
    for interval_id, (a, b) in enumerate(video_labels.values()):
        intervals[a:b] = interval_id
    observed = np.any(points != 0, axis=(1, 2, 3))
    candidates, common = {}, np.ones(len(frame_ids), dtype=bool)
    for offset in OFFSETS_MS:
        match, gap, valid = nearest_timestamp_pairs(seconds, seconds[frame_ids] + offset / 1000, .075)
        safe = np.maximum(match, 0)
        positions = np.clip(safe - start, 0, end - start)
        valid &= ((safe >= start) & (safe <= end) & observed[positions] &
                  (intervals[frame_ids] >= 0) & (intervals[safe] == intervals[frame_ids]))
        selected = points[positions].reshape(-1, 196, 5).copy()
        candidates[offset] = {"points": selected, "mask": np.any(selected != 0, axis=-1),
                              "valid": valid, "gap_ms": gap * 1000, "matched_frame_ids": safe}
        common &= valid
    return candidates, common


def error_metrics(prediction, target, subject_ids):
    if prediction.shape != target.shape or target.shape[-1] != 51 or len(target) != len(subject_ids):
        raise ValueError("Prediction/target/subject shape mismatch")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all() or not len(target):
        raise ValueError("Invalid evaluation values")
    errors = np.linalg.norm((prediction - target).reshape(-1, 17, 3), axis=-1)
    return {"MPJPE_mm": float(errors.mean()),
            "frame_error_p95_mm": float(np.percentile(errors.mean(-1), 95)),
            "per_joint_mean_mm": errors.mean(0).tolist(),
            "per_validation_subject_MPJPE_mm": {
                str(s): float(errors[subject_ids == s].mean()) for s in np.unique(subject_ids)}}


def model_for(name):
    if name == "rgb_matched":
        return MatchedRGBRegressor()
    return PoseRegressor("fusion" if name == "fusion_shuffled" else name)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def aggregate_reports(runs, subjects):
    result = {"models": {}, "paired_comparisons": {}, "injected_lag": {}}
    for name in MODELS:
        result["models"][name] = {}
        for condition in ("clean", "dark"):
            values = [r["models"][name]["metrics"][condition]["MPJPE_mm"] for r in runs]
            result["models"][name][condition] = {
                "mean_MPJPE_mm": float(np.mean(values)), "seed_sd_mm": float(np.std(values, ddof=1)),
                "per_seed_MPJPE_mm": values,
                "seed_mean_per_subject_MPJPE_mm": {
                    str(s): float(np.mean([r["models"][name]["metrics"][condition]["per_validation_subject_MPJPE_mm"][str(s)] for r in runs]))
                    for s in subjects}}
    for comparator in ("rgb", "rgb_matched", "fusion_shuffled"):
        result["paired_comparisons"][comparator + "_minus_fusion"] = {}
        for condition in ("clean", "dark"):
            paired = np.array([
                result["models"][comparator][condition]["seed_mean_per_subject_MPJPE_mm"][str(s)] -
                result["models"]["fusion"][condition]["seed_mean_per_subject_MPJPE_mm"][str(s)] for s in subjects])
            # Resample subjects after averaging seeds; adjacent frames are not units.
            rng = np.random.default_rng(20261013)
            means = paired[rng.integers(0, len(paired), size=(10000, len(paired)))].mean(1)
            result["paired_comparisons"][comparator + "_minus_fusion"][condition] = {
                "mean_gain_mm": float(paired.mean()), "per_subject_gain_mm": dict(zip(map(str, subjects), paired.tolist())),
                "exploratory_subject_bootstrap_95pct_mm": np.percentile(means, [2.5, 97.5]).tolist(),
                "independent_subject_count": len(subjects),
                "interpretation": "Exploratory: four development subjects and one split, not confirmatory inference"}
    for condition in ("clean", "dark"):
        result["injected_lag"][condition] = []
        for offset in OFFSETS_MS:
            values = [r["models"]["fusion"]["injected_lag"][condition][str(offset)] for r in runs]
            result["injected_lag"][condition].append({
                "offset_ms": offset,
                "mean_uncorrected_MPJPE_mm": float(np.mean([v["MPJPE_mm"] for v in values])),
                "mean_restored_MPJPE_mm": float(np.mean([v["restored_MPJPE_mm"] for v in values])),
                "mean_restoration_gain_mm": float(np.mean([v["restoration_gain_mm"] for v in values])),
                "per_seed_restoration_gain_mm": [v["restoration_gain_mm"] for v in values]})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/research_execution_20261012/rgb_features")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research_validation_20261013")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    args = parser.parse_args()
    if not 1 <= args.epochs <= 200 or len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Require fixed 1..200 epochs and at least two distinct initialization seeds")
    if args.output.exists():
        raise ValueError("Refusing to overwrite an experiment")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    split, rows = load_development(args.cache)
    train, validation = split["canonical_ids_train"], split["canonical_ids_validation"]
    with zipfile.ZipFile(ROOT / "dataset_release.zip") as archive:
        for subject in validation:
            labels = load_labels(archive.read(f"dataset_release/aligned_data/pose_labels/subject{subject}_all_labels.cpl"))
            timestamps = np.load(io.BytesIO(archive.read(f"dataset_release/raw_data/unixtime/subject{subject}_unix.npy")), allow_pickle=False)
            points = np.load(io.BytesIO(archive.read(f"dataset_release/features/radar/subject{subject}_featuremap.npy")), allow_pickle=False)
            rows[subject]["lag"], rows[subject]["common"] = lag_candidates(
                points, timestamps, labels["radar_avail_frames"], labels["video_label"], rows[subject]["frame_ids"])
            np.testing.assert_array_equal(rows[subject]["points"], rows[subject]["lag"][0]["points"])

    def stack(ids, key):
        return np.concatenate([rows[s][key] for s in ids])

    point_mean, point_std = point_normalization(stack(train, "points"), stack(train, "mask"))
    rgb_mean, rgb_std = stack(train, "clean").mean(0), np.maximum(stack(train, "clean").std(0), 1e-8)
    target_mean = stack(train, "target").mean(0)
    np.savez_compressed(args.output / "normalization.npz", point_mean=point_mean, point_std=point_std,
                        rgb_mean=rgb_mean, rgb_std=rgb_std, target_mean_mm=target_mean)

    def tensors(rgb, points, mask):
        normalized = np.zeros_like(points, dtype=np.float32)
        normalized[mask] = (points[mask] - point_mean) / point_std
        return (torch.tensor((rgb - rgb_mean) / rgb_std, dtype=torch.float32),
                torch.tensor(normalized), torch.tensor(mask, dtype=torch.bool))

    training = tensors(stack(train, "clean"), stack(train, "points"), stack(train, "mask"))
    train_target = torch.tensor((stack(train, "target") - target_mean) / 1000, dtype=torch.float32)
    validation_target = stack(validation, "target")
    subject_ids = np.concatenate([np.full(len(rows[s]["target"]), s) for s in validation])
    all_frame_ids = stack(validation, "frame_ids")
    common = stack(validation, "common")
    if not common.any() or any(not rows[s]["common"].any() for s in validation):
        raise ValueError("Need common lag frames in each validation subject")
    values = {condition: tensors(stack(validation, condition), stack(validation, "points"), stack(validation, "mask"))
              for condition in ("clean", "dark")}
    lag_values = {}
    for condition in values:
        lag_values[condition] = {}
        for offset in OFFSETS_MS:
            points = np.concatenate([rows[s]["lag"][offset]["points"] for s in validation])[common]
            mask = np.concatenate([rows[s]["lag"][offset]["mask"] for s in validation])[common]
            lag_values[condition][offset] = tensors(stack(validation, condition)[common], points, mask)

    parameters = {name: sum(p.numel() for p in model_for(name).parameters()) for name in MODELS}
    report = {"status": "running", "date": "2026-10-13", "scope": "Development controls; no learned alignment/variance method or final test",
              "data_split_seed": split["seed"], "initialization_seeds": args.seeds,
              "train_subjects": train, "validation_subjects": validation, "test_subjects_used": [],
              "train_frames": len(train_target), "validation_frames": len(validation_target),
              "training": {"epochs": args.epochs, "selection": "last epoch; no validation checkpoint selection",
                           "optimizer": "AdamW", "lr": .001, "weight_decay": .0001, "batch_size": 64,
                           "normalization": "clean training inputs only", "augmentation": "none",
                           "batch_order": "separate seeded generator; same per seed across models"},
              "parameters": parameters, "capacity_match_relative_difference": abs(parameters["rgb_matched"] / parameters["fusion"] - 1),
              "shuffled_control": "Same fusion architecture; radar permuted across training rows each epoch, and across validation rows once per seed; never across train/validation",
              "dark_condition": "Pixels multiplied by 0.18 before frozen RTMO; not a complete realistic low-light corruption",
              "target": "Published hip-centred 17-joint pose, native metres converted to mm; previous fixed development structural screen",
              "lag_protocol": {"offsets_ms": OFFSETS_MS, "max_pair_gap_ms": 75, "common_frames": int(common.sum()),
                               "per_subject_common_frames": {str(s): int(rows[s]["common"].sum()) for s in validation},
                               "pairing": "published camera-frame correspondence, with injected acquisition-time shifts within the same action interval",
                               "restoration": "restore the known injected pairing, without GT-based candidate choice; negative gains retained",
                               "quantization_gap_median_ms": {str(k): float(np.median(np.concatenate([rows[s]["lag"][k]["gap_ms"] for s in validation])[common])) for k in OFFSETS_MS}},
              "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": "CPU", "threads": 4},
              "source_sha256": {p.name: sha256(p) for p in (Path(__file__), ROOT / "research/train_mri_point_baselines.py", ROOT / "research/pose_fusion_protocol.json")},
              "cache_sha256": {f"subject{s}.npz": sha256(args.cache / f"subject{s}.npz") for s in train + validation},
              "runs": []}
    # Save the resolved protocol before any training or metric-based decisions.
    write_json(args.output / "run_summary.json", report)

    def predict(model, inputs):
        model.eval()
        with torch.no_grad():
            return model(*inputs).numpy() * 1000 + target_mean

    for seed in args.seeds:
        seed_dir = args.output / f"seed{seed}"
        seed_dir.mkdir()
        validation_order = np.random.default_rng(seed + 100000).permutation(len(validation_target))
        run = {"seed": seed, "status": "running", "models": {}}
        for name in MODELS:
            start = time.perf_counter()
            torch.manual_seed(seed)
            model = model_for(name)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            order_generator = torch.Generator().manual_seed(seed)
            radar_rng = np.random.default_rng(seed + 200000)
            history = []
            for epoch in range(1, args.epochs + 1):
                model.train()
                radar_order = torch.tensor(radar_rng.permutation(len(train_target))) if name == "fusion_shuffled" else None
                losses = []
                for indices in torch.randperm(len(train_target), generator=order_generator).split(64):
                    radar_indices = indices if radar_order is None else radar_order[indices]
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(training[0][indices], training[1][radar_indices], training[2][radar_indices])
                    loss = (prediction - train_target[indices]).square().mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    losses.append(loss.item())
                if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
                    history.append({"epoch": epoch, "train_MSE_m2": float(np.mean(losses))})
            predictions = {"target_mm": validation_target, "subject_ids": subject_ids, "frame_ids": all_frame_ids,
                           "lag_common": common, "radar_permutation": validation_order}
            metrics, interventions, lag = {}, {}, {}
            for condition, original in values.items():
                paired = original if name != "fusion_shuffled" else (original[0], original[1][validation_order], original[2][validation_order])
                prediction = predict(model, paired)
                predictions[condition + "_prediction_mm"] = prediction
                metrics[condition] = error_metrics(prediction, validation_target, subject_ids)
                if name == "fusion":
                    interventions[condition] = {}
                    for label, pts, mask in (("radar_permuted", original[1][validation_order], original[2][validation_order]),
                                             ("radar_zeroed", torch.zeros_like(original[1]), torch.zeros_like(original[2]))):
                        intervention = predict(model, (original[0], pts, mask))
                        predictions[condition + "_" + label + "_prediction_mm"] = intervention
                        interventions[condition][label] = error_metrics(intervention, validation_target, subject_ids)
                    lag[condition] = {}
                    restored = predict(model, lag_values[condition][0])
                    restored_metric = error_metrics(restored, validation_target[common], subject_ids[common])
                    for offset in OFFSETS_MS:
                        shifted = predict(model, lag_values[condition][offset])
                        predictions[f"{condition}_lag_{offset}_prediction_mm"] = shifted
                        metric = error_metrics(shifted, validation_target[common], subject_ids[common])
                        metric["restored_MPJPE_mm"] = restored_metric["MPJPE_mm"]
                        metric["restoration_gain_mm"] = metric["MPJPE_mm"] - restored_metric["MPJPE_mm"]
                        lag[condition][str(offset)] = metric
            np.savez_compressed(seed_dir / f"{name}_weights.npz", **{key: value.detach().numpy() for key, value in model.state_dict().items()})
            np.savez_compressed(seed_dir / f"{name}_predictions.npz", **predictions)
            run["models"][name] = {"parameters": parameters[name], "seconds": time.perf_counter() - start,
                                   "history": history, "metrics": metrics,
                                   "interventions": interventions, "injected_lag": lag}
            write_json(seed_dir / "training.json", run)
            print(json.dumps({"seed": seed, "model": name, "clean_MPJPE_mm": metrics["clean"]["MPJPE_mm"],
                              "dark_MPJPE_mm": metrics["dark"]["MPJPE_mm"], "seconds": run["models"][name]["seconds"]}), flush=True)
        run["status"] = "complete"
        write_json(seed_dir / "training.json", run)
        report["runs"].append(run)
        write_json(args.output / "run_summary.json", report)
    report["aggregate"] = aggregate_reports(report["runs"], validation)
    report["status"] = "complete"
    write_json(args.output / "run_summary.json", report)


if __name__ == "__main__":
    main()
