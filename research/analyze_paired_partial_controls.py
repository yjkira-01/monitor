"""Saved-array comparison of completed partial geometry/confidence controls.

No models, GPU training or original labels are read. The original hip reference
retains its own runner/source contract while sharing data and architecture.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from analyze_paired_representations import ROOT, DEFAULT_CACHE, DEFAULT_HIP, group_metrics, load_validation_masks, mean_sd

METHODS = ["rgb_matched", "concat", "joint_gate"]
SEEDS = [7, 17, 27]
REPRESENTATIONS = {"geometry": "hip_centroid_fallback", "confidence": "hip_confidence_only"}


def safe_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def equal_statistics(left, right):
    return set(left) == set(right) and all(
        left[key].shape == right[key].shape and left[key].dtype == right[key].dtype
        and np.array_equal(left[key], right[key]) for key in left)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_receipt(path):
    """Bind paired arrays to the independently replayed, unchanged experiment."""
    path = Path(path)
    receipt_path = path / "analysis/verification.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or receipt.get("run_summary_sha256") != file_sha256(path / "run_summary.json"):
        raise ValueError("Complete replay receipt with matching summary SHA required: " + str(path))
    if receipt.get("normalization_sha256") != file_sha256(path / "normalization.npz"):
        raise ValueError("Normalization changed after independent replay: " + str(path))
    rows = {(row["seed"], row["method"]): row for row in receipt["models"]}
    for seed in SEEDS:
        for method in METHODS:
            row = rows.get((seed, method))
            if row is None or not row.get("all_predictions_and_metrics_match") or not row.get("finite_state_and_strict_reload"):
                raise ValueError("Missing independent replay result: " + str((seed, method)))
            if row["prediction_sha256"] != file_sha256(path / f"seed{seed}/{method}_predictions.npz"):
                raise ValueError("Saved predictions changed after independent replay: " + str((seed, method)))
    return {"receipt_path": str(receipt_path), "receipt_SHA256": file_sha256(receipt_path),
            "run_summary_SHA256": receipt["run_summary_sha256"], "normalization_SHA256": receipt["normalization_sha256"],
            "prediction_SHA256": {f"seed{seed}/{method}": rows[(seed, method)]["prediction_sha256"] for seed in SEEDS for method in METHODS}}


def compare_partial_contracts(geometry, confidence, hip=None):
    issues = []
    common = ("data_split_seed", "train_subjects", "validation_subjects", "initialization_seeds", "train_frames",
              "validation_frames", "methods", "parameters", "training", "evaluation_conditions", "score_conditions",
              "cache_sha256", "input_audit_sha256", "environment", "source_sha256", "runner_file", "protocol_sha256",
              "rgb_normalization_key", "model_factory", "representation_control")
    for key in common:
        if geometry.get(key) != confidence.get(key):
            issues.append("Different partial run contract: " + key)
    for name, report in (("geometry", geometry), ("confidence", confidence)):
        if report.get("representation") != REPRESENTATIONS[name] or report.get("rgb_normalization_key") != "hip_clean":
            issues.append(name + ": wrong representation or normalization source")
        if report.get("methods") != METHODS or report.get("initialization_seeds") != SEEDS or report.get("data_split_seed") != 7:
            issues.append(name + ": expected frozen methods/seeds/split")
        if report.get("train_frames") != 1152 or report.get("validation_frames") != 384 or report.get("final_test_subjects_used"):
            issues.append(name + ": wrong development scope")
        training, environment = report.get("training", {}), report.get("environment", {})
        if training.get("epochs") != 80 or training.get("batch_size") != 64:
            issues.append(name + ": expected fixed80 epochs/batch64")
        if (environment.get("device") != "cuda" or environment.get("deterministic") is not True
                or environment.get("mixed_precision") is not False or environment.get("FP32_matmul") != "IEEE, TF32 disabled"):
            issues.append(name + ": expected deterministic IEEE FP32 CUDA")
        runs = report.get("runs", [])
        if len(runs) != 3 or {run.get("seed") for run in runs} != set(SEEDS) or any(run.get("status") != "complete" for run in runs):
            issues.append(name + ": expected three complete runs")
    if hip is not None:
        if hip.get("representation") != "hip" or hip.get("status") != "complete" or hip.get("final_test_subjects_used"):
            issues.append("Original hip reference is not a complete development hip stage")
        for key in ("data_split_seed", "train_subjects", "validation_subjects", "initialization_seeds", "train_frames",
                    "validation_frames", "evaluation_conditions", "score_conditions", "cache_sha256", "input_audit_sha256", "environment"):
            if geometry.get(key) != hip.get(key):
                issues.append("Different original hip reference contract: " + key)
        for key, value in geometry.get("training", {}).items():
            if key != "normalization" and hip.get("training", {}).get(key) != value:
                issues.append("Different original hip training field: " + key)
        for method in METHODS:
            if geometry.get("parameters", {}).get(method) != hip.get("parameters", {}).get(method):
                issues.append("Different original hip parameter count: " + method)
        if geometry.get("model_factory", "fusion_models.FusionModel") != hip.get("model_factory", "fusion_models.FusionModel"):
            issues.append("Different original hip model factory")
        for filename in ("fusion_models.py", "fusion_pose_representation.py"):
            if geometry.get("source_sha256", {}).get(filename) != hip.get("source_sha256", {}).get(filename):
                issues.append("Different original hip shared source: " + filename)
    return issues


def paired_group_aggregates(per_seed, left, right, masks, subjects):
    result = {}
    fields = ("MPJPE_mm", "visible_joint_error_mm", "invisible_joint_error_mm")
    for method in METHODS:
        result[method] = {}
        for group, mask in masks.items():
            a = [per_seed[str(seed)][left][method][group] for seed in SEEDS]
            b = [per_seed[str(seed)][right][method][group] for seed in SEEDS]
            row = {name: {field: mean_sd([record[field] for record in records]) for field in fields}
                   for name, records in ((left, a), (right, b))}
            row["left_minus_right_gain"] = {field: mean_sd([x[field] - y[field] for x, y in zip(a, b) if x[field] is not None])
                                             for field in fields}
            row["per_subject_MPJPE_gain_mm"] = {}
            for subject in subjects:
                values = [x["per_subject"][str(subject)]["MPJPE_mm"] - y["per_subject"][str(subject)]["MPJPE_mm"]
                          for x, y in zip(a, b) if x["per_subject"][str(subject)]["MPJPE_mm"] is not None
                          and y["per_subject"][str(subject)]["MPJPE_mm"] is not None]
                row["per_subject_MPJPE_gain_mm"][str(subject)] = float(np.mean(values)) if values else None
            row["contribution_to_full_occlusion_gain_mm"] = float(mask.mean()) * row["left_minus_right_gain"]["MPJPE_mm"]["mean"]
            result[method][group] = row
        total = sum(result[method][group]["contribution_to_full_occlusion_gain_mm"] for group in ("hip_valid", "partial", "all_raw_missing"))
        if not np.isclose(total, result[method]["all"]["left_minus_right_gain"]["MPJPE_mm"]["mean"], atol=1e-9, rtol=0):
            raise ValueError("Group contributions do not reconstruct the full-frame gain")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--hip-reference", type=Path, default=DEFAULT_HIP)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/fusion_representation_audit_20261013/partial_paired_results.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Historical paired result must not be overwritten; choose a new --output: " + str(args.output))
    paths = {"geometry": args.geometry, "confidence": args.confidence, "hip": args.hip_reference}
    reports = {name: json.loads((path / "run_summary.json").read_text(encoding="utf-8")) for name, path in paths.items()}
    if any(report.get("status") != "complete" for report in reports.values()):
        raise RuntimeError("All formal input directories must be complete before reading paired results")
    independent_receipts = {name: verified_receipt(path) for name, path in paths.items()}
    result = {"status": "checking", "checked_unix": time.time(), "directories": {k: str(v) for k, v in paths.items()},
              "source_SHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "shared_analysis_dependency_SHA256": file_sha256(Path(__file__).with_name("analyze_paired_representations.py")),
              "independent_replay_receipts": independent_receipts,
              "comparability_issues": compare_partial_contracts(reports["geometry"], reports["confidence"], reports["hip"]),
              "joint_availability_definition": "Raw predicted RTMO confidence>=0.3, not independently established physical visibility",
              "interpretation": ["Primary geometry gain is confidence_minus_geometry: positive means lower error with local XY geometry.",
                  "Both partial controls share raw confidence/availability/loss support and original hip-clean statistics. Raw confidence-only XY0 maps to fixed -hip_mean/std in observed normalized fields.",
                  "Hip-valid inputs are equal but separately trained outputs can differ due to partial-frame training gradients.",
                  "All-raw-missing fusion outputs equal each separately trained radar branch; changes there are not restored visual information.",
                  "Four validation subjects, one split. Seeds, frames and joints are not independent subject replicates. Original hip comparisons also change availability/loss support."],
              "final_test_subjects_used": []}
    prior = json.loads((args.cache / "cache_verification.json").read_text(encoding="utf-8"))
    prior_hashes = {str(row["subject"]): row["cache_sha256"] for row in prior["subjects"]}
    norms = {}
    for name, report in reports.items():
        if report.get("cache_sha256") != prior_hashes:
            result["comparability_issues"].append(name + ": cache SHA differs from prior complete verification")
        for source, checksum in report.get("source_sha256", {}).items():
            snapshot = paths[name] / "source_snapshot" / source
            if not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != checksum:
                result["comparability_issues"].append(name + ": invalid source snapshot " + source)
        if name != "hip":
            protocol = paths[name] / "source_snapshot/fusion_partial_protocol_20261013.md"
            if not protocol.exists() or hashlib.sha256(protocol.read_bytes()).hexdigest() != report.get("protocol_sha256"):
                result["comparability_issues"].append(name + ": invalid frozen protocol snapshot")
        norms[name] = safe_arrays(paths[name] / "normalization.npz")
    for name in ("geometry", "confidence"):
        if not equal_statistics(norms[name], norms["hip"]):
            result["comparability_issues"].append(name + ": full normalization differs from original hip-clean reference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if result["comparability_issues"]:
        result["status"] = "failed_comparability"
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise ValueError("Paired controls are not comparable: " + "; ".join(result["comparability_issues"]))

    subjects = reports["geometry"]["validation_subjects"]
    frame_ids, subject_ids, visible, masks = load_validation_masks(args.cache, subjects)
    if {key: int(mask.sum()) for key, mask in masks.items()} != {"all": 384, "hip_valid": 21, "partial": 74, "all_raw_missing": 289}:
        raise ValueError("Unexpected original-cache subgroups")
    result["subgroup_counts"] = {key: {"frames": int(mask.sum()), "visible_joints": int(visible[mask].sum()),
        "invisible_joints": int((~visible[mask]).sum()), "per_subject_frames": {str(s): int((mask & (subject_ids == s)).sum()) for s in subjects}}
        for key, mask in masks.items()}
    result["per_seed_groups"], result["fallback_exact_checks"] = {}, []
    common_target = None
    for seed in SEEDS:
        result["per_seed_groups"][str(seed)] = {}
        for name, path in paths.items():
            result["per_seed_groups"][str(seed)][name] = {}
            for method in METHODS:
                saved = safe_arrays(path / f"seed{seed}/{method}_predictions.npz")
                np.testing.assert_array_equal(saved["frame_ids"], frame_ids)
                np.testing.assert_array_equal(saved["subject_ids"], subject_ids)
                target = saved["target_mm"]
                if common_target is None:
                    common_target = target
                np.testing.assert_array_equal(target, common_target)
                error = np.linalg.norm((saved["occlusion_fused_prediction_mm"] - target).reshape(-1, 17, 3), axis=-1)
                if not np.isfinite(error).all():
                    raise ValueError("Nonfinite saved prediction errors")
                result["per_seed_groups"][str(seed)][name][method] = {group: group_metrics(error, mask, visible, subjects, subject_ids)
                    for group, mask in masks.items()}
                if method != "rgb_matched":
                    absent = ~visible.any(1) if name != "hip" else ~(visible[:, 11] & visible[:, 12])
                    run = next(row for row in reports[name]["runs"] if row["seed"] == seed)
                    actual_valid = run["models"][method]["metrics"]["occlusion"]["image_available_frames"]
                    if actual_valid != (21 if name == "hip" else 95):
                        raise ValueError("Unexpected reported image availability: " + name + "/" + str(seed) + "/" + method)
                    exact = np.array_equal(saved["occlusion_fused_prediction_mm"][absent], saved["occlusion_radar_prediction_mm"][absent])
                    missing = np.array_equal(saved["video_missing_fused_prediction_mm"], saved["video_missing_radar_prediction_mm"])
                    if not exact or not missing:
                        raise ValueError("Incorrect radar fallback: " + name + "/" + str(seed) + "/" + method)
                    result["fallback_exact_checks"].append({"control": name, "seed": seed, "method": method,
                        "occlusion_absent_frames": int(absent.sum()), "occlusion_exact": exact, "video_missing_exact": missing,
                        "reported_image_available_frames": actual_valid})
    result["aggregates"] = {"confidence_minus_geometry": paired_group_aggregates(result["per_seed_groups"], "confidence", "geometry", masks, subjects),
                            "hip_minus_geometry": paired_group_aggregates(result["per_seed_groups"], "hip", "geometry", masks, subjects),
                            "hip_minus_confidence": paired_group_aggregates(result["per_seed_groups"], "hip", "confidence", masks, subjects)}
    result["status"] = "complete"
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "fallback_checks": len(result["fallback_exact_checks"]),
                     "confidence_minus_geometry_occlusion_gain_mm": {m: r["all"]["left_minus_right_gain"]["MPJPE_mm"]["mean"]
                         for m, r in result["aggregates"]["confidence_minus_geometry"].items()}}, indent=2))


if __name__ == "__main__":
    main()
