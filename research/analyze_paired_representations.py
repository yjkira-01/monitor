"""Compare completed hip/centroid runs using subject-aware input subgroups.

Only saved arrays are read: no models are instantiated and no GPU work runs.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HIP = ROOT / "artifacts/fusion_method_research_20261013_expanded_hip"
DEFAULT_CENTROID = ROOT / "artifacts/fusion_method_research_20261013_expanded_centroid"
DEFAULT_CACHE = ROOT / "artifacts/fusion_rgb_expanded_20261013"


def compare_contracts(hip, centroid):
    issues = []
    fields = ("data_split_seed", "train_subjects", "validation_subjects", "initialization_seeds",
              "train_frames", "validation_frames", "methods", "parameters", "training",
              "evaluation_conditions", "score_conditions", "cache_sha256", "input_audit_sha256",
              "environment", "source_sha256", "model_factory")
    for field in fields:
        if hip.get(field) != centroid.get(field):
            issues.append("Different run contract: " + field)
    for name, report, expected_representation in (("hip", hip, "hip"), ("centroid", centroid, "visible_centroid")):
        if report.get("representation") != expected_representation:
            issues.append(name + ": unexpected representation")
        if report.get("final_test_subjects_used"):
            issues.append(name + ": final-test data were used")
        if report.get("data_split_seed") != 7 or report.get("initialization_seeds") != [7, 17, 27]:
            issues.append(name + ": expected split7 and initialization seeds7/17/27")
        if report.get("train_frames") != 1152 or report.get("validation_frames") != 384:
            issues.append(name + ": expected 1152/384 actual-frame budgets")
        training = report.get("training", {})
        if training.get("epochs") != 80 or training.get("batch_size") != 64:
            issues.append(name + ": expected fixed80 epochs/batch64")
        environment = report.get("environment", {})
        if (environment.get("device") != "cuda" or environment.get("deterministic") is not True or
                environment.get("mixed_precision") is not False or environment.get("FP32_matmul") != "IEEE, TF32 disabled"):
            issues.append(name + ": expected deterministic CUDA without mixed precision")
        runs = report.get("runs", [])
        if len(runs) != 3 or {r.get("seed") for r in runs} != {7, 17, 27} or any(r.get("status") != "complete" for r in runs):
            issues.append(name + ": three complete initialization runs required")
    return issues


def exact_npz_comparison(left, right):
    differences = []
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            return ["Different array keys"]
        for key in a.files:
            x, y = a[key], b[key]
            if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y):
                differences.append(key)
    return differences


def simple_metrics(errors, visible):
    if errors.shape != visible.shape or errors.ndim != 2 or errors.shape[1] != 17:
        raise ValueError("Expected aligned frame/joint errors and visibility")
    observed = int(visible.sum())
    invisible = int((~visible).sum())
    return {"frames": len(errors), "visible_joints": observed, "invisible_joints": invisible,
            "MPJPE_mm": float(errors.mean()) if errors.size else None,
            "visible_joint_error_mm": float(errors[visible].mean()) if observed else None,
            "invisible_joint_error_mm": float(errors[~visible].mean()) if invisible else None}


def group_metrics(errors, frame_mask, visible, subjects, subject_ids):
    result = simple_metrics(errors[frame_mask], visible[frame_mask])
    result["per_subject"] = {str(subject): simple_metrics(errors[frame_mask & (subject_ids == subject)],
                                                          visible[frame_mask & (subject_ids == subject)])
                             for subject in subjects}
    return result


def mean_sd(values):
    values = [v for v in values if v is not None]
    return None if not values else {"mean": float(np.mean(values)),
                                   "initialization_seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.}


def load_validation_masks(cache, subjects):
    frame_ids, subject_ids, poses = [], [], []
    for subject in subjects:
        with np.load(cache / f"subject{subject}.npz", allow_pickle=False) as saved:
            ids = saved["frame_ids"].copy()
            frame_ids.append(ids)
            subject_ids.append(np.full(len(ids), subject))
            poses.append(saved["occlusion_poses"].copy())
    visible = np.concatenate(poses)[..., 2] >= .3
    hip_valid = visible[:, 11] & visible[:, 12]
    raw_any = visible.any(1)
    masks = {"all": np.ones(len(visible), dtype=bool), "hip_valid": hip_valid,
             "partial": ~hip_valid & raw_any, "all_raw_missing": ~raw_any}
    return np.concatenate(frame_ids), np.concatenate(subject_ids), visible, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hip", type=Path, default=DEFAULT_HIP)
    parser.add_argument("--centroid", type=Path, default=DEFAULT_CENTROID)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/fusion_representation_audit_20261013/paired_results.json")
    args = parser.parse_args()
    inputs = {"hip": args.hip, "centroid": args.centroid}
    reports = {name: json.loads((path / "run_summary.json").read_text(encoding="utf-8")) for name, path in inputs.items()}
    if any(report.get("status") != "complete" for report in reports.values()):
        raise RuntimeError("Both formal directories must be complete before this analysis runs")
    result = {"status": "checking", "hip_directory": str(args.hip), "centroid_directory": str(args.centroid),
              "checked_unix": time.time(), "comparability_issues": compare_contracts(reports["hip"], reports["centroid"]),
              "cache_integrity_recheck": False, "radar_single_exact_checks": {}, "fallback_exact_checks": [],
              "joint_availability_definition": "Visible/invisible in this report means raw predicted RTMO confidence >=0.3 / <0.3, not independently verified physical visibility or occlusion.",
              "interpretation": ["Centroid changes anchors on all observed frames and changes video/candidate/utility loss support; gains are not solely local visual information.",
                  "All-raw-missing frames use the learned radar branch in both representations; changes there can reflect coupled training gradients.",
                  "Visible/invisible joint means weight joints within each subgroup; primary MPJPE retains every selected frame.",
                  "Per-subject paired gains are descriptive for four subjects. Frame/joint counts and initialization seeds are not independent-subject replicates."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if result["comparability_issues"]:
        result["status"] = "failed_comparability"
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise ValueError("Paired run contract failed: " + "; ".join(result["comparability_issues"]))
    prior = json.loads((args.cache / "cache_verification.json").read_text(encoding="utf-8"))
    prior_hashes = {str(row["subject"]): row["cache_sha256"] for row in prior["subjects"]}
    for name, report in reports.items():
        if report.get("cache_sha256") != prior_hashes:
            result["comparability_issues"].append(name + ": cache hashes differ from prior complete cache record")
        for source, checksum in report.get("source_sha256", {}).items():
            snapshot = inputs[name] / "source_snapshot" / source
            if not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != checksum:
                result["comparability_issues"].append(name + ": invalid source snapshot " + source)
    normalization = {}
    for name, path in inputs.items():
        with np.load(path / "normalization.npz", allow_pickle=False) as saved:
            normalization[name] = {key: saved[key].copy() for key in saved.files}
    for key in ("point_mean", "point_std", "target_mean_mm"):
        if not np.array_equal(normalization["hip"][key], normalization["centroid"][key]):
            result["comparability_issues"].append("Different normalization: " + key)
    for key in ("rgb_mean", "rgb_std"):
        if not np.array_equal(normalization["hip"][key][34:], normalization["centroid"][key][34:]):
            result["comparability_issues"].append("Different clean-training confidence normalization: " + key)
    result["xy_normalization_change"] = {key: float(np.max(np.abs(normalization["hip"][key][:34] - normalization["centroid"][key][:34])))
                                         for key in ("rgb_mean", "rgb_std")}
    run_maps = {name: {run["seed"]: run for run in report["runs"]} for name, report in reports.items()}
    for seed in reports["hip"]["initialization_seeds"]:
        checks = {}
        for kind in ("weights", "predictions"):
            left, right = (path / f"seed{seed}/radar_single_{kind}.npz" for path in inputs.values())
            checks[kind + "_differing_arrays"] = exact_npz_comparison(left, right)
        checks["history_exact_equal"] = run_maps["hip"][seed]["models"]["radar_single"]["history"] == run_maps["centroid"][seed]["models"]["radar_single"]["history"]
        result["radar_single_exact_checks"][str(seed)] = checks
        if checks["weights_differing_arrays"] or checks["predictions_differing_arrays"] or not checks["history_exact_equal"]:
            result["comparability_issues"].append(f"seed{seed}: independent radar weights/predictions/history differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if result["comparability_issues"]:
        result["status"] = "failed_comparability"
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise ValueError("Paired run comparability failed: " + "; ".join(result["comparability_issues"]))
    subjects = reports["hip"]["validation_subjects"]
    ids, subject_ids, visible, masks = load_validation_masks(args.cache, subjects)
    if {key: int(mask.sum()) for key, mask in masks.items()} != {"all": 384, "hip_valid": 21, "partial": 74, "all_raw_missing": 289}:
        raise ValueError("Expected original384/21/74/289 input subgroups")
    result["subgroup_counts"] = {key: {"frames": int(mask.sum()), "visible_joints": int(visible[mask].sum()),
        "invisible_joints": int((~visible[mask]).sum()), "per_subject_frames": {
            str(subject): int((mask & (subject_ids == subject)).sum()) for subject in subjects}}
        for key, mask in masks.items()}
    result["per_seed_groups"] = {}
    for seed in reports["hip"]["initialization_seeds"]:
        result["per_seed_groups"][str(seed)] = {}
        for name, path in inputs.items():
            result["per_seed_groups"][str(seed)][name] = {}
            for method in reports[name]["methods"]:
                with np.load(path / f"seed{seed}/{method}_predictions.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["subject_ids"], subject_ids)
                    np.testing.assert_array_equal(saved["frame_ids"], ids)
                    target = saved["target_mm"].copy()
                    prediction = saved["occlusion_fused_prediction_mm"].copy()
                    with np.load(inputs["hip"] / f"seed{seed}/{method}_predictions.npz", allow_pickle=False) as common:
                        np.testing.assert_array_equal(target, common["target_mm"])
                    errors = np.linalg.norm((prediction - target).reshape(-1, 17, 3), axis=-1)
                    result["per_seed_groups"][str(seed)][name][method] = {group: group_metrics(errors, mask, visible, subjects, subject_ids)
                                                                        for group, mask in masks.items()}
                    if not method.startswith("rgb_") and method != "radar_single":
                        absent = (~(visible[:, 11] & visible[:, 12])) if name == "hip" else ~visible.any(1)
                        exact = np.array_equal(prediction[absent], saved["occlusion_radar_prediction_mm"][absent])
                        missing_exact = np.array_equal(saved["video_missing_fused_prediction_mm"], saved["video_missing_radar_prediction_mm"])
                        expected_valid = 21 if name == "hip" else 95
                        actual_valid = run_maps[name][seed]["models"][method]["metrics"]["occlusion"]["image_available_frames"]
                        if not exact or not missing_exact or actual_valid != expected_valid:
                            raise ValueError(f"Unexpected fallback/valid behavior: {name},seed{seed},{method}")
                        result["fallback_exact_checks"].append({"representation": name, "seed": seed, "method": method,
                            "occlusion_absent_frames": int(absent.sum()), "fused_equals_radar_exact": exact,
                            "video_missing_fused_equals_radar_exact": missing_exact, "actual_image_available_frames": actual_valid})
    result["aggregates"] = {}
    fields = ("MPJPE_mm", "visible_joint_error_mm", "invisible_joint_error_mm")
    for method in reports["hip"]["methods"]:
        method_result = {}
        for group in masks:
            group_result = {name: {field: mean_sd([result["per_seed_groups"][str(seed)][name][method][group][field]
                for seed in reports[name]["initialization_seeds"]]) for field in fields} for name in inputs}
            group_result["hip_minus_centroid_gain"] = {field: mean_sd([
                result["per_seed_groups"][str(seed)]["hip"][method][group][field] - result["per_seed_groups"][str(seed)]["centroid"][method][group][field]
                for seed in reports["hip"]["initialization_seeds"] if result["per_seed_groups"][str(seed)]["hip"][method][group][field] is not None]) for field in fields}
            group_result["per_subject_MPJPE_gain_mm"] = {str(subject): float(np.mean([
                result["per_seed_groups"][str(seed)]["hip"][method][group]["per_subject"][str(subject)]["MPJPE_mm"] -
                result["per_seed_groups"][str(seed)]["centroid"][method][group]["per_subject"][str(subject)]["MPJPE_mm"]
                for seed in reports["hip"]["initialization_seeds"]])) for subject in subjects}
            group_result["contribution_to_full_occlusion_gain_mm"] = float(masks[group].mean()) * group_result["hip_minus_centroid_gain"]["MPJPE_mm"]["mean"]
            method_result[group] = group_result
        total = sum(method_result[group]["contribution_to_full_occlusion_gain_mm"] for group in ("hip_valid", "partial", "all_raw_missing"))
        if not np.isclose(total, method_result["all"]["hip_minus_centroid_gain"]["MPJPE_mm"]["mean"], atol=1e-9, rtol=0):
            raise ValueError("Subgroup contributions do not reconstruct the full-frame result")
        result["aggregates"][method] = method_result
    result["status"] = "complete"
    result["test_subjects_used"] = []
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "independent_radar_exact": True,
        "occlusion_gains_hip_minus_centroid_mm": {method: {group: row[group]["hip_minus_centroid_gain"]["MPJPE_mm"]["mean"]
        for group in masks} for method, row in result["aggregates"].items()}}, indent=2))


if __name__ == "__main__":
    main()
