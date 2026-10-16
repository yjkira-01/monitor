"""Supplementary prediction-hip-recentered metrics after a complete replay audit.

The original predictions, primary metrics and selection rule are never changed.
This operation uses only the prediction's hip joints; it is not Procrustes.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_fusion_trials import compare_tree, independent_metrics, safe_npz, sha256, write_json


def prediction_hip_recenter(prediction):
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape[1] != 51 or not np.isfinite(prediction).all():
        raise ValueError("Expected finite N by 51 predictions in millimetres")
    pose = prediction.reshape(-1, 17, 3)
    hip = (pose[:, 11] + pose[:, 12]) / 2.
    return (pose - hip[:, None]).reshape(-1, 51), hip


def signed_metric_difference(recentered, primary):
    if isinstance(primary, dict):
        if not isinstance(recentered, dict) or set(recentered) != set(primary):
            raise ValueError("Metric keys disagree")
        return {key: signed_metric_difference(recentered[key], primary[key]) for key in primary}
    value = np.asarray(recentered) - np.asarray(primary)
    return value.tolist() if value.ndim else float(value)


def supplementary_metrics(prediction, target, subjects):
    target = np.asarray(target, dtype=np.float64)
    primary = independent_metrics(prediction, target, subjects)
    target_pose = target.reshape(-1, 17, 3)
    if np.max(np.abs((target_pose[:, 11] + target_pose[:, 12]) / 2.)) > 1e-8:
        raise ValueError("Supplement expects original already hip-centred target")
    recentered, hip = prediction_hip_recenter(prediction)
    metrics = independent_metrics(recentered, target, subjects)
    hip_distance = np.linalg.norm(hip, axis=-1)
    return {"primary_metrics": primary, "prediction_hip_recentered_metrics": metrics,
            "signed_delta_recentered_minus_primary_mm": signed_metric_difference(metrics, primary),
            "prediction_hip_offset_norm_mm": {"mean": float(hip_distance.mean()),
                                               "p95": float(np.percentile(hip_distance, 95))}}


def audit(experiment, analysis=None):
    experiment = Path(experiment).resolve()
    analysis = Path(analysis).resolve() if analysis else experiment / "analysis"
    destination = analysis / "prediction_hip_recentered_metrics.json"
    source_copy = analysis / "hip_recenter_auditor_source.py"
    if destination.exists() or source_copy.exists():
        raise ValueError("Refusing to overwrite supplementary audit")
    report_path, verification_path = experiment / "run_summary.json", analysis / "verification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (report.get("status") != "complete" or verification.get("status") != "complete" or
            verification["run_summary_sha256"] != sha256(report_path) or report["final_test_subjects_used"]):
        raise ValueError("Matching complete development-only replay receipt required")
    manifest = {(item["seed"], item["method"]): item for item in verification["models"]}
    expected_pairs = {(run["seed"], method) for run in report["runs"] for method in report["methods"]}
    if set(manifest) != expected_pairs or len(manifest) != len(verification["models"]):
        raise ValueError("Independent replay model manifest differs")
    checks, runs, predictions = [], [], []
    for run in report["runs"]:
        result = {"seed": run["seed"], "models": {}}
        for method in report["methods"]:
            path = experiment / f"seed{run['seed']}" / f"{method}_predictions.npz"
            digest = sha256(path)
            if digest != manifest[run["seed"], method]["prediction_sha256"]:
                raise ValueError("Independently verified saved predictions changed")
            saved = safe_npz(path)
            if set(np.unique(saved["subject_ids"]).tolist()) != set(report["validation_subjects"]):
                raise ValueError("Validation subject population differs")
            result["models"][method] = {}
            for condition in report["evaluation_conditions"]:
                metrics = supplementary_metrics(saved[condition + "_fused_prediction_mm"], saved["target_mm"], saved["subject_ids"])
                recorded = run["models"][method]["metrics"][condition]
                compare_tree(metrics["primary_metrics"], {key: recorded[key] for key in metrics["primary_metrics"]},
                             f"seed{run['seed']}/{method}/{condition}/unchanged_primary", checks)
                result["models"][method][condition] = metrics
            predictions.append({"seed": run["seed"], "method": method, "sha256": digest})
        runs.append(result)
    aggregate = {}
    for method in report["methods"]:
        aggregate[method] = {}
        for condition in report["evaluation_conditions"]:
            values = [run["models"][method][condition] for run in runs]
            primary = [value["primary_metrics"]["MPJPE_mm"] for value in values]
            recentered = [value["prediction_hip_recentered_metrics"]["MPJPE_mm"] for value in values]
            delta = [value["signed_delta_recentered_minus_primary_mm"]["MPJPE_mm"] for value in values]
            aggregate[method][condition] = {"primary_MPJPE_seed_mean_mm": float(np.mean(primary)),
                "prediction_hip_recentered_MPJPE_seed_mean_mm": float(np.mean(recentered)),
                "signed_delta_recentered_minus_primary_seed_mean_mm": float(np.mean(delta)),
                "per_seed_signed_delta_mm": delta,
                "prediction_hip_recentered_frame_p95_seed_mean_mm": float(np.mean([
                    value["prediction_hip_recentered_metrics"]["frame_error_p95_mm"] for value in values])),
                "prediction_hip_recentered_per_subject_seed_mean_mm": {str(subject): float(np.mean([
                    value["prediction_hip_recentered_metrics"]["per_validation_subject_MPJPE_mm"][str(subject)]
                    for value in values])) for subject in report["validation_subjects"]}}
    result = {"status": "complete", "experiment": str(experiment), "source_sha256": sha256(Path(__file__)),
        "run_summary_sha256": sha256(report_path), "independent_replay_receipt_sha256": sha256(verification_path),
        "operation": "Subtract midpoint of prediction joints 11/12 from every predicted joint; original target already hip-centred",
        "units": "millimetres", "primary_metrics_selection_and_predictions_unchanged": True,
        "scope": "Supplementary evaluation only, not Procrustes, checkpoint selection, deployment claim or revised primary evidence",
        "signed_delta_convention": "recentered minus primary; negative means lower error, positive means worse; retain both signs",
        "final_test_subjects_used": [], "independent_validation_subject_count": len(report["validation_subjects"]),
        "initialization_seeds_are_not_subject_splits": True, "representation": report.get("representation", "hip"),
        "prediction_manifest": predictions, "unchanged_primary_metric_check_count": len(checks),
        "max_primary_metric_difference_mm": max((item.get("max_absolute_difference", 0.) for item in checks), default=0.),
        "runs": runs, "aggregate_seed_descriptive_means": aggregate}
    source_copy.write_bytes(Path(__file__).read_bytes())
    write_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    result = audit(args.experiment, args.analysis)
    print(json.dumps({"status": result["status"], "models": len(result["prediction_manifest"]),
                      "unchanged_primary_checks": result["unchanged_primary_metric_check_count"]}), flush=True)


if __name__ == "__main__":
    main()
