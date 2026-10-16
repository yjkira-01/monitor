"""Separate saved-state CUDA replay and independent NumPy metrics audit."""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps/research_cuda"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from run_xfi_accuracy_experiments import CONDITIONS, EVALUATION, SCORE, SOURCES, load_tokens, new_model, sha, state_fingerprint, write_json
from xfi_accuracy_models import native_radar_batch, token_prediction
from xfi_paper_baseline import source_manifest


def safe_arrays(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 1024 or sum(row.file_size for row in members) > 512 * 1024 ** 2:
            raise ValueError("Unexpected native NPZ size")
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key].copy() for key in saved.files}
    if any(value.dtype.kind not in "buif" or not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Invalid native saved arrays")
    return result


def independent_metrics(prediction, target, subjects):
    prediction, target = [np.asarray(x, dtype=np.float64).reshape(-1, 17, 3) for x in (prediction, target)]
    errors = np.sqrt(np.sum((prediction - target) ** 2, axis=2))
    return {"MPJPE_mm": float(errors.mean()), "frame_error_p95_mm": float(np.percentile(errors.mean(1), 95)),
            "per_joint_mean_mm": errors.mean(0).tolist(), "per_validation_subject_MPJPE_mm": {
                str(subject): float(errors[subjects == subject].mean()) for subject in np.unique(subjects)}}


def check_metric(record, expected):
    differences = []
    for key, value in expected.items():
        if isinstance(value, dict):
            for item, number in value.items():
                differences.append(abs(record[key][item] - number))
        else:
            differences.append(float(np.max(np.abs(np.asarray(record[key]) - np.asarray(value)))))
    return max(differences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    analysis = args.experiment / "analysis"
    if analysis.exists():
        raise ValueError("Refusing to overwrite verification")
    analysis.mkdir()
    report_path = args.experiment / "run_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "complete" or report["official_source_sha256"] != source_manifest():
        raise ValueError("Complete source-matched native experiment required")
    recorded_sources = set(report["source_sha256"])
    allowed_source_sets = (set(SOURCES), set(SOURCES) | {"run_xfi_stability_experiments.py"},
                           set(SOURCES) | {"run_xfi_accuracy_resume_20261016.py", "xfi_resume_support.py"})
    if recorded_sources not in allowed_source_sets:
        raise ValueError("Unexpected reconstruction source set")
    for name in sorted(recorded_sources):
        expected = report["source_sha256"][name]
        if sha(ROOT / "research" / name) != expected or sha(args.experiment / "source_snapshot" / name) != expected:
            raise ValueError("Native reconstruction source changed")
    if "resume_provenance" in report:
        from xfi_resume_support import load_prior
        provenance = report["resume_provenance"]
        prior, checked = load_prior(Path(provenance["source_directory"]), Path(report["cache"]))
        generation = json.loads((ROOT / "research/xfi_resume_generation_20261016.json").read_text(encoding="utf-8"))
        if (generation["original_sha256"] != report["source_sha256"]["run_xfi_accuracy_experiments.py"] or
                generation["continuation_sha256"] != report["source_sha256"].get("run_xfi_accuracy_resume_20261016.py") or
                generation["support_sha256"] != report["source_sha256"].get("xfi_resume_support.py") or
                not generation["training_loop_AST_identical"]):
            raise ValueError("Continuation source differs from preserved training loop")
        for key in ("source_run_summary_sha256", "stop_receipt_sha256", "completed_models_reused", "missing_models_to_train"):
            if provenance[key] != checked[key]:
                raise ValueError("Continuation provenance changed")
        expected_copies = {}
        for run in report["runs"]:
            for method, row in run["models"].items():
                if (run["seed"], method) in prior:
                    if row != prior[run["seed"], method]:
                        raise ValueError("Reused complete model metadata changed")
                    for kind in ("weights", "predictions"):
                        name = f"seed{run['seed']}/{method}_{kind}.npz"
                        expected_copies[name] = row[kind + "_sha256"]
        recorded_copies = {row["path"]: row["sha256"] for row in provenance["copied_files"]}
        if len(provenance["copied_files"]) != 10 or recorded_copies != expected_copies:
            raise ValueError("Continuation did not preserve all five original models")
    if "control_protocol_sha256" in report:
        protocol = ROOT / "research/xfi_stability_protocol_20261013.md"
        snapshot = args.experiment / "source_snapshot/xfi_stability_protocol_20261013.md"
        generation = json.loads((ROOT / "research/xfi_stability_runner_generation.json").read_text(encoding="utf-8"))
        if (sha(protocol) != report["control_protocol_sha256"] or sha(snapshot) != report["control_protocol_sha256"] or
                generation["original_sha256"] != report["source_sha256"]["run_xfi_accuracy_experiments.py"] or
                generation["control_sha256"] != report["source_sha256"].get("run_xfi_stability_experiments.py") or
                generation["control_protocol_sha256"] != report["control_protocol_sha256"] or
                report["training"]["lr"] != .0001 or report["methods"] != ["native_mse"]):
            raise ValueError("Fixed optimizer control source, protocol or configuration changed")
    cache = Path(report["cache"])
    cache_report = json.loads((cache / "cache_report.json").read_text(encoding="utf-8"))
    if sha(cache / "cache_report.json") != report["cache_report_sha256"]:
        raise ValueError("Frozen full native cache receipt changed")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    validation = load_tokens(cache, cache_report, report["validation_subjects"], "cuda")
    backbones = safe_arrays(cache / "calibrated_backbones.npz")
    result = {"status": "running", "backend": "Independent saved-state CUDA replay; separate NumPy metric computation",
              "prediction_tolerance_mm": .01, "metric_tolerance_mm": 1e-8,
              "cross_backend_limitation_retained": "Earlier full native CPU reconstruction did not meet .01mm; not claimed passed",
              "run_summary_sha256": sha(report_path), "auditor_sha256": sha(__file__), "models": [],
              "max_prediction_difference_mm": 0., "max_metric_difference_mm": 0., "fallback_exact_checks": 0,
              "initialization_and_schedule_matched": True, "final_test_subjects_used": [],
              "independent_subject_scores": {method: [] for method in report["methods"]}}
    for run in report["runs"]:
        initial, schedule = None, None
        for method in report["methods"]:
            row = run["models"][method]
            if initial is None:
                initial, schedule = row["initial_state_sha256"], row["schedule_sha256"]
            if initial != row["initial_state_sha256"] or schedule != row["schedule_sha256"]:
                raise ValueError("Objective controls lost pairing")
            seed_dir = args.experiment / f"seed{run['seed']}"
            weights_path, predictions_path = seed_dir / f"{method}_weights.npz", seed_dir / f"{method}_predictions.npz"
            if sha(weights_path) != row["weights_sha256"] or sha(predictions_path) != row["predictions_sha256"]:
                raise ValueError("Saved native weights/predictions changed")
            weights, predictions = safe_arrays(weights_path), safe_arrays(predictions_path)
            for key, value in backbones.items():
                np.testing.assert_array_equal(weights[key], value)
            model = new_model(run["seed"], backbones, "cuda")
            if state_fingerprint(model) != row["initial_state_sha256"]:
                raise ValueError("Native initial state does not reconstruct")
            model.load_state_dict({key: torch.from_numpy(value) for key, value in weights.items()}, strict=True)
            model.eval()
            primary_subject_scores, centered_subject_scores = [], []
            for key in ("target_mm", "frame_ids", "subject_ids"):
                np.testing.assert_array_equal(predictions[key], validation[key])
            with torch.no_grad():
                for condition in EVALUATION:
                    condition_index = CONDITIONS.index(condition) if condition in CONDITIONS else 0
                    chunks = []
                    for start in range(0, len(validation["target_mm"]), 16):
                        rgb = validation["rgb"][condition_index, start:start + 16]
                        radar = validation["radar_sparse" if condition == "radar_sparse" else "radar"][start:start + 16]
                        video_valid = torch.full((len(rgb),), condition != "video_missing", dtype=torch.bool, device="cuda")
                        radar_valid = validation["radar_available"][start:start + 16] & (condition != "radar_missing")
                        value = token_prediction(model, rgb, radar, video_valid, radar_valid, root=method == "root_mpjpe")
                        chunks.append(value.cpu().numpy().astype(np.float64).reshape(-1, 51) * 1000)
                    rebuilt = np.concatenate(chunks)
                    saved = predictions[condition + "_fused_prediction_mm"]
                    difference = float(np.max(np.abs(rebuilt - saved)))
                    if difference > .01:
                        raise ValueError("Native saved-state CUDA replay exceeds .01mm: " + str(difference))
                    result["max_prediction_difference_mm"] = max(difference, result["max_prediction_difference_mm"])
                    metrics = independent_metrics(saved, predictions["target_mm"], predictions["subject_ids"])
                    metric_difference = check_metric(row["metrics"][condition], metrics)
                    centered = predictions[condition + "_recentered_prediction_mm"]
                    pose = saved.reshape(-1, 17, 3)
                    independent_centered = (pose - ((pose[:, 11] + pose[:, 12]) / 2)[:, None]).reshape(-1, 51)
                    np.testing.assert_allclose(centered, independent_centered, rtol=0, atol=1e-8)
                    if method == "root_mpjpe" and np.max(np.abs((pose[:, 11] + pose[:, 12]) / 2)) > .0001:
                        raise ValueError("Saved root-constrained prediction violates its coordinate constraint")
                    centered_metric = independent_metrics(centered, predictions["target_mm"], predictions["subject_ids"])
                    metric_difference = max(metric_difference, abs(centered_metric["MPJPE_mm"] - row["metrics"][condition]["prediction_recentered_MPJPE_mm"]))
                    for subject, value in centered_metric["per_validation_subject_MPJPE_mm"].items():
                        metric_difference = max(metric_difference, abs(value - row["metrics"][condition]["prediction_recentered_per_subject_mm"][subject]))
                    if condition in SCORE:
                        primary_subject_scores.append([metrics["per_validation_subject_MPJPE_mm"][str(s)] for s in report["validation_subjects"]])
                        centered_subject_scores.append([centered_metric["per_validation_subject_MPJPE_mm"][str(s)] for s in report["validation_subjects"]])
                    if metric_difference > 1e-8:
                        raise ValueError("Independent native metrics disagree")
                    result["max_metric_difference_mm"] = max(metric_difference, result["max_metric_difference_mm"])
                if not np.array_equal(predictions["video_missing_fused_prediction_mm"], predictions["video_missing_radar_prediction_mm"]):
                    raise ValueError("Native radar-only fallback lost exactness")
                mask = validation["radar_available"].cpu().numpy()
                if not np.array_equal(predictions["radar_missing_fused_prediction_mm"], predictions["radar_missing_video_prediction_mm"]):
                    raise ValueError("Native image-only fallback lost exactness")
                result["fallback_exact_checks"] += 2
                both_absent = torch.zeros(16, dtype=torch.bool, device="cuda")
                absent = token_prediction(model, validation["rgb"][0, :16], validation["radar"][:16], both_absent, both_absent, root=method == "root_mpjpe")
                prior = model.training_prior_m[None].expand(16, -1, -1)
                if method == "root_mpjpe":
                    prior = prior - ((prior[:, 11] + prior[:, 12]) / 2)[:, None]
                if not torch.equal(absent, prior):
                    raise ValueError("Native both-absent fallback differs from training-only prior")
                result.setdefault("both_absent_prior_exact_checks", 0)
                result["both_absent_prior_exact_checks"] += 1
                # Real first 16 frames reconstruct the complete image/radar path,
                # preserving the cache's original native collate/extraction cohort.
                if method == report["methods"][0]:
                    subject = report["validation_subjects"][0]
                    with np.load(cache / "raw" / f"subject{subject}.npz", allow_pickle=False) as raw:
                        images = torch.tensor(raw["clean_images"][:16].transpose(0, 3, 1, 2), dtype=torch.float32, device="cuda")
                        points, _ = native_radar_batch(raw["points"][:16])
                        points = torch.tensor(points, dtype=torch.float32, device="cuda")
                    raw_prediction = model.native_subset(images, points)
                    if method == "root_mpjpe":
                        raw_prediction -= ((raw_prediction[:, 11] + raw_prediction[:, 12]) / 2)[:, None]
                    delta = float(np.max(np.abs(raw_prediction.cpu().numpy().astype(np.float64).reshape(-1, 51) * 1000 - predictions["clean_fused_prediction_mm"][:16])))
                    if delta > .01:
                        raise ValueError("Native raw-image/radar reconstruction differs from token pipeline: " + str(delta))
                    result.setdefault("raw_native_replay_checks", []).append({"seed": run["seed"], "subject": subject, "frames": 16, "max_difference_mm": delta})
            result["models"].append({"seed": run["seed"], "method": method, "weights_sha256": row["weights_sha256"],
                                      "predictions_sha256": row["predictions_sha256"], "initial_state_and_frozen_backbone_verified": True})
            result["independent_subject_scores"][method].append({"seed": run["seed"],
                    "primary_condition_by_subject_mm": primary_subject_scores,
                    "posthoc_condition_by_subject_mm": centered_subject_scores})
            write_json(analysis / "verification_in_progress.json", result)
            del model, weights, predictions
    max_score_difference = 0.
    primary, posthoc = {}, {}
    for method, rows in result["independent_subject_scores"].items():
        primary[method] = np.asarray([row["primary_condition_by_subject_mm"] for row in rows])
        posthoc[method] = np.asarray([row["posthoc_condition_by_subject_mm"] for row in rows])
        published = report["aggregate"]["models"][method]
        for arrays, key in ((primary[method], "selection_score"), (posthoc[method], "posthoc_recentered_selection_score")):
            independent_seed_score = arrays.mean((1, 2))
            max_score_difference = max(max_score_difference,
                    abs(float(independent_seed_score.mean()) - published[key]["mean_mm"]),
                    float(np.max(np.abs(independent_seed_score - published[key]["per_seed_mm"]))))
    if max_score_difference > 1e-8:
        raise ValueError("Independent subject/condition aggregate does not reproduce published scores")
    result["max_aggregate_score_difference_mm"] = max_score_difference
    if "root_mpjpe" in primary and "native_mse" in posthoc and "native_mpjpe" in posthoc:
        reference = min(("native_mse", "native_mpjpe"), key=lambda m: posthoc[m].mean())
        gain = posthoc[reference] - primary["root_mpjpe"]
        clean_change = float(primary["root_mpjpe"][:, 0].mean() - posthoc[reference][:, 0].mean())
        gate = bool(gain.mean() >= max(1., .02 * posthoc[reference].mean()) and
                    clean_change <= max(1., .02 * posthoc[reference][:, 0].mean()) and
                    np.count_nonzero(gain.mean((0, 1)) > 0) >= 3 and
                    np.count_nonzero(gain.mean((1, 2)) > 0) >= 2)
        published = report["aggregate"]["root_training_vs_best_posthoc_control"]
        if (published["reference"] != reference + "+prediction_hip_recenter" or
                abs(gain.mean() - published["score_gain_mm"]) > 1e-8 or gate != published["resource_gate_passed"]):
            raise ValueError("Root training vs. strongest posthoc control decision does not reproduce")
        result["root_training_vs_best_posthoc_control_independently_verified"] = True
    result["status"] = "complete"
    write_json(analysis / "verification.json", result)
    (analysis / "auditor_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": "complete", "models": len(result["models"]),
                      "max_prediction_difference_mm": result["max_prediction_difference_mm"],
                      "max_metric_difference_mm": result["max_metric_difference_mm"]}), flush=True)


if __name__ == "__main__":
    main()
