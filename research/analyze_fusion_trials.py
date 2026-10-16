"""Read-only reconstruction and post-hoc diagnostics for completed fusion trials.

No fitting, checkpoint selection or deployable GT-dependent prediction occurs.
Diagnostics are emitted only after every saved weight/prediction/metric matches.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_ATOL_MM = 1e-5
CUDA_CPU_RECONSTRUCTION_ATOL_MM = .01
METRIC_ATOL_MM = 1e-8
GAIN_EPS_MM = 1e-6
ORACLE_ATOL_MM = 1e-4


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def safe_npz(path):
    """Only bounded numeric/bool arrays; no object loading or pickle execution."""
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 256 or sum(m.file_size for m in members) > 128 * 1024 ** 2:
            raise ValueError(f"Unexpected NPZ size: {path}")
        if len({m.filename for m in members}) != len(members):
            raise ValueError(f"Duplicate NPZ members: {path}")
    with np.load(path, allow_pickle=False) as saved:
        result = {}
        for key in saved.files:
            value = saved[key]
            if value.dtype.kind not in "buif" or not np.isfinite(value).all():
                raise ValueError(f"Nonfinite/non-numeric NPZ array: {path}, {key}")
            result[key] = value.copy()
    return result


def strict_load_weights(model, path):
    arrays = safe_npz(path)
    expected = model.state_dict()
    if set(arrays) != set(expected):
        raise ValueError(f"State keys differ: {path}")
    state = {}
    for key, reference in expected.items():
        value = arrays[key]
        if value.shape != tuple(reference.shape) or value.dtype != reference.detach().numpy().dtype:
            raise ValueError(f"State shape/dtype differs: {path}, {key}")
        if (key.endswith(".adjacency") or key == "gate.scale") and not np.array_equal(value, reference.detach().numpy()):
            raise ValueError(f"Saved fixed buffer differs from declared control: {path}, {key}")
        state[key] = torch.from_numpy(value)
    model.load_state_dict(state, strict=True)
    model.eval()
    return sha256(path)


def check_array(actual, saved, name, atol, receipt):
    actual, saved = np.asarray(actual), np.asarray(saved)
    if actual.shape != saved.shape or not np.isfinite(actual).all() or not np.isfinite(saved).all():
        raise ValueError(f"Array shape/finite mismatch: {name}")
    difference = float(np.max(np.abs(actual.astype(np.float64) - saved.astype(np.float64)))) if actual.size else 0.
    if difference > atol:
        raise ValueError(f"Array mismatch {name}: max difference {difference}, tolerance {atol}")
    receipt.append({"name": name, "max_absolute_difference": difference, "absolute_tolerance": atol})


def independent_metrics(prediction, target, subjects):
    prediction, target, subjects = np.asarray(prediction), np.asarray(target), np.asarray(subjects)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 51 or len(subjects) != len(target):
        raise ValueError("Metric input shapes disagree")
    if not len(target) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Invalid metric inputs")
    error = np.sqrt(np.square((prediction - target).reshape(-1, 17, 3)).sum(-1))
    return {"MPJPE_mm": float(error.mean()), "frame_error_p95_mm": float(np.percentile(error.mean(1), 95)),
            "per_joint_mean_mm": error.mean(0).tolist(),
            "per_validation_subject_MPJPE_mm": {str(s): float(error[subjects == s].mean()) for s in np.unique(subjects)}}


def compare_tree(actual, expected, name, receipt, atol=METRIC_ATOL_MM):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Metric keys differ: {name}")
        for key in expected:
            compare_tree(actual[key], expected[key], name + "/" + key, receipt, atol)
    elif expected is None:
        if actual is not None:
            raise ValueError(f"Null metric mismatch: {name}")
        receipt.append({"name": name, "null_match": True})
    else:
        check_array(actual, expected, name, atol, receipt)


def roc_auc(labels, scores):
    """Mann-Whitney ROC AUC with average ranks for ties, no inference."""
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid AUC inputs")
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores, ranks = scores[order], np.empty(len(scores), dtype=np.float64)
    start = 0
    for stop in range(1, len(scores) + 1):
        if stop == len(scores) or sorted_scores[stop] != sorted_scores[start]:
            ranks[order[start:stop]] = (start + 1 + stop) / 2
            start = stop
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def signed_gain_summary(gain, active_frames, subjects):
    gain, active_frames = np.asarray(gain), np.asarray(active_frames, dtype=bool)
    selected = gain[active_frames]
    if not selected.size:
        return None
    harmful = selected < -GAIN_EPS_MM
    excess = -selected[harmful]
    by_subject = {str(s): float(gain[active_frames & (subjects == s)].mean())
        if (active_frames & (subjects == s)).any() else None for s in np.unique(subjects)}
    return {"joint_count": int(selected.size), "mean_gain_mm": float(selected.mean()),
            "subject_macro_mean_gain_mm": float(np.mean([v for v in by_subject.values() if v is not None])),
            "per_joint_mean_gain_mm": selected.mean(0).tolist(),
            "beneficial_joint_fraction": float((selected > GAIN_EPS_MM).mean()),
            "harmful_joint_fraction": float(harmful.mean()),
            "harmful_excess_mm": scalar_summary(excess),
            "mean_harmful_excess_over_all_active_joints_mm": float(np.maximum(-selected, 0.).mean()),
            "per_subject_mean_gain_mm": by_subject}


def segment_oracle(video, candidate, target, active_frames, actual):
    """GT-only best position on each joint's [video,candidate] line segment.

    Missing/inactive frames keep the actual learned fallback exactly. This is
    neither a deployed predictor nor globally optimal unrestricted 3-D pose.
    """
    video, candidate, target, actual = [np.asarray(x, dtype=np.float64) for x in (video, candidate, target, actual)]
    if video.shape != candidate.shape or target.shape != video.shape or actual.shape != video.shape or video.shape[-2:] != (17, 3):
        raise ValueError("Oracle pose shapes disagree")
    active_frames = np.asarray(active_frames, dtype=bool)
    if active_frames.shape != (len(video),) or not all(np.isfinite(x).all() for x in (video, candidate, target, actual)):
        raise ValueError("Invalid oracle inputs")
    delta = candidate - video
    denominator = np.square(delta).sum(-1)
    numerator = ((target - video) * delta).sum(-1)
    weight = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.)
    weight = np.clip(weight, 0., 1.)
    weight[~active_frames] = 0.
    prediction = actual.copy()
    prediction[active_frames] = (video + weight[..., None] * delta)[active_frames]
    return prediction, weight


def gate_diagnostics(gate, full_gain, actual_gain, active_frames):
    if gate is None or not active_frames.any():
        return None
    scores, full, actual = gate[active_frames].reshape(-1), full_gain[active_frames].reshape(-1), actual_gain[active_frames].reshape(-1)
    selected = scores > .5
    labels = full > GAIN_EPS_MM
    margin_labels = full > 1.
    bins = []
    for low, high in zip((0., .2, .4, .6, .8), (.2, .4, .6, .8, 1.000001)):
        take = (scores >= low) & (scores < high)
        bins.append({"gate_lower": low, "gate_upper": min(high, 1.), "count": int(take.sum()),
                     "full_candidate_beneficial_fraction": float(labels[take].mean()) if take.any() else None,
                     "actual_beneficial_fraction": float((actual[take] > GAIN_EPS_MM).mean()) if take.any() else None,
                     "mean_actual_gain_mm": float(actual[take].mean()) if take.any() else None})
    return {"active_weight": scalar_summary(scores), "selected_gate_gt_0_5_count": int(selected.sum()),
            "selected_fraction": float(selected.mean()),
            "beneficial_precision_full_candidate_gain_gt_0": float(labels[selected].mean()) if selected.any() else None,
            "beneficial_precision_full_candidate_gain_gt_1mm": float(margin_labels[selected].mean()) if selected.any() else None,
            "selected_actual_harmful_fraction": float((actual[selected] < -GAIN_EPS_MM).mean()) if selected.any() else None,
            "descriptive_ROC_AUC_full_candidate_gain_gt_0": roc_auc(labels, scores),
            "descriptive_ROC_AUC_full_candidate_gain_gt_1mm": roc_auc(margin_labels, scores),
            "bins": bins,
            "interpretation": "Descriptive discrimination on correlated frame-joints; no CI, p-value or calibrated probability claim"}


def diagnose_output(output, target, subjects, template, method, saved_predictions=None, oracle_atol_mm=ORACLE_ATOL_MM):
    """Use GT only here, after weight/prediction/metric verification."""
    def millimetres(key):
        if saved_predictions is not None and key in saved_predictions:
            return saved_predictions[key]
        # Match runner's float32 multiplication followed by float64 template.
        return output[key].reshape(-1, 51).detach().numpy() * 1000 + template

    video, actual = millimetres("video").reshape(-1, 17, 3), millimetres("fused").reshape(-1, 17, 3)
    target = target.reshape(-1, 17, 3)
    active = output["both"].numpy()
    video_valid = output["video_valid"].numpy()
    actual_error = np.linalg.norm(actual - target, axis=-1)
    video_error = np.linalg.norm(video - target, axis=-1)
    actual_gain = video_error - actual_error
    diagnostic = {"both_active_frames": int(active.sum()), "both_active_joints": int(active.sum()) * 17,
        "inactive_or_missing_frames": int((~active).sum()),
        "actual_gain_vs_video_on_both": signed_gain_summary(actual_gain, active, subjects),
        "actual_gain_vs_video_when_video_available": signed_gain_summary(actual_gain, video_valid, subjects)}
    if "candidate" in output:
        candidate, endpoint = millimetres("candidate").reshape(-1, 17, 3), "ungated_attention_candidate"
        gate = (saved_predictions["gate"] if saved_predictions is not None and "gate" in saved_predictions else output["gate"].numpy()) if "gate" in output else np.ones_like(actual_gain)
        gate_source = "learned_gate" if "gate" in output else "implicit_g_equals_1"
    elif "gate" in output:
        candidate, endpoint = millimetres("radar").reshape(-1, 17, 3), "radar_pose_branch"
        gate = saved_predictions["gate"] if saved_predictions is not None and "gate" in saved_predictions else output["gate"].numpy()
        gate_source = "learned_radar_pose_weight"
    elif method == "late_average":
        candidate, endpoint = millimetres("radar").reshape(-1, 17, 3), "radar_pose_branch"
        gate, gate_source = np.full_like(actual_gain, .5), "fixed_g_equals_0_5"
    else:
        diagnostic.update(candidate_endpoint=None, gate=None, oracle=None,
                          reason="No explicit ungated correction/pose-mixture segment; absent single branches are placeholders")
        return diagnostic
    full_gain = video_error - np.linalg.norm(candidate - target, axis=-1)
    oracle, oracle_weight = segment_oracle(video, candidate, target, active, actual)
    oracle_error = np.linalg.norm(oracle - target, axis=-1)
    if not np.array_equal(oracle[~active], actual[~active]):
        raise ValueError("Oracle changed inactive fallbacks")
    if active.any() and np.any(oracle_error[active] > np.minimum(video_error, np.linalg.norm(candidate - target, axis=-1))[active] + oracle_atol_mm):
        raise ValueError("Segment oracle is worse than an endpoint")
    if active.any() and np.any(oracle_error[active] > actual_error[active] + oracle_atol_mm):
        raise ValueError("Segment oracle is worse than the actual segment prediction")
    oracle_gain = video_error - oracle_error
    missed_partial = (full_gain <= GAIN_EPS_MM) & (oracle_gain > GAIN_EPS_MM)
    full_harmful = full_gain < -GAIN_EPS_MM
    diagnostic.update(candidate_endpoint=endpoint, gate_source=gate_source,
        full_candidate_g_equals_1_gain=signed_gain_summary(full_gain, active, subjects),
        gate=gate_diagnostics(gate, full_gain, actual_gain, active),
        oracle={"kind": "per_joint_GT_segment_projection_clipped_to_[0,1]",
            "not_deployable": True, "not_global_pose_optimum": True, "inactive_fallbacks_unchanged": True,
            "global_gate_model_relaxed_to_per_joint_oracle": method == "utility_global",
            "numeric_comparison_tolerance_mm": oracle_atol_mm,
            "oracle_weight_on_both": scalar_summary(oracle_weight[active]),
            "oracle_MPJPE_all_frames_mm": float(oracle_error.mean()),
            "actual_MPJPE_all_frames_mm": float(actual_error.mean()),
            "recoverable_gain_vs_actual_all_frames_mm": float((actual_error - oracle_error).mean()),
            "gain_vs_video_on_both": signed_gain_summary(oracle_gain, active, subjects),
            "recoverable_gain_vs_actual_on_both": signed_gain_summary(actual_error - oracle_error, active, subjects),
            "full_candidate_not_beneficial_but_partial_oracle_beneficial_fraction_on_both": float(missed_partial[active].mean()) if active.any() else None,
            "full_candidate_harmful_but_partial_oracle_beneficial_fraction_among_harmful": float(missed_partial[active][full_harmful[active]].mean()) if (active[:, None] & full_harmful).any() else None,
            "interpretation": "Best position along a fixed joint residual segment, GT-only diagnostic. Full-candidate binary sign can miss useful partial corrections."})
    return diagnostic


def load_verified_modules(experiment, report, receipt):
    paths = {}
    for name, digest in report["source_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Source hash names must be basenames")
        candidates = (experiment / "source_snapshot" / name, ROOT / "research" / name)
        path = next((p for p in candidates if p.exists() and sha256(p) == digest.lower()), None)
        if path is None:
            raise ValueError(f"No source matching saved SHA: {name}")
        paths[name] = path
        receipt["verified_sources"][name] = {"path": str(path), "sha256": digest}

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    from types import SimpleNamespace
    runner_file = report.get("runner_file", "run_fusion_experiments.py")
    if Path(runner_file).name != runner_file or runner_file not in paths:
        raise ValueError("Declared runner file is not pinned in source hashes")
    aliases = ("fusion_models", "fusion_graph_models", "fusion_scaled_gate_models", "fusion_pose_representation",
               "fusion_partial_representation")
    if report.get("representation") in ("hip_centroid_fallback", "hip_confidence_only") and "fusion_partial_representation.py" not in paths:
        raise ValueError("Partial representation source is not pinned in report")
    previous = {name: sys.modules.get(name) for name in aliases}
    previous_path = sys.path[:]
    loaded = {}
    try:
        for name in aliases:
            if name + ".py" in paths:
                loaded[name] = load_module("_verified_" + name, paths[name + ".py"])
                sys.modules[name] = loaded[name]
        model_module = loaded["fusion_models"]
        runner = load_module("_verified_fusion_runner", paths[runner_file])
        factory_name = report.get("model_factory", "fusion_models.FusionModel")
        if factory_name == "fusion_graph_models.GraphFusionModel":
            if "fusion_graph_models" not in loaded:
                raise ValueError("Graph factory source is not pinned in report")
            graph = loaded["fusion_graph_models"]
            model_module = SimpleNamespace(METHODS=graph.GRAPH_METHODS, FusionModel=graph.GraphFusionModel)
        elif factory_name == "fusion_scaled_gate_models.ScaledGateFusionModel":
            if "fusion_scaled_gate_models" not in loaded:
                raise ValueError("Scaled factory source is not pinned in report")
            scaled = loaded["fusion_scaled_gate_models"]
            model_module = SimpleNamespace(METHODS=scaled.SCALED_METHODS, FusionModel=scaled.ScaledGateFusionModel)
        elif factory_name != "fusion_models.FusionModel":
            raise ValueError(f"Unsupported model factory: {factory_name}")
        receipt["model_factory"] = factory_name
        receipt["runner_file"] = runner_file
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        sys.path[:] = previous_path
    return model_module, runner


def rebuild_data(report, runner, norm, receipt):
    cache = Path(report["input_cache"])
    ids = report["train_subjects"] + report["validation_subjects"]
    if set(report["cache_sha256"]) != set(map(str, ids)):
        raise ValueError("Cache hash subject set differs")
    for subject in ids:
        path = cache / f"subject{subject}.npz"
        if sha256(path) != report["cache_sha256"][str(subject)].lower():
            raise ValueError(f"Cache checksum differs, subject {subject}")
    audit_path = cache / "rgb_audit.json"
    if sha256(audit_path) != report["input_audit_sha256"].lower():
        raise ValueError("Cache audit checksum differs")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("conditions") != report["input_audit_conditions"]:
        raise ValueError("Pixel audit condition metadata differs")
    if "representation" in report:
        split, rows = runner.load_data(cache, report["training"]["conditions"], report["representation"])
    else:
        split, rows = runner.load_data(cache, report["training"]["conditions"])
    if split["canonical_ids_train"] != report["train_subjects"] or split["canonical_ids_validation"] != report["validation_subjects"] or report["final_test_subjects_used"]:
        raise ValueError("Development subject identities differ or test used")
    if set(ids) & set(split["canonical_ids_test"]):
        raise ValueError("Final subjects present in diagnostic loader")

    def stack(subjects, key):
        return np.concatenate([rows[s][key] for s in subjects])

    train, validation = report["train_subjects"], report["validation_subjects"]
    expected_shapes = {"point_mean": (5,), "point_std": (5,), "rgb_mean": (51,), "rgb_std": (51,), "target_mean_mm": (51,)}
    if set(norm) != set(expected_shapes) or any(norm[k].shape != shape for k, shape in expected_shapes.items()):
        raise ValueError("Normalization keys/shapes differ")
    if (norm["point_std"] <= 0).any() or (norm["rgb_std"] <= 0).any():
        raise ValueError("Nonpositive normalization scale")
    pm, ps = runner.point_normalization(stack(train, "points"), stack(train, "mask"))
    rgb_normalization_key = report.get("rgb_normalization_key", "clean")
    if rgb_normalization_key not in ("clean", "hip_clean") or any(rgb_normalization_key not in rows[s] for s in train):
        raise ValueError("Missing or unsupported RGB normalization source")
    if report.get("representation") in ("hip_centroid_fallback", "hip_confidence_only") and rgb_normalization_key != "hip_clean":
        raise ValueError("Partial controls require shared original hip-clean RGB normalization")
    rgb_reference = stack(train, rgb_normalization_key)
    reconstructed_norm = {"point_mean": pm, "point_std": ps,
        "rgb_mean": rgb_reference.mean(0), "rgb_std": np.maximum(rgb_reference.std(0), 1e-8),
        "target_mean_mm": stack(train, "target").mean(0)}
    for name in norm:
        check_array(reconstructed_norm[name], norm[name], "train_only_normalization/" + name, 0., receipt["array_checks"])
    target = stack(validation, "target")
    subjects = np.concatenate([np.full(len(rows[s]["target"]), s) for s in validation])
    frame_ids = stack(validation, "frame_ids")
    if len(target) != report["validation_frames"] or len(stack(train, "target")) != report["train_frames"]:
        raise ValueError("Experiment sample count differs")
    if len(set(zip(subjects.tolist(), frame_ids.tolist()))) != len(target):
        raise ValueError("Duplicate subject/frame observations")
    values = {c: runner.normalize_inputs(stack(validation, c), stack(validation, "points"), stack(validation, "mask"), norm)
              for c in report["training"]["conditions"]}
    clean = values["clean"]
    values["video_missing"] = (torch.zeros_like(clean[0]), clean[1], clean[2], torch.zeros_like(clean[3]))
    values["radar_missing"] = (clean[0], torch.zeros_like(clean[1]), torch.zeros_like(clean[2]), clean[3])
    thin = torch.tensor(runner.sparse_mask(stack(validation, "mask")))
    values["radar_sparse"] = (clean[0], clean[1].masked_fill(~thin[..., None], 0.), thin, clean[3])
    if list(values) != report["evaluation_conditions"]:
        raise ValueError("Evaluation conditions/order differ")
    receipt["cache_provenance"] = {"input_cache": str(cache), "all_cache_sha_match": True,
        "representation": report.get("representation", "hip"), "rgb_normalization_key": rgb_normalization_key,
        "input_audit_sha_match": True, "train_only_normalization_match": True,
        "final_test_subjects_used": [], "validation_subjects": validation,
        "per_subject_frame_count": {str(s): int((subjects == s).sum()) for s in validation}}
    return target, subjects, frame_ids, values


def aggregate_diagnostics(runs, methods, conditions):
    result = {}
    paths = {"active_actual_gain_mm": ("actual_gain_vs_video_on_both", "mean_gain_mm"),
        "active_actual_harmful_fraction": ("actual_gain_vs_video_on_both", "harmful_joint_fraction"),
        "active_full_candidate_gain_mm": ("full_candidate_g_equals_1_gain", "mean_gain_mm"),
        "active_gate_mean": ("gate", "active_weight", "mean"),
        "descriptive_gate_AUC": ("gate", "descriptive_ROC_AUC_full_candidate_gain_gt_0"),
        "gate_gt_0_5_beneficial_precision": ("gate", "beneficial_precision_full_candidate_gain_gt_0"),
        "oracle_recoverable_all_frames_gain_mm": ("oracle", "recoverable_gain_vs_actual_all_frames_mm"),
        "active_oracle_gain_mm": ("oracle", "gain_vs_video_on_both", "mean_gain_mm"),
        "partial_benefit_missed_by_full_candidate_sign_fraction": ("oracle", "full_candidate_not_beneficial_but_partial_oracle_beneficial_fraction_on_both")}
    for method in methods:
        result[method] = {}
        for condition in conditions:
            row = {}
            for name, path in paths.items():
                values = []
                for run in runs:
                    value = run["models"][method][condition]
                    for field in path:
                        value = value.get(field) if isinstance(value, dict) else None
                    if value is not None:
                        values.append(float(value))
                row[name] = {"seed_mean": float(np.mean(values)), "seed_values": values,
                             "contributing_seeds": len(values)} if values else None
            result[method][condition] = row
    return result


def analyze(experiment, output_directory=None):
    experiment = Path(experiment).resolve()
    report_path = experiment / "run_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError("Training report must be complete before diagnostic reconstruction")
    output_directory = Path(output_directory).resolve() if output_directory else experiment / "analysis"
    if output_directory.exists():
        raise ValueError(f"Refusing to overwrite diagnostic output: {output_directory}")
    output_directory.mkdir(parents=True)
    (output_directory / "analyzer_source.py").write_bytes(Path(__file__).read_bytes())
    receipt = {"status": "verifying", "experiment": str(experiment), "run_summary_sha256": sha256(report_path),
        "analyzer_sha256": sha256(Path(__file__)), "prediction_atol_mm": PREDICTION_ATOL_MM,
        "metric_atol_mm": METRIC_ATOL_MM, "verified_sources": {}, "array_checks": [], "metric_checks": [], "models": [],
        "diagnostics_written_only_after_all_matches": True}
    receipt_path = output_directory / "verification.json"
    try:
        is_cuda_trial = report["environment"]["device"].startswith("cuda")
        prediction_atol = CUDA_CPU_RECONSTRUCTION_ATOL_MM if is_cuda_trial else PREDICTION_ATOL_MM
        gate_atol = 1e-5 if is_cuda_trial else 1e-7
        oracle_atol = max(ORACLE_ATOL_MM, 2 * prediction_atol)
        receipt.update(prediction_atol_mm=prediction_atol, gate_absolute_tolerance=gate_atol,
            reconstruction_backend="CPU", trial_backend=report["environment"]["device"],
            cross_backend_policy="CPU replay of CUDA FP32 outputs allows 0.01mm absolute prediction difference; metrics computed directly from original saved arrays remain strict at 1e-8mm. Actual/gate diagnostics use saved CUDA outputs; unsaved candidate is CPU reconstruction.",
            oracle_numeric_tolerance_mm=oracle_atol)
        torch.set_num_threads(report["environment"]["threads"])
        torch.use_deterministic_algorithms(True)
        modules, runner = load_verified_modules(experiment, report, receipt)
        if report.get("representation") in ("hip_centroid_fallback", "hip_confidence_only"):
            protocol = experiment / "source_snapshot/fusion_partial_protocol_20261013.md"
            if not protocol.exists() or sha256(protocol) != report.get("protocol_sha256", "").lower():
                raise ValueError("Partial frozen protocol snapshot does not match declared SHA")
            receipt["frozen_protocol"] = {"path": str(protocol), "sha256": sha256(protocol), "declared_sha_match": True}
        receipt["current_dependency_sha256"] = {p.name: sha256(p) for p in (ROOT / "research").glob("*.py")
            if p.name in ("train_mri_point_baselines.py", "run_mri_baselines.py", "audit_mri.py", "cache_mri_rgb.py", "validate_mri_point_baselines.py")}
        receipt["current_protocol_sha256"] = sha256(ROOT / "research/pose_fusion_protocol.json")
        receipt["dependency_limit"] = "Imported dependency hashes recorded now; original trial did not pin these hashes. Exact prediction reconstruction additionally checks behavior."
        norm = safe_npz(experiment / "normalization.npz")
        receipt["normalization_sha256"] = sha256(experiment / "normalization.npz")
        target, subjects, frame_ids, values = rebuild_data(report, runner, norm, receipt)
        if [r["seed"] for r in report["runs"]] != report["initialization_seeds"] or any(r["status"] != "complete" for r in report["runs"]):
            raise ValueError("Completed seed reports differ")
        pending = []
        for run in report["runs"]:
            if set(run["models"]) != set(report["methods"]):
                raise ValueError("Model set differs across seeds")
            reconstructed_run = {"seed": run["seed"], "models": {}}
            for method in report["methods"]:
                if method not in modules.METHODS:
                    raise ValueError(f"Method unavailable in verified source: {method}")
                model = modules.FusionModel(method)
                seed_dir = experiment / f"seed{run['seed']}"
                weights_path, predictions_path = seed_dir / f"{method}_weights.npz", seed_dir / f"{method}_predictions.npz"
                weight_sha = strict_load_weights(model, weights_path)
                if sum(p.numel() for p in model.parameters()) != report["parameters"][method]:
                    raise ValueError("Model parameter count differs")
                check_array(model.target_template_m.numpy(), (norm["target_mean_mm"].reshape(17, 3) / 1000).astype(np.float32),
                            f"seed{run['seed']}/{method}/target_template", 0., receipt["array_checks"])
                saved = safe_npz(predictions_path)
                label = f"seed{run['seed']}/{method}"
                for key, value in (("target_mm", target), ("subject_ids", subjects), ("frame_ids", frame_ids)):
                    check_array(value, saved[key], label + "/" + key, 0., receipt["array_checks"])
                outputs = {}
                for condition, inputs in values.items():
                    with torch.no_grad():
                        prediction_output = model(*inputs)
                    outputs[condition] = prediction_output
                    predictions = {name: prediction_output[name].reshape(-1, 51).detach().numpy() * 1000 + norm["target_mean_mm"]
                                   for name in ("fused", "video", "radar")}
                    for name, value in predictions.items():
                        check_array(value, saved[condition + "_" + name + "_prediction_mm"],
                                    label + "/" + condition + "/" + name, prediction_atol, receipt["array_checks"])
                    metrics = independent_metrics(saved[condition + "_fused_prediction_mm"], saved["target_mm"], saved["subject_ids"])
                    expected = run["models"][method]["metrics"][condition]
                    compare_tree(metrics, {k: expected[k] for k in metrics}, label + "/" + condition + "/core_metrics", receipt["metric_checks"])
                    authoritative = {name: saved[condition + "_" + name + "_prediction_mm"] for name in ("fused", "video", "radar")}
                    ef = np.linalg.norm((authoritative["fused"] - target).reshape(-1, 17, 3), axis=-1).mean(1)
                    ev = np.linalg.norm((authoritative["video"] - target).reshape(-1, 17, 3), axis=-1).mean(1)
                    available = prediction_output["video_valid"].numpy()
                    extras = {"video_branch_MPJPE_mm": float(ev.mean()),
                        "negative_fusion_vs_available_video_fraction": float((ef[available] > ev[available]).mean()) if available.any() else None,
                        "both_modalities_missing_frames": int((~prediction_output["video_valid"] & ~prediction_output["radar_valid"]).sum()),
                        "image_available_frames": int(prediction_output["video_valid"].sum()),
                        "radar_available_frames": int(prediction_output["radar_valid"].sum())}
                    if "gate" in prediction_output:
                        gate = prediction_output["gate"].numpy()
                        check_array(gate, saved[condition + "_gate"], label + "/" + condition + "/gate", gate_atol, receipt["array_checks"])
                        authoritative["gate"] = saved[condition + "_gate"]
                        extras["mean_gate_weight"] = float(authoritative["gate"].mean())
                    compare_tree(extras, {k: expected[k] for k in extras}, label + "/" + condition + "/extra_metrics", receipt["metric_checks"])
                    outputs[condition] = (prediction_output, authoritative)
                pending.append((reconstructed_run, method, outputs))
                receipt["models"].append({"seed": run["seed"], "method": method, "weight_sha256": weight_sha,
                    "prediction_sha256": sha256(predictions_path), "finite_state_and_strict_reload": True,
                    "all_predictions_and_metrics_match": True})
                print(json.dumps({"seed": run["seed"], "method": method, "verification": "matched"}), flush=True)
            receipt.setdefault("reconstructed_runs", []).append(reconstructed_run)
        # Compute no gate-quality/oracle conclusions until EVERY model matched.
        diagnostic_runs = receipt.pop("reconstructed_runs")
        for run, method, outputs in pending:
            run["models"][method] = {condition: diagnose_output(out, target, subjects, norm["target_mean_mm"], method, saved_outputs, oracle_atol)
                                     for condition, (out, saved_outputs) in outputs.items()}
        diagnostics = {"status": "complete", "scope": "Post-hoc development diagnostics after all saved arrays/metrics matched",
            "experiment": str(experiment), "verification_receipt": str(receipt_path), "final_test_subjects_used": [],
            "independent_validation_subject_count": len(report["validation_subjects"]),
            "initialization_seeds_are_not_subject_splits": True, "gain_epsilon_mm": GAIN_EPS_MM,
            "representation": report.get("representation", "hip"), "rgb_normalization_key": report.get("rgb_normalization_key", "clean"),
            "trial_backend": report["environment"]["device"],
            "actual_outputs": "Original verified saved predictions/gates; missing fallback unchanged",
            "unsaved_candidate": "CPU forward reconstruction; CUDA trials have declared FP32 replay tolerance. Near-zero endpoint signs can vary within this numeric tolerance.",
            "active_summaries": "Conditional on each model's both-available frames; joint-weighted mean and separate subject-macro mean. Inactive gates excluded.",
            "oracle": "GT-only per-joint best point on the fixed video-to-candidate line segment; inactive fallbacks unchanged. Not deployable or globally optimal pose.",
            "auc": "Descriptive correlated frame-joint discrimination, not inference/calibration; degenerate classes return null.",
            "runs": diagnostic_runs,
            "aggregate_seed_descriptive_means": aggregate_diagnostics(diagnostic_runs, report["methods"], list(values))}
        write_json(output_directory / "diagnostics.json", diagnostics)
        receipt.update(status="complete", verified_model_count=len(receipt["models"]),
            array_check_count=len(receipt["array_checks"]), metric_check_count=len(receipt["metric_checks"]),
            diagnostics_sha256=sha256(output_directory / "diagnostics.json"),
            diagnostics_created_after_all_models_verified=True)
        write_json(receipt_path, receipt)
        return diagnostics, receipt
    except Exception as error:
        receipt.pop("reconstructed_runs", None)
        receipt.update(status="failed", error=str(error), diagnostics_created_after_all_models_verified=False)
        write_json(receipt_path, receipt)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, help="New directory; defaults to experiment/analysis")
    args = parser.parse_args()
    _, receipt = analyze(args.experiment, args.output)
    print(json.dumps({"status": receipt["status"], "verified_models": receipt["verified_model_count"],
                      "array_checks": receipt["array_check_count"], "metric_checks": receipt["metric_check_count"]}), flush=True)


if __name__ == "__main__":
    main()
