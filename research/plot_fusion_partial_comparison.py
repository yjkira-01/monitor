"""Plot the matched partial-geometry factor, with confidence/availability held fixed.

No models, training runtime or GT labels are loaded. Only completed summaries,
source snapshots, caches, normalizers and saved prediction metadata are read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_fusion_results import (COLORS, CONDITION_NAMES, MARKERS, METHOD_NAMES,
                                 extract, save_figure, sha256, write_csv)


CONTROL_METHODS = ("rgb_matched", "concat", "joint_gate")
COMMON_KEYS = ("cache_sha256", "input_audit_sha256", "input_audit_conditions", "data_split_seed",
               "train_subjects", "validation_subjects", "train_frames", "validation_frames",
               "initialization_seeds", "evaluation_conditions", "score_conditions", "training",
               "environment", "source_sha256", "runner_file", "protocol_sha256",
               "rgb_normalization_key", "representation_control")


def read_report(directory, representation):
    source = directory / "run_summary.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("representation") != representation:
        raise ValueError(f"Expected complete {representation} experiment")
    if report.get("final_test_subjects_used"):
        raise ValueError("This development control cannot include final-test subjects")
    return source, sha256(source), report


def verified_replay(directory, summary_digest, report):
    """Require the complete independent replay already requested for this pair."""
    path = directory / "analysis/verification.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {(seed, method) for seed in report["initialization_seeds"] for method in report["methods"]}
    models = receipt.get("models", [])
    manifest = {(row["seed"], row["method"]): row for row in models}
    if (receipt.get("status") != "complete" or receipt.get("run_summary_sha256") != summary_digest or
            set(manifest) != expected or len(models) != len(expected) or
            any(not row.get("finite_state_and_strict_reload") or not row.get("all_predictions_and_metrics_match")
                for row in models)):
        raise ValueError("A matching complete independent replay is required before partial paired plotting")
    if receipt.get("normalization_sha256") != sha256(directory / "normalization.npz"):
        raise ValueError("Independently verified normalization file changed")
    return path, sha256(path), manifest


def pinned_partial_function(directory, report):
    """Load only the two SHA-verified NumPy representation snapshots."""
    snapshot = directory / "source_snapshot"
    for name, expected in report["source_sha256"].items():
        if sha256(snapshot / name) != expected:
            raise ValueError(f"Source snapshot checksum differs: {name}")
    pose_spec = importlib.util.spec_from_file_location("_plot_pinned_pose", snapshot / "fusion_pose_representation.py")
    pose = importlib.util.module_from_spec(pose_spec)
    pose_spec.loader.exec_module(pose)
    partial_spec = importlib.util.spec_from_file_location("_plot_pinned_partial", snapshot / "fusion_partial_representation.py")
    partial = importlib.util.module_from_spec(partial_spec)
    original = sys.modules.get("fusion_pose_representation")
    sys.modules["fusion_pose_representation"] = pose
    try:
        partial_spec.loader.exec_module(partial)
    finally:
        if original is None:
            sys.modules.pop("fusion_pose_representation", None)
        else:
            sys.modules["fusion_pose_representation"] = original
    return partial.hip_preserving_fallback_features


def verify_control(geometry_directory, geometry, confidence_directory, confidence, methods, verified_models):
    for key in COMMON_KEYS:
        if key not in geometry or key not in confidence or geometry[key] != confidence[key]:
            raise ValueError(f"Partial representation control differs or lacks metadata: {key}")
    if geometry.get("model_factory") != confidence.get("model_factory"):
        raise ValueError("Model factories differ")
    if geometry["rgb_normalization_key"] != "hip_clean":
        raise ValueError("Both partial controls must use original hip-clean normalization")
    if any(geometry["parameters"][m] != confidence["parameters"][m] for m in methods):
        raise ValueError("Paired model parameter counts differ")
    normalizers = [d / "normalization.npz" for d in (geometry_directory, confidence_directory)]
    with np.load(normalizers[0], allow_pickle=False) as a, np.load(normalizers[1], allow_pickle=False) as b:
        expected = {"rgb_mean", "rgb_std", "point_mean", "point_std", "target_mean_mm"}
        if set(a.files) != expected or set(b.files) != expected:
            raise ValueError("Unexpected normalization fields")
        for field in sorted(expected):
            if a[field].dtype != b[field].dtype or not np.array_equal(a[field], b[field]) or not np.isfinite(a[field]).all():
                raise ValueError(f"Shared normalization is not exact: {field}")
    geometric = pinned_partial_function(geometry_directory, geometry)
    confidence_only = pinned_partial_function(confidence_directory, confidence)
    cache_checks = []
    raw_conditions = geometry["training"]["conditions"]
    cache_paths = [Path(r["input_cache"]).resolve() for r in (geometry, confidence)]
    for subject in geometry["train_subjects"] + geometry["validation_subjects"]:
        paths = [cache / f"subject{subject}.npz" for cache in cache_paths]
        expected_sha = geometry["cache_sha256"][str(subject)]
        if any(sha256(path) != expected_sha for path in paths):
            raise ValueError("Raw paired cache checksum differs")
        with np.load(paths[0], allow_pickle=False) as saved:
            for condition in raw_conditions:
                raw, poses = saved[condition + "_features"], saved[condition + "_poses"]
                # Confidence and availability do not depend on the positive focal.
                # Unit focal checks only that factor; actual XY replay belongs to
                # the independent model audit, not this plotting process.
                a = geometric(poses, np.ones(2), raw, mode="centroid")
                b = confidence_only(poses, np.ones(2), raw, mode="confidence_only")
                hip_valid = (raw[:, 34:] >= .3).any(1)
                visible = a[:, 34:] >= .3
                if not np.array_equal(a[:, 34:], b[:, 34:]):
                    raise ValueError("Raw confidence differs between partial controls")
                if not np.array_equal((b[:, 34:] >= .3), visible):
                    raise ValueError("Visibility/availability/loss support differs")
                if not np.array_equal(a[hip_valid], raw[hip_valid]) or not np.array_equal(b[hip_valid], raw[hip_valid]):
                    raise ValueError("Hip-valid input changed")
                partial = ~hip_valid & visible.any(1)
                if not np.array_equal(b[partial, :34], np.zeros_like(b[partial, :34])):
                    raise ValueError("Confidence-only partial raw XY is not zero")
                cache_checks.append({"subject": subject, "condition": condition, "cache_sha256": expected_sha,
                                     "raw_confidence_exact_match": True, "visible_joint_masks_exact_match": True,
                                     "hip_valid_raw_features_preserved": True, "confidence_only_partial_raw_XY_zero": True})
    prediction_checks = []
    for seed in geometry["initialization_seeds"]:
        for method in methods:
            files = [d / f"seed{seed}" / f"{method}_predictions.npz"
                     for d in (geometry_directory, confidence_directory)]
            for path, manifest in zip(files, verified_models):
                if sha256(path) != manifest[seed, method]["prediction_sha256"]:
                    raise ValueError("Independently verified paired prediction file changed")
            with np.load(files[0], allow_pickle=False) as a, np.load(files[1], allow_pickle=False) as b:
                for field in ("frame_ids", "subject_ids", "target_mm"):
                    if a[field].dtype != b[field].dtype or not np.array_equal(a[field], b[field]):
                        raise ValueError(f"Prediction metadata not exactly paired: {method}/seed{seed}/{field}")
            prediction_checks.append({"method": method, "seed": seed, "frame_subject_target_exact_match": True,
                                      "geometry_predictions_sha256": sha256(files[0]),
                                      "confidence_predictions_sha256": sha256(files[1])})
    return {"controlled_keys_checked": list(COMMON_KEYS), "normalization_arrays_exact_match": True,
            "normalization_sha256": [sha256(p) for p in normalizers], "raw_input_control_checks": cache_checks,
            "confidence_check_focal": [1., 1.],
            "focal_scope": "Confidence, masks, hip-preservation and raw-zero checks only; no claim to replay actual XY",
            "paired_prediction_metadata": prediction_checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(CONTROL_METHODS))
    parser.add_argument("--prefix", default="fig_partial_geometry")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix):
        raise ValueError("Prefix must be a plain filename")
    directories = args.geometry.resolve(), args.confidence.resolve()
    gs, gd, geometry = read_report(directories[0], "hip_centroid_fallback")
    cs, cd, confidence = read_report(directories[1], "hip_confidence_only")
    replays = [verified_replay(directory, digest, report) for directory, digest, report in
               ((directories[0], gd, geometry), (directories[1], cd, confidence))]
    methods = args.methods
    if len(methods) != len(set(methods)) or not set(methods) <= set(geometry["methods"]) & set(confidence["methods"]):
        raise ValueError("Choose distinct methods available in both controls")
    conditions, seeds = geometry["evaluation_conditions"], geometry["initialization_seeds"]
    if len(seeds) < 2:
        raise ValueError("Paired seed SD needs at least two seeds")
    for report in (geometry, confidence):
        # Verify saved condition/subject metrics; do not recompute predictions.
        extract(report, methods, conditions, methods[0], include_gain=False)
    verification = verify_control(directories[0], geometry, directories[1], confidence, methods,
                                  [replay[2] for replay in replays])
    verification["independent_replay_receipts"] = [{"path": str(path), "sha256": digest}
                                                    for path, digest, _ in replays]
    g_runs = {run["seed"]: run for run in geometry["runs"]}
    c_runs = {run["seed"]: run for run in confidence["runs"]}
    rows, seed_rows = [], []
    for method in methods:
        for condition in conditions:
            gains = []
            for seed in seeds:
                g = g_runs[seed]["models"][method]["metrics"][condition]["MPJPE_mm"]
                c = c_runs[seed]["models"][method]["metrics"][condition]["MPJPE_mm"]
                gains.append(c - g)
                seed_rows.append({"method": method, "condition": condition, "initialization_seed": seed,
                                  "hip_confidence_only_MPJPE_mm": c, "hip_centroid_fallback_MPJPE_mm": g,
                                  "local_geometry_gain_confidence_minus_geometry_mm": c - g})
            rows.append({"method": method, "condition": condition,
                         "local_geometry_gain_confidence_minus_geometry_mm": float(np.mean(gains)),
                         "paired_seed_sample_SD_mm": float(np.std(gains, ddof=1)), "n_paired_initialization_seeds": len(seeds),
                         "interpretation": "Partial XY factor with common raw confidence, visibility, normalizers and budget"})
    output = directories[0] / "figures/local_geometry_comparison"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, records in (("source", rows), ("seed_source", seed_rows)):
        path = output / f"{args.prefix}_{suffix}.csv"
        write_csv(path, records)
        paths.append(path)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10., "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(max(8.5, 1.5 * len(conditions)), 5))
    lookup = {(r["method"], r["condition"]): r for r in rows}
    x = np.arange(len(conditions))
    offsets = np.linspace(-.25, .25, len(methods)) if len(methods) > 1 else [0.]
    for index, (method, offset) in enumerate(zip(methods, offsets)):
        ax.errorbar(x + offset, [lookup[method, c]["local_geometry_gain_confidence_minus_geometry_mm"] for c in conditions],
                    yerr=[lookup[method, c]["paired_seed_sample_SD_mm"] for c in conditions],
                    fmt=MARKERS[index % len(MARKERS)], color=COLORS[index % len(COLORS)],
                    markersize=4.8, capsize=2.5, label=METHOD_NAMES.get(method, method))
    ax.axhline(0., color="#444444", linewidth=1., linestyle="--")
    ax.set_xticks(x, [CONDITION_NAMES.get(c, c) for c in conditions])
    ax.set_xlabel("Evaluation input condition; shared confidence and availability")
    ax.set_ylabel("MPJPE reduction from partial XY geometry / mm\n(confidence-only minus centroid fallback)")
    ax.grid(axis="y", alpha=.22)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.17), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=.22)
    fig.text(.5, .01, "Paired seed mean ± sample SD; local geometry factor; one development split", ha="center", fontsize=9)
    paths += save_figure(fig, output, args.prefix + "_local_geometry")
    caption = output / f"{args.prefix}_caption.md"
    caption.write_text(
        f"# Partial geometry control\n\nGeometry summary: {gs}. Confidence-only summary: {cs}.\n\n"
        "The plotted difference is hip_confidence_only MPJPE minus hip_centroid_fallback MPJPE. Positive values favor "
        "adding partial relative XY. Points average seed-paired differences; bars show their sample SD, not a confidence "
        "interval. Both groups preserve all hip-valid features exactly and retain identical partial raw confidence, "
        "visible joint masks, availability and auxiliary-valid support. They use common original hip-clean RGB statistics, "
        "point/target statistics, cache, frame/subject/target pairing, model capacity, budget, augmentation reference, "
        "source code and backend. Confidence-only has raw XY=0 on partial frames, which generally becomes fixed "
        "-hip_mean/std in its observed normalized XY fields. It does not mean normalized XY is zero.\n\n"
        "Relative to the original hip-only representation, both partial controls add confidence information and visual "
        "availability on partly visible root-unavailable frames, which can expand the valid auxiliary training population. "
        "That shared factor is held fixed in this paired plot; it is distinct from the local XY factor shown here. "
        "Neither the availability change nor the local coordinate gain establishes fusion innovation.\n\n"
        "The geometry control is specific to root-unavailable but partly visible inputs. Training it may also change "
        "predictions on other frames, so gains on fully missing inputs cannot be ascribed to new observed geometry. "
        "These comparisons are within each architecture. They do not establish a particular fusion component's effect "
        "or novelty. No pure visible-centroid or old hip experiment is pooled. No predictions or model inference are "
        "recomputed here. Matching completed independent replay receipts are required for both groups; this plot additionally "
        "verifies source, raw-input controls and saved prediction metadata pairing.\n\n"
        f"Budget: {geometry['training']['epochs']} epochs, {geometry['training']['checkpoint']}; "
        f"training frames {geometry['train_frames']}; validation frames {geometry['validation_frames']}; "
        f"initialization seeds {seeds}; one fixed subject split. Short smoke runs verify execution only, not pose performance.\n",
        encoding="utf-8")
    paths.append(caption)
    if sha256(gs) != gd or sha256(cs) != cd:
        raise RuntimeError("Input summary changed during plotting")
    receipt = {"status": "complete", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "geometry_summary": str(gs), "geometry_summary_sha256": gd,
               "confidence_summary": str(cs), "confidence_summary_sha256": cd,
               "plot_script_sha256": sha256(__file__), "common_plot_script_sha256": sha256(Path(__file__).with_name("plot_fusion_results.py")),
               "methods": methods, "conditions": conditions, "initialization_seeds": seeds,
               "uncertainty": "Sample SD of seed-paired differences; no bootstrap or significance claim", "verification": verification,
               "outputs": [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in paths]}
    receipt_path = output / f"{args.prefix}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "receipt": str(receipt_path), "methods": methods}, ensure_ascii=False))


if __name__ == "__main__":
    main()
