"""Plot one completed fusion experiment, preserving source data and stage boundaries.

Usage: python research/plot_fusion_results.py --experiment artifacts/... --stage stage1
This reads saved metrics without recomputing predictions or loading models.
Representation comparison also checks saved frame/subject/target arrays for pairing.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("concat", "quality_gate", "precision", "joint_gate", "utility_gate", "anatomicalgraph")
GRAPH_METHODS = ("graph_self", "graph_anatomical", "graph_permuted")
SCALED_METHODS = ("scaled_joint_gate", "scaled_utility_gate", "scaled_utility_random")
BOOTSTRAP_SEED = 20261013
BOOTSTRAP_REPLICATES = 10000
METHOD_NAMES = {"rgb_single": "Video only", "rgb_matched": "Capacity-matched video", "radar_single": "Radar only",
                "late_average": "Equal late average", "joint_attention": "Joint attention", "utility_random": "Random-label gate",
                "concat": "Concatenation", "quality_gate": "Joint quality/disagreement", "precision": "Precision weighting",
                "joint_gate": "Joint gate", "utility_gate": "Utility gate", "anatomicalgraph": "Anatomical graph",
                "graph_self": "Self-only graph", "graph_anatomical": "Anatomical graph", "graph_permuted": "Permuted graph",
                "scaled_joint_gate": "Scaled joint gate", "scaled_utility_gate": "Scaled utility gate",
                "scaled_utility_random": "Scaled random-label gate"}
CONDITION_NAMES = {"clean": "Clean", "dark": "Brightness\n×0.18", "noisy_dark": "Dark +\nnoise",
                   "occlusion": "Occlusion", "video_missing": "Video\nmissing", "radar_missing": "Radar\nmissing",
                   "radar_sparse": "Sparse\nradar"}
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000",
          "#332288", "#882255", "#44AA99", "#999933")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", ">", "<", "h", "*")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing an empty source table")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def near(actual, expected, label):
    if not np.isfinite(actual) or not np.isclose(actual, expected, rtol=1e-9, atol=1e-9):
        raise ValueError(f"Saved aggregate disagrees with saved runs: {label}")


def extract(report, methods, conditions, reference, include_gain=True):
    """Validate saved aggregates and expose paired gains from a named reference.

    Copy existing concat intervals when available. For other references, derive a
    separately documented exploratory bootstrap from saved subject means.
    """
    runs = report["runs"]
    seeds = report["initialization_seeds"]
    subjects = list(map(str, report["validation_subjects"]))
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("At least two distinct completed seeds are needed for seed SD")
    if len(runs) != len(seeds) or [r["seed"] for r in runs] != seeds:
        raise ValueError("Completed runs do not match the declared initialization seeds")
    if any(r.get("status") != "complete" for r in runs):
        raise ValueError("An individual run is incomplete")
    aggregate = report["aggregate"]
    metrics, seed_rows, gain_rows, subject_rows, recipes = [], [], [], [], []
    score_conditions = report["score_conditions"]
    for method in methods:
        for condition in conditions:
            saved = aggregate["models"][method][condition]
            values = [r["models"][method]["metrics"][condition]["MPJPE_mm"] for r in runs]
            near(saved["MPJPE_mean_mm"], float(np.mean(values)), f"{method}/{condition}/mean")
            near(saved["seed_sd_mm"], float(np.std(values, ddof=1)), f"{method}/{condition}/SD")
            if set(saved["per_subject_seed_mean_mm"]) != set(subjects):
                raise ValueError("Saved subject means do not match the declared validation subjects")
            for subject in subjects:
                subject_value = float(np.mean([r["models"][method]["metrics"][condition]
                                               ["per_validation_subject_MPJPE_mm"][subject] for r in runs]))
                near(saved["per_subject_seed_mean_mm"][subject], subject_value,
                     f"{method}/{condition}/subject {subject}")
            metrics.append({"method": method, "condition": condition, "MPJPE_mean_mm": saved["MPJPE_mean_mm"],
                            "seed_sample_SD_mm": saved["seed_sd_mm"], "n_seeds": len(seeds)})
            for seed, value in zip(seeds, values):
                seed_rows.append({"method": method, "condition": condition, "initialization_seed": seed, "MPJPE_mm": value})
        if not include_gain:
            continue
        differences = {subject: float(np.mean([
            aggregate["models"][reference][c]["per_subject_seed_mean_mm"][subject] -
            aggregate["models"][method][c]["per_subject_seed_mean_mm"][subject]
            for c in score_conditions])) for subject in subjects}
        saved_comparisons = aggregate.get("comparisons_to_concat", {}) if reference == "concat" else {}
        comparison = saved_comparisons.get(method)
        if comparison is not None:
            if set(comparison["per_subject_score_gain_mm"]) != set(subjects):
                raise ValueError("Subject gains do not match the declared validation subjects")
            for subject in subjects:
                near(comparison["per_subject_score_gain_mm"][subject], differences[subject],
                     method + "/subject " + subject)
            near(comparison["score_gain_mm"], float(np.mean(list(differences.values()))), method + "/paired gain")
            low, high = comparison["exploratory_subject_bootstrap_95pct_mm"]
            recipe = {"method": method, "reference_method": reference, "origin": "copied from saved comparisons_to_concat",
                      "bootstrap_recipe": "Original experiment interval retained unchanged"}
        else:
            values = np.array([differences[s] for s in subjects], dtype=float)
            indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(subjects),
                                                                     size=(BOOTSTRAP_REPLICATES, len(subjects)))
            low, high = np.percentile(values[indices].mean(axis=1), [2.5, 97.5]).tolist()
            recipe = {"method": method, "reference_method": reference, "origin": "newly derived from saved subject means",
                      "bootstrap_recipe": "Subject resampling with replacement after averaging seeds and score conditions",
                      "random_seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
                      "percentiles": [2.5, 97.5], "subject_order": subjects, "rng": "numpy.default_rng/PCG64",
                      "seeds_resampled": False, "conditions_resampled": False}
        recipes.append(recipe)
        if not np.isfinite([low, high]).all() or low > high:
            raise ValueError("Invalid saved bootstrap interval")
        gain_rows.append({"method": method, "reference_method": reference,
                          "gain_vs_reference_mm": float(np.mean(list(differences.values()))),
                          "bootstrap_95pct_low_mm": low, "bootstrap_95pct_high_mm": high,
                          "n_validation_subjects": len(subjects), "n_initialization_seeds": len(seeds),
                          "score_conditions": ";".join(score_conditions),
                          "interval_origin": recipe["origin"],
                          "interpretation": "Exploratory subject bootstrap; one split; not significance"})
        for subject in subjects:
            subject_rows.append({"method": method, "reference_method": reference, "validation_subject": subject,
                                 "paired_score_gain_vs_reference_mm": differences[subject],
                                 "score_conditions": ";".join(score_conditions)})
    return metrics, seed_rows, gain_rows, subject_rows, recipes


def save_figure(fig, output, stem):
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_mpjpe(rows, methods, conditions, output, prefix):
    fig, ax = plt.subplots(figsize=(max(8.5, 1.5 * len(conditions)), 5))
    x = np.arange(len(conditions), dtype=float)
    offsets = np.linspace(-.30, .30, len(methods)) if len(methods) > 1 else [0.]
    lookup = {(r["method"], r["condition"]): r for r in rows}
    for index, (method, offset) in enumerate(zip(methods, offsets)):
        values = [lookup[method, c]["MPJPE_mean_mm"] for c in conditions]
        spread = [lookup[method, c]["seed_sample_SD_mm"] for c in conditions]
        ax.errorbar(x + offset, values, yerr=spread, fmt=MARKERS[index % len(MARKERS)], markersize=4.8, capsize=2.5,
                    elinewidth=1., color=COLORS[index % len(COLORS)], label=METHOD_NAMES.get(method, method))
    ax.set_xticks(x, [CONDITION_NAMES.get(c, c.replace("_", " ")) for c in conditions])
    ax.set_xlabel("Evaluation input condition")
    ax.set_ylabel("MPJPE / mm (lower is better)")
    ax.set_ylim(bottom=0.)
    ax.set_xlim(-.55, len(conditions) - .45)
    ax.grid(axis="y", alpha=.22)
    legend_columns = 4 if len(methods) > 8 else 3
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.17), ncol=legend_columns, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=.22)
    fig.text(.5, .01, "Points: seed means; error bars: sample SD across initialization seeds", ha="center", fontsize=9)
    return save_figure(fig, output, prefix + "_mpjpe_conditions")


def plot_gain(gain_rows, subject_rows, methods, reference, output, prefix):
    selected = [m for m in methods if m != reference]
    fig, ax = plt.subplots(figsize=(max(8.5, 1.45 * len(selected)), 5))
    lookup = {r["method"]: r for r in gain_rows}
    for index, method in enumerate(selected):
        row = lookup[method]
        low, high = row["bootstrap_95pct_low_mm"], row["bootstrap_95pct_high_mm"]
        color = COLORS[methods.index(method) % len(COLORS)]
        ax.vlines(index, low, high, color=color, linewidth=1.5)
        ax.hlines([low, high], index-.06, index+.06, color=color, linewidth=1.)
        values = [s["paired_score_gain_vs_reference_mm"] for s in subject_rows if s["method"] == method]
        ax.scatter(index + np.linspace(-.17, .17, len(values)), values, facecolors="none", edgecolors=color,
                   alpha=.7, s=24., linewidths=.8)
        ax.scatter([index], [row["gain_vs_reference_mm"]], c=[color], marker="D", s=42., zorder=3)
    ax.axhline(0., color="#444444", linewidth=1., linestyle="--")
    labels = [textwrap.fill(METHOD_NAMES.get(m, m).replace("quality/disagreement", "quality / disagreement"), width=17)
              for m in selected]
    ax.set_xticks(np.arange(len(selected)), labels)
    ax.set_xlabel("Fusion method; common saved selection conditions")
    ax.set_ylabel(f"Paired gain over {METHOD_NAMES.get(reference, reference).lower()} / mm\n(higher is better)")
    ax.grid(axis="y", alpha=.22)
    n_subjects = gain_rows[0]["n_validation_subjects"]
    fig.subplots_adjust(bottom=.22)
    fig.text(.5, .01, f"Diamond: paired mean; circles: subjects; 95% subject bootstrap, n={n_subjects}, exploratory", ha="center", fontsize=9)
    return save_figure(fig, output, prefix + "_paired_gain")


def compare_representations(centroid_source, centroid, hip_directory, methods, conditions, output, prefix):
    """Compare matched representations as a separate factor, never as algorithm gain."""
    hip_directory = hip_directory.resolve()
    hip_source = hip_directory / "run_summary.json"
    hip_digest = sha256(hip_source)
    hip = json.loads(hip_source.read_text(encoding="utf-8"))
    if hip.get("status") != "complete" or centroid.get("status") != "complete":
        raise ValueError("Both representation experiments must be complete")
    if hip.get("representation") != "hip" or centroid.get("representation") != "visible_centroid":
        raise ValueError("Representation comparison requires recorded hip and visible_centroid experiments")
    common_keys = ("cache_sha256", "input_audit_sha256", "data_split_seed", "train_subjects", "validation_subjects",
                   "train_frames", "validation_frames", "initialization_seeds", "evaluation_conditions", "score_conditions",
                   "training", "environment", "source_sha256", "runner_file", "model_factory")
    for key in common_keys:
        if hip.get(key) != centroid.get(key):
            raise ValueError(f"Representation comparison is not controlled: {key} differs")
    if not set(methods) <= set(hip["methods"]):
        raise ValueError("Requested comparison methods are absent from the hip experiment")
    if any(hip["parameters"][method] != centroid["parameters"][method] for method in methods):
        raise ValueError("Model parameters differ between representations")
    hip_runs = {run["seed"]: run for run in hip["runs"]}
    centroid_runs = {run["seed"]: run for run in centroid["runs"]}
    seeds = centroid["initialization_seeds"]
    if set(hip_runs) != set(seeds) or set(centroid_runs) != set(seeds):
        raise ValueError("Completed representation seeds are not paired")
    checks = []
    for method in methods:
        for seed in seeds:
            hip_file = hip_directory / f"seed{seed}" / f"{method}_predictions.npz"
            centroid_file = centroid_source.parent / f"seed{seed}" / f"{method}_predictions.npz"
            with np.load(hip_file, allow_pickle=False) as a, np.load(centroid_file, allow_pickle=False) as b:
                for field in ("subject_ids", "frame_ids", "target_mm"):
                    if not np.array_equal(a[field], b[field]):
                        raise ValueError(f"Unpaired representation predictions: {method}/seed{seed}/{field}")
            checks.append({"method": method, "seed": seed, "frame_subject_target_exact_match": True,
                           "hip_predictions_sha256": sha256(hip_file), "centroid_predictions_sha256": sha256(centroid_file)})
    rows, seed_rows = [], []
    for method in methods:
        for condition in conditions:
            values = []
            for seed in seeds:
                hip_value = hip_runs[seed]["models"][method]["metrics"][condition]["MPJPE_mm"]
                centroid_value = centroid_runs[seed]["models"][method]["metrics"][condition]["MPJPE_mm"]
                values.append(hip_value - centroid_value)
                seed_rows.append({"method": method, "condition": condition, "initialization_seed": seed,
                                  "hip_MPJPE_mm": hip_value, "visible_centroid_MPJPE_mm": centroid_value,
                                  "representation_gain_hip_minus_centroid_mm": hip_value - centroid_value})
            rows.append({"method": method, "condition": condition, "representation_gain_hip_minus_centroid_mm": float(np.mean(values)),
                         "paired_seed_sample_SD_mm": float(np.std(values, ddof=1)), "n_paired_seeds": len(seeds),
                         "interpretation": "Independent representation/availability change; not fusion mechanism gain"})
    reference_check = None
    if "radar_single" in set(hip["methods"]) & set(centroid["methods"]):
        differences = [hip_runs[s]["models"]["radar_single"]["metrics"][c]["MPJPE_mm"] -
                       centroid_runs[s]["models"]["radar_single"]["metrics"][c]["MPJPE_mm"]
                       for s in seeds for c in conditions]
        reference_check = {"max_abs_independent_radar_metric_change_mm": float(np.max(np.abs(differences))),
                           "within_1e-7_mm": bool(np.max(np.abs(differences)) <= 1e-7),
                           "interpretation": "Negative control: hip-reference radar augmentation should preserve independent radar training"}
    pair_output = output / "representation_comparison"
    pair_output.mkdir(exist_ok=True)
    paths = []
    for suffix, records in (("source", rows), ("seed_source", seed_rows)):
        path = pair_output / f"{prefix}_representation_{suffix}.csv"
        write_csv(path, records)
        paths.append(path)
    fig, ax = plt.subplots(figsize=(max(8.5, 1.5 * len(conditions)), 5))
    lookup = {(r["method"], r["condition"]): r for r in rows}
    x = np.arange(len(conditions), dtype=float)
    offsets = np.linspace(-.30, .30, len(methods))
    for index, (method, offset) in enumerate(zip(methods, offsets)):
        ax.errorbar(x+offset, [lookup[method, c]["representation_gain_hip_minus_centroid_mm"] for c in conditions],
                    yerr=[lookup[method, c]["paired_seed_sample_SD_mm"] for c in conditions],
                    fmt=MARKERS[index % len(MARKERS)], markersize=4.8,
                    capsize=2.5, elinewidth=1., color=COLORS[index % len(COLORS)], label=METHOD_NAMES.get(method, method))
    ax.axhline(0., color="#444444", linewidth=1., linestyle="--")
    ax.set_xticks(x, [CONDITION_NAMES.get(c, c) for c in conditions])
    ax.set_xlabel("Evaluation input condition; paired cache, subjects and seeds")
    ax.set_ylabel("MPJPE reduction from representation change / mm\n(hip minus visible centroid)")
    ax.grid(axis="y", alpha=.22)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.17), ncol=4 if len(methods) > 8 else 3,
              frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=.22)
    fig.text(.5, .01, "Independent representation/availability factor; paired seed mean ± sample SD; not mechanism gain", ha="center", fontsize=9)
    paths += save_figure(fig, pair_output, prefix + "_representation_change")
    caption = pair_output / f"{prefix}_representation_caption.md"
    caption.write_text(
        "# Independent representation comparison\n\n"
        f"Hip source: {hip_source}. Visible-centroid source: {centroid_source}.\n\n"
        "Dots show paired seed means of hip MPJPE minus visible-centroid MPJPE; positive values favor visible centroid. "
        "Error bars show sample SD of the seed-paired differences, not unpaired SD and not a subject confidence interval. "
        "Both groups use the same cache, frame/subject/target arrays, seeds, model parameters, training budget, backend and source code. "
        "Only the visual representation differs. Centroid retains some hip-unavailable predictions, changes centering even when hip is present, "
        "and can change availability/fallback and valid training-loss populations. This figure therefore describes a representation experiment; "
        "it does not establish a fusion mechanism improvement. Formal mechanism comparisons remain within each representation.\n\n"
        "No old 64-frame CPU result is combined with the 96-frame GPU comparison. No model predictions are recomputed. "
        "No new bootstrap or significance claim is introduced.\n", encoding="utf-8")
    paths.append(caption)
    if sha256(hip_source) != hip_digest:
        raise RuntimeError("Hip summary changed during paired plotting")
    receipt = {"hip_summary": str(hip_source), "hip_summary_sha256": hip_digest,
               "centroid_summary": str(centroid_source), "centroid_summary_sha256": sha256(centroid_source),
               "controlled_keys_checked": list(common_keys), "paired_prediction_metadata": checks,
               "independent_radar_negative_control": reference_check, "methods": methods, "conditions": conditions,
               "interpretation": "Representation/availability change only; not algorithm gain", "plot_script_sha256": sha256(__file__),
               "outputs": [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in paths]}
    receipt_path = pair_output / f"{prefix}_representation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths + [receipt_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--stage", default=None, help="Caption-only stage label; stages are never merged")
    parser.add_argument("--prefix", default="fig_fusion")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--reference-method", help="Paired reference within this experiment; defaults to concat, graph_self or scaled_joint_gate")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--compare-hip", type=Path, help="Optional matched hip experiment; current experiment must be visible_centroid")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix):
        raise ValueError("Prefix must be a plain alphanumeric/underscore/hyphen filename")
    experiment = args.experiment.resolve()
    source = experiment / "run_summary.json"
    source_digest = sha256(source)
    report = json.loads(source.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or "aggregate" not in report:
        raise ValueError("Only a completed experiment with saved aggregate can be plotted")
    if report.get("final_test_subjects_used"):
        raise ValueError("This script is intended for development experiments; final test needs a separate caption protocol")
    available = set(report["methods"])
    if available <= set(GRAPH_METHODS):
        default_methods, default_reference = GRAPH_METHODS, "graph_self"
        if len({report["parameters"][method] for method in report["methods"]}) != 1:
            raise ValueError("Graph adjacency controls must have equal parameter counts")
    elif available <= set(SCALED_METHODS):
        default_methods, default_reference = SCALED_METHODS, "scaled_joint_gate"
    else:
        default_methods, default_reference = METHODS, "concat"
    reference = args.reference_method or default_reference
    methods = args.methods or [m for m in default_methods if m in available]
    # Show all seven GPU evaluation conditions; selection-score gain remains the declared five-condition score.
    conditions = args.conditions or report["evaluation_conditions"]
    if reference not in methods or len(methods) < 2 or len(methods) != len(set(methods)):
        raise ValueError("Choose the reference and at least one distinct available comparison method")
    if not set(methods) <= set(report["methods"]) or not set(conditions) <= set(report["evaluation_conditions"]):
        raise ValueError("Requested method/condition not present in this experiment")
    if len(conditions) != len(set(conditions)):
        raise ValueError("Plot conditions must be distinct")
    metrics, seed_rows, gains, subjects, recipes = extract(report, methods, conditions, reference)
    output = experiment / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10., "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "savefig.dpi": 300})
    outputs = []
    for suffix, rows in (("condition_source", metrics), ("seed_source", seed_rows),
                         ("gain_source", gains), ("subject_source", subjects)):
        path = output / f"{args.prefix}_{suffix}.csv"
        write_csv(path, rows)
        outputs.append(path)
    outputs += plot_mpjpe(metrics, methods, conditions, output, args.prefix)
    outputs += plot_gain(gains, subjects, methods, reference, output, args.prefix)
    stage = args.stage or experiment.name
    scaled_caption = (
        "This scaled-gate stage changes fixed units of the existing gate evidence inputs. "
        "It examines input conditioning and optimization within existing joint/utility/random-label gates, "
        "not a new gating mechanism. Scaling constants are recorded in the model source snapshot. "
        "Only methods within this completed 96-frame GPU stage are shown; older 64-frame CPU stages are not pooled.\n\n"
        if set(methods) <= set(SCALED_METHODS) else "")
    partial_caption = (
        "This is a hip-preserving partial-input control, not the pure visible-centroid representation. "
        "Hip-valid raw visual inputs remain exactly the original hip features; hip-unavailable but partly visible inputs "
        "retain raw confidence and availability, with either centroid-centered partial XY or confidence-only raw XY=0. "
        "The two partial controls share original hip-clean normalizers, confidence and visible masks. "
        "Raw-zero observed XY generally normalizes to fixed -hip_mean/std. Training weights can change, so predictions "
        "on otherwise identical hip-valid or fully missing inputs can change. Within-stage model differences describe "
        "the complete compared architectures. The later partial-to-partial comparison isolates local XY under shared "
        "availability; its gain is not evidence of a new fusion mechanism.\n\n"
        if report.get("representation") in ("hip_centroid_fallback", "hip_confidence_only") else "")
    caption = output / f"{args.prefix}_captions.md"
    caption.write_text(
        f"# Figure captions and interpretation\n\nExperiment/stage: {stage}. Source: {source}.\n\n"
        f"The experiment uses {report['train_frames']} training frames, {report['validation_frames']} validation frames, "
        f"{len(report['validation_subjects'])} validation subjects and {len(report['initialization_seeds'])} initialization seeds "
        "on one fixed subject split. Each figure is a separate artifact; no other stage is pooled.\n\n"
        f"Training budget: {report['training']['epochs']} epochs, {report['training']['checkpoint']}. "
        f"Recorded representation: {report.get('representation', 'not recorded in older run')}. "
        f"Compute device: {report.get('environment', {}).get('device', 'not recorded')}.\n\n"
        "MPJPE condition figure: dots are the saved mean MPJPE in mm; error bars are sample SD across initialization seeds "
        f"({report['initialization_seeds']}). SD describes initialization variability, not population uncertainty. "
        f"Shown conditions: {', '.join(conditions)}.\n\n"
        f"Paired gain figure: positive values favor the displayed method over {METHOD_NAMES.get(reference, reference)} ({reference}). "
        "Each subject's error is averaged over seeds and the saved selection conditions, then differenced against the reference "
        "on that same subject. Diamonds show the mean paired gain, open circles show individual subject gains, and bars show "
        "the 95% subject-bootstrap percentile interval. Existing concatenation intervals are copied unchanged; missing or "
        f"non-concatenation intervals are newly derived from saved subject means using {BOOTSTRAP_REPLICATES} replacement "
        f"resamples and numpy.default_rng seed {BOOTSTRAP_SEED}. They are recorded separately in the receipt and do not alter "
        "the experiment summary. The small subject count makes these exploratory intervals; they are not "
        f"formal significance claims. Selection conditions: {', '.join(report['score_conditions'])}.\n\n"
        f"Recorded graph control: {report.get('graph_control', 'not applicable')}. "
        f"Parameters per shown method: {', '.join(m + '=' + str(report['parameters'][m]) for m in methods)}. "
        "When graph controls are present, this compares fixed adjacency within one graph residual architecture; "
        "it does not compare a deeper graph head with an older smaller residual head.\n\n"
        + scaled_caption + partial_caption +
        "These are local research adaptations, not official reproductions of published methods. Neither diagram establishes "
        "novelty or deployed robustness. Stage differences in cache, training samples, conditions or methods prevent direct "
        "fair stage-to-stage comparison.\n", encoding="utf-8")
    outputs.append(caption)
    if args.compare_hip is not None:
        outputs += compare_representations(source, report, args.compare_hip, methods, conditions, output, args.prefix)
    if sha256(source) != source_digest:
        raise RuntimeError("Input summary changed while plotting; rerun after the experiment is stable")
    receipt = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
               "input_summary": str(source), "input_sha256": source_digest, "input_status": report["status"],
               "plot_script_sha256": sha256(Path(__file__)), "methods": methods, "reference_method": reference, "conditions": conditions,
               "gain_score_conditions": report["score_conditions"], "validation_subjects": report["validation_subjects"],
               "initialization_seeds": report["initialization_seeds"], "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
               "representation": report.get("representation", "not recorded in older experiment"), "environment": report.get("environment"),
               "representation_reference": str(args.compare_hip.resolve()) if args.compare_hip else None,
               "uncertainty": {"MPJPE": "sample SD across seeds", "gain": "exploratory subject-bootstrap 95% interval"},
               "paired_gain_recipes": recipes, "parameters": {m: report["parameters"][m] for m in methods},
               "graph_control": report.get("graph_control"),
               "verification": "Source immutable; means/SD/gains checked against saved run and subject metrics; no predictions recomputed",
               "formats": {"PNG": "300 dpi", "PDF": "vector with embedded fonts", "SVG": "editable text and vector geometry"},
               "outputs": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in outputs]}
    receipt_path = output / f"{args.prefix}_plot_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "figures": str(output), "methods": methods, "outputs": len(outputs),
                      "receipt": str(receipt_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
