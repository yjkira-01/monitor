"""CPU preflight and saved-output pairing for fixed-unit gate controls."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_fusion_trials import (compare_tree, independent_metrics, load_verified_modules,
                                  rebuild_data, safe_npz, sha256, write_json)

METHOD_PAIRS = {"scaled_joint_gate": "joint_gate", "scaled_utility_gate": "utility_gate",
                "scaled_utility_random": "utility_random"}
FIXED_SCALE = np.array([10.] * 6 + [25.] * 3 + [1., 1., 25.], dtype=np.float32)


def completed_report(experiment):
    experiment = Path(experiment).resolve()
    path = experiment / "run_summary.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    receipt = json.loads((experiment / "analysis/verification.json").read_text(encoding="utf-8"))
    if (report.get("status") != "complete" or receipt.get("status") != "complete" or
            receipt["run_summary_sha256"] != sha256(path) or report["final_test_subjects_used"]):
        raise ValueError("Complete independently replayed development experiment required")
    return experiment, report, receipt


def compare_contracts(base, scaled):
    fields = ("input_cache", "data_split_seed", "train_subjects", "validation_subjects", "initialization_seeds",
              "train_frames", "validation_frames", "training", "environment", "source_sha256", "runner_file",
              "evaluation_conditions", "score_conditions", "cache_sha256", "input_audit_sha256", "input_audit_conditions")
    for field in fields:
        if base.get(field) != scaled.get(field):
            raise ValueError("Paired scale contract differs: " + field)
    if (base.get("representation") != "hip" or scaled.get("representation") != "hip" or
            base.get("rgb_normalization_key", "clean") != scaled.get("rgb_normalization_key", "clean")):
        raise ValueError("Expected paired hip representation and identical normalization source")
    if (base.get("model_factory", "fusion_models.FusionModel") != "fusion_models.FusionModel" or
            scaled.get("model_factory") != "fusion_scaled_gate_models.ScaledGateFusionModel"):
        raise ValueError("Unexpected base/scaled factories")
    if set(scaled["methods"]) != set(METHOD_PAIRS) or not set(METHOD_PAIRS.values()) <= set(base["methods"]):
        raise ValueError("Expected all three semantic scale pairs")
    for scaled_method, semantic_method in METHOD_PAIRS.items():
        if scaled["parameters"][scaled_method] != base["parameters"][semantic_method]:
            raise ValueError("Trainable parameter budget differs")
    if (base["environment"]["device"] != "cuda" or base["training"]["epochs"] != 80 or
            base["training"]["batch_size"] != 64 or base["initialization_seeds"] != [7, 17, 27] or
            base["train_frames"] != 1152 or base["validation_frames"] != 384):
        raise ValueError("Expected 96-frame CUDA fixed-budget control group")
    return list(fields)


def check_initialization(base_models, scaled_models, seeds, inputs, target_mean):
    checks = []
    for seed in seeds:
        same_scale_state, same_scale_output = None, None
        for scaled_method, semantic_method in METHOD_PAIRS.items():
            torch.manual_seed(seed)
            base = base_models.FusionModel(semantic_method)
            torch.manual_seed(seed)
            scaled = scaled_models.FusionModel(scaled_method)
            template = torch.tensor(target_mean.reshape(17, 3) / 1000, dtype=torch.float32)
            for model in (base, scaled):
                model.target_template_m.copy_(template)
                model.eval()
            base_state, scaled_state = base.state_dict(), scaled.state_dict()
            if not np.array_equal(scaled_state["gate.scale"].numpy(), FIXED_SCALE):
                raise ValueError("Scaled gate buffer does not match fixed unit control")
            canonical = {key.replace("gate.network.", "gate."): value for key, value in scaled_state.items()
                         if key != "gate.scale"}
            if set(canonical) != set(base_state) or any(not torch.equal(canonical[key], value) for key, value in base_state.items()):
                raise ValueError("Semantic base/scaled initial state differs")
            if scaled.method != semantic_method or sum(p.numel() for p in base.parameters()) != sum(p.numel() for p in scaled.parameters()):
                raise ValueError("Semantic loss or parameter count differs")
            with torch.no_grad():
                base_out, scaled_out = base(*inputs), scaled(*inputs)
            for key in ("video", "radar", "candidate", "video_valid", "radar_valid", "both"):
                if not torch.equal(base_out[key], scaled_out[key]):
                    raise ValueError("Initial shared branch/candidate or validity differs: " + key)
            if same_scale_state is None:
                same_scale_state, same_scale_output = scaled_state, scaled_out
            elif (set(same_scale_state) != set(scaled_state) or
                  any(not torch.equal(same_scale_state[key], value) for key, value in scaled_state.items()) or
                  any(not torch.equal(same_scale_output[key], scaled_out[key])
                      for key in ("video", "radar", "candidate", "gate", "fused", "both"))):
                raise ValueError("Same-scale joint/utility/random initial state or gate function differs")
            checks.append({"seed": seed, "scaled_method": scaled_method, "semantic_method": semantic_method,
                "mapped_state_array_count": len(canonical), "state_arrays_exact_equal": True,
                "shared_video_radar_candidate_validity_exact_equal": True,
                "both_models_parameter_count": sum(p.numel() for p in base.parameters()),
                "fixed_gate_scale": FIXED_SCALE.tolist(),
                "same_scale_reference_method": "scaled_joint_gate",
                "same_scale_initial_state_candidate_gate_fused_exact_equal": True,
                "initial_max_gate_difference": float((base_out["gate"] - scaled_out["gate"]).abs().max()),
                "gate_difference_expected": "Only the input-unit scaling changes initial gate logits; shared candidate is identical"})
    return checks


def audit(base_experiment, scaled_experiment=None, output=None):
    base_experiment, base_report, base_receipt = completed_report(base_experiment)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite scale-control analysis")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    source_receipt = {"verified_sources": {}, "array_checks": []}
    base_models, runner = load_verified_modules(base_experiment, base_report, source_receipt)
    factory_report = dict(base_report, model_factory="fusion_scaled_gate_models.ScaledGateFusionModel")
    scaled_models, _ = load_verified_modules(base_experiment, factory_report, {"verified_sources": {}})
    norm = safe_npz(base_experiment / "normalization.npz")
    _, _, _, values = rebuild_data(base_report, runner, norm, source_receipt)
    initialization = check_initialization(base_models, scaled_models, base_report["initialization_seeds"],
                                        values["clean"], norm["target_mean_mm"])
    result = {"status": "complete", "base_experiment": str(base_experiment), "source_sha256": sha256(Path(__file__)),
        "base_summary_sha256": sha256(base_experiment / "run_summary.json"),
        "base_replay_receipt_sha256": sha256(base_experiment / "analysis/verification.json"),
        "verification_device": "CPU; no training or GPU execution", "final_test_subjects_used": [],
        "fixed_scale": FIXED_SCALE.tolist(), "initialization_checks": initialization,
        "scope": "Preflight only; initialized states and candidate checks are not trained performance results",
        "interpretation": "Parameter arrays match across unit scales, while the initial gate function usually differs. Same-scale joint/utility/random initial functions match. Fixed units change parameterization, effective initialization and training gradients, not observed information or a new fusion mechanism"}
    if scaled_experiment is not None:
        scaled_experiment, scaled_report, scaled_receipt = completed_report(scaled_experiment)
        result.update(scaled_experiment=str(scaled_experiment),
            scaled_summary_sha256=sha256(scaled_experiment / "run_summary.json"),
            scaled_replay_receipt_sha256=sha256(scaled_experiment / "analysis/verification.json"),
            contract_fields_matched=compare_contracts(base_report, scaled_report),
            scope="Completed saved-prediction comparison after independent strict replay; exploratory paired initialization seeds")
        scaled_norm = safe_npz(scaled_experiment / "normalization.npz")
        if set(norm) != set(scaled_norm) or any(not np.array_equal(norm[key], scaled_norm[key]) for key in norm):
            raise ValueError("Paired training normalization differs")
        for name, experiment, report in (("base", base_experiment, base_report), ("scaled", scaled_experiment, scaled_report)):
            for filename, digest in report["source_sha256"].items():
                if sha256(experiment / "source_snapshot" / filename) != digest:
                    raise ValueError("Paired source snapshot differs: " + name + "/" + filename)
        manifests = [{(row["seed"], row["method"]): row for row in receipt["models"]}
                     for receipt in (base_receipt, scaled_receipt)]
        lookup = [{run["seed"]: run for run in report["runs"]} for report in (base_report, scaled_report)]
        pairs, metric_checks = [], []
        for scaled_method, semantic_method in METHOD_PAIRS.items():
            pair = {"scaled_method": scaled_method, "semantic_method": semantic_method, "runs": []}
            for seed in base_report["initialization_seeds"]:
                saved = []
                for index, (experiment, method) in enumerate(((base_experiment, semantic_method), (scaled_experiment, scaled_method))):
                    path = experiment / f"seed{seed}" / f"{method}_predictions.npz"
                    if sha256(path) != manifests[index][seed, method]["prediction_sha256"]:
                        raise ValueError("Previously replayed prediction changed")
                    saved.append(safe_npz(path))
                for key in ("target_mm", "subject_ids", "frame_ids"):
                    if saved[0][key].dtype != saved[1][key].dtype or not np.array_equal(saved[0][key], saved[1][key]):
                        raise ValueError("Unpaired saved observations: " + key)
                metrics = []
                for index, method in enumerate((semantic_method, scaled_method)):
                    condition_metrics = {}
                    for condition in base_report["evaluation_conditions"]:
                        value = independent_metrics(saved[index][condition + "_fused_prediction_mm"], saved[index]["target_mm"], saved[index]["subject_ids"])
                        recorded = lookup[index][seed]["models"][method]["metrics"][condition]
                        compare_tree(value, {key: recorded[key] for key in value},
                                     f"seed{seed}/{method}/{condition}", metric_checks)
                        condition_metrics[condition] = value
                    metrics.append(condition_metrics)
                by_condition = {condition: {
                    "base_MPJPE_mm": metrics[0][condition]["MPJPE_mm"], "scaled_MPJPE_mm": metrics[1][condition]["MPJPE_mm"],
                    "signed_gain_base_minus_scaled_mm": metrics[0][condition]["MPJPE_mm"] - metrics[1][condition]["MPJPE_mm"],
                    "per_subject_signed_gain_mm": {str(subject): metrics[0][condition]["per_validation_subject_MPJPE_mm"][str(subject)] -
                        metrics[1][condition]["per_validation_subject_MPJPE_mm"][str(subject)] for subject in base_report["validation_subjects"]}}
                    for condition in base_report["evaluation_conditions"]}
                score = [float(np.mean([metrics[index][c]["per_validation_subject_MPJPE_mm"][str(s)]
                         for c in base_report["score_conditions"] for s in base_report["validation_subjects"]])) for index in (0, 1)]
                pair["runs"].append({"seed": seed, "conditions": by_condition, "base_score_mm": score[0],
                                     "scaled_score_mm": score[1], "signed_score_gain_base_minus_scaled_mm": score[0] - score[1]})
            pair["aggregate_conditions"] = {c: {
                "signed_gain_seed_mean_mm": float(np.mean([run["conditions"][c]["signed_gain_base_minus_scaled_mm"] for run in pair["runs"]])),
                "paired_initialization_gain_sample_SD_mm": float(np.std([run["conditions"][c]["signed_gain_base_minus_scaled_mm"] for run in pair["runs"]], ddof=1)),
                "per_subject_seed_mean_gain_mm": {str(s): float(np.mean([run["conditions"][c]["per_subject_signed_gain_mm"][str(s)]
                    for run in pair["runs"]])) for s in base_report["validation_subjects"]}}
                for c in base_report["evaluation_conditions"]}
            pair["signed_score_gain_seed_mean_mm"] = float(np.mean([run["signed_score_gain_base_minus_scaled_mm"] for run in pair["runs"]]))
            pairs.append(pair)
        pair_lookup = {pair["scaled_method"]: pair for pair in pairs}
        utility_runs = pair_lookup["scaled_utility_gate"]["runs"]
        within_scale = {}
        for control in ("scaled_joint_gate", "scaled_utility_random"):
            control_runs = pair_lookup[control]["runs"]
            score_gain = [left["scaled_score_mm"] - right["scaled_score_mm"] for left, right in zip(control_runs, utility_runs)]
            within_scale[control] = {"signed_score_gain_control_minus_utility_seed_mean_mm": float(np.mean(score_gain)),
                "per_seed_signed_score_gain_mm": score_gain, "positive_initialization_seeds": int((np.asarray(score_gain) > 0).sum()),
                "signed_condition_gain_control_minus_utility_seed_mean_mm": {condition: float(np.mean([
                    left["conditions"][condition]["scaled_MPJPE_mm"] - right["conditions"][condition]["scaled_MPJPE_mm"]
                    for left, right in zip(control_runs, utility_runs)])) for condition in base_report["evaluation_conditions"]}}
        result.update(pairs=pairs, same_scale_utility_comparisons=within_scale, unchanged_primary_metric_check_count=len(metric_checks),
            max_primary_metric_difference_mm=max(item.get("max_absolute_difference", 0.) for item in metric_checks),
            independent_validation_subject_count=len(base_report["validation_subjects"]),
            initialization_seeds_are_not_subject_splits=True,
            signed_gain_convention="base minus scaled; positive favors scaled, negative favors base; retain both signs",
            within_scaled_risk_comparison="Use the scaled report's joint/utility/random same-scale comparisons; a scale gain does not establish the utility-label mechanism")
    output.mkdir(parents=True)
    (output / "auditor_source.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "scale_control_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--scaled", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.base, args.scaled, args.output)
    print(json.dumps({"status": result["status"], "initialization_pairs": len(result["initialization_checks"]),
                      "trained_performance_pairs": len(result.get("pairs", []))}), flush=True)


if __name__ == "__main__":
    main()
