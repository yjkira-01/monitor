"""Fixed-budget native-paper baseline and ordinary accuracy controls."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps/research_cuda"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from validate_mri_point_baselines import error_metrics
from xfi_accuracy_models import METHODS, recenter_prediction, token_prediction, training_loss
from xfi_paper_baseline import SOURCE_FILES, OFFICIAL, XFiRGBRadar, source_manifest

CONDITIONS = ("clean", "dark", "noisy_dark", "occlusion")
EVALUATION = CONDITIONS + ("video_missing", "radar_missing", "radar_sparse")
SCORE = ("clean", "noisy_dark", "occlusion", "video_missing", "radar_missing")
SOURCES = ("run_xfi_accuracy_experiments.py", "xfi_accuracy_models.py", "xfi_paper_baseline.py")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_tokens(cache, cache_report, subjects, device):
    expected = {row["subject"]: row["tokens_sha256"] for row in cache_report["subjects"]}
    rgb, radar, sparse, available, targets, frames, subject_ids = [[] for _ in range(7)]
    for subject in subjects:
        path = cache / f"subject{subject}_tokens.npz"
        if sha(path) != expected[subject]:
            raise ValueError("Frozen native token cache changed")
        with np.load(path, allow_pickle=False) as saved:
            rgb.append(torch.tensor(np.stack([saved[c + "_rgb_CxT"] for c in CONDITIONS]), device=device))
            radar.append(torch.tensor(saved["radar_tokens"], device=device))
            sparse.append(torch.tensor(saved["radar_sparse_tokens"], device=device))
            available.append(torch.tensor(saved["radar_available"], dtype=torch.bool, device=device))
            targets.append(saved["target_mm"].copy())
            frames.append(saved["frame_ids"].copy())
            subject_ids.append(saved["subject_ids"].copy())
    result = {"rgb": torch.cat(rgb, dim=1), "radar": torch.cat(radar), "radar_sparse": torch.cat(sparse),
              "radar_available": torch.cat(available), "target_mm": np.concatenate(targets),
              "frame_ids": np.concatenate(frames), "subject_ids": np.concatenate(subject_ids)}
    result["target_m"] = torch.tensor(result["target_mm"].reshape(-1, 17, 3) / 1000, dtype=torch.float32, device=device)
    return result


def new_model(seed, backbone_arrays, device):
    torch.manual_seed(seed)
    model = XFiRGBRadar().to(device)
    state = model.state_dict()
    expected = {key for key in state if key.startswith(("rgb_extractor.", "radar_extractor.")) or key == "training_prior_m"}
    if set(backbone_arrays) != expected:
        raise ValueError("Calibrated native backbone state is incomplete")
    state.update({key: torch.from_numpy(value.copy()).to(device) for key, value in backbone_arrays.items()})
    model.load_state_dict(state, strict=True)
    model.freeze_backbones()
    return model


def schedule_digest(digest, indices, condition, subset):
    for value in (indices, condition):
        digest.update(value.numpy().tobytes())
    digest.update(bytes([subset]))


@torch.no_grad()
def evaluate(model, data, method):
    metrics, saved = {}, {"target_mm": data["target_mm"], "subject_ids": data["subject_ids"], "frame_ids": data["frame_ids"]}
    for condition in EVALUATION:
        source_condition = CONDITIONS.index(condition) if condition in CONDITIONS else 0
        point_tokens = data["radar_sparse"] if condition == "radar_sparse" else data["radar"]
        predictions, video_predictions, radar_predictions, offset = [], [], [], []
        for start in range(0, len(data["target_mm"]), 16):
            rgb = data["rgb"][source_condition, start:start + 16]
            radar = point_tokens[start:start + 16]
            image_valid = torch.full((len(rgb),), condition != "video_missing", device=rgb.device, dtype=torch.bool)
            point_valid = data["radar_available"][start:start + 16] & (condition != "radar_missing")
            root = method == "root_mpjpe"
            out = token_prediction(model, rgb, radar, image_valid, point_valid, root=root)
            video = token_prediction(model, rgb, radar, torch.ones_like(image_valid), torch.zeros_like(point_valid), root=root)
            ronly = token_prediction(model, rgb, radar, torch.zeros_like(image_valid), data["radar_available"][start:start + 16], root=root)
            for collection, value in ((predictions, out), (video_predictions, video), (radar_predictions, ronly)):
                collection.append(value.cpu().numpy().astype(np.float64).reshape(-1, 51) * 1000)
            offset.extend((((out[:, 11] + out[:, 12]) / 2).norm(dim=-1) * 1000).cpu().numpy().tolist())
        prediction, video, radar = [np.concatenate(rows) for rows in (predictions, video_predictions, radar_predictions)]
        primary = error_metrics(prediction, data["target_mm"], data["subject_ids"])
        pose = prediction.reshape(-1, 17, 3)
        centered = (pose - ((pose[:, 11] + pose[:, 12]) / 2)[:, None]).reshape(-1, 51)
        centered_metrics = error_metrics(centered, data["target_mm"], data["subject_ids"])
        primary["prediction_hip_offset_mean_mm"] = float(np.mean(offset))
        primary["prediction_recentered_MPJPE_mm"] = centered_metrics["MPJPE_mm"]
        primary["prediction_recentered_per_subject_mm"] = centered_metrics["per_validation_subject_MPJPE_mm"]
        primary["image_available_frames"] = len(prediction) if condition != "video_missing" else 0
        primary["radar_available_frames"] = int(data["radar_available"].sum()) if condition != "radar_missing" else 0
        metrics[condition] = primary
        for name, value in (("fused", prediction), ("video", video), ("radar", radar), ("recentered", centered)):
            saved[condition + "_" + name + "_prediction_mm"] = value
    return metrics, saved


def aggregate(runs, methods, subjects):
    result = {"models": {}, "paired_comparisons": {}}
    for method in methods:
        rows = [run["models"][method] for run in runs]
        combined = {}
        for condition in EVALUATION:
            values = [row["metrics"][condition] for row in rows]
            combined[condition] = {"MPJPE_mean_mm": float(np.mean([v["MPJPE_mm"] for v in values])),
                                   "seed_sd_mm": float(np.std([v["MPJPE_mm"] for v in values], ddof=1)),
                                   "prediction_recentered_MPJPE_mean_mm": float(np.mean([v["prediction_recentered_MPJPE_mm"] for v in values])),
                                   "prediction_recentered_per_subject_seed_mean_mm": {str(s): float(np.mean([v["prediction_recentered_per_subject_mm"][str(s)] for v in values])) for s in subjects},
                                   "per_subject_seed_mean_mm": {str(s): float(np.mean([v["per_validation_subject_MPJPE_mm"][str(s)] for v in values])) for s in subjects}}
        scores = [float(np.mean([row["metrics"][c]["per_validation_subject_MPJPE_mm"][str(s)] for c in SCORE for s in subjects])) for row in rows]
        centered_scores = [float(np.mean([row["metrics"][c]["prediction_recentered_per_subject_mm"][str(s)] for c in SCORE for s in subjects])) for row in rows]
        combined["selection_score"] = {"conditions": list(SCORE), "mean_mm": float(np.mean(scores)), "per_seed_mm": scores,
                                       "seed_sd_mm": float(np.std(scores, ddof=1))}
        combined["posthoc_recentered_selection_score"] = {"mean_mm": float(np.mean(centered_scores)), "per_seed_mm": centered_scores}
        result["models"][method] = combined
    if "native_mse" in methods:
        base = result["models"]["native_mse"]
        for method in methods:
            candidate = result["models"][method]
            gain = base["selection_score"]["mean_mm"] - candidate["selection_score"]["mean_mm"]
            subject_gain = {str(s): float(np.mean([base[c]["per_subject_seed_mean_mm"][str(s)] - candidate[c]["per_subject_seed_mean_mm"][str(s)] for c in SCORE])) for s in subjects}
            seed_gain = np.asarray(base["selection_score"]["per_seed_mm"]) - candidate["selection_score"]["per_seed_mm"]
            clean_change = candidate["clean"]["MPJPE_mean_mm"] - base["clean"]["MPJPE_mean_mm"]
            result["paired_comparisons"][method] = {"score_gain_vs_native_mse_mm": gain,
                    "clean_change_mm": clean_change, "per_subject_score_gain_mm": subject_gain,
                    "per_seed_score_gain_mm": seed_gain.tolist(),
                    "score_gain_vs_native_mse_posthoc_recenter_mm": base["posthoc_recentered_selection_score"]["mean_mm"] - candidate["selection_score"]["mean_mm"],
                    "score_gain_vs_native_mpjpe_posthoc_recenter_mm": (result["models"]["native_mpjpe"]["posthoc_recentered_selection_score"]["mean_mm"] - candidate["selection_score"]["mean_mm"]) if "native_mpjpe" in methods else None,
                    "resource_gate_passed": bool(gain >= max(1., .02 * base["selection_score"]["mean_mm"]) and
                                                 clean_change <= max(1., .02 * base["clean"]["MPJPE_mean_mm"]) and
                                                 sum(v > 0 for v in subject_gain.values()) >= 3 and int((seed_gain > 0).sum()) >= 2),
                    "scope": "Development allocation only, one subject split; also inspect posthoc controls"}
        if "root_mpjpe" in methods and "native_mpjpe" in methods:
            reference_method = min(("native_mse", "native_mpjpe"), key=lambda m: result["models"][m]["posthoc_recentered_selection_score"]["mean_mm"])
            reference, candidate = result["models"][reference_method], result["models"]["root_mpjpe"]
            reference_score = reference["posthoc_recentered_selection_score"]["mean_mm"]
            gain = reference_score - candidate["selection_score"]["mean_mm"]
            clean_change = candidate["clean"]["MPJPE_mean_mm"] - reference["clean"]["prediction_recentered_MPJPE_mean_mm"]
            subject_gain = {str(s): float(np.mean([reference[c]["prediction_recentered_per_subject_seed_mean_mm"][str(s)] - candidate[c]["per_subject_seed_mean_mm"][str(s)] for c in SCORE])) for s in subjects}
            seed_gain = np.asarray(reference["posthoc_recentered_selection_score"]["per_seed_mm"]) - candidate["selection_score"]["per_seed_mm"]
            result["root_training_vs_best_posthoc_control"] = {"reference": reference_method + "+prediction_hip_recenter",
                    "score_gain_mm": gain, "clean_change_mm": clean_change, "per_subject_score_gain_mm": subject_gain,
                    "per_seed_score_gain_mm": seed_gain.tolist(),
                    "resource_gate_passed": bool(gain >= max(1., .02 * reference_score) and
                                                 clean_change <= max(1., .02 * reference["clean"]["prediction_recentered_MPJPE_mean_mm"]) and
                                                 sum(v > 0 for v in subject_gain.values()) >= 3 and int((seed_gain > 0).sum()) >= 2),
                    "scope": "Best of two predefined ordinary controls on development S; no significance or novelty claim"}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        raise ValueError("New output and at least two distinct initialization seeds required")
    if not 1 <= args.epochs <= 100 or (not args.smoke and (args.epochs != 100 or args.seeds != [7, 17, 27])):
        raise ValueError("Formal budget is three specified initialization seeds and 100 epochs")
    report_path = args.cache / "cache_report.json"
    cache_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (cache_report["status"] != "complete" or cache_report["official_source_sha256"] != source_manifest()
            or cache_report["protocol_sha256"] != sha(ROOT / "research/xfi_accuracy_protocol_20261013.md")
            or cache_report["calibration_subjects"] != cache_report["train_subjects"]
            or cache_report["backbone_initialization"]["BN_calibration_frames"] != 1152
            or not cache_report["calibration_backbone_parameters_unchanged"]):
        raise ValueError("Complete native training-only calibrated cache required")
    for name in ("xfi_accuracy_models.py", "xfi_paper_baseline.py"):
        if cache_report["source_sha256"][name] != sha(ROOT / "research" / name):
            raise ValueError("Native cache dependency changed")
    if sha(args.cache / "calibrated_backbones.npz") != cache_report["calibrated_backbones_sha256"]:
        raise ValueError("Frozen pretrained native parameters changed")
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train = load_tokens(args.cache, cache_report, cache_report["train_subjects"], device)
    validation = load_tokens(args.cache, cache_report, cache_report["validation_subjects"], device)
    with np.load(args.cache / "calibrated_backbones.npz", allow_pickle=False) as saved:
        backbone_arrays = {key: saved[key].copy() for key in saved.files}
    args.output.mkdir(parents=True)
    report = {"status": "running", "scope": "Execution smoke, no pose-performance claim" if args.smoke else "X-Fi source-preserving mRI adaptation and ordinary objective controls; development, no official reproduction",
              "cache": str(args.cache.resolve()), "cache_report_sha256": sha(report_path), "source_sha256": {name: sha(ROOT / "research" / name) for name in SOURCES},
              "official_source_sha256": source_manifest(), "protocol_sha256": cache_report["protocol_sha256"],
              "initialization_seeds": args.seeds, "train_subjects": cache_report["train_subjects"],
              "validation_subjects": cache_report["validation_subjects"], "final_test_subjects_used": [],
              "train_frames": len(train["target_mm"]), "validation_frames": len(validation["target_mm"]),
              "methods": args.methods, "training": {"epochs": args.epochs, "batch_size": 16, "checkpoint": "last epoch, no validation selection",
              "optimizer": "AdamW", "lr": .001, "weight_decay": .01, "trainable": "native projector and X_Fusion only",
              "conditions": list(CONDITIONS), "modal_subsets": "equal-probability batch-wise RGB-only, radar-only, both",
              "loss": "native_mse MSE; native_mpjpe/root_mpjpe mean Euclidean joint distance", "precision": "CUDA IEEE FP32; no AMP/TF32",
              "determinism": "warn_only native pool backward; explicit deterministic cuDNN"},
              "evaluation_conditions": list(EVALUATION), "score_conditions": list(SCORE), "runs": []}
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for name in SOURCES:
        (snapshot / name).write_bytes((ROOT / "research" / name).read_bytes())
    (snapshot / "xfi_accuracy_protocol_20261013.md").write_bytes((ROOT / "research/xfi_accuracy_protocol_20261013.md").read_bytes())
    for name in SOURCE_FILES:
        path = snapshot / "official" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((OFFICIAL / name).read_bytes())
    write_json(args.output / "run_summary.json", report)
    for seed in args.seeds:
        seed_dir = args.output / f"seed{seed}"
        seed_dir.mkdir()
        run = {"seed": seed, "models": {}}
        initial_reference, schedule_reference = None, None
        for method in args.methods:
            started = time.perf_counter()
            model = new_model(seed, backbone_arrays, device)
            initial = state_fingerprint(model)
            if initial_reference is None:
                initial_reference = initial
            if initial != initial_reference:
                raise ValueError("Objective controls have unequal initial states")
            optimizer = torch.optim.AdamW([{"params": model.linear_projector.parameters()}, {"params": model.X_Fusion_block.parameters()}], lr=.001, weight_decay=.01)
            order = torch.Generator().manual_seed(seed)
            augmentation = torch.Generator().manual_seed(seed + 10000)
            digest, history = hashlib.sha256(), []
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses = []
                for cpu_indices in torch.randperm(len(train["target_mm"]), generator=order).split(16):
                    condition = torch.randint(4, (len(cpu_indices),), generator=augmentation)
                    subset = int(torch.randint(3, (1,), generator=augmentation).item())
                    schedule_digest(digest, cpu_indices, condition, subset)
                    indices = cpu_indices.to(device)
                    rgb = train["rgb"][condition.to(device), indices]
                    radar = train["radar"][indices]
                    rgb_valid = torch.full((len(indices),), subset != 1, dtype=torch.bool, device=device)
                    radar_valid = train["radar_available"][indices] & (subset != 0)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = token_prediction(model, rgb, radar, rgb_valid, radar_valid, root=method == "root_mpjpe")
                    loss = training_loss(prediction, train["target_m"][indices], method)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite native paper-model training loss")
                    if loss.requires_grad:
                        loss.backward()
                        optimizer.step()
                    losses.append(loss.item())
                if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                    item = {"epoch": epoch, "training_loss": float(np.mean(losses)), "elapsed_seconds": time.perf_counter() - started}
                    history.append(item)
                    print(json.dumps({"seed": seed, "method": method, **item}), flush=True)
            schedule = digest.hexdigest()
            if schedule_reference is None:
                schedule_reference = schedule
            if schedule != schedule_reference:
                raise ValueError("Objective controls have unequal data or augmentation schedule")
            for name, value in model.state_dict().items():
                if name in backbone_arrays:
                    np.testing.assert_array_equal(value.detach().cpu().numpy(), backbone_arrays[name])
            model.eval()
            metrics, saved = evaluate(model, validation, method)
            if method == "root_mpjpe" and any(row["prediction_hip_offset_mean_mm"] > .0001 for row in metrics.values()):
                raise ValueError("Root constraint violated")
            np.savez_compressed(seed_dir / f"{method}_weights.npz", **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
            np.savez_compressed(seed_dir / f"{method}_predictions.npz", **saved)
            run["models"][method] = {"metrics": metrics, "history": history, "initial_state_sha256": initial,
                                      "schedule_sha256": schedule, "frozen_backbone_parameters_and_buffers_unchanged": True,
                                      "parameters": sum(p.numel() for p in model.parameters()),
                                      "seconds": time.perf_counter() - started,
                                      "training_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                      "weights_sha256": sha(seed_dir / f"{method}_weights.npz"),
                                      "predictions_sha256": sha(seed_dir / f"{method}_predictions.npz")}
            write_json(seed_dir / "training.json", run)
            print(json.dumps({"seed": seed, "method": method, "status": "complete", "clean_MPJPE_mm": metrics["clean"]["MPJPE_mm"],
                              "seconds": run["models"][method]["seconds"]}), flush=True)
            del optimizer, model
        report["runs"].append(run)
        write_json(args.output / "run_summary.json", report)
    report["aggregate"] = aggregate(report["runs"], args.methods, report["validation_subjects"])
    report["status"] = "complete"
    write_json(args.output / "run_summary.json", report)


if __name__ == "__main__":
    main()
