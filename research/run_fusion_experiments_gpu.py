"""Exploratory controlled fusion experiments, retaining every selected frame.

The held-out final-test subjects are never evaluated here. Different cache sizes
are new experiments, with all compared methods retrained on the same new data.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

# Isolated, workspace-only CUDA package; the base environment remains intact.
_TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TASK_ROOT / ".deps/research_cuda"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from audit_mri import load_labels
from fusion_models import METHODS, FusionModel, fusion_loss
from fusion_graph_models import GRAPH_METHODS, GraphFusionModel
from fusion_scaled_gate_models import SCALED_METHODS, ScaledGateFusionModel
from fusion_pose_representation import visible_centroid_features
from train_mri_point_baselines import ROOT, load_development, point_normalization
from validate_mri_point_baselines import error_metrics, write_json


def normalize_inputs(features, points, mask, statistics):
    """Read raw confidences before scaling; zero every missing normalized field."""
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 51 or not np.isfinite(features).all():
        raise ValueError("Expected finite 34 XY + 17 raw confidence features")
    confidence = features[:, 34:].copy().astype(np.float32)
    visible = confidence >= .3
    observed_fields = np.concatenate((np.repeat(visible, 2, -1), visible), -1)
    rgb = (features - statistics["rgb_mean"]) / statistics["rgb_std"]
    rgb = np.where(observed_fields, rgb, 0.).astype(np.float32)
    normalized = np.zeros_like(points, dtype=np.float32)
    normalized[mask] = (points[mask] - statistics["point_mean"]) / statistics["point_std"]
    return tuple(torch.from_numpy(x) for x in (rgb, normalized, mask.copy(), confidence))


def sparse_mask(mask, seed=20261013):
    rng = np.random.default_rng(seed)
    reduced = mask & (rng.random(mask.shape) < .2)
    for index in np.flatnonzero(mask.any(1) & ~reduced.any(1)):
        reduced[index, np.flatnonzero(mask[index])[0]] = True
    return reduced


def load_data(cache, conditions, representation="hip"):
    split, rows = load_development(cache)
    with zipfile.ZipFile(ROOT / "dataset_release.zip") as archive:
        for subject, row in rows.items():
            focal = None
            if representation == "visible_centroid":
                labels = load_labels(archive.read(f"dataset_release/aligned_data/pose_labels/subject{subject}_all_labels.cpl"))
                focal = np.diag(labels["camera_matrix"]["cam1_intrinsic"])[:2]
            with np.load(cache / f"subject{subject}.npz", allow_pickle=False) as saved:
                for condition in conditions:
                    key = condition + "_features"
                    if key not in saved or saved[key].shape != (len(row["target"]), 51) or not np.isfinite(saved[key]).all():
                        raise ValueError(f"Missing/invalid actual cached condition: {condition}, subject {subject}")
                    row[condition] = (visible_centroid_features(saved[condition + "_poses"], focal)
                                      if representation == "visible_centroid" else saved[key].copy())
                    # Keep the radar augmentation schedule independent of representation.
                    row["reference_available_" + condition] = (saved[key][:, 34:] >= .3).any(1)
    return split, rows


def augmented_batch(training, indices, augmentation, reference_available=None):
    """Identical condition, missing-modality and point-thinning draws per method."""
    condition = torch.randint(len(training), (len(indices),), generator=augmentation)
    values = [torch.stack([training[c][i][indices] for c in range(len(training))])[condition, torch.arange(len(indices))]
              for i in range(4)]
    rgb, points, mask, confidence = [x.clone() for x in values]
    modality = torch.rand(len(indices), generator=augmentation)
    drop_video = modality < .1
    reference = ((confidence >= .3).any(1) if reference_available is None
                 else reference_available[condition, indices])
    drop_radar = (modality >= .1) & (modality < .2) & reference
    point_keep_probability = torch.where(torch.rand(len(indices), generator=augmentation) < .2, .3, 1.)
    keep = torch.rand(mask.shape, generator=augmentation) < point_keep_probability[:, None]
    reduced = mask & keep
    for index in torch.nonzero(mask.any(1) & ~reduced.any(1)).flatten():
        reduced[index, torch.nonzero(mask[index]).flatten()[0]] = True
    mask = reduced & ~drop_radar[:, None]
    points = points.masked_fill(~mask[..., None], 0.)
    rgb[drop_video], confidence[drop_video] = 0., 0.
    return rgb, points, mask, confidence


def summarise(runs, methods, conditions, score_conditions, subjects):
    aggregate = {"models": {}, "comparisons_to_concat": {}}
    for method in methods:
        aggregate["models"][method] = {}
        for condition in conditions:
            records = [r["models"][method]["metrics"][condition] for r in runs]
            aggregate["models"][method][condition] = {
                "MPJPE_mean_mm": float(np.mean([v["MPJPE_mm"] for v in records])),
                "seed_sd_mm": float(np.std([v["MPJPE_mm"] for v in records], ddof=1)),
                "frame_p95_mean_mm": float(np.mean([v["frame_error_p95_mm"] for v in records])),
                "per_subject_seed_mean_mm": {str(s): float(np.mean([v["per_validation_subject_MPJPE_mm"][str(s)] for v in records])) for s in subjects}}
        scores = [np.mean([r["models"][method]["metrics"][c]["per_validation_subject_MPJPE_mm"][str(s)]
                           for c in score_conditions for s in subjects]) for r in runs]
        aggregate["models"][method]["selection_score"] = {"conditions": score_conditions,
            "mean_mm": float(np.mean(scores)), "seed_sd_mm": float(np.std(scores, ddof=1)), "per_seed_mm": list(map(float, scores))}
    if "concat" in methods:
        for method in methods:
            difference = np.array([
                np.mean([aggregate["models"]["concat"][c]["per_subject_seed_mean_mm"][str(s)] -
                         aggregate["models"][method][c]["per_subject_seed_mean_mm"][str(s)] for c in score_conditions]) for s in subjects])
            rng = np.random.default_rng(20261013)
            bootstrap = difference[rng.integers(0, len(subjects), (10000, len(subjects)))].mean(1)
            base, candidate = aggregate["models"]["concat"], aggregate["models"][method]
            gain = base["selection_score"]["mean_mm"] - candidate["selection_score"]["mean_mm"]
            clean_change = candidate["clean"]["MPJPE_mean_mm"] - base["clean"]["MPJPE_mean_mm"]
            per_seed = np.array(base["selection_score"]["per_seed_mm"]) - candidate["selection_score"]["per_seed_mm"]
            aggregate["comparisons_to_concat"][method] = {"score_gain_mm": gain, "clean_change_mm": clean_change,
                "per_subject_score_gain_mm": dict(zip(map(str, subjects), difference.tolist())),
                "exploratory_subject_bootstrap_95pct_mm": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                "positive_subjects": int((difference > 0).sum()), "positive_seeds": int((per_seed > 0).sum()),
                "expansion_gate_passed": bool(gain >= max(1., .02 * base["selection_score"]["mean_mm"]) and
                    clean_change <= max(1., .02 * base["clean"]["MPJPE_mean_mm"]) and (difference > 0).sum() >= 3 and (per_seed > 0).sum() >= 2),
                "interpretation": "Exploratory allocation rule, four subjects and one split; not statistical significance"}
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/research_execution_20261012/rgb_features")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=["clean", "dark"])
    parser.add_argument("--family", choices=("base", "graph", "scaled"), default="base")
    parser.add_argument("--representation", choices=("hip", "visible_centroid"), default="hip")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--methods", nargs="+", choices=METHODS + GRAPH_METHODS + SCALED_METHODS, default=None)
    parser.add_argument("--score-conditions", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    factory, supported = {"base": (FusionModel, METHODS), "graph": (GraphFusionModel, GRAPH_METHODS),
                          "scaled": (ScaledGateFusionModel, SCALED_METHODS)}[args.family]
    args.methods = list(supported) if args.methods is None else args.methods
    if not set(args.methods) <= set(supported):
        raise ValueError("Methods do not belong to declared model family")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Isolated CUDA environment is not ready")
    device = torch.device(args.device)
    if device.type == "cuda":
        # Explicit IEEE FP32; no hidden TF32 or mixed-precision comparison.
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    if args.output.exists() or not 1 <= args.epochs <= 200 or len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("New output directory, fixed 1..200 epochs and at least two distinct seeds required")
    if "clean" not in args.conditions or len(set(args.methods)) != len(args.methods) or len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Require clean condition and distinct models/conditions")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    split, rows = load_data(args.cache, args.conditions, args.representation)
    train, validation = split["canonical_ids_train"], split["canonical_ids_validation"]

    def stack(ids, key):
        return np.concatenate([rows[s][key] for s in ids])

    point_mean, point_std = point_normalization(stack(train, "points"), stack(train, "mask"))
    stats = {"point_mean": point_mean, "point_std": point_std, "rgb_mean": stack(train, "clean").mean(0),
             "rgb_std": np.maximum(stack(train, "clean").std(0), 1e-8), "target_mean_mm": stack(train, "target").mean(0)}
    np.savez_compressed(args.output / "normalization.npz", **stats)
    training = [normalize_inputs(stack(train, c), stack(train, "points"), stack(train, "mask"), stats) for c in args.conditions]
    reference_available = torch.tensor(np.stack([stack(train, "reference_available_" + c) for c in args.conditions]))
    train_target = torch.tensor((stack(train, "target") - stats["target_mean_mm"]) / 1000, dtype=torch.float32, device=device)
    target = stack(validation, "target")
    subject_ids = np.concatenate([np.full(len(rows[s]["target"]), s) for s in validation])
    values = {c: normalize_inputs(stack(validation, c), stack(validation, "points"), stack(validation, "mask"), stats) for c in args.conditions}
    clean = values["clean"]
    values["video_missing"] = (torch.zeros_like(clean[0]), clean[1], clean[2], torch.zeros_like(clean[3]))
    values["radar_missing"] = (clean[0], torch.zeros_like(clean[1]), torch.zeros_like(clean[2]), clean[3])
    thin = torch.tensor(sparse_mask(stack(validation, "mask")))
    values["radar_sparse"] = (clean[0], clean[1].masked_fill(~thin[..., None], 0.), thin, clean[3])
    score_conditions = args.score_conditions or (["clean", "noisy_dark", "occlusion", "video_missing", "radar_missing"]
                            if "noisy_dark" in values and "occlusion" in values else ["clean", "dark", "video_missing", "radar_missing", "radar_sparse"])
    if not set(score_conditions) <= set(values):
        raise ValueError("Selection conditions unavailable")
    counts = {m: sum(p.numel() for p in factory(m).parameters()) for m in args.methods}
    report = {"status": "running", "date": "2026-10-13", "scope": "Exploratory local fusion adaptations; not official reproductions, proposed novelty or final test",
        "input_cache": str(args.cache), "representation": args.representation,
        "runner_file": Path(__file__).name, "data_split_seed": split["seed"], "initialization_seeds": args.seeds,
        "train_subjects": train, "validation_subjects": validation, "final_test_subjects_used": [],
        "train_frames": len(train_target), "validation_frames": len(target), "methods": args.methods, "parameters": counts,
        "training": {"epochs": args.epochs, "checkpoint": "last epoch, no validation selection", "batch_size": 64,
            "optimizer": "AdamW", "lr": .001, "weight_decay": .0001, "conditions": args.conditions,
            "condition_mix": "uniform across supplied actual cached pixel conditions", "augmentation": "10% image missing, 10% radar missing only when image available; 20% examples retain 30% radar points",
            "augmentation_reference": "Original hip-cache availability governs radar dropout in every representation; independent radar model therefore has identical input schedule",
            "normalization": "clean training subjects only, normalized missing fields explicitly zeroed",
            "loss": "fused MSE + .25*(video MSE + radar MSE); attention modes add .25*ungated candidate MSE; utility adds .001 BCE; precision adds .001 variance NLL",
            "utility_labels": "online ungated candidate vs video error difference, detached; margin1mm/temp10mm; NOT OOF or calibrated risk"},
        "evaluation_conditions": list(values), "score_conditions": score_conditions,
        "expansion_rule": "score improvement >=max(1mm,2%), clean worsening <=max(1mm,2%), >=3/4 subjects and >=2/3 seeds improve; exploratory only",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT / "research/fusion_models.py",
            ROOT / "research/fusion_graph_models.py", ROOT / "research/fusion_scaled_gate_models.py", ROOT / "research/fusion_pose_representation.py")},
        "input_audit_sha256": hashlib.sha256((args.cache / "rgb_audit.json").read_bytes()).hexdigest(),
        "input_audit_conditions": json.loads((args.cache / "rgb_audit.json").read_text(encoding="utf-8")).get("conditions"),
        "cache_sha256": {str(s): hashlib.sha256((args.cache / f"subject{s}.npz").read_bytes()).hexdigest() for s in train + validation},
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "threads": args.threads, "device": str(device),
            "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "CUDA_version": torch.version.cuda, "deterministic": True, "mixed_precision": False,
            "FP32_matmul": "IEEE, TF32 disabled"}, "runs": []}
    if args.family != "base":
        report["model_factory"] = {"graph": "fusion_graph_models.GraphFusionModel", "scaled": "fusion_scaled_gate_models.ScaledGateFusionModel"}[args.family]
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for source in report["source_sha256"]:
        (snapshot / source).write_bytes((ROOT / "research" / source).read_bytes())
    (snapshot / "run_fusion_experiments.py").write_bytes((ROOT / "research/run_fusion_experiments.py").read_bytes())
    write_json(args.output / "run_summary.json", report)

    for seed in args.seeds:
        run = {"seed": seed, "status": "running", "models": {}}
        seed_dir = args.output / f"seed{seed}"
        seed_dir.mkdir()
        for method in args.methods:
            started = time.perf_counter()
            torch.manual_seed(seed)
            model = factory(method)
            model.target_template_m.copy_(torch.tensor(stats["target_mean_mm"].reshape(17, 3) / 1000, dtype=torch.float32))
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            order_rng, aug_rng = (torch.Generator().manual_seed(seed + k) for k in (0, 10000))
            risk_rng = torch.Generator(device=device).manual_seed(seed + 20000)
            history = []
            for epoch in range(1, args.epochs + 1):
                model.train()
                epoch_losses, parts = [], {}
                for indices in torch.randperm(len(train_target), generator=order_rng).split(64):
                    batch = tuple(value.to(device) for value in augmented_batch(training, indices, aug_rng, reference_available))
                    optimizer.zero_grad(set_to_none=True)
                    output = model(*batch)
                    semantic_method = method.removeprefix("scaled_") if args.family == "scaled" else method
                    loss, terms = fusion_loss(output, train_target[indices], semantic_method, risk_rng)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    epoch_losses.append(loss.item())
                    for key, value in terms.items():
                        parts.setdefault(key, []).append(value.item())
                if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
                    history.append({"epoch": epoch, "loss": float(np.mean(epoch_losses)), **{k: float(np.mean(v)) for k, v in parts.items()}})
                    print(json.dumps({"seed": seed, "method": method, **history[-1]}), flush=True)
            model.eval()
            metrics, saved = {}, {"target_mm": target, "subject_ids": subject_ids, "frame_ids": stack(validation, "frame_ids")}
            for condition, inputs in values.items():
                with torch.no_grad():
                    out = model(*(value.to(device) for value in inputs))
                prediction = out["fused"].reshape(-1, 51).cpu().numpy() * 1000 + stats["target_mean_mm"]
                video = out["video"].reshape(-1, 51).cpu().numpy() * 1000 + stats["target_mean_mm"]
                radar = out["radar"].reshape(-1, 51).cpu().numpy() * 1000 + stats["target_mean_mm"]
                metric = error_metrics(prediction, target, subject_ids)
                e_f = np.linalg.norm((prediction - target).reshape(-1, 17, 3), axis=-1).mean(1)
                e_v = np.linalg.norm((video - target).reshape(-1, 17, 3), axis=-1).mean(1)
                available = out["video_valid"].cpu().numpy()
                metric.update(video_branch_MPJPE_mm=float(e_v.mean()),
                    negative_fusion_vs_available_video_fraction=float((e_f[available] > e_v[available]).mean()) if available.any() else None,
                    both_modalities_missing_frames=int((~out["video_valid"] & ~out["radar_valid"]).sum()),
                    image_available_frames=int(out["video_valid"].sum()), radar_available_frames=int(out["radar_valid"].sum()))
                for name, value in (("fused", prediction), ("video", video), ("radar", radar)):
                    saved[condition + "_" + name + "_prediction_mm"] = value
                if "gate" in out:
                    gate = out["gate"].cpu().numpy()
                    metric["mean_gate_weight"] = float(gate.mean())
                    saved[condition + "_gate"] = gate
                metrics[condition] = metric
            np.savez_compressed(seed_dir / f"{method}_weights.npz", **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
            np.savez_compressed(seed_dir / f"{method}_predictions.npz", **saved)
            run["models"][method] = {"metrics": metrics, "history": history, "seconds": time.perf_counter() - started}
            write_json(seed_dir / "training.json", run)
            print(json.dumps({"seed": seed, "method": method, "status": "complete", "clean_MPJPE_mm": metrics["clean"]["MPJPE_mm"], "seconds": run["models"][method]["seconds"]}), flush=True)
        run["status"] = "complete"
        write_json(seed_dir / "training.json", run)
        report["runs"].append(run)
        write_json(args.output / "run_summary.json", report)
    report["aggregate"] = summarise(report["runs"], args.methods, list(values), score_conditions, validation)
    report["status"] = "complete"
    write_json(args.output / "run_summary.json", report)


if __name__ == "__main__":
    main()
