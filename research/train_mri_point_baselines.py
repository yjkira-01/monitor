"""Fixed-budget development probes, with point-set radar inputs and frozen RGB."""

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit_mri import interval_bounds, load_labels
from cache_mri_rgb import MODEL_SHA, VIDEO_SHA, file_checks
from run_mri_baselines import subject_samples

ROOT = Path(__file__).resolve().parents[1]


class PointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.points = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())

    def forward(self, points, mask):
        if points.shape[:-1] != mask.shape or points.shape[-1] != 5 or mask.dtype != torch.bool or not torch.isfinite(points[mask]).all():
            raise ValueError("Invalid point cloud or mask")
        features = self.points(torch.where(mask[..., None], points, 0.))
        mean = (features * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        maximum = features.masked_fill(~mask[..., None], torch.finfo(features.dtype).min).max(1).values
        maximum = torch.where(mask.any(1)[:, None], maximum, 0.)
        return torch.cat([mean, maximum], dim=-1)


class PoseRegressor(nn.Module):
    def __init__(self, modality):
        super().__init__()
        if modality not in {"rgb", "radar", "fusion"}:
            raise ValueError("Unknown modality")
        self.modality = modality
        self.radar = PointEncoder() if modality != "rgb" else None
        self.rgb = nn.Sequential(nn.Linear(51, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()) if modality != "radar" else None
        self.head = nn.Sequential(nn.Linear(256 if modality == "fusion" else 128, 128), nn.ReLU(), nn.Linear(128, 51))
        nn.init.normal_(self.head[-1].weight, std=.001)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, rgb, points, mask):
        features = []
        if self.rgb is not None:
            features.append(self.rgb(rgb))
        if self.radar is not None:
            features.append(self.radar(points, mask))
        return self.head(torch.cat(features, dim=-1))


def point_normalization(points, mask):
    points, mask = np.asarray(points), np.asarray(mask)
    if points.ndim != 3 or points.shape[-1] != 5 or mask.shape != points.shape[:-1] or mask.dtype != np.bool_ or not mask.any() or not np.isfinite(points[mask]).all():
        raise ValueError("Invalid training point cloud")
    valid = points[mask]
    return valid.mean(0), np.maximum(valid.std(0), 1e-6)


def load_development(cache_directory):
    split = json.loads((ROOT / "research/pose_fusion_protocol.json").read_text(encoding="utf-8"))["splits"][0]
    train, validation = split["canonical_ids_train"], split["canonical_ids_validation"]
    if set(train) & set(validation) or (set(train) | set(validation)) & set(split["canonical_ids_test"]):
        raise ValueError("Subject leakage")
    audit = json.loads((cache_directory / "rgb_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "complete" or audit["video_archive_sha256"] != VIDEO_SHA or set(s["subject"] for s in audit["subjects"]) != set(train + validation):
        raise ValueError("Complete independent RGB development cache required")
    data_audit = json.loads((ROOT / "artifacts/research_execution_20261012/mri_data_audit.json").read_text(encoding="utf-8"))
    if data_audit["status"] != "complete" or file_checks(ROOT / "dataset_release.zip")[0] != data_audit["archive_sha256"]:
        raise ValueError("Source data checksum mismatch")
    rows = {}
    with zipfile.ZipFile(ROOT / "dataset_release.zip") as archive:
        for subject in train + validation:
            samples = subject_samples(archive, subject, split["seed"])
            with np.load(cache_directory / f"subject{subject}.npz", allow_pickle=False) as cache:
                if str(cache["model_sha256"]) != MODEL_SHA or str(cache["video_archive_sha256"]) != VIDEO_SHA:
                    raise ValueError("RGB cache provenance mismatch")
                ids = cache["frame_ids"].copy()
                chosen = np.searchsorted(samples["frame_ids"], ids)
                if (chosen >= len(samples["frame_ids"])).any() or not np.array_equal(samples["frame_ids"][chosen], ids):
                    raise ValueError("Frame selection mismatch")
                rows[subject] = {"clean": cache["clean_features"].copy(), "dark": cache["dark_features"].copy(),
                                 "target": samples["target_mm"][chosen], "frame_ids": ids}
            labels = load_labels(archive.read(f"dataset_release/aligned_data/pose_labels/subject{subject}_all_labels.cpl"))
            start, _ = interval_bounds(labels["radar_avail_frames"], len(labels["refined_gt_kps"]))
            points = np.load(io.BytesIO(archive.read(f"dataset_release/features/radar/subject{subject}_featuremap.npy")), allow_pickle=False)
            rows[subject]["points"] = points[ids - start].reshape(-1, 196, 5)
            rows[subject]["mask"] = np.any(rows[subject]["points"] != 0, axis=-1)
    return split, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/research_execution_20261012/rgb_features")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research_execution_20261012/point_development_baselines")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 200:
        raise ValueError("Development probe requires 1..200 fixed epochs")
    if args.output.exists():
        raise ValueError("Refusing to overwrite an experiment")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    split, rows = load_development(args.cache)
    train, validation = split["canonical_ids_train"], split["canonical_ids_validation"]

    def stack(ids, key):
        return np.concatenate([rows[s][key] for s in ids])

    train_points, train_mask = stack(train, "points"), stack(train, "mask")
    point_mean, point_std = point_normalization(train_points, train_mask)
    rgb_mean, rgb_std = stack(train, "clean").mean(0), np.maximum(stack(train, "clean").std(0), 1e-8)
    target_mean = stack(train, "target").mean(0)
    np.savez_compressed(args.output / "normalization.npz", point_mean=point_mean, point_std=point_std,
                        rgb_mean=rgb_mean, rgb_std=rgb_std, target_mean_mm=target_mean)

    def tensors(ids, condition):
        points, mask = stack(ids, "points"), stack(ids, "mask")
        normalized = np.where(mask[..., None], (points - point_mean) / point_std, 0.)
        return (torch.tensor((stack(ids, condition) - rgb_mean) / rgb_std, dtype=torch.float32),
                torch.tensor(normalized, dtype=torch.float32), torch.tensor(mask),
                torch.tensor((stack(ids, "target") - target_mean) / 1000., dtype=torch.float32))

    training = tensors(train, "clean")
    validation_tensors = {condition: tensors(validation, condition) for condition in ("clean", "dark")}
    report = {"status": "running", "scope": "Single-seed, small matched development neural probes; not proposed method or final test",
              "train_subjects": train, "validation_subjects": validation, "test_subjects_used": [],
              "train_frames": len(training[0]), "validation_frames": len(validation_tensors["clean"][0]),
              "input": "Independent frozen RTMO features and published five-channel point clouds; no GT boxes/keypoints as inputs",
              "target": "Published root-relative 17-joint pose, mm; earlier fixed development label screen",
              "point_encoder": "Per-point 5->32->64 ReLU, masked mean/max pooling; no temporal input",
              "training": {"seed": 7, "epochs": args.epochs, "selection": "last epoch, no validation checkpoint selection",
                           "optimizer": "AdamW", "lr": .001, "weight_decay": .0001, "batch_size": 64,
                           "normalization": "training data only", "augmentation": "none; train clean only"},
              "device": "CPU", "torch": torch.__version__, "models": {}}

    def evaluate(model, condition):
        model.eval()
        values = validation_tensors[condition]
        with torch.no_grad():
            prediction = model(*values[:3]).numpy() * 1000 + target_mean
        errors = np.linalg.norm((prediction - stack(validation, "target")).reshape(-1, 17, 3), axis=-1)
        offset, per_subject = 0, {}
        for subject in validation:
            length = len(rows[subject]["target"])
            per_subject[str(subject)] = float(errors[offset:offset + length].mean())
            offset += length
        return {"MPJPE_mm": float(errors.mean()), "frame_error_p95_mm": float(np.percentile(errors.mean(-1), 95)),
                "per_validation_subject_MPJPE_mm": per_subject}

    for modality in ("rgb", "radar", "fusion"):
        torch.manual_seed(7)
        model = PoseRegressor(modality)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        history = []
        start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for indices in torch.randperm(len(training[0])).split(64):
                batch = [tensor[indices] for tensor in training]
                optimizer.zero_grad(set_to_none=True)
                loss = (model(*batch[:3]) - batch[3]).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(loss.item())
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                row = {"epoch": epoch, "train_MSE_m2": float(np.mean(losses)), "clean_validation_MPJPE_mm": evaluate(model, "clean")["MPJPE_mm"]}
                history.append(row)
                print(json.dumps({"model": modality, **row}), flush=True)
                report["models"][modality] = {"status": "running", "history": history}
                (args.output / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        metrics = {condition: evaluate(model, condition) for condition in validation_tensors}
        # Safe array weights, without loading arbitrary Torch pickle checkpoints.
        np.savez_compressed(args.output / f"{modality}_weights.npz", **{name: value.detach().numpy() for name, value in model.state_dict().items()})
        report["models"][modality] = {"status": "complete", "parameters": sum(p.numel() for p in model.parameters()),
                                     "seconds": time.perf_counter() - start, "history": history, "metrics": metrics}
        (args.output / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["status"] = "complete"
    (args.output / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
