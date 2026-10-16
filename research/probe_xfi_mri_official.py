"""Real training-subject forward/backward feasibility; no pose-performance claim."""

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps/research_cuda"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch

from audit_mri import interval_bounds, load_labels
from cache_mri_rgb import file_checks
from run_mri_baselines import subject_samples
from xfi_paper_baseline import COMMIT, OFFICIAL, SOURCE_FILES, XFiRGBRadar, source_manifest


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("New separate feasibility output required")
    args.output.mkdir(parents=True)
    cache = ROOT / "artifacts/fusion_rgb_expanded_20261013"
    audit = json.loads((cache / "rgb_audit.json").read_text(encoding="utf-8"))
    source_rows = {row["subject"]: row for row in audit["subjects"]}
    prior_verification = json.loads((cache / "cache_verification.json").read_text(encoding="utf-8"))
    prior_cache_sha = {row["subject"]: row["cache_sha256"] for row in prior_verification["subjects"]}
    images, point_rows, targets, manifest = [], [], [], []
    with zipfile.ZipFile(ROOT / "dataset_release.zip") as archive:
        for subject in (1, 2):
            cache_path = cache / f"subject{subject}.npz"
            if file_checks(cache_path)[0] != prior_cache_sha[subject]:
                raise ValueError("Previously verified development cache changed")
            video = ROOT / "artifacts/research_execution_20261012/rgb_audit" / f"subject{subject}_color0.mp4"
            digest, crc = file_checks(video)
            if digest != source_rows[subject]["source_sha256"] or crc != source_rows[subject]["source_crc32"]:
                raise ValueError("Previously verified source video changed")
            with np.load(cache_path, allow_pickle=False) as saved:
                frame_ids = saved["frame_ids"][[0, 47]]
            samples = subject_samples(archive, subject, 7)
            positions = np.searchsorted(samples["frame_ids"], frame_ids)
            np.testing.assert_array_equal(samples["frame_ids"][positions], frame_ids)
            labels = load_labels(archive.read(f"dataset_release/aligned_data/pose_labels/subject{subject}_all_labels.cpl"))
            start, _ = interval_bounds(labels["radar_avail_frames"], len(labels["refined_gt_kps"]))
            radar = np.load(io.BytesIO(archive.read(f"dataset_release/features/radar/subject{subject}_featuremap.npy")), allow_pickle=False)
            capture = cv2.VideoCapture(str(video))
            for index, frame in enumerate(frame_ids):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                success, image = capture.read()
                position_after = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if not success or position_after != int(frame) + 1:
                    raise ValueError("Failed exact source-frame decoding")
                shape = image.shape
                image = cv2.flip(image, 1)
                # Native official RGB reshaping expects 480x640; retain BGR and
                # 0..255 float scale as the actual published collate implementation.
                if image.shape != (480, 640, 3):
                    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
                images.append(image.transpose(2, 0, 1).copy())
                point_rows.append(radar[int(frame) - start].reshape(196, 5))
                targets.append(samples["target_mm"][positions[index]].reshape(17, 3) / 1000)
                manifest.append({"subject": subject, "frame_id": int(frame), "source_shape": list(shape),
                                 "video_sha256": digest, "decoded_index_verified": True})
            capture.release()
    device = torch.device("cuda")
    torch.set_num_threads(2)
    # Native AdaptiveAvgPool2d backward is nondeterministic on this CUDA build.
    # Preserve the official operation and disclose it; strict FP32 still applies.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(7)
    image_tensor = torch.tensor(np.array(images), dtype=torch.float32, device=device)
    point_tensor = torch.tensor(np.array(point_rows), dtype=torch.float32, device=device)
    target = torch.tensor(np.array(targets), dtype=torch.float32, device=device)
    model = XFiRGBRadar(target.mean(0), args.pretrained, recalibrate_bn=args.pretrained is not None).to(device)
    backbone_parameters_before = {key: value.detach().clone() for key, value in model.named_parameters()
                                  if key.startswith(("rgb_extractor.", "radar_extractor."))}
    if args.pretrained is not None:
        model.recalibrate_backbones([(image_tensor, point_tensor)])
        model.freeze_backbones()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    history = []
    for step in range(2):
        model.train()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(image_tensor, point_tensor)
        loss = (output - target).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native X-Fi probe loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError("Nonfinite native gradients")
        optimizer.step()
        torch.cuda.synchronize()
        history.append({"step": step + 1, "loss_MSE_m2": loss.item(), "seconds": time.perf_counter() - started})
        print(json.dumps(history[-1]), flush=True)
    training_peak_bytes = torch.cuda.max_memory_allocated()
    backbone_unchanged = all(torch.equal(value, dict(model.named_parameters())[key]) for key, value in backbone_parameters_before.items())
    if args.pretrained is not None and not backbone_unchanged:
        raise ValueError("Frozen author backbone parameters changed")
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor, point_tensor)
        native = model.native_subset(image_tensor, point_tensor)
        torch.testing.assert_close(prediction, native, rtol=0, atol=0)
        missing_checks = []
        for image_flag, radar_flag in ((True, False), (False, True), (False, False)):
            flags = [torch.full((len(target),), value, dtype=torch.bool, device=device) for value in (image_flag, radar_flag)]
            actual = model(image_tensor, point_tensor, *flags)
            expected = model.native_subset(image_tensor, point_tensor, image_flag, radar_flag)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            missing_checks.append({"rgb": image_flag, "radar": radar_flag, "exact_native_subset_or_prior": True})
    weights = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    np.savez_compressed(args.output / "probe_weights.npz", **weights)
    reloaded = XFiRGBRadar().to(device)
    with np.load(args.output / "probe_weights.npz", allow_pickle=False) as saved:
        reloaded.load_state_dict({key: torch.from_numpy(saved[key].copy()) for key in saved.files}, strict=True)
    if model.backbones_frozen:
        reloaded.freeze_backbones()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[key], rtol=0, atol=0)
    reloaded.eval()
    with torch.no_grad():
        replay = reloaded(image_tensor, point_tensor)
    torch.testing.assert_close(replay, prediction, rtol=0, atol=0)
    np.savez_compressed(args.output / "probe_observations_predictions.npz", images=np.array(images),
                        points=np.array(point_rows), training_target_m=np.array(targets), prediction_m=prediction.cpu().numpy())
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for name in SOURCE_FILES:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((OFFICIAL / name).read_bytes())
    for name in ("xfi_paper_baseline.py", Path(__file__).name):
        (snapshot / name).write_bytes((ROOT / "research" / name).read_bytes())
    report = {"status": "complete", "scope": "Four real training-subject frames, two optimization steps; feasibility only, not HPE benchmark",
              "paper": "X-Fi, ICLR2025", "official_repository": "https://github.com/NTUMARS/X-Fi", "commit": COMMIT,
              "source_sha256": source_manifest(), "original_definition_AST_manifest": model.official_definitions,
              "adapter_sha256": hashlib.sha256((ROOT / "research/xfi_paper_baseline.py").read_bytes()).hexdigest(),
              "probe_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "backbone_initialization": model.backbone_initialization,
              "pretrained_backbone_parameters_unchanged": backbone_unchanged if args.pretrained is not None else None,
              "adaptations": ["RGB+radar only on mRI, inactive native slots retained", "Author MM-Fi pretrained parameters retained; training-only BN recalibration" if args.pretrained else "Scratch backbone initialization, no MM-Fi pretrained weights",
                              "Native full-frame BGR float 0..255, horizontal reflection restored", "No GT crop or GT root in inputs",
                              "Explicit per-example missing-modality routing to native subsets; training prior if both absent"],
              "initialization_scope": "Author MM-Fi pretrained backbones with four-training-frame BN recalibration; frozen parameters; not formal recalibration" if args.pretrained else "Scratch backbone initialization",
              "native_dimensions": {"token_dim": 512, "tokens_per_modality": 32, "heads": 8, "fusion_depth": 4, "output_joints": 17},
              "parameters": sum(p.numel() for p in model.parameters()),
              "active_gradient_parameters_last_step": sum(p.numel() for p in model.parameters() if p.grad is not None),
              "GPU": torch.cuda.get_device_name(), "peak_allocated_bytes_training": training_peak_bytes,
              "peak_allocated_bytes_including_reload": torch.cuda.max_memory_allocated(),
              "deterministic_policy": "warn_only; native CUDA AdaptiveAvgPool2d backward lacks strict deterministic implementation",
              "strict_deterministic_backward": False,
              "precision": "IEEE FP32; no AMP/TF32", "training_frames": manifest, "validation_subjects_used": [],
              "final_test_subjects_used": [], "history": history, "original_native_both_prediction_exact": True,
              "missing_checks": missing_checks, "safe_weight_reload_exact": True,
              "weights_sha256": file_checks(args.output / "probe_weights.npz")[0],
              "observations_predictions_sha256": file_checks(args.output / "probe_observations_predictions.npz")[0]}
    write_json(args.output / "probe_receipt.json", report)
    print(json.dumps({"status": "complete", "parameters": report["parameters"],
                      "training_peak_GB": report["peak_allocated_bytes_training"] / 1024 ** 3,
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
