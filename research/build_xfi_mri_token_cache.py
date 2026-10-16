"""Full development native-X-Fi token cache; labels never enter calibration."""

import argparse
import hashlib
import json
import os
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps/research_cuda"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch

from cache_mri_rgb import file_checks
from cache_mri_robust_rgb import pixel_probes
from run_fusion_experiments_gpu import sparse_mask
from train_mri_point_baselines import load_development
from xfi_paper_baseline import COMMIT, OFFICIAL, SOURCE_FILES, XFiRGBRadar, source_manifest
from xfi_accuracy_models import native_radar_batch

CONDITIONS = ("clean", "dark", "noisy_dark", "occlusion")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def tensor(array):
    return torch.tensor(np.asarray(array), dtype=torch.float32, device="cuda")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, default=ROOT / "artifacts/fusion_rgb_expanded_20261013")
    parser.add_argument("--reuse-raw", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("New cache directory required")
    args.output.mkdir(parents=True)
    raw_directory = args.output / "raw"
    raw_directory.mkdir()
    audit = json.loads((args.source_cache / "rgb_audit.json").read_text(encoding="utf-8"))
    verified = json.loads((args.source_cache / "cache_verification.json").read_text(encoding="utf-8"))
    expected_cache = {row["subject"]: row["cache_sha256"] for row in verified["subjects"]}
    source_videos = {row["subject"]: row for row in audit["subjects"]}
    split, rows = load_development(args.source_cache)
    train, validation = split["canonical_ids_train"], split["canonical_ids_validation"]
    if set(train + validation) & set(split["canonical_ids_test"]):
        raise ValueError("Final-test leakage")
    report = {"status": "decoding", "paper": "X-Fi ICLR2025 mRI RGB/radar adaptation", "commit": COMMIT,
              "train_subjects": train, "validation_subjects": validation, "final_test_subjects_used": [],
              "train_frames": sum(len(rows[s]["target"]) for s in train),
              "validation_frames": sum(len(rows[s]["target"]) for s in validation),
              "protocol_sha256": sha(ROOT / "research/xfi_accuracy_protocol_20261013.md"),
              "source_sha256": {name: sha(ROOT / "research" / name) for name in
                                (Path(__file__).name, "xfi_paper_baseline.py", "xfi_accuracy_models.py", "cache_mri_robust_rgb.py", "train_mri_point_baselines.py")},
              "official_source_sha256": source_manifest(), "source_cache_verification_sha256": sha(args.source_cache / "cache_verification.json"),
              "conditions": audit["conditions"], "source_cache_sha256": {}, "raw": [], "subjects": []}
    write_json(args.output / "cache_report.json", report)
    previous_raw = None
    if args.reuse_raw is not None:
        previous_raw = json.loads((args.reuse_raw / "cache_report.json").read_text(encoding="utf-8"))
        if (len(previous_raw["raw"]) != 16 or previous_raw["train_subjects"] != train or previous_raw["validation_subjects"] != validation
                or sha(args.reuse_raw / "raw_stage_builder_source.py") != previous_raw["source_sha256"][Path(__file__).name]
                or previous_raw["source_cache_verification_sha256"] != report["source_cache_verification_sha256"]):
            raise ValueError("Complete authenticated raw stage required for reuse")
        report["raw_stage_reused_from"] = str(args.reuse_raw.resolve())
        report["raw_stage_previous_report_sha256"] = sha(args.reuse_raw / "cache_report.json")
        previous_raw = {row["subject"]: row for row in previous_raw["raw"]}
    for subject in train + validation:
        cache_path = args.source_cache / f"subject{subject}.npz"
        if sha(cache_path) != expected_cache[subject]:
            raise ValueError("Previously verified pose-cache changed")
        report["source_cache_sha256"][str(subject)] = expected_cache[subject]
        if previous_raw is not None:
            row = previous_raw[subject]
            source = args.reuse_raw / "raw" / f"subject{subject}.npz"
            if sha(source) != row["raw_cache_sha256"] or not row["pixel_hashes_exact_match_previous_4_conditions"]:
                raise ValueError("Authenticated resized raw cache changed")
            with np.load(source, allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw["frame_ids"], rows[subject]["frame_ids"])
                np.testing.assert_array_equal(raw["points"], rows[subject]["points"])
                np.testing.assert_array_equal(raw["target_mm"], rows[subject]["target"])
            destination = raw_directory / f"subject{subject}.npz"
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
            if sha(destination) != row["raw_cache_sha256"]:
                raise ValueError("Raw cache reuse did not preserve bytes")
            report["raw"].append(row)
            write_json(args.output / "cache_report.json", report)
            print(json.dumps({"stage": "reused_authenticated_raw", "subject": subject}), flush=True)
            continue
        video = ROOT / "artifacts/research_execution_20261012/rgb_audit" / f"subject{subject}_color0.mp4"
        digest, crc = file_checks(video)
        if digest != source_videos[subject]["source_sha256"] or crc != source_videos[subject]["source_crc32"]:
            raise ValueError("Video differs from authenticated source")
        capture = cv2.VideoCapture(str(video))
        images = {condition: [] for condition in CONDITIONS}
        shapes, pixel_hashes = [], {condition: [] for condition in CONDITIONS}
        with np.load(cache_path, allow_pickle=False) as previous:
            expected_pixels = {condition: previous[condition + "_pixel_sha256"].copy() for condition in CONDITIONS}
        for index, frame in enumerate(rows[subject]["frame_ids"]):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = capture.read()
            if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != int(frame) + 1:
                raise ValueError("Exact frame decode failed")
            shapes.append(list(image.shape))
            image = cv2.flip(image, 1)
            probes = pixel_probes(image, subject, int(frame))[0]
            probes.update(clean=image, dark=(image.astype(np.float32) * .18).astype(np.uint8))
            for condition in CONDITIONS:
                digest_pixels = hashlib.sha256(probes[condition].tobytes()).hexdigest()
                if digest_pixels != str(expected_pixels[condition][index]):
                    raise ValueError("Source-resolution pixels differ from certified perturbation cache")
                pixel_hashes[condition].append(digest_pixels)
                images[condition].append(cv2.resize(probes[condition], (640, 480), interpolation=cv2.INTER_LINEAR))
            if (index + 1) % 32 == 0:
                print(json.dumps({"stage": "decode_pixels", "subject": subject, "frames": index + 1}), flush=True)
        capture.release()
        destination = raw_directory / f"subject{subject}.npz"
        np.savez_compressed(destination, **{condition + "_images": np.asarray(images[condition], dtype=np.uint8) for condition in CONDITIONS},
                            frame_ids=rows[subject]["frame_ids"], points=rows[subject]["points"],
                            target_mm=rows[subject]["target"], point_mask=rows[subject]["mask"])
        report["raw"].append({"subject": subject, "frames": len(rows[subject]["target"]), "raw_cache_sha256": sha(destination),
                              "video_sha256": digest, "video_crc32": crc, "decoded_positions_verified": True,
                              "source_shape": shapes[0], "all_source_shapes_same": all(x == shapes[0] for x in shapes),
                              "source_resolution_pixel_sha256": pixel_hashes,
                              "pixel_hashes_exact_match_previous_4_conditions": True,
                              "resize": "Source-resolution perturbation then INTER_LINEAR to 480x640; BGR 0..255 retained"})
        write_json(args.output / "cache_report.json", report)
        print(json.dumps({"stage": "decoded", "subject": subject, "frames": len(rows[subject]["target"])}), flush=True)
        del images, probes

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(7)
    weights = ROOT / "artifacts/paper_baseline_sources_20261013/xfi_pretrained"
    prior = np.concatenate([rows[s]["target"] for s in train]).mean(0).reshape(17, 3) / 1000
    model = XFiRGBRadar(prior, weights, recalibrate_bn=True).to("cuda")
    before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
              if name.startswith(("rgb_extractor.", "radar_extractor."))}

    def calibration_batches():
        for subject in train:
            with np.load(raw_directory / f"subject{subject}.npz", allow_pickle=False) as saved:
                image, points = saved["clean_images"], saved["points"]
                for start in range(0, len(image), 16):
                    observations, _ = native_radar_batch(points[start:start + 16])
                    yield tensor(image[start:start + 16].transpose(0, 3, 1, 2)), tensor(observations)

    model.recalibrate_backbones(calibration_batches())
    model.freeze_backbones()
    model.eval()
    if model.backbone_initialization["BN_calibration_frames"] != report["train_frames"]:
        raise ValueError("Incomplete training calibration population")
    if not all(torch.equal(value, dict(model.named_parameters())[name].detach().cpu()) for name, value in before.items()):
        raise ValueError("Published backbone parameters changed during calibration")
    del before
    backbone_state = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()
                      if name.startswith(("rgb_extractor.", "radar_extractor.")) or name == "training_prior_m"}
    np.savez_compressed(args.output / "calibrated_backbones.npz", **backbone_state)
    report["backbone_initialization"] = model.backbone_initialization
    report["calibration_subjects"] = train
    report["calibration_backbone_parameters_unchanged"] = True
    report["calibrated_backbones_sha256"] = sha(args.output / "calibrated_backbones.npz")
    report["environment"] = {"torch": torch.__version__, "numpy": np.__version__, "device": "cuda",
                              "GPU": torch.cuda.get_device_name(), "FP32": "IEEE", "mixed_precision": False,
                              "deterministic": "warn_only; native pool backward limitation", "cuDNN_deterministic": True}
    report["status"] = "extracting_native_tokens"
    write_json(args.output / "cache_report.json", report)
    sparse = {}
    for group in (train, validation):
        combined = sparse_mask(np.concatenate([rows[s]["mask"] for s in group]))
        start = 0
        for subject in group:
            count = len(rows[subject]["target"])
            sparse[subject] = combined[start:start + count]
            start += count
    for subject in train + validation:
        saved_tokens = {"frame_ids": rows[subject]["frame_ids"], "target_mm": rows[subject]["target"],
                        "subject_ids": np.full(len(rows[subject]["target"]), subject, dtype=np.int32),
                        "radar_available": rows[subject]["mask"].any(1)}
        pixels_manifest = []
        with np.load(raw_directory / f"subject{subject}.npz", allow_pickle=False) as raw:
            points = raw["points"]
            frame_count = len(points)
            with torch.no_grad():
                for condition in CONDITIONS:
                    image_batch, token_parts = raw[condition + "_images"], []
                    for start in range(0, frame_count, 16):
                        values = model.rgb_extractor(tensor(image_batch[start:start + 16].transpose(0, 3, 1, 2)))
                        token_parts.append(values.view(len(values), 512, 49).cpu().numpy())
                    tokens = np.concatenate(token_parts)
                    if not np.isfinite(tokens).all():
                        raise ValueError("Nonfinite native RGB tokens")
                    saved_tokens[condition + "_rgb_CxT"] = tokens
                    pixels_manifest.append({"condition": condition, "pixel_sha256": hashlib.sha256(image_batch.tobytes()).hexdigest()})
                    del image_batch
                for name, observations in (("radar", points), ("radar_sparse", np.where(sparse[subject][..., None], points, 0.))):
                    values, lengths = [], []
                    for start in range(0, frame_count, 16):
                        observed, count = native_radar_batch(observations[start:start + 16])
                        values.append(model.radar_extractor(tensor(observed))[0].cpu().numpy())
                        lengths.extend(count.tolist())
                    tokens = np.concatenate(values)
                    if tokens.shape != (frame_count, 32, 512) or not np.isfinite(tokens).all():
                        raise ValueError("Invalid native radar tokens")
                    saved_tokens[name + "_tokens"] = tokens
                    saved_tokens[name + "_observation_counts"] = np.asarray(lengths, dtype=np.int32)
                if subject == train[0]:
                    again = model.rgb_extractor(tensor(raw["clean_images"][:16].transpose(0, 3, 1, 2))).view(16, 512, 49).cpu().numpy()
                    np.testing.assert_array_equal(again, saved_tokens["clean_rgb_CxT"][:16])
                    observed, _ = native_radar_batch(points[:16])
                    radar_again = model.radar_extractor(tensor(observed))[0].cpu().numpy()
                    np.testing.assert_array_equal(radar_again, saved_tokens["radar_tokens"][:16])
                    report["same_GPU_first16_native_token_replay_exact"] = True
        destination = args.output / f"subject{subject}_tokens.npz"
        np.savez_compressed(destination, **saved_tokens)
        report["subjects"].append({"subject": subject, "frames": len(rows[subject]["target"]),
                                   "tokens_sha256": sha(destination), "pixels": pixels_manifest,
                                   "RGB_storage": "N x 512 x 49 preserves native contiguous channel/token layout",
                                   "radar_storage": "N x 32 x 512; native TD on actual point rows, padded to max observed count in original 16-frame extraction batches"})
        write_json(args.output / "cache_report.json", report)
        print(json.dumps({"stage": "tokens_complete", "subject": subject}), flush=True)
        del saved_tokens
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for name in report["source_sha256"]:
        (snapshot / name).write_bytes((ROOT / "research" / name).read_bytes())
    for name in SOURCE_FILES:
        path = snapshot / "official" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((OFFICIAL / name).read_bytes())
    (snapshot / "xfi_accuracy_protocol_20261013.md").write_bytes((ROOT / "research/xfi_accuracy_protocol_20261013.md").read_bytes())
    if source_manifest() != report["official_source_sha256"] or any(sha(ROOT / "research" / name) != digest for name, digest in report["source_sha256"].items()):
        raise ValueError("Source code changed during cache build")
    report["status"] = "complete"
    write_json(args.output / "cache_report.json", report)
    print(json.dumps({"status": "complete", "train_frames": report["train_frames"], "validation_frames": report["validation_frames"]}), flush=True)


if __name__ == "__main__":
    main()
