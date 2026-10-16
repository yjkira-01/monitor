"""Expand only the seed-7 mRI development RGB cache with fixed pixel probes.

The earlier complete archive SHA authentication is reused. This run checks archive
size/central-directory metadata and each extracted source video's SHA/CRC; it
does not stream the 15 GB video archive again or extract any additional video.
"""

import argparse
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from audit_mri import load_labels
from cache_mri_rgb import MODEL_SHA, VIDEO_SHA, file_checks, independent_pose, pose_features
from run_mri_baselines import subject_samples

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("clean", "dark", "noisy_dark", "occlusion")
NOISE_SEED = 20261013
VIDEO_BYTES = 14988106753
DEFAULT_OUTPUT = ROOT / "artifacts/fusion_rgb_expanded_20261013"


def expanded_ids(sample_ids, old_ids):
    sample_ids, old_ids = np.asarray(sample_ids), np.asarray(old_ids)
    if (sample_ids.shape != (256,) or old_ids.shape != (64,) or
            not np.all(np.diff(sample_ids) > 0) or
            not np.array_equal(old_ids, sample_ids[np.linspace(0, 255, 64, dtype=int)])):
        raise ValueError("Expected the original 64 positions within fixed 256 samples")
    remaining = sample_ids[~np.isin(sample_ids, old_ids)]
    new_ids = remaining[np.linspace(0, 191, 32, dtype=int)]
    ids = np.sort(np.concatenate([old_ids, new_ids]))
    if len(np.unique(ids)) != 96:
        raise ValueError("Expansion must contain 96 distinct actual frame IDs")
    return ids, np.isin(ids, old_ids)


def pixel_probes(image, subject, frame_id):
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("Pixel probes require a decoded uint8 BGR frame")
    seed_words = [NOISE_SEED, int(subject), int(frame_id)]
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    noisy = image.astype(np.float32) * .12
    noisy += rng.normal(0., 8., size=image.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    occluded = image.copy()
    height, width = image.shape[:2]
    occluded[int(.15 * height):int(.90 * height), int(.40 * width):int(.60 * width)] = 114
    return {"noisy_dark": noisy, "occlusion": occluded}, seed_words


def verify_archive_directory(path, entries, expected_bytes, dataset=False):
    """Check directory metadata without reading or extracting member payloads."""
    if path.stat().st_size != expected_bytes:
        raise ValueError("Previously authenticated archive byte size changed")
    expected = {}
    for entry in entries:
        name = entry["name" if dataset else "member"]
        if name in expected:
            raise ValueError("Duplicate member in authenticated archive manifest")
        expected[name] = entry
    with zipfile.ZipFile(path) as archive:
        actual = archive.infolist()
        if not dataset:
            actual = [info for info in actual if not info.is_dir()]
        if len(actual) != len(expected) or len({info.filename for info in actual}) != len(actual):
            raise ValueError("Archive directory count or uniqueness changed")
        for info in actual:
            entry = expected.get(info.filename)
            crc = int(entry["crc32"], 16) if dataset and entry else entry["crc32"] if entry else None
            if (entry is None or info.file_size != entry["bytes"] or info.CRC != crc or
                    info.flag_bits & 1 or
                    (dataset and info.compress_size != entry["compressed_bytes"]) or
                    (not dataset and info.compress_type != entry["compression_method"])):
                raise ValueError("Previously authenticated archive directory metadata changed")


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pixel_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def validate_completed(cache_path, row, ids, protocol_sha, old_sha, source_sha):
    if file_checks(cache_path)[0] != row["cache_sha256"]:
        raise ValueError("Completed output cache checksum changed")
    with np.load(cache_path, allow_pickle=False) as cache:
        if (str(cache["protocol_sha256"]) != protocol_sha or str(cache["old_cache_sha256"]) != old_sha or
                str(cache["source_sha256"]) != source_sha or not np.array_equal(cache["frame_ids"], ids)):
            raise ValueError("Completed output cache provenance or frame selection changed")
        for condition in CONDITIONS:
            if (cache[condition + "_poses"].shape != (96, 17, 3) or
                    cache[condition + "_features"].shape != (96, 51) or
                    cache[condition + "_available"].shape != (96, 17) or
                    cache[condition + "_missing"].shape != (96,) or
                    not np.isfinite(cache[condition + "_poses"]).all() or
                    not np.isfinite(cache[condition + "_features"]).all()):
                raise ValueError("Invalid completed cache arrays")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--old-cache", type=Path, default=ROOT / "artifacts/research_execution_20261012/rgb_features")
    parser.add_argument("--video-directory", type=Path, default=ROOT / "artifacts/research_execution_20261012/rgb_audit")
    args = parser.parse_args()
    args.output = args.output.resolve()
    for protected in (args.old_cache.resolve(), args.video_directory.resolve()):
        if args.output == protected or protected in args.output.parents or args.output in protected.parents:
            raise ValueError("Output must be separate from existing source caches and videos")
    if ROOT not in args.output.parents:
        raise ValueError("Output must stay within the research workspace")
    split = json.loads((ROOT / "research/pose_fusion_protocol.json").read_text(encoding="utf-8"))["splits"][0]
    subjects = split["canonical_ids_train"] + split["canonical_ids_validation"]
    if split["seed"] != 7 or len(subjects) != 16 or len(set(subjects)) != 16 or set(subjects) & set(split["canonical_ids_test"]):
        raise ValueError("Only the fixed seed-7 development split is allowed")
    old_audit = json.loads((args.old_cache / "rgb_audit.json").read_text(encoding="utf-8"))
    if (old_audit["status"] != "complete" or old_audit["video_archive_sha256"] != VIDEO_SHA or
            old_audit["model_sha256"] != MODEL_SHA or
            set(row["subject"] for row in old_audit["subjects"]) != set(subjects) or old_audit["test_subjects_used"]):
        raise ValueError("A complete authenticated development-only source RGB cache is required")
    video_manifest = json.loads((args.old_cache / "video_manifest.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((ROOT / "artifacts/research_execution_20261012/mri_archive_manifest.json").read_text(encoding="utf-8"))
    data_audit = json.loads((ROOT / "artifacts/research_execution_20261012/mri_data_audit.json").read_text(encoding="utf-8"))
    if (video_manifest["archive_sha256"] != VIDEO_SHA or data_audit["status"] != "complete" or
            not data_manifest["sha256_matches_dryad"] or data_manifest["sha256"] != data_audit["archive_sha256"]):
        raise ValueError("Prior full archive authentication is missing")
    verify_archive_directory(ROOT / "blurred_videos.zip", video_manifest["entries"], VIDEO_BYTES)
    verify_archive_directory(ROOT / "dataset_release.zip", data_manifest["index"], data_manifest["bytes"], dataset=True)
    model_path = ROOT / "desktop/CareMonitor/Assets/Realtime/rtmo.onnx"
    if file_checks(model_path)[0] != MODEL_SHA:
        raise ValueError("RTMO checksum mismatch")
    protocol = {"schema_version": 1, "train_subjects": split["canonical_ids_train"],
                "validation_subjects": split["canonical_ids_validation"], "split_seed": 7,
                "test_subjects_used": [], "frames_per_subject": 96, "reused_frames_per_subject": 64,
                "new_frames_per_subject": 32, "sampling": "original64 union evenly-spaced32 among remaining192 of fixed subject_samples256; sorted",
                "video_archive_sha256": VIDEO_SHA, "data_archive_sha256": data_audit["archive_sha256"],
                "model_sha256": MODEL_SHA, "old_cache_directory": str(args.old_cache.resolve()),
                "video_directory": str(args.video_directory.resolve()),
                "old_audit_sha256": file_checks(args.old_cache / "rgb_audit.json")[0],
                "input": "horizontal reflection undone before full-frame RTMO; no GT boxes/keypoints or GT-based detections",
                "conditions": {"clean": "unchanged reflected decoded BGR uint8 pixels",
                               "dark": "floor(pixel*0.18), no enhancement; original64 features and poses reused exactly",
                               "noisy_dark": "clip(float32(pixel)*0.12 + Gaussian(0,8) float32,0,255).astype(uint8); noise at full source resolution",
                               "occlusion": "fixed source-resolution rectangle x[0.40,0.60), y[0.15,0.90), integer-floor boundaries; fill BGR114"},
                "noise_rng": "numpy.default_rng(SeedSequence[20261013,subject,frame_id]); per-pixel, per-BGR-channel independent",
                "missing_rule": "root hips11&12 confidence>=0.3; retain every selected frame regardless of outputs",
                "archive_authentication": "reuse prior complete SHA certification; this run verifies ZIP sizes and directory CRC/size metadata, then each extracted video SHA/CRC; video ZIP payload not rehashed",
                "software": {"numpy": np.__version__, "opencv": cv2.__version__, "onnxruntime": ort.__version__}}
    protocol_sha = digest_json(protocol)
    if args.output.exists():
        existing_protocol = args.output / "protocol.json"
        if not existing_protocol.exists() or digest_json(json.loads(existing_protocol.read_text(encoding="utf-8"))) != protocol_sha:
            raise ValueError("Refusing to overwrite an unrelated or differently configured output directory")
    else:
        args.output.mkdir(parents=True)
        save_json(args.output / "protocol.json", protocol)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 1
    model = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    cv2.setNumThreads(1)
    report = {"status": "running", "scope": "Expanded real-pixel development cache; synthetic probes, not final test or actual night/occlusion collection",
              "protocol_sha256": protocol_sha, "video_archive_sha256": VIDEO_SHA, "model_sha256": MODEL_SHA,
              "data_archive_sha256": data_audit["archive_sha256"], "conditions": protocol["conditions"],
              "input": protocol["input"], "sampling": protocol["sampling"], "test_subjects_used": [],
              "archive_authentication": protocol["archive_authentication"], "expected_subjects": subjects,
              "expected_frames": 1536, "subjects": [], "started_unix": time.time(), "current_subject": None}

    def persist():
        report["completed_subjects"] = len(report["subjects"])
        report["completed_frames"] = sum(row["sampled_frames"] for row in report["subjects"])
        report["updated_unix"] = time.time()
        save_json(args.output / "rgb_audit.json", report)
        save_json(args.output / "status.json", report)

    persist()
    try:
        with zipfile.ZipFile(ROOT / "dataset_release.zip") as archive:
            for subject in subjects:
                report["current_subject"] = subject
                report["current_subject_decoded_frames"] = 0
                persist()
                print(json.dumps({"event": "subject_start", "subject": subject}), flush=True)
                samples = subject_samples(archive, subject, 7)
                old_path = args.old_cache / f"subject{subject}.npz"
                old_sha = file_checks(old_path)[0]
                with np.load(old_path, allow_pickle=False) as old:
                    old_values = {key: old[key].copy() for key in old.files}
                if str(old_values["model_sha256"]) != MODEL_SHA or str(old_values["video_archive_sha256"]) != VIDEO_SHA:
                    raise ValueError("Original cache provenance mismatch")
                ids, reused = expanded_ids(samples["frame_ids"], old_values["frame_ids"])
                path = args.video_directory / f"subject{subject}_color0.mp4"
                entry = next(entry for entry in video_manifest["entries"] if entry["member"].endswith(f"/subject{subject}_color0.mp4"))
                source_sha, crc = file_checks(path)
                if path.stat().st_size != entry["bytes"] or crc != entry["crc32"] or source_sha != str(old_values["source_sha256"]):
                    raise ValueError("Extracted source video differs from verified source cache")
                cache_path = args.output / f"subject{subject}.npz"
                row_path = args.output / f"subject{subject}.json"
                if cache_path.exists():
                    if not row_path.exists():
                        raise ValueError("Output cache without its completion record; refusing overwrite")
                    row = json.loads(row_path.read_text(encoding="utf-8"))
                    validate_completed(cache_path, row, ids, protocol_sha, old_sha, source_sha)
                    report["subjects"].append(row)
                    persist()
                    print(json.dumps({"event": "subject_resumed", "subject": subject}), flush=True)
                    continue
                labels = load_labels(archive.read(f"dataset_release/aligned_data/pose_labels/subject{subject}_all_labels.cpl"))
                timestamps = np.load(io.BytesIO(archive.read(f"dataset_release/raw_data/unixtime/subject{subject}_unix.npy")), allow_pickle=False)
                focal = np.diag(labels["camera_matrix"]["cam1_intrinsic"])[:2]
                if not np.array_equal(timestamps[old_values["frame_ids"]], old_values["timestamps_ms"]):
                    raise ValueError("Original frame timestamps changed")
                cap = cv2.VideoCapture(str(path))
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not cap.isOpened() or count != len(timestamps) or count != len(labels["refined_gt_kps"]) or (width, height) != (1920, 1080) or not np.isfinite(fps) or fps <= 0:
                    cap.release()
                    raise ValueError("Source video header differs from published frame sequence")
                poses = {key: np.zeros((96, 17, 3), dtype=np.float32) for key in CONDITIONS}
                old_positions = np.searchsorted(old_values["frame_ids"], ids[reused])
                for condition in ("clean", "dark"):
                    poses[condition][reused] = old_values[condition + "_poses"][old_positions]
                positions, seed_words = [], []
                pixel_hashes = {key: [] for key in CONDITIONS}
                durations = {key: [] for key in CONDITIONS}
                started = time.perf_counter()
                try:
                    for index, frame_id in enumerate(ids):
                        if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id)):
                            raise ValueError("Video frame seek failed")
                        ok, image = cap.read()
                        position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        if not ok or position != int(frame_id) + 1:
                            raise ValueError("Selected actual frame did not decode at expected index")
                        image = cv2.flip(image, 1)
                        probes, words = pixel_probes(image, subject, frame_id)
                        probes["clean"] = image
                        probes["dark"] = (image.astype(np.float32) * .18).astype(np.uint8)
                        seed_words.append(words)
                        positions.append(position)
                        for condition in CONDITIONS:
                            pixel_hashes[condition].append(pixel_hash(probes[condition]))
                            if reused[index] and condition in ("clean", "dark"):
                                continue
                            infer_started = time.perf_counter()
                            poses[condition][index] = independent_pose(model, probes[condition])
                            durations[condition].append((time.perf_counter() - infer_started) * 1000)
                        if (index + 1) % 16 == 0:
                            report["current_subject_decoded_frames"] = index + 1
                            persist()
                            print(json.dumps({"event": "frame_progress", "subject": subject, "decoded": index + 1,
                                              "elapsed_s": time.perf_counter() - started}), flush=True)
                finally:
                    cap.release()
                values = {"frame_ids": ids, "timestamps_ms": timestamps[ids], "reused_clean_dark": reused,
                          "decode_next_frame_positions": np.asarray(positions), "pixel_noise_seed_words": np.asarray(seed_words),
                          "source_sha256": source_sha, "source_crc32": np.uint32(crc), "model_sha256": MODEL_SHA,
                          "video_archive_sha256": VIDEO_SHA, "data_archive_sha256": data_audit["archive_sha256"],
                          "old_cache_sha256": old_sha, "protocol_sha256": protocol_sha}
                condition_summary = {}
                for condition in CONDITIONS:
                    features, available = pose_features(poses[condition], focal)
                    if condition in ("clean", "dark"):
                        if not np.array_equal(features[reused], old_values[condition + "_features"][old_positions]):
                            raise ValueError("Reused feature coordinates differ from source cache")
                        features[reused] = old_values[condition + "_features"][old_positions]
                    values[condition + "_poses"] = poses[condition]
                    values[condition + "_features"] = features
                    values[condition + "_available"] = available
                    values[condition + "_missing"] = ~available[:, 11]
                    values[condition + "_pixel_sha256"] = np.asarray(pixel_hashes[condition])
                    condition_summary[condition] = {"root_available_frames": int(available[:, 11].sum()),
                                                    "missing_frames": int((~available[:, 11]).sum()),
                                                    "visible_joint_fraction": float((poses[condition][..., 2] >= .3).mean()),
                                                    "new_inference_count": len(durations[condition]),
                                                    "inference_median_ms": float(np.median(durations[condition]))}
                temporary = cache_path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **values)
                cache_sha = file_checks(temporary)[0]
                row = {"subject": subject, "sampled_frames": 96, "reused_clean_dark_frames": 64, "additional_frames": 32,
                       "header_frames": count, "header_fps": fps, "decoded_frame_index_verified": True,
                       "extracted_video_crc_verified": True, "extracted_video_sha256_matches_original_cache": True,
                       "source_sha256": source_sha, "source_crc32": crc, "source_bytes": path.stat().st_size,
                       "old_cache_sha256": old_sha, "cache_sha256": cache_sha, "seconds": time.perf_counter() - started,
                       "conditions": condition_summary}
                save_json(row_path, row)
                os.replace(temporary, cache_path)
                report["subjects"].append(row)
                persist()
                print(json.dumps({"event": "subject_complete", **row}), flush=True)
        report["status"] = "complete"
        report["current_subject"] = None
        report["current_subject_decoded_frames"] = 0
        report["software"] = protocol["software"]
        persist()
        print(json.dumps({"event": "complete", "subjects": 16, "frames": 1536}), flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        persist()
        raise


if __name__ == "__main__":
    main()
