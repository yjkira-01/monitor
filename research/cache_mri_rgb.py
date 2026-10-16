"""Extract independent image poses from verified mRI development videos."""

import argparse
import hashlib
import io
import json
import re
import time
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from audit_mri import load_labels
from run_mri_baselines import subject_samples

ROOT = Path(__file__).resolve().parents[1]
VIDEO_SHA = "1a95d3993d564ba028aa7f7f53c1a23ac35edd4f7196a951399529738873149e"
MODEL_SHA = "750d872151ff4652f02c738efc3a547e112ce9b688fe920bc17f948e8c3afdac"


def file_checks(path):
    digest = hashlib.sha256()
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return digest.hexdigest(), crc


def video_manifest(path):
    if path.stat().st_size != 14988106753 or file_checks(path)[0] != VIDEO_SHA:
        raise ValueError("Video archive differs from the verified Dryad release")
    with zipfile.ZipFile(path) as archive:
        members = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            match = re.fullmatch(r"blurred_videos/subject(\d+)_\d{4}/subject\1_color([01])\.mp4", info.filename)
            if not match or not 0 < info.file_size <= 1024 ** 3 or info.flag_bits & 1:
                raise ValueError("Unexpected video member")
            key = (int(match[1]), int(match[2]))
            if key in members:
                raise ValueError("Duplicate video member")
            members[key] = {"member": info.filename, "bytes": info.file_size,
                            "crc32": info.CRC, "compression_method": info.compress_type}
        if set(members) != {(s, v) for s in range(1, 21) for v in (0, 1)}:
            raise ValueError("Expected two views for all twenty subjects")
    return members


def independent_pose(model, image):
    ratio = min(640 / image.shape[0], 640 / image.shape[1])
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    small = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
    padded[:small.shape[0], :small.shape[1]] = small
    dets, poses = model.run(None, {model.get_inputs()[0].name: padded.transpose(2, 0, 1)[None].astype(np.float32)})
    if (dets.ndim != 3 or poses.shape != (1, dets.shape[1], 17, 3) or
            dets.shape[0] != 1 or dets.shape[2] != 5 or
            not np.isfinite(dets).all() or not np.isfinite(poses).all()):
        raise ValueError("Unexpected RTMO output")
    # Single-person research: select by detector score, never by GT proximity.
    eligible = np.flatnonzero(dets[0, :, 4] >= .7)
    if not eligible.size:
        return np.zeros((17, 3), dtype=np.float32)
    selected = eligible[np.argmax(dets[0, eligible, 4])]
    result = poses[0, selected].copy()
    result[:, :2] /= ratio
    outside = ((result[:, :2] < 0) | (result[:, :2] >= [image.shape[1], image.shape[0]])).any(-1)
    result[outside, 2] = 0
    return result


def pose_features(poses, focal):
    poses = np.asarray(poses, dtype=np.float64)
    focal = np.asarray(focal, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (17, 3) or focal.shape != (2,) or not np.isfinite(poses).all() or not np.isfinite(focal).all() or (focal <= 0).any():
        raise ValueError("Invalid image poses or focal lengths")
    visible = poses[..., 2] >= .3
    root_available = visible[:, 11] & visible[:, 12]
    available = visible & root_available[:, None]
    xy = (poses[..., :2] - poses[:, [11, 12], :2].mean(1)[:, None]) / focal
    xy = np.where(available[..., None], xy, 0.)
    confidence = np.where(available, poses[..., 2], 0.)
    return np.concatenate([xy.reshape(len(poses), -1), confidence], axis=-1), available


def projection_errors(labels, frame_ids):
    # Published GT metres -> calibration millimetres; x/y axes are reversed.
    points = labels["refined_gt_kps"][frame_ids].transpose(0, 2, 1) * [-1000., -1000., 1000.]
    points = np.concatenate([points, np.ones((*points.shape[:2], 1))], axis=-1)
    result = {}
    for camera, side in (("cam1", "r"), ("cam2", "l")):
        projected = points @ labels["camera_matrix"][camera + "_proj_mat"].T
        if not np.isfinite(projected).all() or (abs(projected[..., 2]) < 1e-8).any():
            raise ValueError("Invalid label projection")
        pixels = projected[..., :2] / projected[..., 2:]
        error = np.linalg.norm(pixels - labels["pose_2d_" + side][frame_ids].transpose(0, 2, 1), axis=-1)
        result[camera] = {"published_side": side, "median_px": float(np.median(error)), "p95_px": float(np.percentile(error, 95))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--video-directory", type=Path, default=ROOT / "artifacts/research_execution_20261012/rgb_audit")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research_execution_20261012/rgb_features")
    args = parser.parse_args()
    split = json.loads((ROOT / "research/pose_fusion_protocol.json").read_text(encoding="utf-8"))["splits"][0]
    allowed = split["canonical_ids_train"] + split["canonical_ids_validation"]
    if not 1 <= args.samples <= 256 or len(set(args.subjects)) != len(args.subjects) or not set(args.subjects) <= set(allowed):
        raise ValueError("Only distinct development subjects and 1..256 samples are allowed")
    args.output.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        if (args.output / f"subject{subject}.npz").exists():
            raise ValueError("Refusing to overwrite an existing feature cache")
    manifest = video_manifest(ROOT / "blurred_videos.zip")
    (args.output / "video_manifest.json").write_text(json.dumps({"archive_sha256": VIDEO_SHA, "entries": list(manifest.values())}, indent=2), encoding="utf-8")
    model_path = ROOT / "desktop/CareMonitor/Assets/Realtime/rtmo.onnx"
    if file_checks(model_path)[0] != MODEL_SHA:
        raise ValueError("RTMO checksum mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 1
    model = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    cv2.setNumThreads(1)
    report = {"status": "running", "scope": "Independent RGB development probe, not final pose accuracy",
              "video_archive_sha256": VIDEO_SHA, "model_sha256": MODEL_SHA,
              "model_source": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
              "input": "color0, horizontal reflection undone, full-frame detector; no GT boxes or GT keypoints",
              "conditions": {"clean": "unchanged decoded pixels after reflection", "dark": "floor(pixel*0.18) before independent pose inference; no enhancement"},
              "sampling": "64 evenly spaced positions within the earlier 256 random screened development frame IDs" if args.samples == 64 else f"{args.samples} positions within the earlier 256 random development frame IDs",
              "test_subjects_used": [], "subjects": []}
    with zipfile.ZipFile(ROOT / "dataset_release.zip") as data:
        for subject in args.subjects:
            prefix = "dataset_release/"
            labels = load_labels(data.read(prefix + f"aligned_data/pose_labels/subject{subject}_all_labels.cpl"))
            timestamps = np.load(io.BytesIO(data.read(prefix + f"raw_data/unixtime/subject{subject}_unix.npy")), allow_pickle=False)
            samples = subject_samples(data, subject, split["seed"])
            chosen = np.linspace(0, len(samples["frame_ids"]) - 1, min(args.samples, len(samples["frame_ids"])), dtype=int)
            frame_ids = samples["frame_ids"][chosen]
            path = args.video_directory / f"subject{subject}_color0.mp4"
            entry = manifest[subject, 0]
            source_hash, crc = file_checks(path)
            if path.stat().st_size != entry["bytes"] or crc != entry["crc32"]:
                raise ValueError("Extracted video CRC or size mismatch")
            cap = cv2.VideoCapture(str(path))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or count != len(timestamps) or count != len(labels["refined_gt_kps"]) or (width, height) != (1920, 1080) or not np.isfinite(fps) or fps <= 0:
                cap.release()
                raise ValueError("Video header and published frame counts do not match")
            poses = {key: [] for key in report["conditions"]}
            durations = []
            try:
                for frame_id in frame_ids:
                    if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id)):
                        raise ValueError("Video frame seek failed")
                    ok, image = cap.read()
                    if not ok or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_id + 1:
                        raise ValueError("Selected video frame did not decode at the expected index")
                    image = cv2.flip(image, 1)
                    for condition in poses:
                        pixels = image if condition == "clean" else (image.astype(np.float32) * .18).astype(np.uint8)
                        started = time.perf_counter()
                        poses[condition].append(independent_pose(model, pixels))
                        durations.append((time.perf_counter() - started) * 1000)
            finally:
                cap.release()
            poses = {key: np.asarray(value) for key, value in poses.items()}
            focal = np.diag(labels["camera_matrix"]["cam1_intrinsic"])[:2]
            published = labels["pose_2d_r"][frame_ids].transpose(0, 2, 1)
            valid = poses["clean"][..., 2] >= .3
            errors = np.linalg.norm(poses["clean"][..., :2] - published, axis=-1)
            features = {key: pose_features(value, focal)[0] for key, value in poses.items()}
            np.savez_compressed(args.output / f"subject{subject}.npz", frame_ids=frame_ids,
                                timestamps_ms=timestamps[frame_ids], clean_poses=poses["clean"], dark_poses=poses["dark"],
                                clean_features=features["clean"], dark_features=features["dark"],
                                source_sha256=source_hash, model_sha256=MODEL_SHA, video_archive_sha256=VIDEO_SHA)
            row = {"subject": subject, "sampled_frames": len(frame_ids), "header_frames": count,
                   "header_fps": fps, "encoded_duration_s": count / fps,
                   "acquisition_span_s": float((timestamps[-1] - timestamps[0]) / 1000),
                   "decoded_frame_index_verified": True, "extracted_video_crc_verified": True,
                   "clean_agreement_with_published_2d_median_px": float(np.median(errors[valid])) if valid.any() else None,
                   "projection_audit": projection_errors(labels, frame_ids),
                   "inference_median_ms": float(np.median(durations)),
                   "conditions": {key: {"root_available_frames": int((pose_features(value, focal)[1][:, 11]).sum()),
                                        "visible_joint_fraction": float((value[..., 2] >= .3).mean())} for key, value in poses.items()}}
            report["subjects"].append(row)
            (args.output / "rgb_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
    report["status"] = "complete"
    report["software"] = {"numpy": np.__version__, "opencv": cv2.__version__, "onnxruntime": ort.__version__}
    (args.output / "rgb_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
