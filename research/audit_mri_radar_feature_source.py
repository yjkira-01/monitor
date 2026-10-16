"""Finite input-source audit on one previously selected development radar frame.

Reads no pose labels and does not authenticate full dataset/video ZIPs again.
The raw/feature correspondence is a local observation, not proof of GT-free
preprocessing throughout the published dataset.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/fusion_representation_audit_20261013/mri_radar_feature_source_audit.json"


def sorted_geometry(rows: np.ndarray) -> np.ndarray:
    return rows[np.lexsort(rows[:, :4].T[::-1]), :4]


def member_metadata(archive: zipfile.ZipFile, name: str) -> dict:
    info = archive.getinfo(name)
    return {"name": name, "bytes": info.file_size, "CRC32": f"{info.CRC:08x}"}


def main() -> None:
    subject = 1
    prior_audit = json.loads((ROOT / "artifacts/research_execution_20261012/mri_data_audit.json").read_text(encoding="utf-8"))
    subject_audit = next(row for row in prior_audit["subjects"] if row["subject"] == subject)
    cache_path = ROOT / f"artifacts/fusion_rgb_expanded_20261013/subject{subject}.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        camera_frame = int(cache["frame_ids"][0])
    feature_name = f"dataset_release/features/radar/subject{subject}_featuremap.npy"
    aligned_name = f"dataset_release/aligned_data/radar/singleframe/subject{subject}.csv"
    raw_name = "dataset_release/raw_data/radar/subject1_0427.csv"
    with zipfile.ZipFile(ROOT / "dataset_release.zip") as archive:
        members = [member_metadata(archive, name) for name in (feature_name, aligned_name, raw_name)]
        features = np.load(io.BytesIO(archive.read(feature_name)), allow_pickle=False)
        feature = features[camera_frame - subject_audit["radar_availability_inclusive"][0]].reshape(196, 5)
        aligned = np.loadtxt(io.BytesIO(archive.read(aligned_name)), delimiter=",", skiprows=1)
        raw = np.loadtxt(io.BytesIO(archive.read(raw_name)), delimiter=",", skiprows=1)

    assert feature.shape == (196, 5) and np.isfinite(feature).all()
    nonzero = feature[np.any(feature != 0, axis=1)]
    camera_rows = aligned[aligned[:, 8] == camera_frame]
    radar_frame = int(np.unique(camera_rows[:, 0]).item())
    nearby_raw = raw[(raw[:, 0] >= radar_frame - 5) & (raw[:, 0] <= radar_frame + 8)]
    distance = np.max(abs(nonzero[:, None, :4] - nearby_raw[None, :, 2:6]), axis=2)
    nearest = distance.argmin(axis=1)
    matched_raw = nearby_raw[nearest]
    maximum_distance = float(distance.min(axis=1).max())
    assert maximum_distance <= 1e-4, "Selected released geometry does not match nearby raw radar"
    coefficient = np.polyfit(matched_raw[:, 6], nonzero[:, 4], 1)
    source_frames, frame_counts = np.unique(matched_raw[:, 0].astype(np.int64), return_counts=True)

    windows = []
    for offsets in ((-2, -1, 0), (-1, 0, 1), (0, 1, 2)):
        rows = np.concatenate([aligned[aligned[:, 8] == camera_frame + offset, 2:7] for offset in offsets])
        equal = len(rows) == len(nonzero) and np.array_equal(sorted_geometry(rows), sorted_geometry(nonzero))
        windows.append({"camera_offsets": list(offsets), "point_rows": len(rows), "XYZ_Doppler_multiset_exact": bool(equal)})
    assert windows[1]["XYZ_Doppler_multiset_exact"]

    selected_aligned = np.concatenate([aligned[aligned[:, 8] == camera_frame + offset] for offset in (-1, 0, 1)])
    frame_receipt = OUTPUT.with_name("mri_radar_selected_frame_receipt.npz")
    receipt_arrays = {"feature_196x5": feature, "matched_raw_rows_for_each_nonzero_feature": matched_raw,
                      "aligned_neighbor_camera_rows": selected_aligned,
                      "camera_frame": np.array(camera_frame, dtype=np.int64),
                      "subject": np.array(subject, dtype=np.int64)}
    np.savez_compressed(frame_receipt, **receipt_arrays)
    array_receipts = {name: {"shape": list(array.shape), "dtype": str(array.dtype),
                            "contiguous_array_SHA256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
                      for name, array in receipt_arrays.items()}

    source_zip = ROOT / "artifacts/research_execution_20261012/mri_source_zip.zip"
    with zipfile.ZipFile(source_zip) as archive:
        source_members = archive.namelist()
    report = {
        "status": "complete",
        "scope": "One already selected development frame, published input source; no training or final-test data reads",
        "subject": subject,
        "camera_frame": camera_frame,
        "aligned_radar_frame": radar_frame,
        "selection": "First frame ID in the frozen expanded subject1 development cache; no GT or performance selection",
        "pose_labels_read": False,
        "final_test_subjects_used": [],
        "full_archive_rehashed": False,
        "archive_SHA256_from_prior_complete_audit": prior_audit["archive_sha256"],
        "members": members,
        "audit_script_SHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selected_frame_receipt": {"path": str(frame_receipt),
                                   "SHA256": hashlib.sha256(frame_receipt.read_bytes()).hexdigest(),
                                   "arrays": array_receipts},
        "feature_dtype": str(feature.dtype),
        "feature_nonzero_rows": len(nonzero),
        "feature_allzero_padding_rows": int(np.all(feature == 0, axis=1).sum()),
        "camera_window_checks": windows,
        "source_raw_frame_counts_for_feature_rows": {str(frame): int(count) for frame, count in zip(source_frames, frame_counts)},
        "row_count_interpretation": "55 is the number of nonzero published feature rows, not 55 independent physical radar observations: raw frame212 rows are repeated twice. Current quality-gate count uses nonzero rows/196 and must not be called independent physical point density.",
        "feature_to_raw_XYZ_Doppler_max_abs_difference": maximum_distance,
        "coordinate_interpretation": "XYZ and Doppler are exactly released aligned-CSV values, matching original raw values within CSV rounding on this frame. No root translation or point crop occurs relative to all rows of camera frames 137,138,139 in this inspected feature.",
        "intensity_affine_fit_from_raw": {
            "slope": float(coefficient[0]),
            "intercept": float(coefficient[1]),
            "maximum_residual": float(abs(np.polyval(coefficient, matched_raw[:, 6]) - nonzero[:, 4]).max()),
            "normalization_fit_population": "Unconfirmed; an empirical affine identity on this frame does not establish the preprocessing statistics or their scope.",
        },
        "includes_next_camera_frame_points": True,
        "causality_limit": "The published feature for camera138 includes camera139 radar points. Do not claim a strictly causal instantaneous input or zero-lookahead realtime latency from this offline cache.",
        "archived_source_checkout": {
            "path": str(source_zip),
            "SHA256": hashlib.sha256(source_zip.read_bytes()).hexdigest(),
            "entries": len(source_members),
            "content": "Root README/license and ActionFormer action_localization implementation; no radar feature-generation implementation found in this archived checkout.",
        },
        "README_preprocessing_detail": "Only states features are generated from synced data and gives (frames,14,14,5) with XYZ/Doppler/intensity.",
        "main_paper_preprocessing_detail": "Section3.1 describes denoised radar point cloud and merging three consecutive frames, 64 to192 points; refers to SupplementA.4.",
        "supplemental_preprocessing_detail": "Official SupplementA.4 (PDFpages4-5) cites reference1 for point-cloud generation/software/hardware/preprocessing and reference2 for continuous-frame fusion to reduce sparsity. It does not specify a GT body crop/root center, the released intensity fitting population, or the exact camera-neighbor window of every feature.",
        "GT_free_published_preprocessing_verified": False,
        "GT_root_centering_for_selected_frame_observed": False,
        "inference_boundary": "Current local loader does not introduce GT cropping/root-centering into radar inputs. This finite correspondence does not verify every published frame, the preprocessing generator, intensity-statistics fitting or all filtering decisions as GT-free.",
    }
    supplemental_receipt_path = OUTPUT.with_name("mri_supplemental_receipt.json")
    if supplemental_receipt_path.exists():
        report["supplemental_source"] = json.loads(supplemental_receipt_path.read_text(encoding="utf-8"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "subject", "camera_frame", "feature_nonzero_rows", "includes_next_camera_frame_points", "GT_free_published_preprocessing_verified")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
