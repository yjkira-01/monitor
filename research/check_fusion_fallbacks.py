"""Exact saved-array fallback identities after a completed strict replay receipt."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_fusion_trials import safe_npz, sha256, write_json


def check_fallbacks(experiment, analysis=None):
    experiment = Path(experiment).resolve()
    analysis = Path(analysis).resolve() if analysis else experiment / "analysis"
    destination = analysis / "fallback_verification.json"
    if destination.exists():
        raise ValueError("Refusing to overwrite fallback receipt")
    report_path = experiment / "run_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verified = json.loads((analysis / "verification.json").read_text(encoding="utf-8"))
    if report["status"] != "complete" or verified["status"] != "complete" or verified["run_summary_sha256"] != sha256(report_path):
        raise ValueError("Matching complete independent replay receipt required")
    clean_available, subject_ids, frame_ids = [], [], []
    cache = Path(report["input_cache"])
    representation = report.get("representation", "hip")
    for subject in report["validation_subjects"]:
        path = cache / f"subject{subject}.npz"
        if sha256(path) != report["cache_sha256"][str(subject)]:
            raise ValueError("Validation cache checksum changed")
        with np.load(path, allow_pickle=False) as saved:
            if representation == "hip":
                confidence = saved["clean_features"][:, 34:].astype(np.float32)
            elif representation in ("visible_centroid", "hip_centroid_fallback", "hip_confidence_only"):
                confidence = saved["clean_poses"][:, :, 2].astype(np.float32)
            else:
                raise ValueError("Unsupported representation availability rule")
            if confidence.shape[1] != 17 or not np.isfinite(confidence).all():
                raise ValueError("Invalid clean confidence array")
            clean_available.append((confidence >= .3).any(1))
            subject_ids.append(np.full(len(confidence), subject))
            frame_ids.append(saved["frame_ids"].copy())
    available, subjects, frames = [np.concatenate(x) for x in (clean_available, subject_ids, frame_ids)]
    checks = []
    manifest = {(r["seed"], r["method"]): r for r in verified["models"]}
    for run in report["runs"]:
        for method in report["methods"]:
            path = experiment / f"seed{run['seed']}" / f"{method}_predictions.npz"
            if sha256(path) != manifest[run["seed"], method]["prediction_sha256"]:
                raise ValueError("Verified prediction file changed")
            saved = safe_npz(path)
            if not np.array_equal(saved["subject_ids"], subjects) or not np.array_equal(saved["frame_ids"], frames):
                raise ValueError("Prediction cache order changed")
            identities = []

            def identity(condition, branch, rows, reason):
                fused = saved[condition + "_fused_prediction_mm"][rows]
                reference = saved[condition + "_" + branch + "_prediction_mm"][rows]
                if not np.array_equal(fused, reference):
                    raise ValueError(f"Fallback identity failed: seed{run['seed']}/{method}/{condition}/{branch}")
                identities.append({"condition": condition, "branch": branch,
                                   "frames": int(rows.sum()), "exact_equal": True, "reason": reason})

            all_rows = np.ones(len(available), dtype=bool)
            if method in ("rgb_single", "rgb_matched", "radar_single"):
                branch = "radar" if method == "radar_single" else "video"
                for condition in report["evaluation_conditions"]:
                    identity(condition, branch, all_rows, "Independent single model cannot borrow the opposite modality")
            else:
                identity("video_missing", "radar", all_rows, "Video unavailable; final where selects radar, including any both-missing prior")
                identity("radar_missing", "video", available, "Radar unavailable with clean video available")
                identity("radar_missing", "radar", ~available, "Both unavailable; declared final radar zero-feature prior, not an observation")
                metrics = run["models"][method]["metrics"]["radar_missing"]
                if metrics["image_available_frames"] != int(available.sum()) or metrics["both_modalities_missing_frames"] != int((~available).sum()):
                    raise ValueError("Fallback availability count differs from recorded metrics")
            checks.append({"seed": run["seed"], "method": method, "identities": identities})
    result = {"status": "complete", "experiment": str(experiment), "source_sha256": sha256(Path(__file__)),
        "run_summary_sha256": sha256(report_path), "independent_replay_receipt_sha256": sha256(analysis / "verification.json"),
        "verification_device": "CPU saved-array checks only", "prediction_model_count": len(checks),
        "representation": representation, "clean_available_frames": int(available.sum()),
        "clean_unavailable_frames": int((~available).sum()), "checks": checks,
        "scope": "Fallback identities and prior policy; a 2-epoch experiment remains a runtime smoke, not pose performance evidence"}
    write_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    result = check_fallbacks(args.experiment, args.analysis)
    print(json.dumps({"status": result["status"], "models": result["prediction_model_count"],
                      "clean_available": result["clean_available_frames"], "clean_unavailable": result["clean_unavailable_frames"]}), flush=True)


if __name__ == "__main__":
    main()
