"""Descriptive activation scales without fitting or final-test access."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite diagnostics")
    report_path = args.cache / "cache_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "complete" or report["final_test_subjects_used"]:
        raise ValueError("Complete development-only cache required")
    hashes = {row["subject"]: row["tokens_sha256"] for row in report["subjects"]}
    groups = {}
    rows = []
    keys = ("clean_rgb_CxT", "dark_rgb_CxT", "noisy_dark_rgb_CxT", "occlusion_rgb_CxT", "radar_tokens", "radar_sparse_tokens")
    for group, subjects in (("train", report["train_subjects"]), ("development", report["validation_subjects"])):
        sums = {key: {"count": 0, "sum": 0., "square_sum": 0., "abs_sum": 0., "max_abs": 0.} for key in keys}
        for subject in subjects:
            path = args.cache / f"subject{subject}_tokens.npz"
            with path.open("rb") as stream:
                observed_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if observed_hash != hashes[subject]:
                raise ValueError("Cache changed")
            with np.load(path, allow_pickle=False) as data:
                for key in keys:
                    values = data[key].astype(np.float64)
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite native activations")
                    record = sums[key]
                    record["count"] += values.size
                    record["sum"] += float(values.sum())
                    record["square_sum"] += float(np.square(values).sum())
                    record["abs_sum"] += float(np.abs(values).sum())
                    record["max_abs"] = max(record["max_abs"], float(np.abs(values).max()))
                    rows.append({"group": group, "subject": subject, "feature": key,
                                 "rms": float(np.sqrt(np.square(values).mean())), "mean": float(values.mean()),
                                 "std": float(values.std()), "max_abs": float(np.abs(values).max())})
        groups[group] = {}
        for key, values in sums.items():
            mean = values["sum"] / values["count"]
            mean_square = values["square_sum"] / values["count"]
            groups[group][key] = {"count": values["count"], "mean": mean, "rms": float(np.sqrt(mean_square)),
                                 "std": float(np.sqrt(max(0., mean_square - mean ** 2))),
                                 "mean_abs": values["abs_sum"] / values["count"], "max_abs": values["max_abs"]}
    result = {"scope": "Descriptive frozen native feature scales; no added training, model selection or final-test evaluation",
              "cache_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
              "groups": groups, "per_subject": rows, "final_test_subjects_used": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "complete", "groups": groups}))


if __name__ == "__main__":
    main()
