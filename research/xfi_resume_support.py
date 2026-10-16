"""Verify stopped experiments and copy completed models without retraining."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_prior(directory, cache):
    directory = directory.resolve()
    report_path = directory / "run_summary.json"
    stop_path = directory / "stop_receipt.json"
    report, stop = [json.loads(p.read_text(encoding="utf-8")) for p in (report_path, stop_path)]
    if (report["status"] != "stopped_by_user" or stop["status"] != "stopped_by_user" or
            report["initialization_seeds"] != [7, 17, 27] or report["training"]["epochs"] != 100 or
            report["training"]["lr"] != .001 or report["final_test_subjects_used"] or
            stop["final_test_subjects_used"] or report["cache_report_sha256"] != sha(cache / "cache_report.json")):
        raise ValueError("Original stopped, source-matched development experiment required")
    for name, expected in report["source_sha256"].items():
        if Path(name).name != name or sha(ROOT / "research" / name) != expected or sha(directory / "source_snapshot" / name) != expected:
            raise ValueError("Original execution source changed")
    protocol = "xfi_accuracy_protocol_20261013.md"
    if (sha(ROOT / "research" / protocol) != report["protocol_sha256"] or
            sha(directory / "source_snapshot" / protocol) != report["protocol_sha256"]):
        raise ValueError("Original frozen protocol changed")
    models = {}
    expected_models = {(7, "native_mse"), (7, "native_mpjpe"), (7, "root_mpjpe"), (17, "native_mse"), (17, "native_mpjpe")}
    receipts = {(row["seed"], row["method"]): row for row in stop["completed_formal_models"]}
    for run in report["runs"]:
        training = json.loads((directory / f"seed{run['seed']}" / "training.json").read_text(encoding="utf-8"))
        if training != run:
            raise ValueError("Stopped summary and per-initialization training record disagree")
        for method, row in run["models"].items():
            key = run["seed"], method
            if key in models or key not in expected_models or row["history"][-1]["epoch"] != 100:
                raise ValueError("Unexpected or incomplete saved model")
            if not row["frozen_backbone_parameters_and_buffers_unchanged"]:
                raise ValueError("Original backbone was not frozen")
            for kind in ("weights", "predictions"):
                path = directory / f"seed{run['seed']}" / f"{method}_{kind}.npz"
                actual = sha(path)
                if actual != row[kind + "_sha256"] or actual != receipts[key][kind + "_sha256"]:
                    raise ValueError("Completed saved model changed")
            models[key] = row
    if set(models) != expected_models or set(receipts) != expected_models:
        raise ValueError("Exactly five completed original models required")
    provenance = {"source_directory": str(directory), "source_run_summary_sha256": sha(report_path),
                  "stop_receipt_sha256": sha(stop_path), "completed_models_reused": 5,
                  "missing_models_to_train": 4, "restart_interrupted_model": "seed17/root_mpjpe from initialization, 100 epochs",
                  "copied_files": []}
    return models, provenance


def stage_completed(seed, source, output, prior, provenance):
    result = {method: row for (old_seed, method), row in prior.items() if seed == old_seed}
    references = {(row["initial_state_sha256"], row["schedule_sha256"]) for row in result.values()}
    if len(references) > 1:
        raise ValueError("Reused objective controls lost pairing")
    for method, row in result.items():
        for kind in ("weights", "predictions"):
            name = f"{method}_{kind}.npz"
            old, new = source / f"seed{seed}" / name, output / f"seed{seed}" / name
            if new.exists():
                raise ValueError("Refusing to overwrite model output")
            shutil.copyfile(old, new)
            if sha(new) != row[kind + "_sha256"]:
                raise ValueError("Reused model copy differs")
            provenance["copied_files"].append({"path": f"seed{seed}/{name}", "sha256": row[kind + "_sha256"]})
    initial, schedule = next(iter(references)) if references else (None, None)
    return result, initial, schedule
