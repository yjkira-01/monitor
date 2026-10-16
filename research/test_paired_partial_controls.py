"""Focused rejection checks for the frozen partial comparison contract."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_paired_partial_controls import METHODS, SEEDS, compare_partial_contracts, equal_statistics, file_sha256, paired_group_aggregates, verified_receipt
from analyze_paired_representations import group_metrics


def reports():
    hip = {"status": "complete", "representation": "hip", "methods": METHODS.copy(),
           "data_split_seed": 7, "initialization_seeds": SEEDS.copy(), "train_frames": 1152, "validation_frames": 384,
           "train_subjects": [1, 2, 4, 7, 8, 9, 11, 13, 14, 15, 16, 17], "validation_subjects": [5, 6, 19, 20],
           "final_test_subjects_used": [], "parameters": dict.fromkeys(METHODS, 100),
           "training": {"epochs": 80, "batch_size": 64, "normalization": "clean"},
           "environment": {"device": "cuda", "deterministic": True, "mixed_precision": False, "FP32_matmul": "IEEE, TF32 disabled"},
           "source_sha256": {"fusion_models.py": "model", "fusion_pose_representation.py": "pose", "old_runner.py": "old"},
           "runner_file": "old_runner.py", "runs": [{"seed": seed, "status": "complete"} for seed in SEEDS]}
    geometry = copy.deepcopy(hip)
    geometry.update(representation="hip_centroid_fallback", rgb_normalization_key="hip_clean",
                    runner_file="partial_runner.py", protocol_sha256="protocol")
    geometry["training"]["normalization"] = "hip_clean"
    geometry["source_sha256"] = {"fusion_models.py": "model", "fusion_pose_representation.py": "pose",
                                 "partial_runner.py": "new", "fusion_partial_representation.py": "helper"}
    confidence = copy.deepcopy(geometry)
    confidence["representation"] = "hip_confidence_only"
    return geometry, confidence, hip


class PartialPairTests(unittest.TestCase):
    def test_new_runner_and_helper_allowed_against_original_hip(self):
        geometry, confidence, hip = reports()
        self.assertEqual(compare_partial_contracts(geometry, confidence, hip), [])

    def test_partial_source_mismatch_rejected(self):
        geometry, confidence, hip = reports()
        confidence["source_sha256"]["fusion_partial_representation.py"] = "changed"
        self.assertIn("Different partial run contract: source_sha256", compare_partial_contracts(geometry, confidence, hip))

    def test_shared_wrong_epoch_budget_rejected(self):
        geometry, confidence, hip = reports()
        for report in (geometry, confidence):
            report["training"]["epochs"] = 1
        self.assertTrue(any("expected fixed80" in issue for issue in compare_partial_contracts(geometry, confidence, hip)))

    def test_full_normalization_requires_confidence_and_geometry_exact(self):
        original = {"rgb_mean": np.arange(51, dtype=np.float64), "rgb_std": np.ones(51)}
        changed = copy.deepcopy(original)
        changed["rgb_std"][34] += .001
        self.assertFalse(equal_statistics(original, changed))
        self.assertTrue(equal_statistics(original, copy.deepcopy(original)))

    def test_weighted_contribution_and_absent_joint_null(self):
        visible = np.zeros((4, 17), dtype=bool)
        visible[:2, :2] = True
        masks = {"all": np.ones(4, dtype=bool), "hip_valid": np.array([True, False, False, False]),
                 "partial": np.array([False, True, False, False]), "all_raw_missing": np.array([False, False, True, True])}
        per_seed = {}
        for seed in SEEDS:
            per_seed[str(seed)] = {}
            for name, values in (("confidence", [10, 20, 30, 40]), ("geometry", [8, 12, 32, 42])):
                errors = np.repeat(np.array(values)[:, None], 17, axis=1)
                per_seed[str(seed)][name] = {method: {group: group_metrics(errors, mask, visible, [5], np.full(4, 5))
                    for group, mask in masks.items()} for method in METHODS}
        result = paired_group_aggregates(per_seed, "confidence", "geometry", masks, [5])["concat"]
        self.assertAlmostEqual(result["all"]["left_minus_right_gain"]["MPJPE_mm"]["mean"], 1.5)
        self.assertAlmostEqual(result["partial"]["contribution_to_full_occlusion_gain_mm"], 2.)
        self.assertAlmostEqual(result["all_raw_missing"]["contribution_to_full_occlusion_gain_mm"], -1.)
        self.assertIsNone(result["all_raw_missing"]["left_minus_right_gain"]["visible_joint_error_mm"])

    def test_empty_subject_subgroup_remains_null(self):
        visible = np.zeros((4, 17), dtype=bool)
        masks = {"all": np.ones(4, dtype=bool), "hip_valid": np.array([True, False, False, False]),
                 "partial": np.array([False, True, False, False]), "all_raw_missing": np.array([False, False, True, True])}
        per_seed = {str(seed): {name: {method: {group: group_metrics(np.ones((4, 17)), mask, visible, [5, 6], np.full(4, 5))
            for group, mask in masks.items()} for method in METHODS} for name in ("confidence", "geometry")} for seed in SEEDS}
        result = paired_group_aggregates(per_seed, "confidence", "geometry", masks, [5, 6])
        self.assertIsNone(result["concat"]["partial"]["per_subject_MPJPE_gain_mm"]["6"])

    def test_changed_prediction_after_independent_replay_rejected(self):
        test_parent = Path(__file__).resolve().parents[1] / "artifacts/partial_pair_test_tmp"
        test_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_parent) as temporary:
            root = Path(temporary)
            assert root.resolve().is_relative_to(test_parent.resolve())
            (root / "analysis").mkdir()
            (root / "run_summary.json").write_text("{}", encoding="utf-8")
            (root / "normalization.npz").write_bytes(b"normalization")
            rows = []
            for seed in SEEDS:
                (root / f"seed{seed}").mkdir()
                for method in METHODS:
                    path = root / f"seed{seed}/{method}_predictions.npz"
                    path.write_bytes(b"prediction")
                    rows.append({"seed": seed, "method": method, "prediction_sha256": file_sha256(path),
                                 "all_predictions_and_metrics_match": True, "finite_state_and_strict_reload": True})
            receipt = {"status": "complete", "run_summary_sha256": file_sha256(root / "run_summary.json"),
                       "normalization_sha256": file_sha256(root / "normalization.npz"), "models": rows}
            (root / "analysis/verification.json").write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(len(verified_receipt(root)["prediction_SHA256"]), 9)
            (root / "seed7/concat_predictions.npz").write_bytes(b"changed prediction")
            with self.assertRaisesRegex(ValueError, "changed after independent replay"):
                verified_receipt(root)


if __name__ == "__main__":
    unittest.main()
