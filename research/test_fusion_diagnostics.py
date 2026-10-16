import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

from analyze_fusion_trials import (analyze, compare_tree, diagnose_output, gate_diagnostics,
    independent_metrics, load_verified_modules, rebuild_data, roc_auc, safe_npz, segment_oracle, sha256, strict_load_weights)

TEST_TEMP_PARENT = Path(__file__).resolve().parents[1] / "artifacts"


class FixedPrediction(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("target_template_m", torch.zeros(17, 3))

    def forward(self, *inputs):
        n = len(inputs[0])
        pose = self.bias.expand(n, 17, 3)
        valid = torch.ones(n, dtype=torch.bool)
        return dict(video=pose, radar=pose, fused=pose,
                    video_valid=valid, radar_valid=valid, both=valid)


class FusionDiagnosticChecks(unittest.TestCase):
    def test_continuous_oracle_finds_partial_benefit_despite_harmful_full_candidate(self):
        video = np.zeros((2, 17, 3))
        candidate, target = video.copy(), video.copy()
        candidate[..., 0], target[..., 0] = 3., 1.
        actual = video + .7 * (candidate - video)
        actual[1] = 9.  # Inactive fallback must be retained exactly.
        oracle, weight = segment_oracle(video, candidate, target, np.array([True, False]), actual)
        np.testing.assert_allclose(weight[0], 1 / 3)
        np.testing.assert_allclose(oracle[0], target[0])
        np.testing.assert_array_equal(oracle[1], actual[1])
        self.assertTrue((np.linalg.norm(candidate[0] - target[0], axis=-1) > np.linalg.norm(video[0] - target[0], axis=-1)).all())

    def test_oracle_zero_delta_and_endpoint_clipping(self):
        video = np.zeros((3, 17, 3))
        candidate, target = video.copy(), video.copy()
        candidate[1:, :, 0] = 3
        target[0, :, 0], target[1, :, 0], target[2, :, 0] = 1, -2, 5
        oracle, weight = segment_oracle(video, candidate, target, np.ones(3, dtype=bool), video)
        np.testing.assert_array_equal(weight[0], 0)
        np.testing.assert_array_equal(weight[1], 0)
        np.testing.assert_array_equal(weight[2], 1)
        self.assertTrue(np.isfinite(oracle).all())

    def test_auc_uses_tie_average_ranks_and_degenerate_classes_are_null(self):
        self.assertEqual(roc_auc([False, True], [.5, .5]), .5)
        self.assertEqual(roc_auc([False, True, False, True], [.1, .9, .2, .8]), 1.)
        self.assertEqual(roc_auc([False, True], [.9, .1]), 0.)
        self.assertIsNone(roc_auc([True, True], [.2, .8]))
        self.assertIsNone(roc_auc([], []))

    def test_inactive_gates_do_not_affect_reliability(self):
        gate = np.array([[.8] * 8 + [.2] * 9, [1.] * 17])
        gain = np.array([[3.] * 8 + [-2.] * 9, [-999.] * 17])
        diagnostics = gate_diagnostics(gate, gain, gain, np.array([True, False]))
        self.assertEqual(diagnostics["selected_gate_gt_0_5_count"], 8)
        self.assertEqual(diagnostics["beneficial_precision_full_candidate_gain_gt_0"], 1.)
        self.assertEqual(diagnostics["descriptive_ROC_AUC_full_candidate_gain_gt_0"], 1.)
        self.assertEqual(diagnostics["active_weight"]["count"], 17)
        self.assertIsNone(gate_diagnostics(gate, gain, gain, np.array([False, False])))

    def test_joint_diagnostic_reports_binary_endpoint_missed_partial_correction(self):
        video = torch.zeros(1, 17, 3)
        candidate = video.clone()
        candidate[..., 0] = .003
        target = np.zeros((1, 51))
        target.reshape(1, 17, 3)[..., 0] = 1.
        output = dict(video=video, radar=video, fused=video + .7 * candidate,
                      candidate=candidate, gate=torch.full((1, 17), .7),
                      video_valid=torch.tensor([True]), radar_valid=torch.tensor([True]), both=torch.tensor([True]))
        diagnostic = diagnose_output(output, target, np.array([5]), np.zeros(51), "utility_gate")
        self.assertLess(diagnostic["full_candidate_g_equals_1_gain"]["mean_gain_mm"], 0)
        self.assertGreater(diagnostic["oracle"]["gain_vs_video_on_both"]["mean_gain_mm"], 0)
        self.assertEqual(diagnostic["oracle"]["full_candidate_not_beneficial_but_partial_oracle_beneficial_fraction_on_both"], 1.)

    def test_strict_safe_weight_reload_rejects_dtype_keys_nonfinite_and_object(self):
        model = nn.Linear(2, 1)
        original = {k: v.detach().numpy() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_PARENT) as directory:
            path = Path(directory) / "weights.npz"
            np.savez_compressed(path, **original)
            strict_load_weights(model, path)
            changed = dict(original, weight=original["weight"].astype(np.float64))
            np.savez_compressed(path, **changed)
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                strict_load_weights(model, path)
            np.savez_compressed(path, **dict(original, unrelated=np.zeros(1)))
            with self.assertRaisesRegex(ValueError, "keys"):
                strict_load_weights(model, path)
            np.savez_compressed(path, bad=np.array([np.nan]))
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                safe_npz(path)
            np.savez_compressed(path, bad=np.array([object()], dtype=object))
            with self.assertRaises(ValueError):
                safe_npz(path)

    def test_independent_metrics_and_receipt_detect_changed_subject_error(self):
        target, prediction = np.zeros((3, 51)), np.zeros((3, 51))
        prediction.reshape(3, 17, 3)[:, :, 0] = np.array([1, 3, 5])[:, None]
        result = independent_metrics(prediction, target, np.array([5, 5, 6]))
        self.assertEqual(result["MPJPE_mm"], 3.)
        self.assertEqual(result["per_validation_subject_MPJPE_mm"], {"5": 2., "6": 5.})
        tampered = dict(result, per_validation_subject_MPJPE_mm={"5": 2., "6": 4.})
        with self.assertRaisesRegex(ValueError, "Array mismatch"):
            compare_tree(result, tampered, "metric", [])

    def test_graph_factory_requires_pinned_source_and_preserves_adjacency(self):
        research = Path(__file__).resolve().parent
        names = ("fusion_models.py", "run_fusion_experiments.py", "fusion_graph_models.py")
        report = {"model_factory": "fusion_graph_models.GraphFusionModel",
                  "source_sha256": {name: sha256(research / name) for name in names}}
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_PARENT) as directory:
            receipt = {"verified_sources": {}}
            modules, _ = load_verified_modules(Path(directory), report, receipt)
            model = modules.FusionModel("graph_anatomical")
            torch.testing.assert_close(model.residual.adjacency.sum(-1), torch.ones(17))
            self.assertEqual(receipt["model_factory"], report["model_factory"])
            arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
            arrays["residual.adjacency"] = np.eye(17, dtype=np.float32)
            bad = Path(directory) / "wrong_graph.npz"
            np.savez_compressed(bad, **arrays)
            with self.assertRaisesRegex(ValueError, "fixed buffer differs"):
                strict_load_weights(model, bad)
            del report["source_sha256"]["fusion_graph_models.py"]
            with self.assertRaisesRegex(ValueError, "not pinned"):
                load_verified_modules(Path(directory), report, {"verified_sources": {}})

    def test_scaled_factory_and_gpu_runner_use_pinned_modules_without_path_side_effect(self):
        import sys
        research = Path(__file__).resolve().parent
        names = ("fusion_models.py", "run_fusion_experiments_gpu.py", "fusion_graph_models.py",
                 "fusion_scaled_gate_models.py", "fusion_pose_representation.py")
        report = {"model_factory": "fusion_scaled_gate_models.ScaledGateFusionModel",
                  "runner_file": "run_fusion_experiments_gpu.py", "representation": "visible_centroid",
                  "source_sha256": {name: sha256(research / name) for name in names}}
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_PARENT) as directory:
            original_path = sys.path[:]
            receipt = {"verified_sources": {}}
            modules, runner = load_verified_modules(Path(directory), report, receipt)
            self.assertEqual(sys.path, original_path)
            self.assertEqual(receipt["runner_file"], "run_fusion_experiments_gpu.py")
            self.assertIn("representation", runner.load_data.__code__.co_varnames)
            model = modules.FusionModel("scaled_utility_gate")
            arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
            arrays["gate.scale"] = np.ones(12, dtype=np.float32)
            bad = Path(directory) / "wrong_scale.npz"
            np.savez_compressed(bad, **arrays)
            with self.assertRaisesRegex(ValueError, "fixed buffer differs"):
                strict_load_weights(model, bad)

    def test_partial_runner_requires_pinned_helper_and_restores_module_alias(self):
        import sys
        research = Path(__file__).resolve().parent
        names = ("fusion_models.py", "run_fusion_partial_experiments_gpu.py", "fusion_graph_models.py",
                 "fusion_scaled_gate_models.py", "fusion_pose_representation.py", "fusion_partial_representation.py")
        report = {"runner_file": "run_fusion_partial_experiments_gpu.py", "representation": "hip_confidence_only",
                  "rgb_normalization_key": "hip_clean", "source_sha256": {name: sha256(research / name) for name in names}}
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_PARENT) as directory:
            original_path = sys.path[:]
            original_partial = sys.modules.get("fusion_partial_representation")
            _, runner = load_verified_modules(Path(directory), report, {"verified_sources": {}})
            self.assertEqual(sys.path, original_path)
            self.assertIs(sys.modules.get("fusion_partial_representation"), original_partial)
            self.assertEqual(runner.load_data.__globals__["hip_preserving_fallback_features"].__module__,
                             "_verified_fusion_partial_representation")
            del report["source_sha256"]["fusion_partial_representation.py"]
            with self.assertRaisesRegex(ValueError, "Partial representation source is not pinned"):
                load_verified_modules(Path(directory), report, {"verified_sources": {}})

    def test_partial_normalization_reconstructs_original_hip_clean_not_changed_representation(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_PARENT) as directory:
            cache = Path(directory)
            for subject in (1, 5):
                (cache / f"subject{subject}.npz").write_bytes(b"fixture cache identity")
            (cache / "rgb_audit.json").write_text(json.dumps({"conditions": ["clean"]}), encoding="utf-8")
            rows = {}
            for subject, length in ((1, 2), (5, 1)):
                reference = np.tile(np.array([-1., 1.])[:length, None], (1, 51))
                rows[subject] = {"hip_clean": reference, "clean": reference + 100., "target": np.zeros((length, 51)),
                    "points": np.zeros((length, 3, 5)), "mask": np.ones((length, 3), dtype=bool),
                    "frame_ids": np.arange(length)}
            report = {"input_cache": str(cache), "train_subjects": [1], "validation_subjects": [5],
                "cache_sha256": {str(s): sha256(cache / f"subject{s}.npz") for s in (1, 5)},
                "input_audit_sha256": sha256(cache / "rgb_audit.json"), "input_audit_conditions": ["clean"],
                "training": {"conditions": ["clean"]}, "representation": "hip_confidence_only",
                "rgb_normalization_key": "hip_clean", "final_test_subjects_used": [], "train_frames": 2,
                "validation_frames": 1, "evaluation_conditions": ["clean", "video_missing", "radar_missing", "radar_sparse"]}
            split = {"canonical_ids_train": [1], "canonical_ids_validation": [5], "canonical_ids_test": [3]}

            def normalized(rgb, points, mask, norm):
                return torch.tensor(rgb), torch.tensor(points), torch.tensor(mask), torch.ones((len(rgb), 17))

            runner = SimpleNamespace(load_data=lambda *args: (split, rows),
                point_normalization=lambda *args: (np.zeros(5), np.ones(5)),
                normalize_inputs=normalized, sparse_mask=lambda mask: mask)
            norm = {"point_mean": np.zeros(5), "point_std": np.ones(5), "rgb_mean": np.zeros(51),
                    "rgb_std": np.ones(51), "target_mean_mm": np.zeros(51)}
            receipt = {"array_checks": []}
            rebuild_data(report, runner, norm, receipt)
            self.assertEqual(receipt["cache_provenance"]["rgb_normalization_key"], "hip_clean")
            with self.assertRaisesRegex(ValueError, "Array mismatch"):
                rebuild_data(report, runner, dict(norm, rgb_mean=np.full(51, 100.)), {"array_checks": []})
            report["rgb_normalization_key"] = "clean"
            with self.assertRaisesRegex(ValueError, "require shared original hip-clean"):
                rebuild_data(report, runner, norm, {"array_checks": []})

    def test_pipeline_never_computes_diagnostics_before_all_predictions_match(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_PARENT) as directory:
            experiment = Path(directory)
            model = FixedPrediction("concat")
            seed_dir = experiment / "seed7"
            seed_dir.mkdir()
            np.savez_compressed(seed_dir / "concat_weights.npz", **{k: v.detach().numpy() for k, v in model.state_dict().items()})
            norm = dict(point_mean=np.zeros(5), point_std=np.ones(5), rgb_mean=np.zeros(51),
                        rgb_std=np.ones(51), target_mean_mm=np.zeros(51))
            np.savez_compressed(experiment / "normalization.npz", **norm)
            target, subjects, frames = np.zeros((1, 51)), np.array([5]), np.array([1])
            saved = dict(target_mm=target, subject_ids=subjects, frame_ids=frames,
                         clean_fused_prediction_mm=np.ones((1, 51)),
                         clean_video_prediction_mm=np.zeros((1, 51)), clean_radar_prediction_mm=np.zeros((1, 51)))
            np.savez_compressed(seed_dir / "concat_predictions.npz", **saved)
            report = dict(status="complete", environment={"threads": 1, "device": "cpu"}, initialization_seeds=[7], methods=["concat"],
                          parameters={"concat": 1}, validation_subjects=[5], final_test_subjects_used=[],
                          runs=[dict(seed=7, status="complete", models={"concat": {"metrics": {}}})])
            (experiment / "run_summary.json").write_text(json.dumps(report), encoding="utf-8")
            modules = SimpleNamespace(METHODS=("concat",), FusionModel=FixedPrediction)
            values = {"clean": (torch.zeros(1, 51),)}
            with mock.patch("analyze_fusion_trials.load_verified_modules", return_value=(modules, None)), \
                 mock.patch("analyze_fusion_trials.rebuild_data", return_value=(target, subjects, frames, values)), \
                 mock.patch("analyze_fusion_trials.diagnose_output") as diagnose:
                with self.assertRaisesRegex(ValueError, "Array mismatch"):
                    analyze(experiment)
                diagnose.assert_not_called()
            output = experiment / "analysis"
            self.assertFalse((output / "diagnostics.json").exists())
            self.assertEqual(json.loads((output / "verification.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
