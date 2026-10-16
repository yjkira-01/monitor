import unittest
import pickle

import numpy as np
import torch

from pose_fusion import alignment_moments, fuse_diagonal_poses, nearest_timestamp_pairs
from audit_mri import interval_bounds, load_labels
from run_mri_baselines import fit_ridge


class PoseFusionChecks(unittest.TestCase):
    def test_equal_entropy_different_motion_risk(self):
        poses = torch.zeros(2, 3, 1, 3, dtype=torch.float64)
        poses[1, :, 0, 0] = torch.tensor([-200., 0., 200.])
        out = alignment_moments(torch.zeros(2, 3, dtype=torch.float64), poses,
                                torch.full_like(poses, 100.), torch.ones(2, 3, dtype=torch.bool))
        torch.testing.assert_close(out["weights"][0], out["weights"][1])
        torch.testing.assert_close(out["between"][0], torch.zeros(1, 3, dtype=torch.float64))
        self.assertAlmostEqual(out["between"][1, 0, 0].item(), 80000 / 3)
        self.assertAlmostEqual(out["variance"][1, 0, 0].item(), 100 + 80000 / 3)

    def test_total_variance_matches_second_moment(self):
        torch.manual_seed(8)
        mu = torch.randn(2, 5, 17, 3, dtype=torch.float64)
        var = torch.rand_like(mu) + .01
        scores = torch.randn(2, 5, dtype=torch.float64)
        out = alignment_moments(scores, mu, var, torch.ones_like(scores, dtype=torch.bool))
        second = (out["weights"][..., None, None] * (var + mu.square())).sum(-3)
        torch.testing.assert_close(out["variance"], second - out["mean"].square())

    def test_missing_candidates_do_not_pollute_or_create_observations(self):
        mu = torch.full((2, 3, 1, 3), torch.nan)
        var = torch.full_like(mu, torch.nan)
        mu[0, 1] = 7.; var[0, 1] = 4.
        valid = torch.tensor([[False, True, False], [False, False, False]])
        scores = torch.full((2, 3), torch.nan); scores[0, 1] = 2.
        out = alignment_moments(scores, mu, var, valid)
        self.assertEqual(out["available"].tolist(), [True, False])
        self.assertEqual(out["weights"].tolist(), [[0., 1., 0.], [0., 0., 0.]])
        torch.testing.assert_close(out["mean"][0], torch.full((1, 3), 7.))
        self.assertTrue(torch.isfinite(out["variance"]).all())

    def test_gradients_are_finite_including_missing_rows(self):
        scores = torch.zeros(2, 3, requires_grad=True)
        mu = torch.randn(2, 3, 2, 3, requires_grad=True)
        var = torch.ones_like(mu)
        valid = torch.tensor([[True, True, False], [False, False, False]])
        out = alignment_moments(scores, mu, var, valid)
        (out["mean"].square().sum() + out["variance"].sum()).backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.isfinite(mu.grad).all())
        torch.testing.assert_close(scores.grad[1], torch.zeros(3))

    def test_single_candidate_and_batched_shape(self):
        mu = torch.randn(2, 4, 1, 17, 3)
        out = alignment_moments(torch.zeros(2, 4, 1), mu, torch.ones_like(mu),
                                torch.ones(2, 4, 1, dtype=torch.bool))
        torch.testing.assert_close(out["mean"], mu.squeeze(-3))
        torch.testing.assert_close(out["between"], torch.zeros(2, 4, 17, 3))

    def test_invalid_candidate_values_and_shapes_are_rejected(self):
        mu = torch.zeros(3, 1, 3); valid = torch.ones(3, dtype=torch.bool)
        for var in (torch.zeros_like(mu), torch.full_like(mu, torch.inf)):
            with self.assertRaises(ValueError):
                alignment_moments(torch.zeros(3), mu, var, valid)
        with self.assertRaises(ValueError):
            alignment_moments(torch.zeros(2), mu, torch.ones_like(mu), valid)
        with self.assertRaises(ValueError):
            alignment_moments(torch.zeros(0), mu[:0], torch.ones_like(mu[:0]), valid[:0])

    def test_fusion_weights_and_single_modality_fallbacks(self):
        v = torch.ones(4, 1, 3); r = torch.full_like(v, 3.)
        vv = torch.full_like(v, 4.); rv = torch.ones_like(v)
        vm = torch.tensor([[True], [True], [False], [False]])
        rm = torch.tensor([[True], [False], [True], [False]])
        v[2:] = torch.nan; vv[2:] = torch.nan
        r[[1, 3]] = torch.nan; rv[[1, 3]] = torch.nan
        out = fuse_diagonal_poses(v, vv, r, rv, vm, rm)
        self.assertAlmostEqual(out["mean"][0, 0, 0].item(), 2.6, places=6)
        self.assertEqual(out["mean"][1, 0, 0].item(), 1.)
        self.assertEqual(out["mean"][2, 0, 0].item(), 3.)
        self.assertFalse(out["available"][3, 0])
        self.assertTrue(torch.isnan(out["mean"][3]).all())

    def test_extreme_valid_variances_do_not_overflow(self):
        v = torch.zeros(1, 3); r = torch.ones_like(v)
        mask = torch.ones(1, dtype=torch.bool)
        out = fuse_diagonal_poses(v, torch.full_like(v, 3e38), r, torch.full_like(v, 3e38), mask, mask)
        torch.testing.assert_close(out["mean"], torch.full_like(v, .5))

    def test_fusion_rejects_invalid_available_inputs(self):
        v = torch.zeros(1, 3); mask = torch.ones(1, dtype=torch.bool)
        with self.assertRaises(ValueError):
            fuse_diagonal_poses(v, torch.zeros_like(v), v, torch.ones_like(v), mask, mask)
        with self.assertRaises(ValueError):
            fuse_diagonal_poses(v, torch.ones_like(v), v, torch.ones_like(v), mask.float(), mask)

    def test_timestamp_pairing_has_bounded_gap_and_no_wraparound(self):
        index, gap, valid = nearest_timestamp_pairs([0., .1, .2], [-.2, .049, .151, .9], .05)
        np.testing.assert_array_equal(index, [-1, 0, 2, -1])
        np.testing.assert_array_equal(valid, [False, True, True, False])
        np.testing.assert_allclose(gap, [.2, .049, .049, .7])

    def test_empty_camera_stream_has_no_matches(self):
        index, gap, valid = nearest_timestamp_pairs([], [1., 2.], .1)
        np.testing.assert_array_equal(index, [-1, -1])
        self.assertTrue(np.isinf(gap).all())
        self.assertFalse(valid.any())

    def test_bad_timebases_are_rejected(self):
        for ts in ([0., 0.], [1., 0.], [0., np.nan]):
            with self.assertRaises(ValueError):
                nearest_timestamp_pairs(ts, [0.], .1)
        with self.assertRaises(ValueError):
            nearest_timestamp_pairs([0.], [0.], -.1)

    def test_published_numpy_container_is_readable_without_general_pickle_execution(self):
        original = {"pose": np.ones((3, 17, 3)), "frames": [0, 2]}
        restored = load_labels(pickle.dumps(original, protocol=4))
        np.testing.assert_array_equal(restored["pose"], original["pose"])
        self.assertEqual(interval_bounds(restored["frames"], 3), (0, 2))

    def test_unknown_pickle_globals_are_rejected_before_execution(self):
        with self.assertRaisesRegex(ValueError, "Unapproved pickle global"):
            load_labels(pickle.dumps(eval))

    def test_inclusive_intervals_reject_out_of_bounds_and_wrong_types(self):
        for value in ([0, 3], [-1, 2], [2, 1], [0.0, 2], [0, 1, 2]):
            with self.assertRaises(ValueError):
                interval_bounds(value, 3)

    def test_ridge_constant_target_and_train_only_normalization(self):
        train = np.array([[0., 2.], [1., 3.], [2., 4.]])
        validation = np.array([[1000., 1002.]])
        result, model = fit_ridge(train, np.full((3, 2), 7.), validation)
        np.testing.assert_allclose(model["normalization_mean"], [1., 3.])
        np.testing.assert_allclose(result, [[7., 7.]], atol=1e-8)

    def test_ridge_nonfinite_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_ridge(np.array([[np.nan]]), np.ones((1, 1)), np.ones((1, 1)))


if __name__ == "__main__":
    unittest.main()
