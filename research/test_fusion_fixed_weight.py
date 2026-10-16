import unittest

import numpy as np

from fusion_fixed_weight import balanced_fit_weights, fit_fixed_weights, fixed_fusion


class FixedWeightTests(unittest.TestCase):
    def test_known_optimum_and_joint_optima(self):
        rng = np.random.default_rng(7)
        video, radar = rng.normal(size=(23, 17, 3)), rng.normal(size=(23, 17, 3))
        desired = np.linspace(.05, .95, 17)
        target = video + desired[None, :, None] * (radar - video)
        alpha, joint = fit_fixed_weights(video, radar, target, np.ones(23))
        np.testing.assert_allclose(joint, desired, atol=1e-12)
        objective = lambda a: ((video + a * (radar - video) - target) ** 2).sum()
        self.assertLessEqual(objective(alpha), min(objective(a) for a in np.linspace(0, 1, 501)) + 1e-9)

    def test_constraints_and_degenerate_experts(self):
        video, radar = np.zeros((3, 17, 3)), np.ones((3, 17, 3))
        for target, expected in ((radar * 2, 1.), (radar * -2, 0.)):
            alpha, joint = fit_fixed_weights(video, radar, target, np.ones(3))
            self.assertEqual(alpha, expected)
            np.testing.assert_array_equal(joint, np.full(17, expected))
        alpha, joint = fit_fixed_weights(video, video, radar, np.ones(3))
        self.assertEqual(alpha, 0.)
        np.testing.assert_array_equal(joint, np.zeros(17))

    def test_exact_fallback_including_both_absent(self):
        video, radar = np.full((4, 17, 3), 3.), np.full((4, 17, 3), 11.)
        fused = fixed_fusion(video, radar, np.array([1, 0, 1, 0], dtype=bool),
                             np.array([1, 1, 0, 0], dtype=bool), .25)
        np.testing.assert_array_equal(fused[:, 0, 0], [5., 11., 3., 11.])

    def test_block_balance_and_missing_examples_cannot_change_fit(self):
        subjects = np.array([1, 1, 1, 2, 2])
        condition = np.zeros(5, dtype=int)
        both = np.array([True, True, False, True, False])
        weights = balanced_fit_weights(subjects, condition, both)
        np.testing.assert_array_equal(weights, [.25, .25, 0., .5, 0.])
        video, radar = np.zeros((5, 17, 3)), np.ones((5, 17, 3))
        target = radar * .3
        target[~both] = 1e10
        alpha, _ = fit_fixed_weights(video, radar, target, weights)
        self.assertAlmostEqual(alpha, .3)


if __name__ == "__main__":
    unittest.main()
