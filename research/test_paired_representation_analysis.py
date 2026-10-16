import tempfile
import copy
import unittest
from pathlib import Path

import numpy as np

from analyze_paired_representations import ROOT, compare_contracts, exact_npz_comparison, simple_metrics


class SavedRepresentationAnalysisChecks(unittest.TestCase):
    def contracts(self):
        hip = {'representation': 'hip', 'data_split_seed': 7, 'initialization_seeds': [7, 17, 27],
               'train_frames': 1152, 'validation_frames': 384, 'final_test_subjects_used': [],
               'training': {'epochs': 80, 'batch_size': 64},
               'environment': {'device': 'cuda', 'deterministic': True, 'mixed_precision': False,
                               'FP32_matmul': 'IEEE, TF32 disabled'},
               'runs': [{'seed': seed, 'status': 'complete'} for seed in (7, 17, 27)]}
        centroid = copy.deepcopy(hip)
        centroid['representation'] = 'visible_centroid'
        return hip, centroid

    def test_paired_contract_rejects_changed_epoch_budget(self):
        hip, centroid = self.contracts()
        self.assertEqual(compare_contracts(hip, centroid), [])
        centroid['training']['epochs'] = 79
        self.assertIn('Different run contract: training', compare_contracts(hip, centroid))

    def test_paired_contract_rejects_tf32_backend(self):
        hip, centroid = self.contracts()
        centroid['environment']['FP32_matmul'] = 'TF32 enabled'
        self.assertIn('Different run contract: environment', compare_contracts(hip, centroid))
        self.assertTrue(any('expected deterministic CUDA' in issue for issue in compare_contracts(hip, centroid)))

    def test_joint_diagnostics_use_their_own_denominators_without_changing_mpjpe(self):
        errors = np.stack((np.full(17, 3.), np.full(17, 5.)))
        visible = np.zeros((2, 17), dtype=bool)
        visible[0, 0] = True
        visible[1, :3] = True
        values = simple_metrics(errors, visible)
        self.assertEqual(values['MPJPE_mm'], 4.)
        self.assertEqual(values['visible_joint_error_mm'], 4.5)
        self.assertAlmostEqual(values['invisible_joint_error_mm'], (16 * 3 + 14 * 5) / 30)
        self.assertEqual(values['visible_joints'], 4)
        self.assertEqual(values['invisible_joints'], 30)

    def test_no_observed_joints_have_null_observed_error(self):
        values = simple_metrics(np.ones((3, 17)) * 5, np.zeros((3, 17), dtype=bool))
        self.assertIsNone(values['visible_joint_error_mm'])
        self.assertEqual(values['invisible_joint_error_mm'], 5.)

    def test_npz_controls_compare_array_values_and_types(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as folder:
            left, right = Path(folder) / 'a.npz', Path(folder) / 'b.npz'
            np.savez_compressed(left, weights=np.arange(6, dtype=np.float32))
            np.savez_compressed(right, weights=np.arange(6, dtype=np.float32))
            self.assertEqual(exact_npz_comparison(left, right), [])
            np.savez_compressed(right, weights=np.arange(6, dtype=np.float64))
            self.assertEqual(exact_npz_comparison(left, right), ['weights'])
            changed = np.arange(6, dtype=np.float32)
            changed[3] += 1
            np.savez_compressed(right, weights=changed)
            self.assertEqual(exact_npz_comparison(left, right), ['weights'])


if __name__ == '__main__':
    unittest.main()
