from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_speedup_training import select_execution, speedup_command, validate_driver


class SpeedupControllerTests(unittest.TestCase):
    def test_selection_excludes_oom_and_keeps_established_optimizer(self):
        report = {'cases': [
            {'passed': False, 'fused': False, 'samples_per_second': 30},
            {'passed': True, 'fused': True, 'samples_per_second': 16.46},
            {'passed': True, 'fused': False, 'samples_per_second': 16.39, 'batch': 16},
            {'passed': True, 'fused': False, 'samples_per_second': 15.77, 'batch': 4},
        ]}
        self.assertEqual(select_execution(report)['batch'], 16)

    def test_transition_preserves_learning_objective_and_original_command(self):
        original = ['python', '/old/scripts/train_video.py', '--batch-size', '2', '--accumulate', '8',
                    '--workers', '2', '--epochs', '120', '--lr', '2e-5', '--risk-loss-weight', '.25',
                    '--manifest', '/frozen.csv', '--strong-augment', '--bounded-sampling', '--domain-selection']
        selected = {'batch': 16, 'accumulate': 1, 'checkpoint_blocks': 10}
        command = speedup_command(original, Path('/run'), Path('/source'), selected, '/local/cache')
        self.assertEqual(original[original.index('--batch-size')+1], '2')
        for flag in ('--epochs', '--lr', '--risk-loss-weight', '--manifest'):
            self.assertEqual(command[command.index(flag)+1], original[original.index(flag)+1])
        self.assertEqual(command[command.index('--batch-size')+1], '16')
        self.assertEqual(command[command.index('--accumulate')+1], '1')
        self.assertEqual(command[command.index('--checkpoint-blocks')+1], '10')
        self.assertIn('--resume', command)
        self.assertIn('--performance-resume', command)
        self.assertIn('--domain-selection', command)

    def test_second_handover_replaces_execution_flags_without_duplicates(self):
        original = ['python', '/run/source/scripts/train_video.py', '--batch-size', '16', '--accumulate', '1',
                    '--workers', '0', '--manifest', '/frozen.csv', '--lr', '2e-5', '--resume', '/run/last.pt',
                    '--performance-resume', '--checkpoint-blocks', '10', '--eval-batch-size', '32',
                    '--cache-clips', '--cache-dir', '/local/cache', '--log-every', '20']
        command = speedup_command(original, Path('/run'), Path('/run/io_source'),
                                  {'batch': 16, 'accumulate': 1, 'checkpoint_blocks': 10}, '/local/cache', '/local/staging')
        for flag in ('--resume', '--performance-resume', '--checkpoint-blocks', '--cache-clips',
                     '--cache-dir', '--eval-batch-size', '--log-every', '--checkpoint-dir'):
            self.assertEqual(command.count(flag), 1)
        self.assertEqual(command[command.index('--checkpoint-dir')+1], '/local/staging')
        self.assertEqual(command[command.index('--manifest')+1], '/frozen.csv')

    def test_handover_driver_requires_matching_project_and_snapshot(self):
        root = Path('/project').resolve()
        run = root/'artifacts/stage_f'
        for name in ('run_moderate_training.py', 'run_speedup_training.py'):
            validate_driver(['python', str(run/'source/scripts'/name), '--project-root', str(root)], root, run)
        with self.assertRaises(RuntimeError):
            validate_driver(['python', '/other/run_speedup_training.py', '--project-root', str(root)], root, run)
        with self.assertRaises(RuntimeError):
            validate_driver(['python', str(run/'source/scripts/run_speedup_training.py'), '--project-root', '/other'], root, run)


if __name__ == '__main__':
    unittest.main()
