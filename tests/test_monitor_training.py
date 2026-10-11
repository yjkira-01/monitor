"""Offline fixtures for the read-only monitor; no torch or GPU required."""
import json
from pathlib import Path
import sys
import tempfile
import unicodedata
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.monitor_training import progress_bar, read_log, render, render_live, snapshot


class MonitorTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='training-monitor-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'project'
        self.run = self.root / 'artifacts/activity_stage_f_internvideo2_base_20261009'
        self.run.mkdir(parents=True)
        self.proc = Path(self.temp.name) / 'proc'
        self.proc.mkdir()

    def write_json(self, name, data):
        (self.run / name).write_text(json.dumps(data), encoding='utf-8')

    def process(self, pid, ppid, args, state='S'):
        entry = self.proc / str(pid)
        entry.mkdir()
        (entry / 'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in args) + b'\0')
        (entry / 'stat').write_text(f'{pid} (python worker) {state} {ppid} 0 0')

    def start_fixture(self):
        self.write_json('status.json', {'state': 'training', 'pid': 10, 'epochs_limit': 120})
        self.process(10, 1, ['python', 'scripts/run_moderate_training.py', '--project-root', str(self.root)])
        trainer = ['python', '/frozen/scripts/train_video.py', '--output', str(self.run / 'internvideo2_activity.pt')]
        self.process(11, 10, trainer)
        self.process(12, 11, trainer)  # DataLoader worker, not a second training job.

    def test_first_epoch_and_old_traceback_do_not_report_current_failure(self):
        self.start_fixture()
        (self.run / 'training.log').write_text(
            'Traceback (most recent call last):\nModuleNotFoundError: old setup\n'
            'epoch=001 batch=100/2686 loss=2.50439\n'
            '[mp3float @ 0xc828a00] Header missing\n'
            'epoch=001 batch=2600/2686 loss=2.18455\n', encoding='utf-8')
        result = snapshot(self.run, self.proc)
        self.assertEqual(result['observed_state'], 'running')
        self.assertEqual(result['processes']['driver_pid'], 10)
        self.assertEqual(result['processes']['trainer_pids'], [11])
        self.assertEqual(result['completed_epochs'], 0)
        self.assertEqual(result['latest_batch']['batch'], 2600)
        self.assertIsNone(result['latest'])
        self.assertIn('验证结果尚未保存', render(result))

    def test_latest_epoch_and_best_rank_are_distinct(self):
        self.write_json('status.json', {'state': 'complete', 'pid': 10})
        first = {'epoch': 1, 'val_macro_f1': .91, 'fall_recall': .97, 'val_acc': .92,
                 'fall_precision': .92, 'selection_rank': [1, .8, .91], 'seconds': 2400,
                 'per_class': {'lying': {'recall': .81, 'support': 65}},
                 'robustness': {'night': {'val_macro_f1': .76}}, 'regression_gate_passed': True}
        second = {**first, 'epoch': 2, 'val_macro_f1': .94, 'selection_rank': [0, .85, .94],
                  'regression_gate_passed': False}
        self.write_json('internvideo2_activity.json', {'history': [first, second]})
        self.write_json('internvideo2_activity_evaluation.json', {'selected_epoch': 1})
        (self.run / 'internvideo2_activity_last.pt').write_bytes(b'checkpoint-stat-only')
        result = snapshot(self.run, self.proc)
        self.assertEqual(result['completed_epochs'], 2)
        self.assertEqual(result['latest']['epoch'], 2)
        self.assertEqual(result['best']['epoch'], 1)
        self.assertEqual(result['observed_state'], 'complete')
        self.assertGreater(result['checkpoints']['internvideo2_activity_last.pt']['bytes'], 0)
        output = render(result)
        self.assertIn('卧姿召回: 81.00%', output)
        self.assertIn('暗光 76.00%', output)
        self.assertIn('不是独立测试集成绩', output)

    def test_stale_status_pid_and_other_runs_do_not_count_as_alive(self):
        self.write_json('status.json', {'state': 'training', 'pid': 10})
        self.process(10, 1, ['python', 'unrelated.py'])
        self.process(11, 1, ['python', 'train_video.py', '--output', str(self.root / 'another.pt')])
        self.process(12, 1, ['python', 'train_video.py', '--output', str(self.run / 'internvideo2_activity.pt')], 'Z')
        result = snapshot(self.run, self.proc)
        self.assertEqual(result['observed_state'], 'interrupted')
        self.assertEqual(result['processes']['trainer_pids'], [])
        self.assertIsNone(result['processes']['driver_pid'])

    def test_detached_or_resumed_trainer_is_found_without_driver(self):
        self.write_json('status.json', {'state': 'failed', 'pid': 10, 'error': 'previous failure'})
        self.process(11, 1, ['python', 'train_video.py', '--output', str(self.run / 'internvideo2_activity.pt')])
        result = snapshot(self.run, self.proc)
        self.assertEqual(result['observed_state'], 'running')
        self.assertEqual(result['processes']['trainer_pids'], [11])
        self.assertIn('状态文件: failed', render(result))

    def test_current_failure_uses_status_instead_of_old_log_messages(self):
        self.write_json('status.json', {'state': 'failed', 'error': 'RuntimeError: CUDA out of memory'})
        result = snapshot(self.run, self.proc)
        self.assertEqual(result['observed_state'], 'failed')
        self.assertIn('CUDA out of memory', render(result))

    def test_partial_json_warns_and_recovers_on_next_refresh(self):
        self.start_fixture()
        (self.run / 'internvideo2_activity.json').write_text('{"history":')
        result = snapshot(self.run, self.proc)
        self.assertEqual(result['observed_state'], 'running')
        self.assertEqual(len(result['read_errors']), 1)
        self.write_json('internvideo2_activity.json', {'history': []})
        self.assertEqual(snapshot(self.run, self.proc)['read_errors'], [])

    def test_log_read_is_bounded_and_discards_partial_first_line(self):
        log = self.run / 'training.log'
        log.write_bytes(b'old error\n' + b'x' * 4096 + b'\nepoch=002 batch=100/2686 loss=1.3\n')
        text, error = read_log(log, limit=128)
        self.assertIsNone(error)
        self.assertLessEqual(len(text), 128)
        self.assertEqual(text, 'epoch=002 batch=100/2686 loss=1.3\n')

    def test_without_proc_does_not_claim_the_cloud_job_stopped(self):
        self.write_json('status.json', {'state': 'training', 'pid': 10})
        result = snapshot(self.run, self.proc / 'missing')
        self.assertEqual(result['observed_state'], 'unverified')
        self.assertIn('不能确认云端进程', render(result))

    def test_progress_bar_has_correct_proportion_and_unknown_total(self):
        self.assertEqual(progress_bar(50, 100, width=4), '[██░░] 50/100  50.00%')
        self.assertEqual(progress_bar(100, 100, width=4), '[████] 100/100  100.00%')
        self.assertIn('等待进度数据', progress_bar(0, None))
        self.assertIn('等待进度数据', progress_bar(0, 0))

    def test_batch_bar_does_not_count_as_a_complete_epoch(self):
        self.start_fixture()
        (self.run / 'training.log').write_text('epoch=001 batch=2600/2686 loss=2.18455\n')
        output = render(snapshot(self.run, self.proc))
        self.assertIn('[░' + '░' * 27 + '] 0/120  0.00%', output)
        self.assertIn('2600/2686  96.80%', output)
        self.assertIn('不包含验证进度', output)

    def test_early_stop_and_saved_epoch_do_not_show_old_batch_as_current(self):
        self.write_json('status.json', {'state': 'complete', 'epochs_limit': 120})
        self.write_json('internvideo2_activity.json', {
            'history': [{'epoch': 25, 'train_loss': .23456, 'train_acc': .93}]
        })
        (self.run / 'training.log').write_text('epoch=025 batch=2600/2686 loss=.20000\n')
        output = render(snapshot(self.run, self.proc))
        self.assertIn('历史批次', output)
        self.assertIn('25/120', output)
        self.assertIn('训练任务已结束', output)
        self.assertIn('平均训练 loss: 0.23456', output)

    def test_live_panel_keeps_progress_visible_in_small_terminal(self):
        self.start_fixture()
        self.write_json('internvideo2_activity.json', {'history': [
            {'epoch': 1, 'val_acc': .44, 'val_macro_f1': .31, 'train_loss': 2.34}
        ]})
        (self.run / 'training.log').write_text('epoch=002 batch=200/2686 loss=2.61912\n')
        data = snapshot(self.run, self.proc)
        data['gpu'] = {'devices': [{'index': '0', 'utilization': '99', 'used_mib': '4051',
                                  'total_mib': '12288', 'temperature': '59'}]}
        for columns, rows in ((80, 24), (60, 16)):
            panel = render_live(data, columns, rows)
            lines = panel.splitlines()
            self.assertLess(len(lines), rows)
            self.assertIn('完整轮次', lines[2])
            self.assertIn('200/2686', lines[3])
            self.assertIn('Ctrl+C', lines[-1])
            for line in lines:
                display_width = sum(2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1 for char in line)
                self.assertLess(display_width, columns)

    def test_speedup_throughput_is_read_and_old_epoch_rate_is_ignored(self):
        self.start_fixture()
        (self.run / 'training.log').write_text(
            'epoch=004 batch=20/336 loss=1.50000\n'
            'throughput epoch=004 samples=320 elapsed=20.000 samples_per_second=16.000\n')
        data = snapshot(self.run, self.proc)
        self.assertEqual(data['train_throughput']['samples_per_second'], 16.)
        self.assertIn('16.00 窗/秒', render_live(data))
        with (self.run / 'training.log').open('a') as log:
            log.write('epoch=005 batch=1/336 loss=1.00000\n')
        self.assertIsNone(snapshot(self.run, self.proc)['train_throughput'])


if __name__ == '__main__':
    unittest.main()
