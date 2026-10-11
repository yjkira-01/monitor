"""Switch Stage F to measured execution settings at a saved epoch boundary."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def running(pid):
    try:
        state = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[0]
        return state not in ('Z', 'X')
    except FileNotFoundError:
        return False


def select_execution(report):
    return max((case for case in report['cases'] if case['passed'] and not case['fused']),
               key=lambda case: case['samples_per_second'])


def speedup_command(original, run, source, chosen, cache_dir, checkpoint_dir=None):
    command = original[:]
    command[command.index(next(arg for arg in command if arg.endswith('/train_video.py')))] = str(source/'scripts/train_video.py')
    for flag, value in (('--batch-size', chosen['batch']), ('--accumulate', chosen['accumulate']), ('--workers', 0)):
        command[command.index(flag)+1] = str(value)
    for flag in ('--resume', '--performance-resume', '--checkpoint-blocks', '--eval-batch-size',
                 '--cache-clips', '--cache-dir', '--log-every', '--checkpoint-dir'):
        while flag in command:
            index = command.index(flag)
            del command[index:index+(1 if flag in ('--performance-resume', '--cache-clips') else 2)]
    command += ['--resume', str(run/'internvideo2_activity_last.pt'), '--performance-resume',
                '--checkpoint-blocks', str(chosen['checkpoint_blocks']), '--eval-batch-size', '32',
                '--cache-clips', '--cache-dir', cache_dir, '--log-every', '20']
    if checkpoint_dir:
        command += ['--checkpoint-dir', checkpoint_dir]
    return command


def validate_driver(argv, root, run):
    scripts = [Path(arg) for arg in argv if arg.endswith('.py')]
    if len(scripts) != 1 or scripts[0].name not in ('run_moderate_training.py', 'run_speedup_training.py'):
        raise RuntimeError('Unexpected training manager; production left running')
    if '--project-root' not in argv or Path(argv[argv.index('--project-root')+1]).resolve() != root:
        raise RuntimeError('Training manager project mismatch; production left running')
    if not scripts[0].resolve().is_relative_to(run):
        raise RuntimeError('Training manager is outside the experiment snapshot; production left running')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--after-epoch', type=int, required=True)
    parser.add_argument('--controller-state', default='controller_status.json',
                        choices=['controller_status.json', 'controller_io_status.json'])
    parser.add_argument('--checkpoint-dir', help='Local scratch directory for single-serialization checkpoint I/O')
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    run = root/'artifacts/activity_stage_f_internvideo2_base_20261009'
    perf = run/'performance_20261009'
    controller = perf/args.controller_state
    previous = read(controller)
    if previous.get('pid') and running(previous['pid']):
        raise RuntimeError('A speedup controller already exists')
    control = {'pid': os.getpid(), 'state': 'waiting_setup', 'target_epoch': args.after_epoch}
    save(controller, control)
    while True:
        cache, bench = read(perf/'cache_report.json'), read(perf/'benchmark_partial.json')
        if cache.get('complete') and bench.get('finished_at'):
            break
        for label in ('cache', 'benchmark_extra'):
            job = read(perf/(label+'_job.json'))
            complete = cache.get('complete') if label == 'cache' else bench.get('finished_at')
            if job.get('pid') and not running(job['pid']) and not complete:
                raise RuntimeError(f'{label} exited without a completed report; production left running')
        time.sleep(1)
    # Fused gains are within benchmark noise; retain the established optimizer implementation.
    chosen = select_execution(bench)
    save(perf/'selected_execution.json', chosen)
    control.update(state='waiting_checkpoint', chosen=chosen)
    save(controller, control)
    while True:
        status = read(run/'status.json')
        history = read(run/'internvideo2_activity.json').get('history', [])
        if history and history[-1]['epoch'] >= args.after_epoch:
            break
        if status.get('state') in ('complete', 'failed'):
            raise RuntimeError('Original job ended before the requested handover epoch')
        time.sleep(.5)
    command = next(step['command'] for step in reversed(status['steps']) if step['label'] == 'training')[:]
    driver_pid = status['pid']
    driver_command = (Path('/proc')/str(driver_pid)/'cmdline').read_bytes()
    driver_args = driver_command.decode().split('\0')
    validate_driver(driver_args, root, run)
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            ppid = int((entry/'stat').read_text().rsplit(')', 1)[1].split()[1])
            argv = (entry/'cmdline').read_bytes().split(b'\0')
            if ppid == driver_pid and b'--model' in argv and b'internvideo2_base' in argv:
                assert Path(argv[argv.index(b'--output')+1].decode()).resolve() == run/'internvideo2_activity.pt'
                matches.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    assert len(matches) == 1, matches
    trainer_pid = matches[0]
    source = Path(__file__).resolve().parents[1]
    assert source.is_relative_to(run), 'Controller must run from an independent experiment snapshot'
    assert (source/'scripts/train_video.py').is_file()
    command = speedup_command(command, run, source, chosen, cache['cache_dir'], args.checkpoint_dir)
    save(perf/(controller.stem+'_before_handover.json'), status)
    os.kill(trainer_pid, signal.SIGSTOP)
    os.kill(driver_pid, signal.SIGTERM)
    os.kill(trainer_pid, signal.SIGTERM)
    os.kill(trainer_pid, signal.SIGCONT)
    for _ in range(40):
        if not running(trainer_pid) and not running(driver_pid):
            break
        time.sleep(.25)
    assert not running(trainer_pid) and not running(driver_pid), 'Inspect handover processes'
    status.update(pid=os.getpid(), state='training', log_every=20, performance=chosen,
                  resume_epoch=history[-1]['epoch'], cache_dir=cache['cache_dir'])
    if args.checkpoint_dir:
        status['checkpoint_dir'] = args.checkpoint_dir
    status['steps'][-1].update(state='superseded', finished_at=time.time())
    step = {'label': 'training', 'state': 'running', 'command': command, 'started_at': time.time()}
    status['steps'].append(step)
    save(run/'status.json', status)
    control.update(state='training', resume_epoch=history[-1]['epoch'])
    save(controller, control)
    try:
        with (run/'training.log').open('a') as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4'})
            step['pid'] = process.pid
            save(run/'status.json', status)
            code = process.wait()
        if code:
            raise RuntimeError(f'Optimized training exited with code {code}; last checkpoint preserved')
        report = read(run/'internvideo2_activity_evaluation.json')
        step.update(state='complete', finished_at=time.time())
        status.update(state='complete', finished_at=time.time(), selected_epoch=report.get('selected_epoch'))
    except Exception as exc:
        status.update(state='failed', error=str(exc), finished_at=time.time())
        raise
    finally:
        save(run/'status.json', status)


if __name__ == '__main__':
    main()
