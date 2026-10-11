"""Read-only Stage F training monitor. Standard library only; never loads weights."""
import argparse
import csv
from datetime import datetime
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unicodedata


DEFAULT_RUN = Path(__file__).resolve().parents[1] / 'artifacts/activity_stage_f_internvideo2_base_20261009'
BATCH_PATTERN = re.compile(r'^epoch=(\d+) batch=(\d+)/(\d+) loss=([\d.eE+-]+)\s*$', re.MULTILINE)
THROUGHPUT_PATTERN = re.compile(r'^throughput epoch=(\d+) samples=(\d+) elapsed=([\d.]+) samples_per_second=([\d.]+)\s*$', re.MULTILINE)
CASES = {
    'night': '暗光', 'upper_body': '仅上半身', 'lower_body': '仅下半身',
    'side_crop': '侧边截断', 'center_occlusion': '中央遮挡',
    'low_quality': '低画质', 'night_low_quality': '暗光+低画质',
}
CLASSES = {
    'fall': '跌倒过程', 'fallen': '倒地', 'lie_down': '躺下', 'lying': '卧姿',
    'other': '其他', 'sit_down': '坐下', 'sitting': '坐姿',
    'stand_up': '起身', 'standing': '站姿', 'walk': '行走',
}
STATES = {'running': '运行中', 'complete': '已完成', 'failed': '失败',
          'interrupted': '未发现对应进程，请检查是否中断', 'unknown': '尚无状态文件',
          'unverified': '进程未核验'}


def read_json(path):
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict):
            raise ValueError('expected a JSON object')
        return data, None
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as exc:
        return {}, f'{path.name}: {exc}'


def file_info(path, now):
    try:
        stat = path.stat()
        return {'bytes': stat.st_size, 'age_seconds': max(0, now - stat.st_mtime)}
    except FileNotFoundError:
        return None
    except OSError as exc:
        return {'error': str(exc)}


def read_log(path, limit=128 * 1024):
    try:
        with path.open('rb') as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size - limit))
            data = handle.read(limit)
        if size > limit:
            data = data.partition(b'\n')[2]
        return data.decode('utf-8', errors='replace'), None
    except FileNotFoundError:
        return '', None
    except OSError as exc:
        return '', f'{path.name}: {exc}'


def argument(args, flag):
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def process_info(entry):
    try:
        args = [part.decode(errors='replace') for part in (entry / 'cmdline').read_bytes().split(b'\0') if part]
        fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
        if not args or fields[0] in ('Z', 'X'):
            return None
        return {'pid': int(entry.name), 'ppid': int(fields[1]), 'args': args,
                'cwd': (entry / 'cwd').resolve()}
    except (OSError, ValueError, IndexError):
        return None


def matches_path(value, target, cwd):
    if value is None:
        return False
    path = Path(value)
    try:
        return (path if path.is_absolute() else cwd / path).resolve() == target
    except (OSError, ValueError):
        return False


def inspect_processes(run, status, proc_root=Path('/proc')):
    if not proc_root.is_dir():
        return {'available': False, 'driver_pid': None, 'trainer_pids': [],
                'error': '当前环境无 /proc，不能确认云端进程'}
    try:
        processes = [info for entry in proc_root.iterdir() if entry.name.isdigit()
                     for info in [process_info(entry)] if info]
    except OSError as exc:
        return {'available': False, 'driver_pid': None, 'trainer_pids': [], 'error': str(exc)}
    driver = None
    trainers = []
    for info in processes:
        args, cwd = info['args'], info['cwd']
        names = {Path(arg).name for arg in args}
        if (info['pid'] == status.get('pid') and names.intersection({'run_moderate_training.py', 'run_speedup_training.py'})
                and matches_path(argument(args, '--project-root'), run.parent.parent, cwd)):
            driver = info['pid']
        if ('train_video.py' in names
                and matches_path(argument(args, '--output'), run / 'internvideo2_activity.pt', cwd)):
            trainers.append(info)
    # DataLoader forks inherit the command line; their parent is the actual trainer.
    ids = {info['pid'] for info in trainers}
    roots = sorted(info['pid'] for info in trainers if info['ppid'] not in ids)
    return {'available': True, 'driver_pid': driver, 'trainer_pids': roots, 'error': None}


def read_gpus():
    command = ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
               '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
        return {'devices': [dict(zip(('index', 'name', 'utilization', 'used_mib', 'total_mib', 'temperature'),
                                    (value.strip() for value in row)))
                            for row in csv.reader(io.StringIO(result.stdout)) if len(row) == 6], 'error': None}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'devices': [], 'error': f'nvidia-smi 不可用: {exc}'}


def snapshot(run, proc_root=Path('/proc')):
    run = run.resolve()
    now = time.time()
    status, status_error = read_json(run / 'status.json')
    metrics, metrics_error = read_json(run / 'internvideo2_activity.json')
    report, report_error = read_json(run / 'internvideo2_activity_evaluation.json')
    log, log_error = read_log(run / 'training.log')
    history = metrics.get('history', [])
    history = [row for row in history if isinstance(row, dict) and isinstance(row.get('epoch'), int)]
    latest = history[-1] if history else None
    best = max(history, key=lambda row: tuple(row.get('selection_rank',
               [row.get('val_macro_f1', -1), row.get('fall_recall', -1), row.get('val_acc', -1)]))) if history else None
    batches = list(BATCH_PATTERN.finditer(log))
    batch = None
    if batches:
        epoch, current, total, loss = batches[-1].groups()
        batch = {'epoch': int(epoch), 'batch': int(current), 'total': int(total), 'loss': float(loss)}
    throughput = None
    rates = list(THROUGHPUT_PATTERN.finditer(log))
    if rates:
        epoch, samples, duration, rate = rates[-1].groups()
        if batch and int(epoch) == batch['epoch']:
            throughput = {'epoch': int(epoch), 'samples': int(samples), 'elapsed_seconds': float(duration),
                          'samples_per_second': float(rate)}
    processes = inspect_processes(run, status, proc_root)
    alive = processes['driver_pid'] is not None or bool(processes['trainer_pids'])
    state = status.get('state', 'unknown')
    if alive:
        observed = 'running'
    elif not processes['available']:
        observed = 'unverified'
    elif state in ('complete', 'failed'):
        observed = state
    elif status:
        observed = 'interrupted'
    else:
        observed = 'unknown'
    return {
        'observed_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'run': str(run),
        'observed_state': observed, 'status': status, 'processes': processes,
        'completed_epochs': latest['epoch'] if latest else 0, 'latest_batch': batch,
        'train_throughput': throughput,
        'latest': latest, 'best': best, 'evaluation': report,
        'log': file_info(run / 'training.log', now), 'log_tail': log.splitlines()[-5:],
        'checkpoints': {name: file_info(run / name, now) for name in
                        ('internvideo2_activity_last.pt', 'internvideo2_activity.pt')},
        'read_errors': [error for error in (status_error, metrics_error, report_error, log_error) if error],
    }


def percent(value):
    return f'{value:.2%}' if isinstance(value, (int, float)) and math.isfinite(value) else '—'


def progress_bar(current, total, width=28):
    if not isinstance(total, (int, float)) or not math.isfinite(total) or total <= 0:
        return '[' + '░' * width + '] 等待进度数据'
    ratio = min(1.0, max(0.0, current / total))
    filled = int(ratio * width)
    return '[' + '█' * filled + '░' * (width - filled) + f'] {current}/{total}  {percent(ratio)}'


def elapsed(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def metric_line(label, row):
    return (f"{label} 第 {row['epoch']} 轮 | 准确率 {percent(row.get('val_acc'))}"
            f" | macro-F1 {percent(row.get('val_macro_f1'))}"
            f" | 风险召回 {percent(row.get('fall_recall'))}"
            f" | 风险精确率 {percent(row.get('fall_precision'))}")


def render(data):
    status, processes = data['status'], data['processes']
    lines = [f"云端训练实时面板  {data['observed_at']}", f"实验目录: {data['run']}",
             f"状态: {STATES[data['observed_state']]} | 状态文件: {status.get('state', '—')}"
             f" | 模型: {status.get('model', '—')}",
             f"进程: 管理 PID {processes['driver_pid'] or '未确认'}"
             f" | 训练 PID {', '.join(map(str, processes['trainer_pids'])) or '未确认'}",
             f"已完成: {data['completed_epochs']} / 上限 {status.get('epochs_limit', '—')} 轮"
             f" | 早停耐心 {status.get('early_stop_patience', '—')} 轮"
             f" | 有效 batch {status.get('effective_batch', '—')}"]
    lines.append('完整轮次  ' + progress_bar(data['completed_epochs'], status.get('epochs_limit')))
    if data['observed_state'] == 'complete':
        lines.append('训练任务已结束（轮数为上限，任务可提前停止）。')
    windows = status.get('windows', {})
    if windows:
        lines.append(f"数据窗: train {windows.get('train', '—')} / val {windows.get('val', '—')}"
                     f" / test {windows.get('test', '—')} | 训练视频 {status.get('training_videos', '—')}")
    batch = data['latest_batch']
    if batch:
        label = '历史批次' if batch['epoch'] <= data['completed_epochs'] else '本轮批次'
        lines.append(f"{label}  第 {batch['epoch']} 轮 " + progress_bar(batch['batch'], batch['total']))
        lines.append(f"最近批次 loss: {batch['loss']:.5f}（日志记录值，批次条不包含验证进度）")
    else:
        lines.append('本轮批次  ' + progress_bar(0, None))
    if data['log'] and 'age_seconds' in data['log']:
        lines.append(f"日志距更新: {elapsed(data['log']['age_seconds'])}"
                     f"（每{status.get('log_every', 100)}批记录；验证期间可能无新批次日志）")
    if data.get('train_throughput'):
        lines.append(f"本轮训练吞吐: {data['train_throughput']['samples_per_second']:.2f} 窗/秒（含读取与计算）")
    latest, best = data['latest'], data['best']
    if latest:
        lines.extend(['', metric_line('最新验证', latest), metric_line('最佳选模', best)])
        train_loss = latest.get('train_loss')
        lines.append(f"最近一轮平均训练 loss: {train_loss:.5f}" if isinstance(train_loss, (int, float))
                     else '最近一轮平均训练 loss: —')
        lines.append(f"最近一轮训练准确率: {percent(latest.get('train_acc'))}")
        lines.append(f"最近一轮耗时: {elapsed(latest.get('seconds', 0))}"
                     f" | 卧姿召回: {percent(latest.get('per_class', {}).get('lying', {}).get('recall'))}"
                     f" | 开发验收门槛: {'通过' if latest.get('regression_gate_passed') else '未通过'}")
        robust = latest.get('robustness', {})
        if robust:
            lines.append('困难条件 macro-F1: ' + ' | '.join(
                f'{label} {percent(robust.get(case, {}).get("val_macro_f1"))}' for case, label in CASES.items()))
        per_class = latest.get('per_class', {})
        if per_class:
            lines.append('各类验证召回率（n=验证窗数）:')
            for key, label in CLASSES.items():
                row = per_class.get(key, {})
                lines.append(f"  {label:<6} {percent(row.get('recall')):>7}  n={row.get('support', '—')}")
    else:
        lines.extend(['', '完整轮次的验证结果尚未保存，暂不能判断准确度。'])
    report = data['evaluation']
    if report:
        lines.append(f"最终报告: 选中第 {report.get('selected_epoch', '—')} 轮"
                     '（此报告是验证结果，不是独立测试集成绩）')
    lines.append('')
    for name, info in data['checkpoints'].items():
        if info and 'bytes' in info:
            text = f"{info['bytes'] / 1024**2:.1f} MiB | 距写入 {elapsed(info['age_seconds'])}"
        else:
            text = info.get('error') if info else '尚未生成'
        lines.append(f'断点 {name}: {text}')
    lines.append('GPU（整张卡，含其他任务）:')
    for gpu in data.get('gpu', {}).get('devices', []):
        lines.append(f"  #{gpu['index']} {gpu['name']} | 使用率 {gpu['utilization']}%"
                     f" | 显存 {gpu['used_mib']}/{gpu['total_mib']} MiB | 温度 {gpu['temperature']}°C")
    errors = data['read_errors'] + [error for error in
             (processes.get('error'), data.get('gpu', {}).get('error')) if error]
    if status.get('error'):
        errors.append(f"状态文件记录的错误: {status['error']}")
    lines.extend(f'提示: {error}' for error in errors)
    lines.extend(['', '日志末尾:'] + [f'  {line[:240]}' for line in data['log_tail']])
    lines.append('只读监控；Ctrl+C 退出监控，云端训练继续。')
    return '\n'.join(lines)


def fit_line(text, columns):
    result, used = [], 0
    for char in text:
        width = 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if used + width >= columns:
            break
        result.append(char)
        used += width
    return ''.join(result)


def render_live(data, columns=80, rows=24, interval=1):
    """Keep the bars visible even in a small SSH terminal; --once has full detail."""
    status, batch, latest = data['status'], data['latest_batch'], data['latest']
    lines = [f"云端训练实时面板  {data['observed_at']}",
             f"{STATES[data['observed_state']]} | {status.get('model', '—')}"
             f" | 训练 PID {','.join(map(str, data['processes']['trainer_pids'])) or '未确认'}",
             '完整轮次 ' + progress_bar(data['completed_epochs'], status.get('epochs_limit'), width=20)]
    if batch:
        label = '历史' if batch['epoch'] <= data['completed_epochs'] else '本轮'
        lines.append(f"{label}第{batch['epoch']}轮 " + progress_bar(batch['batch'], batch['total'], width=20))
        rate = data.get('train_throughput')
        lines.append(f"批次 loss {batch['loss']:.5f} | " + (f"训练 {rate['samples_per_second']:.2f} 窗/秒（含读取）"
                     if rate else '批次条为日志记录值，不含验证进度'))
    else:
        lines.append('本轮批次 ' + progress_bar(0, None, width=20))
    if latest:
        lines.append(f"最新验证 E{latest['epoch']} | acc {percent(latest.get('val_acc'))}"
                     f" | macro-F1 {percent(latest.get('val_macro_f1'))}")
        lines.append(f"风险召回 {percent(latest.get('fall_recall'))}"
                     f" | 风险精确率 {percent(latest.get('fall_precision'))}"
                     f" | 卧姿召回 {percent(latest.get('per_class', {}).get('lying', {}).get('recall'))}")
        train_loss = latest.get('train_loss')
        lines.append((f"轮均训练 loss {train_loss:.5f}" if isinstance(train_loss, (int, float)) else '轮均训练 loss —')
                     + f" | 训练 acc {percent(latest.get('train_acc'))}"
                     + f" | 门槛 {'通过' if latest.get('regression_gate_passed') else '未通过'}")
        best = data['best']
        lines.append(f"最佳 E{best['epoch']} | acc {percent(best.get('val_acc'))}"
                     f" | macro-F1 {percent(best.get('val_macro_f1'))}")
        for heading, values in (
            ('困难条件 macro-F1', [f'{label} {percent(latest.get("robustness", {}).get(case, {}).get("val_macro_f1"))}'
                                  for case, label in CASES.items()]),
            ('动作召回率（验证窗数）', [f'{label} {percent(latest.get("per_class", {}).get(key, {}).get("recall"))}'
                                       f' ({latest.get("per_class", {}).get(key, {}).get("support", "—")})'
                                       for key, label in CLASSES.items()]),
        ):
            lines.append(heading)
            lines.extend(' | '.join(values[index:index+2]) for index in range(0, len(values), 2))
    else:
        lines.append('完整轮次验证结果尚未保存。')
    for gpu in data.get('gpu', {}).get('devices', []):
        lines.append(f"GPU#{gpu['index']} {gpu['utilization']}% | 显存 {gpu['used_mib']}/{gpu['total_mib']} MiB"
                     f" | {gpu['temperature']}°C（含其他任务）")
    if data['log'] and 'age_seconds' in data['log']:
        lines.append(f"日志距更新 {elapsed(data['log']['age_seconds'])} | 每{status.get('log_every', 100)}批记录，轮末更新验证")
    errors = data['read_errors'] + [error for error in
              (data['processes'].get('error'), data.get('gpu', {}).get('error'), status.get('error')) if error]
    if errors:
        lines.insert(2, '提示: ' + errors[0])
    footer = f'{interval:g}秒刷新 | 完整数据: --once | Ctrl+C退出监控，训练继续'
    visible = lines[:max(1, rows - 2)] + [footer]
    return '\n'.join(fit_line(line, max(2, columns)) for line in visible)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN, help='Stage F experiment directory')
    parser.add_argument('--interval', type=float, default=1, help='refresh interval in seconds (default: 1)')
    parser.add_argument('--once', action='store_true', help='print one snapshot and exit')
    parser.add_argument('--json', action='store_true', help='one JSON snapshot; implies --once')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error('--interval must be a finite number >= 1')
    if not args.run.is_dir():
        parser.error(f'experiment directory does not exist: {args.run}')
    try:
        while True:
            data = snapshot(args.run)
            data['gpu'] = read_gpus()
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                if sys.stdout.isatty() and not args.once:
                    size = shutil.get_terminal_size()
                    print('\033[2J\033[H', end='')
                    print(render_live(data, size.columns, size.lines, args.interval), flush=True)
                else:
                    print(render(data), flush=True)
            if args.once or args.json:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n监控已退出；训练未受影响。')


if __name__ == '__main__':
    main()
