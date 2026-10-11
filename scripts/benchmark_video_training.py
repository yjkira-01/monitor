"""Isolated, synchronized throughput probe; never writes production checkpoints."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from yanglao_monitor.foundation_video import build_internvideo2, normalize_video, layer_parameters
from yanglao_monitor.video import VideoManifestDataset
from train_video import activity_loss, prepare_batch, cache_dataset, restore_optimizer_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pause-pid', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA is required')
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'effective_batch': 16,
              'purpose': 'resource/throughput probe; not accuracy evaluation', 'cases': [], 'evaluation': []}
    paused = False
    def interrupted(signum, frame):
        raise SystemExit(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.pause_pid:
            command = (Path('/proc') / str(args.pause_pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            assert 'train_video.py' in command and '--model internvideo2_base' in command
            assert str(Path(args.manifest).resolve()) in command or Path(args.manifest).name in command
            os.kill(args.pause_pid, signal.SIGSTOP)
            paused = True
            time.sleep(.3)
        dataset = VideoManifestDataset(args.manifest, 'train', 8, (224, 224), preserve_aspect=True)
        examples = [dataset[round(index * (len(dataset)-1) / 15)] for index in range(16)]
        raw = torch.stack([x for x, _ in examples]).cuda()
        targets = torch.tensor([y for _, y in examples], device='cuda')
        model = build_internvideo2(10, args.weights).cuda()

        # Same input, seed and parameters: checkpointing may recompute but must preserve gradients.
        report['numerical_checks'] = []
        torch.backends.cuda.matmul.allow_tf32 = False
        for amp in (False, True):
            probe_batch = 2 if amp else 1
            reference = None
            differences = []
            error_squared, reference_squared = 0., 0.
            for checkpointing in (True, False):
                for block in model.blocks:
                    block.with_cp = checkpointing
                model.train()
                model.zero_grad(set_to_none=True)
                torch.manual_seed(7)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    prediction = model(normalize_video(raw[:probe_batch].transpose(1, 2)))
                    loss = activity_loss(prediction, targets[:probe_batch], dataset.classes, .25)
                loss.backward()
                gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
                assert all(torch.isfinite(grad).all() for grad in gradients.values())
                if reference is None:
                    reference, reference_prediction = gradients, prediction.detach().cpu().clone()
                else:
                    torch.testing.assert_close(prediction.detach().cpu(), reference_prediction,
                                               rtol=.005 if amp else 1e-4, atol=.005 if amp else 1e-5)
                    for name, gradient in gradients.items():
                        delta = gradient-reference[name]
                        differences.append(float(delta.abs().max()))
                        error_squared += float(delta.square().sum(dtype=torch.float64))
                        reference_squared += float(reference[name].square().sum(dtype=torch.float64))
            check = {'dtype': 'BF16' if amp else 'FP32', 'gradient_max_abs_difference': max(differences),
                     'gradient_relative_l2_error': (error_squared/max(reference_squared,1e-30))**.5}
            report['numerical_checks'].append(check)
            Path(args.output).write_text(json.dumps(report, indent=2))
            assert check['gradient_relative_l2_error'] < (.02 if amp else 1e-4), check
            print(json.dumps(check), flush=True)
        del reference, gradients
        model.zero_grad(set_to_none=True)
        base = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        parameter = torch.nn.Parameter(torch.randn(512, device='cuda'))
        original = torch.optim.AdamW([parameter], lr=2e-5, weight_decay=.05)
        parameter.grad = torch.randn_like(parameter)
        original.step()
        saved = copy.deepcopy(original.state_dict())
        results = []
        gradient = torch.randn_like(parameter)
        for fused in (False, True):
            value = torch.nn.Parameter(parameter.detach().clone())
            candidate = torch.optim.AdamW([value], lr=2e-5, weight_decay=.05, fused=fused)
            candidate.load_state_dict(copy.deepcopy(saved))
            restore_optimizer_execution(candidate, fused)
            value.grad = gradient.clone()
            candidate.step()
            results.append(value.detach().clone())
        torch.testing.assert_close(results[1], results[0], rtol=1e-5, atol=1e-7)
        report['fused_resume_max_abs_difference'] = float((results[1]-results[0]).abs().max())
        del original, candidate, saved
        cases = [(2, 12, False), (4, 0, False), (8, 6, False), (8, 8, False),
                 (16, 8, False), (16, 10, False), (16, 12, False),
                 (8, 6, True), (16, 8, True), (16, 10, True)]
        for batch, checkpoint_blocks, fused in cases:
            case = {'batch': batch, 'accumulate': 16 // batch, 'checkpoint_blocks': checkpoint_blocks, 'fused': fused}
            optimizer = None
            try:
                model.load_state_dict(base)
                for index, block in enumerate(model.blocks):
                    block.with_cp = index < checkpoint_blocks
                model.train()
                optimizer = torch.optim.AdamW(layer_parameters(model, 2e-5), weight_decay=.05, fused=fused)
                def step():
                    optimizer.zero_grad(set_to_none=True)
                    for start in range(0, 16, batch):
                        clips = prepare_batch(raw[start:start+batch], normalize_video, augment=True,
                                              strong=True, low_light=True, partial=True, quality=True)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            logits = model(clips)
                            loss = activity_loss(logits, targets[start:start+batch], dataset.classes, .25)
                        if not torch.isfinite(loss):
                            raise RuntimeError('nonfinite loss')
                        (loss * batch / 16).backward()
                        # Include the per-microbatch host synchronization of the production loop.
                        int((logits.argmax(1) == targets[start:start+batch]).sum())
                        loss.item()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step()
                step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    step()
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - began)
                case.update(passed=True, seconds_per_effective_step=timings,
                            samples_per_second=16/statistics.mean(timings),
                            peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                            peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
            except torch.cuda.OutOfMemoryError as exc:
                case.update(passed=False, error='CUDA OOM: ' + str(exc).splitlines()[0])
            finally:
                model.zero_grad(set_to_none=True)
                del optimizer
                gc.collect()
                torch.cuda.empty_cache()
            report['cases'].append(case)
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(json.dumps(case), flush=True)
        model.eval()
        for block in model.blocks:
            block.with_cp = False
        for batch in (2, 8, 16, 32):
            inputs = raw[torch.arange(batch, device='cuda') % 16]
            timings = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                clips = normalize_video(inputs.transpose(1, 2))
                model(clips)
                for _ in range(3):
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    model(clips)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - began)
            report['evaluation'].append({'batch': batch, 'samples_per_second': batch/statistics.mean(timings)})
        validation = VideoManifestDataset(args.manifest, 'val', 8, (224, 224),
                                          classes=dataset.classes, preserve_aspect=True)
        subset = Subset(validation, list(range(min(32, len(validation)))))
        cached = cache_dataset(subset, 2)
        for name, data, workers in (('decode', subset, 2), ('uint8_cached', cached, 0)):
            began = time.perf_counter()
            count = 0
            for _ in range(2):
                for clips, labels in DataLoader(data, batch_size=8, num_workers=workers):
                    count += len(labels)
            report.setdefault('input', {})[name] = {'samples': count, 'seconds': time.perf_counter()-began}
        report['finished_at'] = time.time()
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report['evaluation']), flush=True)
        print(json.dumps(report['input']), flush=True)
    finally:
        if paused:
            try:
                os.kill(args.pause_pid, signal.SIGCONT)
            except ProcessLookupError:
                pass


if __name__ == '__main__':
    main()
