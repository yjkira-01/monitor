"""Isolated CPU/CUDA compatibility and synthetic timing; no dataset training.

Invoke base and cuda modes in SEPARATE Python processes. CUDA mode only inserts
the workspace target into that process's module path; base packages are intact.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tiny_model(torch):
    return torch.nn.Sequential(torch.nn.Linear(51, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 51))


def environment(torch):
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "torch_version": torch.__version__, "torch_module": torch.__file__, "torch_cuda_build": torch.version.cuda,
        "numpy_version": np.__version__, "cpu_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available()}


def assert_numeric_match(actual, expected, name, atol, checks):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"Invalid numeric comparison: {name}")
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    checks.append({"name": name, "max_absolute_error": error, "absolute_tolerance": atol, "matched": error <= atol})
    if error > atol:
        raise ValueError(f"CPU/CUDA discrepancy at {name}: {error} > {atol}")


def timing(torch, step, device, repeats=20, warmup=5):
    for _ in range(warmup):
        step()
    values = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        begin = time.perf_counter()
        step()
        if device == "cuda":
            torch.cuda.synchronize()
        values.append((time.perf_counter() - begin) * 1000)
    return {"median_ms": float(np.median(values)), "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)), "repeats": repeats, "warmup_steps": warmup,
        "wall_time_including_host_checks_and_optimizer": True}


def fusion_timings(torch, device):
    from fusion_models import FusionModel, fusion_loss
    rng = np.random.default_rng(20261013)
    rows = []
    for method in ("concat", "utility_gate"):
        for batch_size in (64, 256, 1024):
            torch.manual_seed(20261013)
            model = FusionModel(method).to(device)
            rgb = torch.tensor(rng.standard_normal((batch_size, 51)), dtype=torch.float32, device=device)
            points = torch.tensor(rng.standard_normal((batch_size, 196, 5)), dtype=torch.float32, device=device)
            mask = torch.tensor(rng.random((batch_size, 196)) > .2, device=device)
            confidence = torch.full((batch_size, 17), .8, device=device)
            target = torch.tensor(rng.standard_normal((batch_size, 51)) * .1, dtype=torch.float32, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)

            def step():
                optimizer.zero_grad(set_to_none=True)
                output = model(rgb, points, mask, confidence)
                loss, _ = fusion_loss(output, target, method)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

            result = timing(torch, step, device)
            result.update(method=method, batch_size=batch_size, device=device,
                examples_per_second=batch_size / (result["median_ms"] / 1000),
                parameters=sum(p.numel() for p in model.parameters()))
            if device == "cuda":
                result["allocated_bytes_after_case"] = torch.cuda.memory_allocated()
                result["reserved_bytes_after_case"] = torch.cuda.memory_reserved()
            rows.append(result)
            print(json.dumps({"device": device, "method": method, "batch": batch_size, "median_ms": result["median_ms"]}), flush=True)
    return rows


def cuda_mechanism_smoke(torch):
    from fusion_models import METHODS, FUSION_METHODS, FusionModel, fusion_loss
    from fusion_graph_models import GRAPH_METHODS, GraphFusionModel
    from fusion_scaled_gate_models import SCALED_METHODS, ScaledGateFusionModel
    rows = []
    for family, methods, factory in (("base", METHODS, FusionModel), ("graph", GRAPH_METHODS, GraphFusionModel),
                                     ("scaled", SCALED_METHODS, ScaledGateFusionModel)):
        for method in methods:
            model = factory(method).cuda()
            rgb, points = torch.randn(4, 51, device="cuda"), torch.randn(4, 196, 5, device="cuda")
            mask, confidence = torch.ones(4, 196, dtype=torch.bool, device="cuda"), torch.ones(4, 17, device="cuda")
            mask[0], mask[2], confidence[1], confidence[2] = False, False, 0., 0.
            output = model(rgb, points, mask, confidence)
            if family != "base" or method in FUSION_METHODS:
                torch.testing.assert_close(output["fused"][0], output["video"][0], atol=0., rtol=0.)
                torch.testing.assert_close(output["fused"][1], output["radar"][1], atol=0., rtol=0.)
                torch.testing.assert_close(output["fused"][2], output["radar"][2], atol=0., rtol=0.)
            semantic = method.removeprefix("scaled_") if family == "scaled" else method
            loss, terms = fusion_loss(output, torch.zeros(4, 51, device="cuda"), semantic,
                                      torch.Generator(device="cuda").manual_seed(20261013))
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(output["fused"]).all():
                raise ValueError(f"Nonfinite CUDA smoke output/loss: {method}")
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError(f"Nonfinite CUDA smoke gradient: {method}")
            if ("utility_BCE" in terms) != semantic.startswith("utility"):
                raise ValueError(f"Wrong utility loss semantics: {method}")
            rows.append({"family": family, "method": method, "forward_loss_backward_finite": True,
                         "missing_fallbacks_checked": family != "base" or method in FUSION_METHODS,
                         "GPU_risk_generator_checked": "random" in method, "utility_BCE_present": "utility_BCE" in terms,
                         "loss": float(loss.detach().cpu())})
    return rows


def base_mode(torch, output):
    if (output / "base_summary.json").exists() or (output / "cpu_numeric_reference.npz").exists():
        raise ValueError("Refusing to overwrite base reference")
    torch.manual_seed(20261013)
    rng = np.random.default_rng(20261013)
    model = tiny_model(torch)
    inputs = torch.tensor(rng.standard_normal((64, 51)), dtype=torch.float32)
    target = torch.tensor(rng.standard_normal((64, 51)) * .1, dtype=torch.float32)
    arrays = {"input": inputs.numpy(), "target": target.numpy()}
    arrays.update({"initial__" + name: value.detach().numpy().copy() for name, value in model.state_dict().items()})
    prediction = model(inputs)
    loss = (prediction - target).square().mean()
    loss.backward()
    arrays["prediction"], arrays["loss"] = prediction.detach().numpy(), loss.detach().numpy()
    arrays.update({"gradient__" + name: value.grad.numpy().copy() for name, value in model.named_parameters()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    optimizer.step()
    arrays.update({"after_adamw__" + name: value.detach().numpy().copy() for name, value in model.state_dict().items()})
    left = torch.tensor(rng.standard_normal((512, 512)) * .01, dtype=torch.float32, requires_grad=True)
    right = torch.tensor(rng.standard_normal((512, 512)) * .01, dtype=torch.float32, requires_grad=True)
    product = left @ right
    product.square().mean().backward()
    arrays.update(matmul_left=left.detach().numpy(), matmul_right=right.detach().numpy(),
                  matmul_product=product.detach().numpy(), matmul_left_gradient=left.grad.numpy(),
                  matmul_right_gradient=right.grad.numpy())
    reference_path = output / "cpu_numeric_reference.npz"
    np.savez_compressed(reference_path, **arrays)
    summary = {"status": "complete", "mode": "base_cpu", "environment": environment(torch),
        "numeric_reference_sha256": digest(reference_path), "timings": fusion_timings(torch, "cpu"),
        "source_sha256": {p.name: digest(p) for p in (Path(__file__), ROOT / "research/fusion_models.py")},
        "scope": "Synthetic compatibility/timing only, no mRI fitting or model deployment"}
    write_json(output / "base_summary.json", summary)


def cuda_mode(torch, output, target_path):
    if (output / "cuda_summary.json").exists():
        raise ValueError("Refusing to overwrite CUDA summary")
    if not Path(torch.__file__).resolve().is_relative_to(target_path.resolve()):
        raise ValueError("Torch was not imported from isolated workspace target")
    if not torch.cuda.is_available():
        raise ValueError("Installed isolated Torch reports no usable CUDA device")
    base = json.loads((output / "base_summary.json").read_text(encoding="utf-8"))
    reference_path = output / "cpu_numeric_reference.npz"
    if digest(reference_path) != base["numeric_reference_sha256"]:
        raise ValueError("CPU numeric reference checksum differs")
    with np.load(reference_path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if any(value.dtype.kind not in "buif" or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Invalid numeric reference arrays")
    model = tiny_model(torch).cuda()
    model.load_state_dict({name: torch.from_numpy(arrays["initial__" + name]) for name in model.state_dict()}, strict=True)
    inputs, target = [torch.from_numpy(arrays[key]).cuda() for key in ("input", "target")]
    prediction = model(inputs)
    loss = (prediction - target).square().mean()
    loss.backward()
    checks = []
    assert_numeric_match(prediction.detach().cpu().numpy(), arrays["prediction"], "MLP_prediction", 1e-6, checks)
    assert_numeric_match(loss.detach().cpu().numpy(), arrays["loss"], "MLP_MSE", 1e-8, checks)
    for name, value in model.named_parameters():
        assert_numeric_match(value.grad.cpu().numpy(), arrays["gradient__" + name], "MLP_gradient/" + name, 1e-6, checks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    optimizer.step()
    for name, value in model.state_dict().items():
        assert_numeric_match(value.detach().cpu().numpy(), arrays["after_adamw__" + name], "MLP_after_AdamW/" + name, 1e-5, checks)
    left = torch.tensor(arrays["matmul_left"], device="cuda", requires_grad=True)
    right = torch.tensor(arrays["matmul_right"], device="cuda", requires_grad=True)
    product = left @ right
    product.square().mean().backward()
    for name, actual in (("matmul_product", product), ("matmul_left_gradient", left.grad), ("matmul_right_gradient", right.grad)):
        assert_numeric_match(actual.detach().cpu().numpy(), arrays[name], name, 1e-6, checks)
    torch.cuda.synchronize()
    mechanism_checks = cuda_mechanism_smoke(torch)
    device = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    rows = fusion_timings(torch, "cuda")
    comparisons = []
    for cpu, gpu in zip(base["timings"], rows):
        if (cpu["method"], cpu["batch_size"]) != (gpu["method"], gpu["batch_size"]):
            raise ValueError("CPU/CUDA timing cases differ")
        comparisons.append({"method": cpu["method"], "batch_size": cpu["batch_size"],
            "cpu_median_ms": cpu["median_ms"], "cuda_median_ms": gpu["median_ms"],
            "same_batch_CPU_over_CUDA_speed_ratio": cpu["median_ms"] / gpu["median_ms"]})
    summary = {"status": "complete", "mode": "isolated_cuda", "environment": environment(torch),
        "isolated_target": str(target_path), "device": {"name": device.name, "compute_capability": [device.major, device.minor],
            "total_bytes": device.total_memory, "free_bytes_after_smoke": free, "driver_visible_total_bytes": total},
        "precision": "float32 IEEE matmul, TF32 disabled, no autocast", "deterministic_algorithms": True,
        "numeric_checks": checks, "all_numeric_checks_passed": all(c["matched"] for c in checks),
        "mechanism_checks": mechanism_checks, "all_19_family_mechanism_checks_passed": len(mechanism_checks) == 19,
        "timings": rows, "same_batch_comparisons": comparisons,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "source_sha256": {p.name: digest(p) for p in (Path(__file__), ROOT / "research/fusion_models.py")},
        "limits": "Synthetic fixed-shape step timing; no accuracy or complete training speed claim. CPU2 threads; same batch cases. Actual model host checks and optimizer included. Different batch sizes are separate cases, not fair accuracy comparisons.",
        "base_environment_unchanged_by_target_install": "Verify separately with a fresh base Python process lacking sys.path insertion"}
    write_json(output / "cuda_summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("base", "cuda"), required=True)
    parser.add_argument("--target", type=Path, default=ROOT / ".deps/research_cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/fusion_acceleration_20261013")
    args = parser.parse_args()
    if args.mode == "cuda":
        sys.path.insert(0, str(args.target.resolve()))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.mode == "cuda":
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "base":
        base_mode(torch, args.output)
    else:
        cuda_mode(torch, args.output, args.target)
    print(json.dumps({"mode": args.mode, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
