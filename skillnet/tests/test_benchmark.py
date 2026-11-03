"""benchmark_kernel: on-device correctness and timing (needs the skill's NVIDIA GPU)."""
import json
from pathlib import Path

import pytest
from skillnet_sdk import call_tool

from conftest import has_gpu
from tools.benchmark import benchmark_kernel

GPU = has_gpu()


@pytest.mark.skipif(GPU, reason="a GPU is present; the no-GPU path is tested on CPU-only machines")
def test_no_gpu_is_a_clear_error(kernels, tmp_path):
    env = call_tool(benchmark_kernel, {"kernel_file": str(kernels["vector_add"])}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "no_gpu"
    assert "simulate_kernel" in env.next_step and "not possible" in env.next_step


@pytest.mark.skipif(not GPU, reason="no NVIDIA GPU visible")
def test_time_and_compare_with_baseline(kernels, tmp_path):
    env = call_tool(benchmark_kernel, {"kernel_file": str(kernels["starter_saxpy"]),
                                       "baseline_file": str(kernels["starter_saxpy"]),
                                       "repeats": 20, "runs": 2}, output_root=tmp_path)
    assert env.status == "ok", env.summary
    d = env.data
    assert d["correct"] is True
    assert d["median_us"] > 0 and d["min_us"] <= d["median_us"]
    assert len(d["per_run_median_us"]) == 2
    assert d["baseline_median_us"] > 0 and d["speedup"] > 0
    report = json.loads(Path(next(o.path for o in env.outputs if o.name == "benchmark_report.json")).read_text())
    assert report


@pytest.mark.skipif(not GPU, reason="no NVIDIA GPU visible")
def test_wrong_kernel_is_not_reported_correct(kernels, tmp_path):
    env = call_tool(benchmark_kernel, {"kernel_file": str(kernels["vector_add_bug"]), "repeats": 10, "runs": 1},
                    output_root=tmp_path)
    assert env.status == "partial"
    assert env.data["correct"] is False


@pytest.mark.skipif(not GPU, reason="no NVIDIA GPU visible")
def test_blackwell_kernel_on_hopper(kernels, tmp_path):
    env = call_tool(benchmark_kernel, {"kernel_file": str(kernels["va_blackwell"]), "repeats": 10, "runs": 1},
                    output_root=tmp_path)
    if env.status == "error":
        assert env.error.code == "arch_unsupported"
        assert "sm_90a" in env.next_step or "simulate" in env.next_step.lower()
    else:  # a Blackwell device can run it
        assert env.data["device_arch"].startswith("sm_10")
