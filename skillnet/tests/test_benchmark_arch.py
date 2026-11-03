"""benchmark_kernel's static architecture check: refuses non-H100 kernels by name before any device work."""
import pytest
from skillnet_sdk import call_tool

from conftest import SOURCES, has_gpu
from tools.benchmark import benchmark_kernel

GPU = has_gpu()

TCGEN05 = '''import numpy as np
import tirx_kernels.tirx_lite as txl


@txl.kernel(warps=1, arch="sm_90a", grid=1)
def copy_tc(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([lane]))
    txl.ptx.tcgen05.fence__after_thread_sync()
    txl.ptx.st.global_.f32(out.ptr_to([lane]), x)


def make_inputs():
    return {"a": np.arange(32, dtype=np.float32), "out": np.zeros(32, np.float32)}


def reference(inputs):
    return {"out": inputs["a"].copy()}
'''


def test_sm100a_kernel_is_refused_by_name(tmp_path):
    env = call_tool(benchmark_kernel, {"source": SOURCES["va_blackwell"], "repeats": 10, "runs": 1},
                    output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "arch_unsupported"  # raised before the GPU is looked for
    assert "sm_100a" in env.error.message
    assert "H100" in env.error.message and "sm_90a" in env.error.message
    assert "Nothing was timed" in env.error.message
    assert "sm_90a" in env.next_step and "simulate_kernel" in env.next_step


def test_sm100a_file_is_refused_too(kernels, tmp_path):
    env = call_tool(benchmark_kernel, {"kernel_file": str(kernels["va_blackwell"])}, output_root=tmp_path)
    assert env.error.code == "arch_unsupported" and "sm_100a" in env.error.message


def test_tcgen05_kernel_is_refused_even_if_declared_sm90a(tmp_path):
    env = call_tool(benchmark_kernel, {"source": TCGEN05}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "arch_unsupported"
    assert "tcgen05" in env.error.message and "H100" in env.error.message


def test_blackwell_baseline_is_refused_naming_the_baseline(tmp_path):
    env = call_tool(benchmark_kernel, {"source": SOURCES["vector_add"], "baseline_source": SOURCES["va_blackwell"]},
                    output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "arch_unsupported"
    assert env.error.message.startswith("Baseline kernel")
    assert "sm_100a" in env.error.message
    assert "baseline" in env.next_step


def test_both_baseline_inputs_are_ambiguous(kernels, tmp_path):
    env = call_tool(benchmark_kernel, {"source": SOURCES["vector_add"], "baseline_source": SOURCES["vector_add"],
                                       "baseline_file": str(kernels["vector_add"])}, output_root=tmp_path)
    assert env.error.code == "ambiguous_kernel_input"
    assert "baseline" in env.error.message


@pytest.mark.skipif(GPU, reason="a GPU is present")
def test_h100_kernel_without_gpu_is_no_gpu(tmp_path):
    env = call_tool(benchmark_kernel, {"source": SOURCES["vector_add"]}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "no_gpu"
    assert "not possible" in env.next_step


@pytest.mark.skipif(not GPU, reason="no NVIDIA GPU visible")
def test_saxpy_starter_source_on_h100(kernels, tmp_path):
    src = kernels["starter_saxpy"].read_text()
    env = call_tool(benchmark_kernel, {"source": src, "baseline_source": src, "repeats": 20, "runs": 2},
                    output_root=tmp_path)
    assert env.status == "ok", env.summary
    d = env.data
    assert d["correct"] is True and d["median_us"] > 0
    assert "H100" in d["device"] and d["device_arch"].startswith("sm_90")
    assert d["speedup"] and d["speedup"] > 0
