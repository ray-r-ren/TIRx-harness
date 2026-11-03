"""export_kernel: standalone CUDA C++ with its host wrapper, optionally built into a shared library."""
import re
from pathlib import Path

import pytest
from skillnet_sdk import call_tool

from conftest import has_nvcc
from tools.export_tool import export_kernel


def test_export_source_and_usage(kernels, tmp_path):
    env = call_tool(export_kernel, {"kernel_file": str(kernels["starter_saxpy"])}, output_root=tmp_path)
    assert env.status == "ok", env.summary
    d = env.data
    assert d["entry_symbol"] == "saxpy" and d["arch"] == "sm_90a" and d["ffi_arch"] == "9.0a"
    assert [a["name"] for a in d["args"]] == ["x", "y", "out", "alpha", "n"]
    assert d["built"] is False
    cu = Path(next(o.path for o in env.outputs if o.name == "saxpy.cu")).read_text()
    assert re.search(r"__global__\s+void", cu)
    assert "__tvm_ffi_saxpy" in cu
    usage = Path(next(o.path for o in env.outputs if o.name == "USAGE.md")).read_text()
    assert "alpha" in usage and "9.0a" in usage


@pytest.mark.skipif(not has_nvcc(), reason="nvcc is not installed")
def test_build_shared_library_fp16(kernels, tmp_path):
    env = call_tool(export_kernel, {"kernel_file": str(kernels["half_scale"]), "build_library": True},
                    output_root=tmp_path)
    assert env.status == "ok", (env.summary, env.data.get("build_error"))
    assert env.data["built"] is True
    so = Path(next(o.path for o in env.outputs if o.name.endswith(".so")))
    assert so.read_bytes()[:4] == b"\x7fELF"
    import subprocess
    syms = subprocess.run(["nm", "-D", "--defined-only", str(so)], capture_output=True, text=True).stdout
    assert "__tvm_ffi_half_scale" in syms  # the FFI entry the USAGE.md call uses
    cu = Path(next(o.path for o in env.outputs if o.name == "half_scale.cu")).read_text()
    assert "#include <cuda_fp16.h>" in cu.split("#ifdef __CUDACC_RTC__")[0]


def test_ambiguous_kernel(kernels, tmp_path):
    env = call_tool(export_kernel, {"kernel_file": str(kernels["two"])}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "ambiguous_kernel"
    assert "kernel_name" in env.next_step
