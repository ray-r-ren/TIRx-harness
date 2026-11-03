"""compile_kernel: CUDA, PTX and SASS plus the assembler's resource report, on CPU."""
import re
from pathlib import Path

from skillnet_sdk import call_tool

from tools.compile_tool import compile_kernel


def _out(env, suffix):
    return Path(next(o.path for o in env.outputs if o.name.endswith(suffix)))


def test_rmsnorm_sm90a_full_report(kernels, tmp_path):
    env = call_tool(compile_kernel, {"kernel_file": str(kernels["starter_rmsnorm"])}, output_root=tmp_path)
    assert env.status == "ok", (env.summary, env.uncertain)
    d = env.data
    assert d["arch"] == "sm_90a" and d["ptx_target"] == "sm_90a"
    assert d["registers"] and d["registers"] > 0
    assert d["spill_stores"] == 0 and d["spill_loads"] == 0
    assert d["sass_instructions"] > 10
    ptx = _out(env, ".ptx").read_text()
    assert re.search(r"^\.target\s+sm_90a", ptx, re.M)
    assert "shfl.sync.bfly" in ptx
    sass = _out(env, ".sass").read_text()
    assert "SHFL.BFLY" in sass
    assert "__global__" in _out(env, ".cu").read_text()
    log = _out(env, ".ptxas.log").read_text()
    assert f"Used {d['registers']} registers" in log


def test_shared_memory_and_barrier_reported(kernels, tmp_path):
    env = call_tool(compile_kernel, {"kernel_file": str(kernels["smem_exchange"])}, output_root=tmp_path)
    assert env.status == "ok"
    assert env.data["dynamic_smem_bytes"] == 64 * 4
    assert env.data["barriers"] == 1


def test_other_arch_and_ptx_only(kernels, tmp_path):
    env = call_tool(compile_kernel, {"kernel_file": str(kernels["vector_add"]), "arch": "sm_80", "stages": "ptx"},
                    output_root=tmp_path)
    assert env.status == "ok", env.summary
    assert env.data["arch"] == "sm_80" and env.data["ptx_target"] == "sm_80"
    assert env.data["registers"] is None  # the register report needs the SASS stage
    names = {o.name for o in env.outputs}
    assert any(n.endswith(".ptx") for n in names) and not any(n.endswith(".sass") for n in names)


def test_fp16_kernel_compiles(kernels, tmp_path):
    env = call_tool(compile_kernel, {"kernel_file": str(kernels["half_scale"])}, output_root=tmp_path)
    assert env.status == "ok", (env.summary, env.data.get("errors"))
    assert env.data["registers"] > 0 and not env.data["errors"]


def test_trace_error(kernels, tmp_path):
    env = call_tool(compile_kernel, {"kernel_file": str(kernels["broken"])}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "kernel_trace_failed"
    assert env.next_step
