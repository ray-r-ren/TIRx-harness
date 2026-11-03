"""inspect_kernel: trace a TIRx-lite file and report its kernels."""
from pathlib import Path

from skillnet_sdk import call_tool

from tools.inspect_tool import inspect_kernel


def test_vector_add_launch_and_params(kernels, tmp_path):
    env = call_tool(inspect_kernel, {"kernel_file": str(kernels["vector_add"])}, output_root=tmp_path)
    assert env.status == "ok", env.summary
    (k,) = env.data["kernels"]
    assert k["name"] == "vector_add"
    assert [(p["name"], p["kind"], p["dtype"]) for p in k["params"]] == [
        ("a", "pointer", "float32"), ("b", "pointer", "float32"), ("out", "pointer", "float32")]
    assert k["grid"] == [1] and k["grid_static"] is True
    assert k["warps"] == 1 and k["threads_per_cta"] == 32
    assert k["arch"] == "sm_90a" and k["blackwell_only"] is False
    assert env.data["has_case_functions"] is True
    assert env.data["case_check"]["ok"] is True
    assert env.data["case_check"]["input_shapes"]["a"] == [32]
    tirx = next(o for o in env.outputs if o.name.endswith(".tirx.txt"))
    text = Path(tirx.path).read_text()
    assert "vector_add" in text and len(text.splitlines()) > 5


def test_scalar_params_and_grid(kernels, tmp_path):
    env = call_tool(inspect_kernel, {"kernel_file": str(kernels["starter_saxpy"])}, output_root=tmp_path)
    assert env.status == "ok"
    (k,) = env.data["kernels"]
    assert k["grid"] == [8] and k["warps"] == 4 and k["threads_per_cta"] == 128
    kinds = {p["name"]: (p["kind"], p["dtype"]) for p in k["params"]}
    assert kinds["alpha"] == ("scalar", "float32")
    assert kinds["n"] == ("scalar", "int32")


def test_two_kernels_are_all_listed(kernels, tmp_path):
    env = call_tool(inspect_kernel, {"kernel_file": str(kernels["two"])}, output_root=tmp_path)
    assert env.status == "ok"
    assert env.data["all_kernel_names"] == ["add_one", "double"]
    by_name = {k["name"]: k for k in env.data["kernels"]}
    assert by_name["double"]["arch"] == "sm_80" and by_name["double"]["grid"] == [4]
    assert by_name["double"]["warps"] == 2
    assert env.data["has_make_inputs"] is True and env.data["has_reference"] is False
    assert env.data["has_case_functions"] is False


def test_select_one_kernel_by_name(kernels, tmp_path):
    env = call_tool(inspect_kernel, {"kernel_file": str(kernels["two"]), "kernel_name": "double"},
                    output_root=tmp_path)
    assert env.status == "ok"
    assert [k["name"] for k in env.data["kernels"]] == ["double"]


def test_unknown_kernel_name(kernels, tmp_path):
    env = call_tool(inspect_kernel, {"kernel_file": str(kernels["two"]), "kernel_name": "triple"},
                    output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "kernel_not_found"
    assert "add_one" in env.next_step and "double" in env.next_step


def test_blackwell_kernel_is_flagged(kernels, tmp_path):
    env = call_tool(inspect_kernel, {"kernel_file": str(kernels["va_blackwell"])}, output_root=tmp_path)
    assert env.status == "ok"
    assert env.data["kernels"][0]["arch"] == "sm_100a"
    assert env.data["kernels"][0]["blackwell_only"] is True


def test_syntax_error_reports_line(kernels, tmp_path):
    env = call_tool(inspect_kernel, {"kernel_file": str(kernels["broken"])}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "kernel_trace_failed"
    assert "line 8" in env.error.message
    assert env.next_step


def test_future_annotations_explained(kernels, tmp_path):
    env = call_tool(inspect_kernel, {"kernel_file": str(kernels["future"])}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "kernel_trace_failed"
    assert "__future__" in (env.error.message + env.next_step)
