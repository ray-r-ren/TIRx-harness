"""Kernels passed as pasted source text instead of an uploaded file."""
from pathlib import Path

import pytest
from skillnet_sdk import call_tool

from conftest import SOURCES
from tools.compile_tool import compile_kernel
from tools.export_tool import export_kernel
from tools.inspect_tool import inspect_kernel
from tools.safety import check_kernel_safety
from tools.simulate import simulate_kernel

VA = SOURCES["vector_add"]

ALL_TOOLS = [inspect_kernel, simulate_kernel, check_kernel_safety, compile_kernel, export_kernel]


def test_inspect_source_matches_file(kernels, tmp_path):
    by_file = call_tool(inspect_kernel, {"kernel_file": str(kernels["vector_add"])}, output_root=tmp_path / "f")
    by_src = call_tool(inspect_kernel, {"source": VA}, output_root=tmp_path / "s")
    assert by_src.status == "ok", by_src.summary
    assert by_src.data["input_kind"] == "source" and by_file.data["input_kind"] == "file"
    assert by_src.data["kernels"] == by_file.data["kernels"]
    assert by_src.data["case_check"]["ok"] is True
    assert by_src.data["kernels"][0]["name"] == "vector_add"


def test_simulate_source_pass(tmp_path):
    env = call_tool(simulate_kernel, {"source": VA}, output_root=tmp_path)
    assert env.status == "ok", env.summary
    assert env.data["verdict"] == "pass" and env.data["total_mismatches"] == 0
    assert float(env.data["outputs"][0]["max_abs_err"]) == 0.0


def test_simulate_source_locates_lane_bug(tmp_path):
    env = call_tool(simulate_kernel, {"source": SOURCES["vector_add_bug"]}, output_root=tmp_path)
    assert env.data["verdict"] == "fail"
    (o,) = env.data["outputs"]
    assert o["mismatches"] == 16 and o["mismatch_ranges"] == [[16, 31]]


def test_safety_source_clean_and_race_lines_match_text(tmp_path):
    clean = call_tool(check_kernel_safety, {"source": SOURCES["smem_exchange"]}, output_root=tmp_path / "c")
    assert clean.status == "ok", clean.summary
    assert clean.data["overall"] == "clean"
    racy_src = SOURCES["racy"]
    env = call_tool(check_kernel_safety, {"source": racy_src}, output_root=tmp_path / "r")
    assert env.data["overall"] == "error"
    lines = {ln for f in env.data["racecheck"]["findings"] for ln in f["lines"]}
    src = racy_src.splitlines()
    store = next(i for i, l in enumerate(src, 1) if "st.shared" in l)
    load = next(i for i, l in enumerate(src, 1) if "ld.shared" in l)
    # Line numbers refer to the text exactly as given (nothing was prepended).
    assert {store, load} <= lines


def test_compile_source(tmp_path):
    env = call_tool(compile_kernel, {"source": VA}, output_root=tmp_path)
    assert env.status == "ok", env.summary
    assert env.data["arch"] == "sm_90a" and env.data["registers"] > 0


def test_export_source_writes_cu_and_kernel_py(tmp_path):
    env = call_tool(export_kernel, {"source": VA}, output_root=tmp_path)
    assert env.status == "ok", env.summary
    names = {o.name for o in env.outputs}
    assert {"vector_add.cu", "USAGE.md", "vector_add.py"} <= names
    py = Path(next(o.path for o in env.outputs if o.name == "vector_add.py")).read_text()
    assert py == VA


@pytest.mark.parametrize("fn", ALL_TOOLS, ids=lambda f: f.__name__)
def test_neither_input_is_missing_kernel(fn, tmp_path):
    env = call_tool(fn, {}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "missing_kernel"
    assert "source=" in env.next_step and "kernel_file=" in env.next_step


@pytest.mark.parametrize("fn", ALL_TOOLS, ids=lambda f: f.__name__)
def test_both_inputs_are_ambiguous(fn, kernels, tmp_path):
    env = call_tool(fn, {"kernel_file": str(kernels["vector_add"]), "source": VA}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "ambiguous_kernel_input"
    assert "exactly one" in env.next_step


def test_syntax_error_line_refers_to_source_text(tmp_path):
    src = SOURCES["broken"]  # the unclosed parenthesis is on line 8 of this text
    assert "local_scalar(\"float32\"" in src.splitlines()[7]
    env = call_tool(inspect_kernel, {"source": src}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "kernel_trace_failed"
    assert "line 8" in env.error.message or "line 9" in env.error.message
    assert "source text" in env.error.message


def test_relative_import_in_source_is_refused(tmp_path):
    src = "from .helpers import tile\n" + VA
    env = call_tool(simulate_kernel, {"source": src}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "relative_import"
    assert "line 1" in env.error.message
    assert "tirx_kernels" in env.next_step


def test_source_too_long(tmp_path):
    env = call_tool(inspect_kernel, {"source": VA + "#" * 200_001}, output_root=tmp_path)
    assert env.status == "error"
    # The input schema caps the text at 200,000 characters; the tool keeps its own check as a backstop.
    assert env.error.code in ("invalid_arguments", "source_too_long")
    assert "source" in env.error.message
