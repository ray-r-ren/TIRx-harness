"""The seven shipped H100 (sm_90a) starter kernels: standalone, simulate pass, safety clean, compile."""
import ast
from pathlib import Path

import pytest
from skillnet_sdk import call_tool

from conftest import STARTERS
from tools.compile_tool import compile_kernel
from tools.find_examples import STARTER_NAMES, find_kernel_examples
from tools.inspect_tool import inspect_kernel
from tools.safety import check_kernel_safety
from tools.simulate import simulate_kernel

NAMES = ["elementwise", "saxpy", "reduction", "softmax", "rmsnorm", "layernorm", "matmul"]


def test_starter_list_is_complete():
    assert sorted(STARTER_NAMES) == sorted(NAMES)
    assert sorted(p.stem for p in STARTERS.glob("*.py")) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_starter_is_standalone(name):
    text = (STARTERS / f"{name}.py").read_text()
    tree = ast.parse(text)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import on line {node.lineno}"
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    assert mods <= {"numpy", "tirx_kernels.tirx_lite"}, mods
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"make_inputs", "reference"} <= funcs
    assert 'arch="sm_90a"' in text
    assert "BYTES_MOVED" in text and "FLOPS" in text


@pytest.mark.parametrize("name", NAMES)
def test_starter_inspects_simulates_and_is_safe(name, kernels, tmp_path):
    path = str(kernels[f"starter_{name}"])
    ins = call_tool(inspect_kernel, {"kernel_file": path}, output_root=tmp_path / "i")
    assert ins.status == "ok", ins.summary
    (k,) = ins.data["kernels"]
    assert k["arch"] == "sm_90a" and k["blackwell_only"] is False
    assert ins.data["case_check"]["ok"] is True

    sim = call_tool(simulate_kernel, {"kernel_file": path}, output_root=tmp_path / "s")
    assert sim.status == "ok", sim.summary
    assert sim.data["verdict"] == "pass", sim.data
    assert sim.data["total_mismatches"] == 0

    safe = call_tool(check_kernel_safety, {"kernel_file": path}, output_root=tmp_path / "c")
    assert safe.status == "ok", safe.summary
    assert safe.data["overall"] == "clean", safe.data


@pytest.mark.parametrize("name", NAMES)
def test_starter_compiles_for_h100(name, kernels, tmp_path):
    env = call_tool(compile_kernel, {"kernel_file": str(kernels[f"starter_{name}"])}, output_root=tmp_path)
    assert env.status == "ok", (env.summary, env.data.get("errors") if env.data else None)
    assert env.data["arch"] == "sm_90a" and env.data["ptx_target"] == "sm_90a"
    assert env.data["registers"] > 0
    assert env.data["spill_stores"] == 0


def test_softmax_query_returns_verified_starter_with_source(tmp_path):
    env = call_tool(find_kernel_examples, {"query": "softmax"}, output_root=tmp_path)
    assert env.status == "ok", env.summary
    top = env.data["matches"][0]
    assert top["path"] == "starter/softmax.py"
    assert top["verified"] is True and top["runs_on_h100"] is True and top["standalone"] is True
    out = Path(next(o.path for o in env.outputs if o.name == top["file"]))
    assert top["source"] == out.read_text() == (STARTERS / "softmax.py").read_text()


@pytest.mark.parametrize("query,expected", [("vector add", "elementwise"), ("sum of each row", "reduction"),
                                            ("matrix multiply", "matmul"), ("gemm", "matmul"),
                                            ("layer norm", "layernorm"), ("rms norm", "rmsnorm")])
def test_plain_words_rank_the_right_starter_first(query, expected, tmp_path):
    env = call_tool(find_kernel_examples, {"query": query}, output_root=tmp_path)
    assert env.data["matches"][0]["path"] == f"starter/{expected}.py"


def test_starter_argument_fetches_exactly_one(tmp_path):
    env = call_tool(find_kernel_examples, {"starter": "matmul"}, output_root=tmp_path)
    assert env.status == "ok", env.summary
    assert [m["path"] for m in env.data["matches"]] == ["starter/matmul.py"]
    m = env.data["matches"][0]
    assert m["source"] and "def matmul" in m["source"]
    assert [o.name for o in env.outputs] == ["starter_matmul.py"]


def test_returned_starter_simulates_as_file_and_as_source(tmp_path):
    env = call_tool(find_kernel_examples, {"starter": "rmsnorm"}, output_root=tmp_path / "f")
    out = next(o.path for o in env.outputs if o.name == "starter_rmsnorm.py")
    as_file = call_tool(simulate_kernel, {"kernel_file": str(out)}, output_root=tmp_path / "a")
    assert as_file.data["verdict"] == "pass", as_file.summary
    as_src = call_tool(simulate_kernel, {"source": env.data["matches"][0]["source"]}, output_root=tmp_path / "b")
    assert as_src.data["verdict"] == "pass", as_src.summary
