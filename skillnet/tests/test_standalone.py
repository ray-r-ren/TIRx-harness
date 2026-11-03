"""Every kernel .py the skill hands out loads on its own: no relative imports, and rewritten imports resolve."""
import ast
import importlib.util
from pathlib import Path

import pytest
from skillnet_sdk import call_tool

from conftest import SOURCES
from tools._standalone import make_standalone
from tools.export_tool import export_kernel
from tools.find_examples import find_kernel_examples


def _site_root() -> Path:
    spec = importlib.util.find_spec("tirx_kernels")
    assert spec and spec.submodule_search_locations
    return Path(list(spec.submodule_search_locations)[0]).parent


def _module_file_exists(dotted: str) -> bool:
    rel = Path(*dotted.split("."))
    root = _site_root()
    return (root / rel).with_suffix(".py").is_file() or (root / rel / "__init__.py").is_file()


def _imports(text: str):
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom):
            yield node


@pytest.mark.parametrize("query", ["gemm", "rmsnorm", "softmax", "flash attention", "reduction"])
def test_find_outputs_have_no_relative_imports(query, tmp_path):
    env = call_tool(find_kernel_examples, {"query": query, "scope": "kernels", "max_results": 10},
                    output_root=tmp_path)
    assert env.status == "ok", env.summary
    pys = [Path(o.path) for o in env.outputs if o.name.endswith(".py")]
    assert pys, "expected at least one kernel copy"
    for p in pys:
        text = p.read_text()
        for node in _imports(text):
            assert node.level == 0, f"{p.name}: relative import left on line {node.lineno}"
            if node.module and node.module.startswith("tirx_kernels"):
                assert _module_file_exists(node.module), f"{p.name}: {node.module} does not exist"
    for m in env.data["matches"]:
        if m["file"]:
            assert m["standalone"] is True


def test_collection_copy_is_rewritten_with_same_line_count(tmp_path):
    env = call_tool(find_kernel_examples, {"query": "gemm", "scope": "kernels", "max_results": 10},
                    output_root=tmp_path)
    coll = [m for m in env.data["matches"] if m["path"].startswith("tirx_kernels/") and m["file"]]
    assert coll, "the collection should have gemm kernels"
    root = _site_root()
    for m in coll:
        original = (root / m["path"]).read_text()
        copy = Path(next(o.path for o in env.outputs if o.name == m["file"])).read_text()
        assert len(copy.splitlines()) == len(original.splitlines())


def test_make_standalone_rewrites_and_flags_missing_targets():
    root = _site_root()
    ok = make_standalone("from .tirx_lite import kernel\nx = 1\n", "tirx_kernels", root)
    assert ok.standalone and ok.rewritten == 1
    assert ok.text.splitlines()[0] == "from tirx_kernels.tirx_lite import kernel"
    bad = make_standalone("from .no_such_module_xyz import thing\n", "tirx_kernels", root)
    assert bad.standalone is False and bad.problems
    multi = make_standalone("from .tirx_lite import (\n    kernel,\n)\ny = 2\n", "tirx_kernels", root)
    assert multi.standalone and multi.text.splitlines()[3] == "y = 2"


def test_export_kernel_py_is_standalone(kernels, tmp_path):
    env = call_tool(export_kernel, {"kernel_file": str(kernels["starter_softmax"])}, output_root=tmp_path)
    assert env.status == "ok", env.summary
    py = Path(next(o.path for o in env.outputs if o.name == "softmax.py")).read_text()
    assert all(n.level == 0 for n in _imports(py))
    assert py == kernels["starter_softmax"].read_text()


def test_kernel_file_with_relative_import_is_refused(tmp_path):
    p = tmp_path / "uses_helpers.py"
    p.write_text("import numpy as np\nfrom .helpers import tile_load\n" + SOURCES["vector_add"])
    from tools.inspect_tool import inspect_kernel
    env = call_tool(inspect_kernel, {"kernel_file": str(p)}, output_root=tmp_path / "o")
    assert env.status == "error"
    assert env.error.code == "relative_import"
    assert "line 2" in env.error.message and "from .helpers import tile_load" in env.error.message
    assert "one file" in env.next_step or "inline" in env.next_step
