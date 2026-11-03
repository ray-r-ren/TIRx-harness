"""find_kernel_examples: search the starters, the kernel collection and the reference docs."""
from pathlib import Path

from skillnet_sdk import call_tool

from tools.find_examples import find_kernel_examples


def test_rmsnorm_finds_starter_first(tmp_path):
    env = call_tool(find_kernel_examples, {"query": "rmsnorm"}, output_root=tmp_path)
    assert env.status == "ok", env.summary
    matches = env.data["matches"]
    assert env.data["total"] >= 3
    top = matches[0]
    assert top["kind"] == "kernel"
    assert top["path"] == "starter/rmsnorm.py"
    assert top["verified"] is True and top["runs_on_h100"] is True
    assert "sm_90a" in top["archs"]
    # The collection's RMSNorm port is also found and flagged as Blackwell.
    coll = [m for m in matches if m["kind"] == "kernel" and m["path"].startswith("tirx_kernels/")]
    assert any("rmsnorm" in m["path"] for m in coll)
    assert all(m["runs_on_h100"] is False for m in coll if m["archs"] == ["sm_100a"])
    # The top source comes back as a file that is real Python defining the kernel.
    names = [o.name for o in env.outputs]
    assert "starter_rmsnorm.py" in names
    src = Path(next(o.path for o in env.outputs if o.name == "starter_rmsnorm.py")).read_text()
    assert "@txl.kernel" in src and "def reference" in src
    compile(src, "starter_rmsnorm.py", "exec")


def test_reference_scope_finds_mbarrier_docs(tmp_path):
    env = call_tool(find_kernel_examples, {"query": "mbarrier", "scope": "reference", "max_results": 5},
                    output_root=tmp_path)
    assert env.status == "ok"
    kinds = {m["kind"] for m in env.data["matches"]}
    assert kinds <= {"doc", "api"} and kinds
    assert any("mbarrier" in (m["excerpt"] + m["name"] + m["path"]).lower() for m in env.data["matches"])
    assert len(env.data["matches"]) <= 5
    assert any(o.name == "reference_matches.md" for o in env.outputs)


def test_supported_ops_table_is_searchable(tmp_path):
    env = call_tool(find_kernel_examples, {"query": "shfl sync supported operations", "scope": "reference"},
                    output_root=tmp_path)
    assert env.status == "ok"
    assert any("SUPPORTED_OPS" in m["path"] for m in env.data["matches"])


def test_nonsense_query_returns_nothing_with_next_step(tmp_path):
    env = call_tool(find_kernel_examples, {"query": "zzqx flibber"}, output_root=tmp_path)
    assert env.status == "ok"
    assert env.data["total"] == 0 and env.data["matches"] == []
    assert env.next_step and "starter='elementwise'" in env.next_step


def test_empty_query_is_rejected(tmp_path):
    env = call_tool(find_kernel_examples, {"query": ""}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "missing_query"
    assert env.next_step and "starter" in env.next_step
