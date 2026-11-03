"""check_kernel_safety: Synccheck + Racecheck on the concrete launch."""
from pathlib import Path

from skillnet_sdk import call_tool

from tools.safety import check_kernel_safety


def test_clean_kernels(kernels, tmp_path):
    for name in ("vector_add", "smem_exchange", "starter_reduction"):
        env = call_tool(check_kernel_safety, {"kernel_file": str(kernels[name])}, output_root=tmp_path / name)
        assert env.status == "ok", env.summary
        assert env.data["overall"] == "clean", (name, env.data)
        assert env.data["synccheck"]["verdict"] == "clean"
        assert env.data["racecheck"]["verdict"] == "clean"
    names = {o.name for o in env.outputs}
    assert {"safety_report.json", "safety_report.txt"} <= names


def test_missing_barrier_is_a_race(kernels, tmp_path):
    env = call_tool(check_kernel_safety, {"kernel_file": str(kernels["racy"])}, output_root=tmp_path)
    assert env.status == "ok"
    assert env.data["overall"] == "error"
    race = env.data["racecheck"]
    assert race["verdict"] == "error"
    kinds = {f["kind"] for f in race["findings"]}
    assert "data_race" in kinds
    lines = {ln for f in race["findings"] for ln in f["lines"]}
    src = Path(kernels["racy"]).read_text().splitlines()
    store = next(i for i, l in enumerate(src, 1) if "st.shared" in l)
    load = next(i for i, l in enumerate(src, 1) if "ld.shared" in l)
    assert {store, load} <= lines
    text = Path(next(o.path for o in env.outputs if o.name == "safety_report.txt")).read_text()
    assert "data_race" in text


def test_barrier_count_deadlock(kernels, tmp_path):
    env = call_tool(check_kernel_safety, {"kernel_file": str(kernels["hang"]), "checks": "sync"},
                    output_root=tmp_path)
    assert env.data["overall"] == "error"
    assert env.data["racecheck"] is None
    assert any(f["kind"] == "deadlock" for f in env.data["synccheck"]["findings"])


def test_opaque_helper_is_incomplete(kernels, tmp_path):
    env = call_tool(check_kernel_safety, {"kernel_file": str(kernels["opaque"])}, output_root=tmp_path)
    assert env.status == "partial"
    assert env.data["overall"] == "incomplete"
    reasons = " ".join(str(env.data[c]["reason"] or "") for c in ("synccheck", "racecheck"))
    assert "my_square" in reasons or "func_call" in reasons


def test_broken_file_is_reported(kernels, tmp_path):
    env = call_tool(check_kernel_safety, {"kernel_file": str(kernels["broken"])}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "kernel_trace_failed"
    assert env.next_step
