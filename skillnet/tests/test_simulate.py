"""simulate_kernel: NumSim run compared with an independent NumPy reference."""
import numpy as np
import pytest
from pathlib import Path

from skillnet_sdk import call_tool

from tools.simulate import simulate_kernel

STARTERS = ["vector_add", "smem_exchange", "scale_add", "starter_saxpy", "starter_reduction"]


@pytest.mark.parametrize("name", STARTERS)
def test_starters_pass(kernels, tmp_path, name):
    env = call_tool(simulate_kernel, {"kernel_file": str(kernels[name])}, output_root=tmp_path)
    assert env.status == "ok", env.summary
    assert env.data["verdict"] == "pass"
    assert env.data["total_mismatches"] == 0
    for o in env.data["outputs"]:
        assert o["mismatches"] == 0
        assert float(o["max_abs_err"]) < 1e-4


def test_vector_add_exact_and_npz_holds_values(kernels, tmp_path):
    env = call_tool(simulate_kernel, {"kernel_file": str(kernels["vector_add"])}, output_root=tmp_path)
    assert env.data["verdict"] == "pass"
    (o,) = env.data["outputs"]
    assert o["name"] == "out" and o["shape"] == [32] and o["dtype"] == "float32"
    assert float(o["max_abs_err"]) == 0.0
    npz = next(o for o in env.outputs if o.name.endswith(".npz"))
    with np.load(npz.path) as z:
        assert np.array_equal(z["out"], z["out__expected"])
    assert any(o.name == "numsim_report.json" for o in env.outputs)


def test_lane_bug_is_located(kernels, tmp_path):
    env = call_tool(simulate_kernel, {"kernel_file": str(kernels["vector_add_bug"])}, output_root=tmp_path)
    assert env.status == "ok"
    assert env.data["verdict"] == "fail"
    (o,) = env.data["outputs"]
    assert o["mismatches"] == 16
    assert o["mismatch_ranges"] == [[16, 31]]
    first = o["first_mismatches"][0]
    assert first["flat_index"] == 16
    assert first["got"] == pytest.approx(15.0) and first["expected"] == pytest.approx(17.0)
    assert "state_id" in env.next_step


def test_case_file_overrides_inputs(kernels, tmp_path):
    env = call_tool(simulate_kernel, {"kernel_file": str(kernels["vector_add"]),
                                      "case_file": str(kernels["va_case"])}, output_root=tmp_path)
    assert env.data["verdict"] == "pass"
    npz = next(o for o in env.outputs if o.name.endswith(".npz"))
    with np.load(npz.path) as z:
        assert np.allclose(z["out"], 0.0, atol=1e-6)  # a + reversed(a) for symmetric linspace


def test_fp16_input_kernel(kernels, tmp_path):
    env = call_tool(simulate_kernel, {"kernel_file": str(kernels["half_scale"])}, output_root=tmp_path)
    assert env.data["verdict"] == "pass", env.summary


def test_out_of_bounds_read_is_a_fault(kernels, tmp_path):
    env = call_tool(simulate_kernel, {"kernel_file": str(kernels["oob"])}, output_root=tmp_path)
    assert env.data["verdict"] == "fail"
    assert env.data["fault"] == "out_of_bounds"
    assert "index" in env.next_step


def test_unsupported_op_is_incomplete_with_rewrite_hint(kernels, tmp_path):
    env = call_tool(simulate_kernel, {"kernel_file": str(kernels["sqrt_kernel"])}, output_root=tmp_path)
    assert env.status == "partial"
    assert env.data["verdict"] == "incomplete"
    assert "tirx.sqrt" in env.data["reason"]
    assert "rsqrt" in env.next_step


def test_missing_reference(kernels, tmp_path):
    env = call_tool(simulate_kernel, {"kernel_file": str(kernels["two"]), "kernel_name": "add_one"},
                    output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "missing_case_functions"
    assert "reference" in env.next_step


def test_wrong_input_dtype(kernels, tmp_path):
    env = call_tool(simulate_kernel, {"kernel_file": str(kernels["vector_add"]),
                                      "case_file": str(kernels["bad_dtype_case"])}, output_root=tmp_path)
    assert env.status == "error"
    assert env.error.code == "invalid_inputs"
    assert "float" in env.error.message
    assert env.next_step


def test_state_reuses_simulator_cache(kernels, tmp_path):
    state = tmp_path / "state"
    first = call_tool(simulate_kernel, {"kernel_file": str(kernels["vector_add"])},
                      output_root=tmp_path / "o1", state_dir=state)
    assert first.data["verdict"] == "pass"
    assert any(state.rglob("*")), "the simulator cache should be kept in the state"
    second = call_tool(simulate_kernel, {"kernel_file": str(kernels["vector_add_bug"])},
                       output_root=tmp_path / "o2", state_dir=state)
    assert second.data["verdict"] == "fail"
    assert second.data["cache_reused"] is True
    assert second.data["same_kernel_as_before"] is False
    # The same text again, now pasted as source: recognised by content, not by file name.
    third = call_tool(simulate_kernel, {"source": kernels["vector_add"].read_text()},
                      output_root=tmp_path / "o3", state_dir=state)
    assert third.data["verdict"] == "pass"
    assert third.data["same_kernel_as_before"] is True
