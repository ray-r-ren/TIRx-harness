"""simulate_kernel: run a kernel in NumSim on CPU and compare its outputs with a NumPy reference."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from skillnet_sdk import ToolContext, ToolInput, ToolOutput, ToolResult, tool

from ._common import output_files, remember, run_engine, seen_before
from ._models import (
    CASE_FILE_DESC,
    KERNEL_FILE_DESC,
    KERNEL_NAME_DESC,
    SOURCE_DESC,
    SOURCE_MAX_CHARS,
    KernelFile,
    OutputCheck,
    kernel_request,
    output_check,
)


class SimulateIn(ToolInput):
    kernel_file: KernelFile | None = Field(default=None, description=KERNEL_FILE_DESC)
    source: str | None = Field(default=None, max_length=SOURCE_MAX_CHARS, description=SOURCE_DESC)
    case_file: KernelFile | None = Field(default=None, description=CASE_FILE_DESC)
    kernel_name: str | None = Field(default=None, max_length=200, description=KERNEL_NAME_DESC)
    rtol: float = Field(default=1e-5, ge=0, le=1, description="Relative tolerance for np.isclose (use ~1e-2 for "
                                                                "fp16/bf16 outputs).")
    atol: float = Field(default=1e-8, ge=0, le=1e6, description="Absolute tolerance for np.isclose.")


class Diagnostic(BaseModel):
    status: str
    kind: str
    count: int
    examples: list[str]


class SimulateOut(ToolOutput):
    verdict: Literal["pass", "fail", "incomplete"] = Field(
        description="pass = every output within tolerance; fail = wrong values or a simulator fault; incomplete = "
                    "the simulator could not model the kernel")
    kernel: str
    outputs: list[OutputCheck]
    total_mismatches: int
    fault: str | None = Field(default=None, description="deadlock, out_of_bounds or execution_fault when the "
                                                        "simulated launch itself failed")
    execution_error: str | None = None
    reason: str | None = Field(default=None, description="Why the result is incomplete")
    diagnostics: list[Diagnostic] = Field(default_factory=list, description="Simulator diagnostics, grouped")
    sim_verdict: str = Field(description="NumSim's own verdict on the run")
    extra_input_keys: list[str] = Field(default_factory=list)
    rtol: float
    atol: float
    simulate_seconds: float | None = Field(default=None, description="CPU simulation time (not GPU latency)")
    cache_reused: bool = Field(description="Simulator build artifacts from an earlier call in the state passed as "
                                           "state_id were available for reuse")
    same_kernel_as_before: bool = Field(default=False, description="This exact kernel text (file or source) was "
                                                                   "already simulated in that state")


def _describe(o: OutputCheck) -> str:
    total = 1
    for s in o.shape:
        total *= s
    if o.missing:
        return f"{o.name}: no value returned"
    if o.shape_mismatch:
        return f"{o.name}: {o.shape_mismatch}"
    if not o.mismatches:
        return f"{o.name}: all {total} match (max_abs_err {o.max_abs_err})"
    where = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in o.mismatch_ranges[:3])
    more = "…" if o.mismatch_range_count > 3 else ""
    return f"{o.name}: {o.mismatches} of {total} wrong (flat {where}{more}; max_abs_err {o.max_abs_err})"


@tool(
    kind="check",
    title="Check a kernel's numbers on CPU (NumSim)",
    long_running=True,
    timeout_s=900,
    state_version=1,
    examples=[{"arguments": {"kernel_file": "my_kernel.py"}, "description": "Verify a kernel against its reference"}],
)
def simulate_kernel(args: SimulateIn, ctx: ToolContext) -> ToolResult[SimulateOut]:
    """Use to verify a kernel computes the right math without a GPU ('is my kernel correct?', 'which elements are
    wrong?'). After inspect_kernel. Takes an uploaded file (kernel_file) or pasted source text (source). Runs
    NumSim, a CPU simulator of the GPU program, on make_inputs() and compares every output with reference(inputs)
    under rtol/atol. Returns pass/fail/incomplete, max abs/rel error per output, mismatch counts, the first
    mismatching indices (got/expected) and mismatch ranges. Simulator time is not GPU latency. Returns a state_id:
    pass it to the next simulate_kernel call to reuse compiled simulator work."""
    ctx.progress(5, "loading the kernel and test case")
    numsim_dir = ctx.state_dir / "numsim"
    req = kernel_request(args.kernel_file, args.source, args.case_file, args.kernel_name)
    reused = bool(ctx.state_restored and numsim_dir.is_dir() and any(numsim_dir.iterdir()))
    same = bool(ctx.state_restored and seen_before(ctx.state_dir, req["_sha256"]))
    req.update({"rtol": args.rtol, "atol": args.atol})
    ctx.progress(15, "simulating on CPU (first run of a new kernel compiles it)")
    data = run_engine(ctx, "simulate", req, timeout_s=860, numsim_dir=numsim_dir)
    remember(ctx.state_dir, req["_sha256"])
    ctx.mark_untrusted("kernel file / source text / simulator output")
    outs = [output_check(o) for o in data.get("outputs", [])]
    shown = outs[:8]
    diags = [Diagnostic(status=d["status"], kind=d["kind"], count=d["count"],
                        examples=[e[:220] for e in d.get("examples", [])[:2]])
             for d in (data.get("diagnostics") or [])[:10]]
    verdict = data["verdict"]
    timing = data.get("timing_s") or {}
    out = SimulateOut(
        verdict=verdict,
        kernel=data["kernel"],
        outputs=shown,
        total_mismatches=sum(o.mismatches for o in outs),
        fault=data.get("fault"),
        execution_error=data.get("execution_error"),
        reason=data.get("reason"),
        diagnostics=diags,
        sim_verdict=data.get("sim_verdict") or "",
        extra_input_keys=data.get("extra_input_keys") or [],
        rtol=args.rtol,
        atol=args.atol,
        simulate_seconds=round(sum(timing.values()), 2) if timing else None,
        cache_reused=reused,
        same_kernel_as_before=same,
    )
    outputs = output_files(data, {
        ".npz": "Simulated outputs and expected values (NumPy .npz; <name> and <name>__expected)",
        "numsim_report.json": "Full simulator report: comparisons, every diagnostic, stats",
    })
    uncertain = [f"The check covers only the inputs from make_inputs() (rtol={args.rtol}, atol={args.atol}); other "
                 "shapes or values are not proven."]
    status = "ok"
    if verdict == "pass":
        summary = f"PASS: kernel {out.kernel} matches the reference. " + "; ".join(_describe(o) for o in shown[:3])
        next_step = ("Run check_kernel_safety if you have not yet; for speed, benchmark_kernel (H100, sm_90a "
                     "kernels). Report max_abs_err and the mismatch count as evidence.")
    elif verdict == "fail" and out.fault:
        summary = f"FAIL: simulated launch of {out.kernel} stopped with {out.fault}: {(out.execution_error or '')[:200]}"
        hints = {
            "deadlock": "Run check_kernel_safety for barrier/mbarrier evidence (Synccheck), fix the wait/arrive "
                        "counts, then simulate again with the state_id.",
            "out_of_bounds": "Check the index math against the buffer sizes in make_inputs() (and tail guards), fix, "
                             "then simulate again with the state_id.",
        }
        next_step = hints.get(out.fault, "Read execution_error, fix the kernel and simulate again with the state_id.")
    elif verdict == "fail":
        summary = f"FAIL: kernel {out.kernel} output differs from the reference. " + "; ".join(
            _describe(o) for o in shown if o.mismatches or o.missing or o.shape_mismatch)[:240]
        if any(d.status == "error" for d in diags) and not out.total_mismatches:
            summary = f"FAIL: simulator reported error diagnostics for {out.kernel}: " + diags[0].kind
        next_step = ("Use first_mismatches and mismatch_ranges to locate the bug (a contiguous lane range, a stride "
                     "pattern or only the tail point to different causes), fix the kernel, and call simulate_kernel "
                     "again with the returned state_id. Confirm reference() is independent NumPy math.")
    else:
        status = "partial"
        summary = f"INCOMPLETE: the simulator could not model kernel {out.kernel}."
        uncertain.append(f"Not simulated: {(out.reason or 'unsupported operation')[:400]}")
        next_step = ("Rewrite the unsupported operation with supported instructions (call find_kernel_examples with "
                     "scope='reference' and the operation name to see the supported-operations table), or tell the "
                     "user the numbers could not be checked on CPU.")
        reason = out.reason or ""
        if "tirx.sqrt" in reason:
            next_step = ("txl.sqrt is not simulated: write sqrt(a) as a * txl.rsqrt(a) (a > 0; add a tiny epsilon if a "
                         "can be 0), then call simulate_kernel again with the state_id.")
        elif "tirx.tanh" in reason:
            next_step = ("txl.tanh is not simulated: write it with exp, e = txl.exp(-2*v); (1 - e) / (1 + e), then "
                         "simulate again with atol around 1e-6.")
    if out.fault is None and verdict != "incomplete":
        uncertain.append("Simulator time is CPU time, not GPU latency; speed comes only from benchmark_kernel.")
    if len(outs) > len(shown):
        uncertain.append(f"{len(outs) - len(shown)} more output(s) are only in numsim_report.json.")
    return ToolResult(data=out, summary=summary[:300], status=status, outputs=outputs, uncertain=uncertain,
                      next_step=next_step)
