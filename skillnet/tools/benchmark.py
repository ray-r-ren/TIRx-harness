"""benchmark_kernel: correctness on the device plus timing, optionally against a baseline version."""

from __future__ import annotations

from pydantic import Field
from skillnet_sdk import ToolContext, ToolInput, ToolOutput, ToolResult, tool

from ._common import output_files, run_engine
from ._models import (
    CASE_FILE_DESC,
    KERNEL_FILE_DESC,
    KERNEL_NAME_DESC,
    SOURCE_DESC,
    SOURCE_MAX_CHARS,
    KernelFile,
    OutputCheck,
    kernel_input,
    kernel_request,
    output_check,
)


class BenchmarkIn(ToolInput):
    kernel_file: KernelFile | None = Field(default=None, description=KERNEL_FILE_DESC)
    source: str | None = Field(default=None, max_length=SOURCE_MAX_CHARS, description=SOURCE_DESC)
    case_file: KernelFile | None = Field(default=None, description="Optional test-case .py with make_inputs()/reference() at real sizes (omit when the "
                                         "kernel file has them); constants BYTES_MOVED / FLOPS add GB/s / TFLOP/s.")
    baseline_file: KernelFile | None = Field(
        default=None,
        description="Optional earlier, verified version of the kernel to time under identical conditions for a "
                    "speedup (same kernel_name, same parameters). Pass this or baseline_source, not both.",
    )
    baseline_source: str | None = Field(
        default=None, max_length=SOURCE_MAX_CHARS,
        description="The baseline kernel's Python source as text, instead of baseline_file (e.g. the version before "
                    "your change).",
    )
    kernel_name: str | None = Field(default=None, max_length=200, description=KERNEL_NAME_DESC)
    repeats: int = Field(default=100, ge=10, le=1000, description="Timed launches per run (10-1000).")
    runs: int = Field(default=3, ge=1, le=5, description="Independent timing runs (1-5); candidate and baseline are "
                                                         "interleaved run by run.")
    rtol: float = Field(default=1e-5, ge=0, le=1, description="Relative tolerance for the on-device correctness check.")
    atol: float = Field(default=1e-8, ge=0, le=1e6, description="Absolute tolerance for the on-device check.")


class BenchmarkOut(ToolOutput):
    kernel: str
    device: str
    device_arch: str
    arch: str = Field(description="The arch the kernel was compiled for on the device")
    kernel_arch: str | None = Field(default=None, description="The arch declared in @txl.kernel")
    correct: bool = Field(description="Device outputs match reference() within rtol/atol")
    max_abs_err: float | str | None = None
    outputs: list[OutputCheck] = Field(default_factory=list)
    median_us: float = Field(description="Median latency over all timed launches (microseconds)")
    min_us: float
    per_run_median_us: list[float]
    repeats: int
    runs: int
    baseline_kernel: str | None = None
    baseline_correct: bool | None = None
    baseline_median_us: float | None = None
    baseline_per_run_median_us: list[float] | None = None
    speedup: float | None = Field(default=None, description="baseline_median_us / median_us (>1 = faster)")
    faster_in_every_run: bool | None = None
    bandwidth_gbs: float | None = None
    tflops: float | None = None


@tool(
    kind="measure",
    title="Benchmark a kernel on the H100 GPU",
    long_running=True,
    timeout_s=1200,
    examples=[{"arguments": {"kernel_file": "rmsnorm_v2.py", "baseline_file": "rmsnorm_v1.py"},
               "description": "Did my change make it faster?"}],
)
def benchmark_kernel(args: BenchmarkIn, ctx: ToolContext) -> ToolResult[BenchmarkOut]:
    """Use for 'how fast is it?' or 'did my change make it faster?'. Last stage, after simulate_kernel passes and
    check_kernel_safety is clean. Runs on the skill's NVIDIA H100 (sm_90a). Takes an uploaded file (kernel_file) or
    pasted source text (source), and the same for an optional baseline. First refuses with arch_unsupported, naming
    the kernel's arch and the H100's, any kernel that targets another GPU (sm_100a / tcgen05 Blackwell). Then checks
    the device outputs against reference() and times interleaved runs (median/min microseconds); with a baseline it
    reports the speedup."""
    ctx.progress(5, "checking the kernel's target architecture")
    req = kernel_request(args.kernel_file, args.source, args.case_file, args.kernel_name)
    req.update({"repeats": args.repeats, "runs": args.runs, "rtol": args.rtol, "atol": args.atol})
    base = kernel_input(args.baseline_file, args.baseline_source, args.kernel_name, role="baseline",
                        file_arg="baseline_file", source_arg="baseline_source", required=False,
                        tmpdirs=req["_tmpdirs"])
    if base is not None:
        req["baseline_file"] = str(base["path"])
        req["baseline_display"] = base["display"]
        req["baseline_is_source"] = base["input_kind"] == "source"
    ctx.progress(15, "compiling, checking and timing on the device")
    data = run_engine(ctx, "benchmark", req, timeout_s=1150, memory_limit=False)
    ctx.mark_untrusted("kernel file / device measurements")
    fields = {k: v for k, v in data.items() if k in BenchmarkOut.model_fields and k != "outputs"}
    out = BenchmarkOut(**fields, outputs=[output_check(o) for o in (data.get("outputs") or [])[:8]])
    outputs = output_files(data, {"benchmark_report.json": "All timing samples, per-run medians and the device check"})
    uncertain = [f"Timings are for the shapes in make_inputs() on {out.device}; other shapes and GPUs differ."]
    if out.kernel_arch and out.kernel_arch != out.arch:
        uncertain.append(f"The kernel declares {out.kernel_arch}; it was compiled and timed for {out.arch} on this "
                         "GPU, so code generated for its declared target may differ.")
    status = "ok"
    summary = (f"{out.kernel} on {out.device}: median {out.median_us:.2f} us (min {out.min_us:.2f}) over "
               f"{out.runs}x{out.repeats} launches; device check {'correct' if out.correct else 'WRONG'}")
    if out.speedup is not None:
        summary += f"; speedup {out.speedup:.3f}x vs baseline ({out.baseline_median_us:.2f} us)"
    if not out.correct:
        status = "partial"
        uncertain.append("The kernel's device outputs do not match reference(): its timing is not meaningful.")
        next_step = ("Do not report the speed as a result. Re-run simulate_kernel and check_kernel_safety to find the "
                     "difference (a race that NumSim did not show, or a precision issue: consider rtol/atol for fp16).")
    elif out.speedup is not None:
        if out.baseline_correct is False:
            uncertain.append("The baseline's device outputs do not match reference().")
        if out.speedup > 1 and out.faster_in_every_run:
            next_step = "Keep the change: it is correct and faster in every run. Report speedup with per-run medians."
        elif out.speedup > 1:
            next_step = ("The median is faster but not in every run: treat the gain as noise unless a rerun with more "
                         "runs/repeats confirms it.")
        else:
            next_step = "The change is not faster: keep the baseline version, or try another optimization."
    else:
        next_step = ("Report median_us with per_run_median_us as evidence; pass the previous version as baseline_file "
                     "or baseline_source to measure a speedup.")
    if out.runs > 1 and out.per_run_median_us:
        spread = (max(out.per_run_median_us) - min(out.per_run_median_us)) / max(out.median_us, 1e-9)
        if spread > 0.1:
            uncertain.append(f"Run-to-run spread is {spread:.0%} of the median; timings are noisy.")
    return ToolResult(data=out, summary=summary[:300], status=status, outputs=outputs, uncertain=uncertain,
                      next_step=next_step)
