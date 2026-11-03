"""check_kernel_safety: Synccheck and Racecheck for one concrete launch, on CPU."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from skillnet_sdk import ToolContext, ToolInput, ToolOutput, ToolResult, tool

from ._common import output_files, run_engine
from ._models import (
    CASE_FILE_DESC,
    KERNEL_FILE_DESC,
    KERNEL_NAME_DESC,
    SOURCE_DESC,
    SOURCE_MAX_CHARS,
    KernelFile,
    kernel_request,
)


class SafetyIn(ToolInput):
    kernel_file: KernelFile | None = Field(default=None, description=KERNEL_FILE_DESC)
    source: str | None = Field(default=None, max_length=SOURCE_MAX_CHARS, description=SOURCE_DESC)
    case_file: KernelFile | None = Field(default=None, description=CASE_FILE_DESC)
    kernel_name: str | None = Field(default=None, max_length=200, description=KERNEL_NAME_DESC)
    checks: Literal["both", "sync", "race"] = Field(
        default="both",
        description="Which checks: 'sync' = deadlocks and barrier misuse (Synccheck, 'will it hang?'), 'race' = "
                    "racing memory accesses (Racecheck), 'both' (default).",
    )


class Finding(BaseModel):
    id: str
    status: str = Field(description="error = definite problem; review = needs a human look")
    kind: str
    message: str
    lines: list[int] = Field(default_factory=list, description="Kernel line numbers involved (of the file, or of "
                                                                "the source text when source was given)")
    code: dict[str, str] = Field(default_factory=dict, description="Source text of those lines")


class CheckerResult(BaseModel):
    verdict: str = Field(description="clean, review, incomplete or error")
    finding_count: int
    counts: dict[str, int] = Field(default_factory=dict, description="Findings per status:kind")
    findings: list[Finding] = Field(default_factory=list, description="Up to 6 distinct findings (all in the report)")
    reason: str | None = Field(default=None, description="Why the checker could not finish")
    seconds: float | None = None


class SafetyOut(ToolOutput):
    kernel: str
    overall: Literal["clean", "review", "incomplete", "error"]
    synccheck: CheckerResult | None = None
    racecheck: CheckerResult | None = None


def _checker(d: dict | None) -> CheckerResult | None:
    if not d:
        return None
    findings = []
    for f in (d.get("findings") or [])[:6]:
        findings.append(Finding(id=f.get("id", ""), status=f.get("status", ""), kind=f.get("kind", ""),
                                message=str(f.get("message", ""))[:320], lines=f.get("lines", [])[:6],
                                code={str(k): str(v)[:160] for k, v in (f.get("code") or {}).items()}))
    verdict = d.get("verdict", "incomplete")
    reason = (d.get("reason") or None) and str(d["reason"])[:500]
    if not reason and verdict == "incomplete":
        why = [f.message for f in findings if f.status == "incomplete"]
        reason = why[0][:500] if why else None
    return CheckerResult(verdict=verdict, finding_count=int(d.get("finding_count", 0)),
                         counts=d.get("counts") or {}, findings=findings, reason=reason, seconds=d.get("seconds"))


@tool(
    kind="check",
    title="Check for deadlocks and memory races",
    long_running=True,
    timeout_s=900,
    state_version=1,
    examples=[{"arguments": {"kernel_file": "my_kernel.py"}, "description": "Check a kernel before running it"}],
)
def check_kernel_safety(args: SafetyIn, ctx: ToolContext) -> ToolResult[SafetyOut]:
    """Use before running a kernel on a GPU and after any change to barriers, pipelines or warp roles ('will this
    hang?', 'is there a race?'). Takes an uploaded file (kernel_file) or pasted source text (source). Runs
    Synccheck (deadlocks, barrier misuse) and Racecheck (racing shared/global memory accesses) on CPU for the launch
    from make_inputs(). Returns each checker's verdict (clean, review, incomplete, error) and findings with kind,
    message and kernel lines, plus full reports as files. A clean result covers only these inputs and this launch.
    Returns a state_id to reuse compiled work."""
    ctx.progress(5, "loading the kernel and test case")
    req = kernel_request(args.kernel_file, args.source, args.case_file, args.kernel_name)
    req["checks"] = args.checks
    ctx.progress(15, "running the synchronization and race checkers")
    data = run_engine(ctx, "safety", req, timeout_s=860, numsim_dir=ctx.state_dir / "numsim")
    ctx.mark_untrusted("kernel file / checker output")
    out = SafetyOut(kernel=data["kernel"], overall=data["overall"], synccheck=_checker(data.get("synccheck")),
                    racecheck=_checker(data.get("racecheck")))
    outputs = output_files(data, {
        "safety_report.json": "Full Synccheck/Racecheck reports (JSON)",
        "safety_report.txt": "Readable checker reports",
    })
    ran = [n for n, r in (("Synccheck", out.synccheck), ("Racecheck", out.racecheck)) if r is not None]
    parts = []
    for label, r in (("Synccheck", out.synccheck), ("Racecheck", out.racecheck)):
        if r is None:
            continue
        top = ", ".join(sorted({f.kind for f in r.findings}))[:80]
        parts.append(f"{label} {r.verdict}" + (f" ({r.finding_count} finding(s): {top})" if r.finding_count else ""))
    summary = f"Safety {out.overall.upper()} for {out.kernel}: " + "; ".join(parts)
    uncertain = ["Checks cover only this launch and the inputs from make_inputs(); other shapes are not proven."]
    if args.checks != "both":
        uncertain.append(f"Only {ran[0] if ran else args.checks} ran.")
    status = "ok"
    if out.overall == "clean":
        next_step = "Run simulate_kernel next (or benchmark_kernel if it already passes)."
    elif out.overall == "error":
        next_step = ("Fix the error findings first (use their lines and code): a missing barrier between a write and "
                     "a read by another thread is the usual race; mismatched arrive/wait counts cause hangs. Then run "
                     "check_kernel_safety again; do not trust simulate_kernel numbers until it is clean.")
    elif out.overall == "review":
        next_step = ("Review each finding against the kernel lines shown; explain to the user why each is benign or "
                     "fix it, then run check_kernel_safety again.")
    else:
        status = "partial"
        for label, r in (("Synccheck", out.synccheck), ("Racecheck", out.racecheck)):
            if r is not None and r.verdict == "incomplete":
                why = r.reason or next((f.message for f in r.findings if f.status == "incomplete"), None)
                uncertain.append(f"{label} could not fully check this kernel: "
                                 f"{(why or 'unsupported operation or missing binding')[:300]}")
        next_step = ("The checkers could not model part of the kernel (unsupported operation or missing binding); see "
                     "the reasons, rewrite with supported operations (find_kernel_examples scope='reference'), or tell "
                     "the user these hazards were not checked.")
    return ToolResult(data=out, summary=summary[:300], status=status, outputs=outputs, uncertain=uncertain,
                      next_step=next_step)
