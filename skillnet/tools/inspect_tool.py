"""inspect_kernel: trace a TIRx-lite file and report its kernels and launch configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from skillnet_sdk import ToolContext, ToolInput, ToolOutput, ToolResult, tool

from ._common import output_files, run_engine
from ._models import KERNEL_FILE_DESC, KERNEL_NAME_DESC, SOURCE_DESC, SOURCE_MAX_CHARS, KernelFile, Param, kernel_request


class InspectIn(ToolInput):
    kernel_file: KernelFile | None = Field(default=None, description=KERNEL_FILE_DESC)
    source: str | None = Field(default=None, max_length=SOURCE_MAX_CHARS, description=SOURCE_DESC)
    kernel_name: str | None = Field(default=None, max_length=200, description=KERNEL_NAME_DESC + " Leave empty to "
                                    "report every kernel in the file.")


class KernelInfo(BaseModel):
    name: str
    params: list[Param]
    grid: list[int | str] = Field(description="CTAs per grid dimension; a name means it depends on a parameter")
    grid_static: bool = Field(description="True when the grid is a constant (the CPU simulator needs this)")
    warps: int
    threads_per_cta: int
    arch: str = Field(description="Target arch from @txl.kernel, e.g. sm_90a (H100) or sm_100a (Blackwell)")
    blackwell_only: bool = Field(description="Targets sm_100+ or uses tcgen05/TMEM: cannot run on the H100")
    uses_wgmma: bool
    uses_tcgen05: bool
    dynamic_smem_bytes: int = Field(default=0, description="Dynamic shared memory per CTA requested at launch")
    tirx_lines: int


class CaseCheck(BaseModel):
    ok: bool = Field(description="make_inputs() ran and matches the kernel's parameters and dtypes")
    problem: str | None = None
    input_shapes: dict[str, list[int]] = Field(default_factory=dict)
    extra_keys: list[str] = Field(default_factory=list, description="make_inputs() keys that are not parameters")


class InspectOut(ToolOutput):
    file: str
    input_kind: Literal["file", "source"] = Field(description="Whether the kernel came as a file or as source text")
    kernels: list[KernelInfo]
    all_kernel_names: list[str]
    has_make_inputs: bool
    has_reference: bool
    has_case_functions: bool = Field(description="Both make_inputs() and reference() are defined in this file")
    case_check: CaseCheck | None = Field(default=None, description="Dry run of make_inputs() (single kernel only)")


@tool(
    kind="inspect",
    title="Inspect a TIRx-lite kernel",
    timeout_s=180,
    examples=[{"arguments": {"kernel_file": "my_kernel.py"}, "description": "Check that a new kernel file loads"},
              {"arguments": {"source": "import numpy as np\nimport tirx_kernels.tirx_lite as txl\n..."},
               "description": "Check kernel code you just wrote, passed as text"}],
)
def inspect_kernel(args: InspectIn, ctx: ToolContext) -> ToolResult[InspectOut]:
    """Use on any kernel before checking or benchmarking it, or when a kernel 'won't load'. After
    find_kernel_examples. Takes the kernel as an uploaded file (kernel_file) or as pasted source text (source), not
    both. Traces it in a time- and memory-limited process and reports each @txl.kernel: parameters with dtypes,
    grid, warps, target arch, whether it is Blackwell-only, and whether make_inputs()/reference() exist and match.
    Returns the traced TIRx text as a file. Fails with kernel_trace_failed (failing line) or relative_import (the
    kernel must be one standalone file)."""
    ctx.progress(10, "tracing the kernel")
    req = kernel_request(args.kernel_file, args.source, None, args.kernel_name)
    data = run_engine(ctx, "inspect", req, timeout_s=150)
    ctx.mark_untrusted("kernel file / source text")
    kernels = [KernelInfo(**{k: v for k, v in kf.items() if k in KernelInfo.model_fields}) for kf in data["kernels"]]
    cc = data.get("case_check")
    case_check = CaseCheck(**cc) if cc else None
    out = InspectOut(
        file=data["file"],
        input_kind=req["_input_kind"],
        kernels=kernels,
        all_kernel_names=data["all_kernel_names"],
        has_make_inputs=data["has_make_inputs"],
        has_reference=data["has_reference"],
        has_case_functions=bool(data["has_make_inputs"] and data["has_reference"]),
        case_check=case_check,
    )
    outputs = output_files(data, {".tirx.txt": "Traced TIRx function text"})
    desc = "; ".join(
        f"{k.name}: {len(k.params)} params, grid {k.grid}, {k.warps} warp(s), {k.arch}"
        + (" (Blackwell-only)" if k.blackwell_only else "")
        for k in kernels[:4]
    )
    summary = f"{data['file']} traced: {len(kernels)} kernel(s). {desc}"
    uncertain = []
    steps = []
    if len(data["all_kernel_names"]) > 1 and not args.kernel_name:
        steps.append(f"The file defines several kernels ({', '.join(data['all_kernel_names'][:8])}); pass kernel_name "
                     "to the other tools.")
    if any(not k.grid_static for k in kernels):
        uncertain.append("A kernel's grid depends on a parameter; simulate_kernel and check_kernel_safety need a "
                         "constant grid.")
        steps.append("Use a constant grid (e.g. grid=4) before simulating.")
    if any(k.blackwell_only for k in kernels):
        uncertain.append("Blackwell-only kernels (sm_100a / tcgen05) can be simulated, checked and compiled here but "
                         "not timed on the H100.")
    if not out.has_case_functions:
        missing = [n for n, ok in (("make_inputs()", out.has_make_inputs), ("reference(inputs)", out.has_reference))
                   if not ok]
        steps.append(f"The file has no {' or '.join(missing)}: add {'it' if len(missing) == 1 else 'them'}, or pass a "
                     "case_file to simulate_kernel and check_kernel_safety.")
    elif case_check is not None and not case_check.ok:
        steps.append(f"Fix make_inputs(): {case_check.problem}")
    if not steps:
        how = "source" if req["_input_kind"] == "source" else "file"
        steps.append(f"Run check_kernel_safety, then simulate_kernel on this {how}.")
    return ToolResult(data=out, summary=summary[:300], outputs=outputs, uncertain=uncertain, next_step=" ".join(steps))
