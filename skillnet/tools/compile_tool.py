"""compile_kernel: compile for an NVIDIA arch and return CUDA/PTX/SASS plus the ptxas resource report."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from skillnet_sdk import ToolContext, ToolInput, ToolOutput, ToolResult, tool

from ._common import output_files, run_engine
from ._models import (
    ARCH_DESC,
    KERNEL_FILE_DESC,
    KERNEL_NAME_DESC,
    SOURCE_DESC,
    SOURCE_MAX_CHARS,
    Arch,
    KernelFile,
    kernel_request,
)


class CompileIn(ToolInput):
    kernel_file: KernelFile | None = Field(default=None, description=KERNEL_FILE_DESC)
    source: str | None = Field(default=None, max_length=SOURCE_MAX_CHARS, description=SOURCE_DESC)
    kernel_name: str | None = Field(default=None, max_length=200, description=KERNEL_NAME_DESC)
    arch: Arch = Field(default="kernel", description=ARCH_DESC)
    stages: Literal["cuda", "ptx", "sass"] = Field(
        default="sass",
        description="How far to go: 'cuda' = generated CUDA C++ only; 'ptx' = CUDA + PTX; 'sass' (default) = "
                    "CUDA + PTX + the assembler's register/spill/shared-memory report + the SASS listing.",
    )


class CompileOut(ToolOutput):
    kernel: str
    arch: str
    kernel_arch: str = Field(description="The arch declared in @txl.kernel")
    symbols: list[str]
    registers: int | None = Field(default=None, description="Registers per thread (ptxas)")
    spill_stores: int | None = Field(default=None, description="Bytes spilled to local memory (stores)")
    spill_loads: int | None = Field(default=None, description="Bytes reloaded from spills")
    stack_bytes: int | None = None
    static_smem_bytes: int | None = Field(default=None, description="Static shared memory per CTA (bytes, ptxas)")
    dynamic_smem_bytes: int = Field(default=0, description="Dynamic shared memory per CTA set at launch (bytes)")
    barriers: int | None = None
    ptx_target: str | None = Field(default=None, description="The .target line of the PTX, e.g. sm_90a")
    sass_instructions: int = 0
    cuda_lines: int = 0
    ptx_lines: int = 0
    stages: str
    errors: list[str] = Field(default_factory=list)
    compile_seconds: float | None = None


@tool(
    kind="inspect",
    title="Compile and show generated GPU code",
    long_running=True,
    timeout_s=600,
    examples=[{"arguments": {"kernel_file": "my_kernel.py", "arch": "sm_90a"}, "description": "Registers on H100"}],
)
def compile_kernel(args: CompileIn, ctx: ToolContext) -> ToolResult[CompileOut]:
    """Use when the user asks about registers, spills, shared memory or the generated PTX/SASS, or to confirm a
    kernel compiles for an architecture (A100 sm_80, L4 sm_89, H100 sm_90a, B200 sm_100a). Takes an uploaded file
    (kernel_file) or pasted source text (source). Works on CPU, no GPU needed. Compiles with the TIRx pipeline and returns the assembler's resource report (registers, spill bytes,
    static shared memory) plus the CUDA, PTX and SASS files. Run it after the kernel passes simulate_kernel, when
    tuning."""
    ctx.progress(10, "compiling the kernel")
    req = kernel_request(args.kernel_file, args.source, None, args.kernel_name)
    req.update({"arch": args.arch, "stages": args.stages})
    data = run_engine(ctx, "compile", req, timeout_s=560)
    ctx.mark_untrusted("kernel file / compiler output")
    out = CompileOut(
        kernel=data["kernel"], arch=data["arch"], kernel_arch=data["kernel_arch"], symbols=data.get("symbols") or [],
        registers=data.get("registers"), spill_stores=data.get("spill_stores"), spill_loads=data.get("spill_loads"),
        stack_bytes=data.get("stack_bytes"), static_smem_bytes=data.get("static_smem_bytes"),
        dynamic_smem_bytes=data.get("dynamic_smem_bytes") or 0,
        barriers=data.get("barriers"), ptx_target=data.get("ptx_target"),
        sass_instructions=data.get("sass_instructions") or 0, cuda_lines=data.get("cuda_lines") or 0,
        ptx_lines=data.get("ptx_lines") or 0, stages=args.stages, errors=[e[:300] for e in data.get("errors") or []][:6],
        compile_seconds=data.get("compile_s"),
    )
    outputs = output_files(data, {
        ".cu": "Generated CUDA C++ device code",
        ".ptx": "PTX assembly",
        ".sass": "SASS machine code (cuobjdump)",
        ".ptxas.log": "ptxas resource report (registers, spills, shared memory)",
    })
    names = {Path(o.path).name for o in outputs}
    uncertain = []
    if args.stages == "sass" and out.registers is None:
        uncertain.append("The assembler's register/spill report was not produced" +
                         (f": {out.errors[0]}" if out.errors else "."))
    if args.stages == "sass" and not any(n.endswith(".sass") for n in names):
        uncertain.append("No SASS listing was produced (the disassembler failed or is unavailable)" +
                         (f": {out.errors[-1]}" if out.errors else "."))
    if args.stages in ("ptx", "sass") and not any(n.endswith(".ptx") for n in names):
        uncertain.append("No PTX was produced.")
    if args.stages != "sass":
        uncertain.append("Register, spill and shared-memory numbers come from the assembler and need stages='sass' "
                         "(the default); this call skipped it.")
    if out.arch != out.kernel_arch:
        uncertain.append(f"Compiled for {out.arch} although the kernel declares {out.kernel_arch}; numbers can differ "
                         "from the declared target.")
    status = "partial" if any(u.startswith("No ") or "not produced" in u for u in uncertain) else "ok"
    if out.registers is not None:
        spill = (out.spill_stores or 0) + (out.spill_loads or 0)
        summary = (f"{out.kernel} compiled for {out.arch}: {out.registers} registers/thread, "
                   f"{'no spills' if not spill else f'{out.spill_stores} B spill stores / {out.spill_loads} B loads'}, "
                   f"{(out.static_smem_bytes or 0) + out.dynamic_smem_bytes} B shared memory per CTA")
    else:
        summary = f"{out.kernel} compiled for {out.arch} (stages: {args.stages}); {len(outputs)} file(s) written"
    if status == "partial":
        next_step = ("Report the numbers that were produced and say which listing is missing; try stages='ptx' or "
                     "arch='kernel' if the target arch is unusual.")
    else:
        next_step = ("Report registers, spills and shared memory with the ptxas log as evidence. Spills or very high "
                     "register counts are tuning targets; verify any change with simulate_kernel and benchmark_kernel.")
    return ToolResult(data=out, summary=summary[:300], status=status, outputs=outputs, uncertain=uncertain,
                      next_step=next_step)
