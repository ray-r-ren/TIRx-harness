"""export_kernel: standalone CUDA C++ with its host launch wrapper (and optionally a shared library)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from skillnet_sdk import OutputFile, ToolContext, ToolInput, ToolOutput, ToolResult, tool

from ._common import ffi_cuda_home, output_files, run_engine
from ._models import (
    ARCH_DESC,
    KERNEL_FILE_DESC,
    KERNEL_NAME_DESC,
    SOURCE_DESC,
    SOURCE_MAX_CHARS,
    Arch,
    KernelFile,
    kernel_request,
    relative_imports,
)


class ExportIn(ToolInput):
    kernel_file: KernelFile | None = Field(default=None, description=KERNEL_FILE_DESC)
    source: str | None = Field(default=None, max_length=SOURCE_MAX_CHARS, description=SOURCE_DESC)
    kernel_name: str | None = Field(default=None, max_length=200, description=KERNEL_NAME_DESC)
    arch: Arch = Field(default="kernel", description=ARCH_DESC)
    build_library: bool = Field(default=False, description="Also build a loadable shared library (.so, tvm-ffi) "
                                                           "for the chosen arch with nvcc.")


class ExportArg(BaseModel):
    name: str
    kind: str
    dtype: str


class ExportOut(ToolOutput):
    kernel: str
    entry_symbol: str = Field(description="Name of the host entry function in the .cu / .so")
    device_kernels: list[str]
    arch: str
    ffi_arch: str = Field(description="Value for TVM_FFI_CUDA_ARCH_LIST when rebuilding")
    args: list[ExportArg] = Field(description="Arguments in call order")
    grid: list[int | str]
    threads_per_cta: int
    cu_lines: int
    built: bool
    build_error: str | None = None
    kernel_py: str | None = Field(default=None, description="Name of the kernel .py returned with the export (the "
                                                           "source as given; one standalone file)")


@tool(
    kind="export",
    title="Export kernel as standalone CUDA",
    long_running=True,
    timeout_s=600,
    examples=[{"arguments": {"kernel_file": "my_kernel.py", "build_library": True},
               "description": "Ship a verified kernel as .cu and .so"}],
)
def export_kernel(args: ExportIn, ctx: ToolContext) -> ToolResult[ExportOut]:
    """Use when the user wants to ship or use the kernel without TVM or TIRx ('give me a .cu I can build'). Final
    step, after simulate_kernel and check_kernel_safety. Takes an uploaded file (kernel_file) or pasted source text
    (source). Writes CUDA C++ holding the device kernel and its host launch wrapper, USAGE.md with the argument
    contract and build/call snippets, and the kernel .py itself (one standalone file, no relative imports);
    optionally builds a tvm-ffi shared library (.so) for the chosen arch with nvcc (no GPU needed)."""
    ctx.progress(10, "generating CUDA source")
    req = kernel_request(args.kernel_file, args.source, None, args.kernel_name)
    req.update({"arch": args.arch, "build_library": args.build_library})
    kernel_text = Path(req["kernel_file"]).read_text(encoding="utf-8", errors="replace")
    extra_env = {}
    if args.build_library:
        home = ffi_cuda_home()
        if home is not None:
            extra_env = {"CUDA_HOME": str(home), "CUDA_PATH": str(home)}
    data = run_engine(ctx, "export", req, timeout_s=560, extra_env=extra_env or None)
    ctx.mark_untrusted("kernel file / source text / generated code")
    cu = next((Path(f) for f in data.get("files") or [] if str(f).endswith(".cu")), None)
    kernel_py = None
    if not relative_imports(kernel_text):  # the engine refuses relative imports; never ship one
        py = ctx.output_path(f"{cu.stem if cu else data['kernel']}.py")
        py.write_text(kernel_text, encoding="utf-8")
        kernel_py = py.name
    out = ExportOut(
        kernel=data["kernel"], entry_symbol=data["entry_symbol"], device_kernels=data.get("device_kernels") or [],
        arch=data["arch"], ffi_arch=data["ffi_arch"],
        args=[ExportArg(**a) for a in data.get("args") or []], grid=data.get("grid") or [],
        threads_per_cta=data.get("threads_per_cta") or 0, cu_lines=data.get("cu_lines") or 0,
        built=bool(data.get("built")), build_error=(data.get("build_error") or None) and data["build_error"][:600],
        kernel_py=kernel_py,
    )
    outputs = output_files(data, {
        ".cu": "Standalone CUDA C++: device kernel + host launch wrapper",
        "USAGE.md": "Argument contract, build and call instructions",
        ".so": f"Shared library built for {out.arch} (load with tvm_ffi.load_module)",
    })
    if kernel_py:
        outputs.append(OutputFile(ctx.output_path(kernel_py), "The kernel's TIRx-lite source (.py, one standalone "
                                                              "file) that this export was generated from"))
    uncertain = ["The exported code was generated, not run here: validate the built library against the same inputs "
                 "and reference on the target GPU before shipping."]
    status = "ok"
    summary = (f"Exported {out.kernel} for {out.arch}: entry {out.entry_symbol}({', '.join(a.name for a in out.args)}),"
               f" {out.cu_lines} lines of CUDA")
    if args.build_library:
        if out.built:
            summary += "; shared library built"
        else:
            status = "partial"
            summary += "; shared library build failed"
            uncertain.append(f"The .so was not built: {(out.build_error or 'unknown error')[:400]}")
    next_step = ("Give the user the .cu, USAGE.md and the kernel .py (and the .so if built); state the argument order "
                 "and dtypes from args.")
    if status == "partial":
        next_step = ("Hand over the .cu and USAGE.md; say the library build failed and quote build_error. The user can "
                     "build it with the snippet in USAGE.md on a machine with a CUDA toolkit for this arch.")
    return ToolResult(data=out, summary=summary[:300], status=status, outputs=outputs, uncertain=uncertain,
                      next_step=next_step)
