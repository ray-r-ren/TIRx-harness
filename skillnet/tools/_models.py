"""Shared input fields and result models for the kernel tools."""

from __future__ import annotations

import ast
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from skillnet_sdk import FileConstraints, InputFile, ToolError

KernelFile = Annotated[InputFile, FileConstraints(extensions=[".py"])]

KERNEL_FILE_DESC = (
    "The TIRx-lite kernel .py file (@txl.kernel functions, usually with make_inputs() and reference()). "
    "An uploaded file_id or an https URL. Pass this or `source`, not both."
)
SOURCE_MAX_CHARS = 200_000
SOURCE_DESC = (
    "The kernel's Python source code as text (the same content as a kernel .py file), e.g. a kernel you just "
    "wrote or a starter's `source` from find_kernel_examples. Use instead of kernel_file; line numbers in "
    "results refer to this text. Max 200,000 characters; upload longer code and pass kernel_file."
)
CASE_FILE_DESC = (
    "Optional test-case .py with make_inputs() (dict of NumPy inputs by parameter name, zeroed outputs) and "
    "reference(inputs) (expected outputs). Omit when the kernel file has them."
)
KERNEL_NAME_DESC = "Which @txl.kernel function to use when the file defines several (its Python function name)."

Arch = Literal["kernel", "sm_80", "sm_89", "sm_90a", "sm_100a"]
ARCH_DESC = (
    "Target GPU architecture: 'kernel' (default) = the arch declared in @txl.kernel; sm_80 = A100, sm_89 = L4/"
    "RTX 40, sm_90a = H100/H200 (Hopper), sm_100a = B200/GB200 (Blackwell)."
)


class Param(BaseModel):
    name: str
    kind: str = Field(description="pointer (global buffer), scalar, tensormap or other")
    dtype: str
    shape: list[str] | None = Field(default=None, description="Symbolic buffer shape (pointers only)")


class Mismatch(BaseModel):
    index: list[int]
    flat_index: int
    got: float | int | str | bool | None
    expected: float | int | str | bool | None


class OutputCheck(BaseModel):
    name: str
    shape: list[int] = Field(default_factory=list)
    dtype: str = ""
    max_abs_err: float | str | None = None
    max_rel_err: float | str | None = Field(default=None, description="Over elements whose expected value is nonzero")
    mismatches: int = 0
    first_mismatches: list[Mismatch] = Field(default_factory=list)
    mismatch_ranges: list[list[int]] = Field(default_factory=list,
                                             description="Runs of consecutive mismatching flat indices [first, last]")
    mismatch_range_count: int = 0
    missing: bool = Field(default=False, description="The simulator returned no value for this output")
    shape_mismatch: str | None = None


def output_check(d: dict[str, Any]) -> OutputCheck:
    keep = {k: v for k, v in d.items() if k in OutputCheck.model_fields}
    if "first_mismatches" in keep:
        keep["first_mismatches"] = [Mismatch(**m) for m in keep["first_mismatches"]]
    return OutputCheck(**keep)


def display_name(path: Path | str | None, fallback: str) -> str:
    if not path:
        return fallback
    name = Path(str(path)).name
    return name if name.endswith(".py") else fallback


def file_request(kernel_file: Path, case_file: Path | None = None, kernel_name: str | None = None) -> dict[str, Any]:
    req: dict[str, Any] = {
        "kernel_file": str(kernel_file),
        "kernel_display": display_name(kernel_file, "kernel.py"),
        "kernel_name": (kernel_name or "").strip() or None,
    }
    if case_file:
        req["case_file"] = str(case_file)
        req["case_display"] = display_name(case_file, "case.py")
    return req


# ----------------------------------------------------------------------------------------------
# Kernel input: an uploaded file or source text (exactly one)
# ----------------------------------------------------------------------------------------------


def _stem(name: str | None, fallback: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", (name or "").strip())[:60] or fallback
    if stem[0].isdigit():
        stem = "_" + stem
    return stem


def kernel_input(
    kernel_file: Path | None,
    source: str | None,
    kernel_name: str | None,
    *,
    role: str = "kernel",
    file_arg: str = "kernel_file",
    source_arg: str = "source",
    required: bool = True,
    tmpdirs: list[str] | None = None,
) -> dict[str, Any] | None:
    """Resolve a kernel given as a file or as source text into a local .py path.

    Source text is written unchanged (no header lines, so line numbers match the text) to a private
    temporary directory as <kernel_name or 'kernel'>.py. It is untrusted code: it is only ever loaded
    by the engine worker in its time- and memory-limited subprocess, never imported here.
    Returns {"path", "display", "input_kind", "sha256"} or None when optional and absent.
    """
    has_source = source is not None and source.strip() != ""
    has_file = kernel_file is not None
    if has_file and has_source:
        raise ToolError(
            "ambiguous_kernel_input",
            f"Both {file_arg} and {source_arg} were given for the {role}.",
            f"Pass exactly one: {source_arg}=<the code as text> for code you have in hand, or {file_arg}=<file_id or "
            f"https URL> for an uploaded file. Drop the other and call again.",
        )
    if not has_file and not has_source:
        if not required:
            return None
        raise ToolError(
            "missing_kernel",
            f"No {role} was given: neither {file_arg} nor {source_arg} is set.",
            f"Call again with {source_arg}=<the kernel's Python code as text>, or {file_arg}=<an uploaded .py "
            "file_id or https URL>. To start from a verified example, call find_kernel_examples and pass a "
            f"starter's `source` as {source_arg}.",
        )
    if has_file:
        path = Path(kernel_file)
        data = path.read_bytes()
        return {"path": path, "display": display_name(path, f"{role}.py"), "input_kind": "file",
                "sha256": hashlib.sha256(data).hexdigest()}
    if len(source) > SOURCE_MAX_CHARS:
        raise ToolError("source_too_long", f"The {role} source is {len(source)} characters; the limit is "
                        f"{SOURCE_MAX_CHARS}.", f"Upload it as a file and pass {file_arg} instead.")
    d = tempfile.mkdtemp(prefix=f"gkf-src-{role}-")
    os.chmod(d, 0o700)
    if tmpdirs is not None:
        tmpdirs.append(d)
    path = Path(d) / f"{_stem(kernel_name, 'kernel')}.py"
    path.write_text(source, encoding="utf-8")
    return {"path": path, "display": path.name, "input_kind": "source",
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()}


def kernel_request(
    kernel_file: Path | None,
    source: str | None,
    case_file: Path | None = None,
    kernel_name: str | None = None,
) -> dict[str, Any]:
    """An engine request for a kernel given as a file or as source text (exactly one)."""
    tmpdirs: list[str] = []
    k = kernel_input(kernel_file, source, kernel_name, tmpdirs=tmpdirs)
    req = file_request(k["path"], case_file, kernel_name)
    req["kernel_display"] = k["display"]
    req["kernel_is_source"] = k["input_kind"] == "source"
    req["_input_kind"] = k["input_kind"]
    req["_sha256"] = k["sha256"]
    req["_tmpdirs"] = tmpdirs
    return req


def relative_imports(text: str) -> list[tuple[int, str]]:
    """(line, source line) of every relative import (from . / from .. ) in Python text; [] if it doesn't parse."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            ln = node.lineno
            out.append((ln, lines[ln - 1].strip() if 0 < ln <= len(lines) else ""))
    return sorted(out)
