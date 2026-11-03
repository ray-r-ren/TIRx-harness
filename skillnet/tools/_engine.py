"""Engine worker for GPU Kernel Forge.

Runs one action on a user's TIRx-lite kernel file in its own process, so that user code
(executed while tracing the kernel), the CPU simulator and the compiler are isolated from
the tool server and bounded by its timeout. Usage:

    python _engine.py request.json response.json

The request names an ``action`` (inspect, simulate, safety, compile, export, benchmark) and
its arguments; the response is ``{"ok": true, "data": {...}}`` or
``{"ok": false, "error": {"code", "message", "next_step", "details"}}``. The process exits 0
whenever it wrote a response.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

MAX_TEXT = 600


class Fail(Exception):
    """An expected failure, reported to the tool with a code and a next step."""

    def __init__(self, code: str, message: str, next_step: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.next_step = next_step
        self.details = details or {}


def clip(text, limit: int = MAX_TEXT) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def num(value):
    """A JSON-safe number: NaN and infinities become strings."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    if isinstance(value, bool):
        return bool(value)
    try:
        import numpy as np

        if isinstance(value, (int, np.integer)):
            return int(value)
    except ImportError:  # pragma: no cover
        pass
    return f


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, float)):
        return num(obj)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return num(obj.item())
        if isinstance(obj, np.ndarray):
            return jsonable(obj.tolist())
    except ImportError:  # pragma: no cover
        pass
    return str(obj)


# ----------------------------------------------------------------------------------------------
# Loading user files
# ----------------------------------------------------------------------------------------------


class Loaded:
    def __init__(self, module, path: Path, display: str, lines: list[str]):
        self.module = module
        self.path = path
        self.display = display
        self.lines = lines

    def line_text(self, lineno) -> str:
        try:
            i = int(lineno)
        except (TypeError, ValueError):
            return ""
        if 1 <= i <= len(self.lines):
            return self.lines[i - 1].strip()
        return ""


def _safe_stem(name: str, fallback: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", Path(name).stem) or fallback
    if stem[0].isdigit():
        stem = "_" + stem
    return stem


def _shadows_module(stem: str) -> bool:
    """A user file named like an importable module (numpy.py, math.py) would shadow it."""
    if stem in sys.stdlib_module_names or stem in sys.modules:
        return True
    try:
        return importlib.util.find_spec(stem) is not None
    except (ImportError, ValueError):
        return True


def check_relative_imports(text: str, display: str, what: str) -> None:
    """Refuse relative imports: a kernel must be one standalone file."""
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return  # reported with its line when the file is loaded
    lines = text.splitlines()
    rel = sorted((n.lineno, n) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level > 0)
    if rel:
        line = rel[0][0]
        src_line = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        raise Fail(
            "relative_import",
            clip(f"The {what} {display} uses a relative import at line {line}: `{src_line}`. Kernels must be one "
                 "standalone .py file", 600) + (f" ({len(rel)} relative imports in all)." if len(rel) > 1 else "."),
            "Inline the imported helpers into this one file, or import them absolutely from the installed "
            "tirx_kernels package (e.g. `import tirx_kernels.tirx_lite as txl`); then call again.",
            {"file": display, "line": line, "source_line": src_line, "lines": [r[0] for r in rel][:20]},
        )


def load_user_file(src: str, display: str, workdir: Path, role: str, is_source: bool = False) -> Loaded:
    """Import a user's .py file from a private copy, mapping failures to kernel_trace_failed."""
    stem = _safe_stem(display, role)
    if _shadows_module(stem):
        stem = f"{stem}_{role}"
    dest = workdir / f"{stem}.py"
    if dest.exists():
        dest = workdir / f"{stem}_{role}.py"
    shutil.copyfile(src, dest)
    text = dest.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    noun = "source text" if is_source else "file"
    what = f"{role} {noun}" if role in ("kernel", "baseline") else "case file"
    check_relative_imports(text, display, what)
    if str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))
    modname = dest.stem
    spec = importlib.util.spec_from_file_location(modname, dest)
    if spec is None or spec.loader is None:
        raise Fail("invalid_file", f"{display} is not a loadable Python file.",
                   "Pass a TIRx-lite kernel written as a .py file.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except SyntaxError as e:
        line = e.lineno or 0
        raise Fail(
            "kernel_trace_failed",
            f"Syntax error in the {what} {display} at line {line}: {e.msg}",
            "Fix the Python syntax at the reported line, then call the tool again.",
            {"file": display, "line": line, "source_line": (e.text or "").strip(), "error": clip(e.msg)},
        ) from None
    except Exception as e:  # noqa: BLE001 - user code can raise anything
        frames = traceback.extract_tb(e.__traceback__)
        mine = [f for f in frames if Path(f.filename).resolve() == dest.resolve()]
        frame = mine[-1] if mine else None
        line = frame.lineno if frame else 0
        err = f"{type(e).__name__}: {e}"
        hint = _trace_hint(err)
        raise Fail(
            "kernel_trace_failed",
            clip(f"Loading the {what} {display} failed" + (f" at line {line}" if line else "") + f": {err}", 900),
            hint + " Fix the reported line, then call the tool again.",
            {
                "file": display,
                "line": line,
                "source_line": (lines[line - 1].strip() if 0 < line <= len(lines) else ""),
                "error": clip(err, 1500),
                "user_frames": [{"line": f.lineno, "function": f.name} for f in mine][-6:],
            },
        ) from None
    return Loaded(module, dest, display, lines)


def _trace_hint(err: str) -> str:
    low = err.lower()
    if "from __future__ import annotations" in low or "pep 563" in low:
        return "Remove `from __future__ import annotations`: @txl.kernel needs live annotation objects."
    if "no module named 'torch'" in low:
        return ("This skill's kernels use NumPy for inputs and references; torch is not installed. "
                "Replace torch tensors in make_inputs()/reference() with NumPy arrays.")
    if "no module named" in low:
        return "The file imports a module that is not installed; kernels may import numpy, tirx_kernels.tirx_lite and the standard library."
    if "low-level ir" in low or "func_call" in low:
        return "The kernel breaks a TIRx-lite rule (see the error); look up the instruction with find_kernel_examples scope=reference."
    return "Read the error and the reported source line; compare with a working example from find_kernel_examples."


def discover_kernels(module):
    from tirx_kernels.tirx_lite.entry import Kernel

    found = []
    seen = set()
    for attr, value in vars(module).items():
        if isinstance(value, Kernel) and id(value) not in seen:
            seen.add(id(value))
            found.append((attr, value))
    return found


def select_kernel(loaded: Loaded, name: str | None):
    kernels = discover_kernels(loaded.module)
    names = [k.name for _, k in kernels]
    if not kernels:
        raise Fail(
            "no_kernels",
            f"{loaded.display} defines no @txl.kernel function.",
            "Decorate the kernel with @txl.kernel(warps=..., arch=..., grid=...) after `import tirx_kernels.tirx_lite as txl`.",
        )
    if name:
        for attr, k in kernels:
            if name in (attr, k.name):
                return k
        raise Fail(
            "kernel_not_found",
            f"No kernel named {name!r} in {loaded.display}; it defines: {', '.join(names)}.",
            f"Call again with kernel_name set to one of: {', '.join(names)}.",
            {"kernels": names},
        )
    if len(kernels) > 1:
        raise Fail(
            "ambiguous_kernel",
            f"{loaded.display} defines {len(kernels)} kernels: {', '.join(names)}.",
            f"Call again with kernel_name set to one of: {', '.join(names)}.",
            {"kernels": names},
        )
    return kernels[0][1]


# ----------------------------------------------------------------------------------------------
# Kernel facts
# ----------------------------------------------------------------------------------------------

_GRID_RE = re.compile(r"T\.cta_id\(\[([^\]]*)\]")


def arch_major(arch: str) -> int:
    m = re.match(r"sm_(\d+)", str(arch or ""))
    if not m:
        return 0
    digits = m.group(1)
    return int(digits[:-1]) if len(digits) >= 2 else int(digits)


def kernel_params(k) -> list[dict]:
    """Each entry parameter: pointer (BufferType), scalar (PrimType) or tensormap/other."""
    params = []
    for p in k.func.params:
        name = str(getattr(p, "name", None) or p)
        ty = getattr(p, "ty", None)
        tname = type(ty).__name__ if ty is not None else ""
        if tname == "BufferType":
            try:
                dtype = str(p.dtype)
            except AttributeError:
                dtype = ""
            shape = getattr(getattr(k.session, "params", {}).get(name), "shape", None) or []
            params.append({"name": name, "kind": "pointer", "dtype": dtype, "shape": [str(s) for s in shape]})
        elif tname == "PrimType":
            dtype = str(getattr(ty, "dtype", ""))
            kind = "scalar" if dtype and dtype != "handle" else "tensormap"
            params.append({"name": name, "kind": kind, "dtype": dtype})
        else:
            kind = "tensormap" if "tensormap" in (tname + str(ty)).lower() else "other"
            params.append({"name": name, "kind": kind, "dtype": str(ty) if ty is not None else ""})
    return params


def kernel_facts(k) -> dict:
    text = str(k.func)
    grid = []
    m = _GRID_RE.search(text)
    if m:
        for part in m.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            grid.append(int(part) if part.isdigit() else part)
    arch = str(k.arch)
    uses_tcgen05 = "tcgen05" in text
    dyn = re.search(r"tirx\.dyn_smem_bytes\"?:\s*T\.int\d+\((\d+)\)", text)
    return {
        "name": k.name,
        "params": kernel_params(k),
        "grid": grid,
        "grid_static": all(isinstance(g, int) for g in grid) and bool(grid),
        "warps": int(k.warps),
        "threads_per_cta": int(k.warps) * 32,
        "arch": arch,
        "blackwell_only": arch_major(arch) >= 10 or uses_tcgen05,
        "uses_tcgen05": uses_tcgen05,
        "uses_wgmma": "wgmma" in text,
        "dynamic_smem_bytes": int(dyn.group(1)) if dyn else 0,
    }


# ----------------------------------------------------------------------------------------------
# Case functions and inputs
# ----------------------------------------------------------------------------------------------


def case_functions(kernel: Loaded, case: Loaded | None, need_reference: bool = True):
    source = case or kernel
    make_inputs = getattr(source.module, "make_inputs", None)
    reference = getattr(source.module, "reference", None)
    missing = []
    if not callable(make_inputs):
        missing.append("make_inputs()")
    if need_reference and not callable(reference):
        missing.append("reference(inputs)")
    if missing:
        raise Fail(
            "missing_case_functions",
            f"{source.display} has no {' or '.join(missing)}.",
            "Add make_inputs() returning a dict of NumPy arrays and scalars keyed by the kernel's parameter names "
            "(output buffers zero-filled), and reference(inputs) returning a dict of expected outputs computed "
            "independently with NumPy; or pass them in a separate case_file.",
            {"missing": missing, "file": source.display},
        )
    return make_inputs, reference


_NP_KIND = {"float": "f", "int": "i", "uint": "u", "bool": "b", "bfloat": "V", "e4m3": "V", "e5m2": "V"}


def _dtype_matches(np_dtype, tvm_dtype: str) -> tuple[bool, str]:
    import numpy as np

    want = str(tvm_dtype)
    try:
        want_np = np.dtype(want)
    except TypeError:
        want_np = None
        try:
            import ml_dtypes  # noqa: F401

            want_np = np.dtype(getattr(ml_dtypes, want))
        except (ImportError, AttributeError, TypeError):
            want_np = None
    if want_np is not None:
        return (np.dtype(np_dtype) == want_np, str(want_np))
    bits = re.findall(r"(\d+)", want)
    if bits:
        return (np.dtype(np_dtype).itemsize * 8 == int(bits[0]), want)
    return (True, want)


def build_inputs(k, make_inputs):
    import numpy as np

    try:
        raw = make_inputs()
    except Exception as e:  # noqa: BLE001
        raise Fail("invalid_inputs", clip(f"make_inputs() raised {type(e).__name__}: {e}"),
                   "Fix make_inputs() so it runs with NumPy alone and returns a dict.") from None
    if not isinstance(raw, dict):
        raise Fail("invalid_inputs", f"make_inputs() returned {type(raw).__name__}, not a dict.",
                   "Return a dict keyed by the kernel's parameter names.")
    params = kernel_params(k)
    names = [p["name"] for p in params]
    missing = [n for n in names if n not in raw]
    if missing:
        raise Fail(
            "invalid_inputs",
            f"make_inputs() has no value for kernel parameter(s): {', '.join(missing)}.",
            "Add every kernel parameter to make_inputs(), including zero-filled output buffers and scalars.",
            {"missing": missing, "parameters": names},
        )
    problems = []
    for p in params:
        value = raw[p["name"]]
        if p["kind"] == "pointer":
            if not isinstance(value, np.ndarray):
                problems.append(f"{p['name']} must be a NumPy array, got {type(value).__name__}")
                continue
            ok, want = _dtype_matches(value.dtype, p["dtype"])
            if not ok:
                problems.append(f"{p['name']} is {value.dtype} but the kernel declares {p['dtype']} (use dtype {want})")
        elif p["kind"] == "scalar":
            if isinstance(value, np.ndarray) and value.ndim != 0:
                problems.append(f"{p['name']} is a scalar parameter but got an array of shape {value.shape}")
            elif not isinstance(value, (int, float, bool, np.generic, np.ndarray)):
                problems.append(f"{p['name']} is a scalar parameter but got {type(value).__name__}")
    if problems:
        raise Fail("invalid_inputs", clip("make_inputs() does not match the kernel: " + "; ".join(problems), 900),
                   "Fix the dtypes/kinds in make_inputs() to match the kernel's parameters (see inspect_kernel).",
                   {"problems": problems[:20]})
    extra = [key for key in raw if key not in names]
    engine_inputs = {n: copy.deepcopy(raw[n]) for n in names}
    return raw, engine_inputs, extra


def expected_outputs(k, raw, reference):
    import numpy as np

    try:
        ref = reference(copy.deepcopy(raw))
    except Exception as e:  # noqa: BLE001
        raise Fail("invalid_reference", clip(f"reference(inputs) raised {type(e).__name__}: {e}"),
                   "Fix reference(inputs) so it computes the expected outputs with NumPy from the inputs dict.") from None
    if not isinstance(ref, dict) or not ref:
        raise Fail("invalid_reference", "reference(inputs) must return a non-empty dict of expected outputs.",
                   "Return e.g. {'out': expected_array}, keyed by the kernel's output buffer names.")
    pointer_names = [p["name"] for p in kernel_params(k) if p["kind"] == "pointer"]
    unknown = [n for n in ref if n not in pointer_names]
    if unknown:
        raise Fail(
            "invalid_reference",
            f"reference() returned key(s) that are not kernel buffer parameters: {', '.join(map(str, unknown))}.",
            f"Key the reference dict by output buffer names; the kernel's buffers are: {', '.join(pointer_names)}.",
        )
    out = {}
    for name, value in ref.items():
        try:
            out[name] = np.asarray(value)
        except Exception as e:  # noqa: BLE001
            raise Fail("invalid_reference", clip(f"reference() value for {name} is not array-like: {e}"),
                       "Return NumPy arrays from reference().") from None
    return out


# ----------------------------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------------------------


def _ranges(flat_idx, limit: int = 8):
    ranges = []
    if len(flat_idx) == 0:
        return ranges, 0
    start = prev = int(flat_idx[0])
    count = 0
    for i in flat_idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
            continue
        count += 1
        if len(ranges) < limit:
            ranges.append([start, prev])
        start = prev = i
    count += 1
    if len(ranges) < limit:
        ranges.append([start, prev])
    return ranges, count


def compare_output(name, got, exp, rtol, atol, max_listed=10) -> dict:
    import numpy as np

    got = np.asarray(got)
    exp = np.asarray(exp)
    info = {"name": name, "shape": list(got.shape), "dtype": str(got.dtype)}
    notes = []
    if exp.shape != got.shape:
        if exp.size == got.size:
            exp = exp.reshape(got.shape)
            notes.append(f"expected array reshaped from {list(np.asarray(exp).shape)} to {list(got.shape)}")
        else:
            info.update({
                "max_abs_err": "n/a", "max_rel_err": "n/a", "mismatches": int(got.size), "first_mismatches": [],
                "mismatch_ranges": [], "mismatch_range_count": 0,
                "shape_mismatch": f"kernel output has shape {list(got.shape)} but reference has {list(exp.shape)}",
            })
            return info
    try:
        g = got.astype(np.float64)
        e = exp.astype(np.float64)
    except (TypeError, ValueError):
        eq = got == exp
        mism = ~np.asarray(eq, dtype=bool)
        g = e = None
    if g is not None:
        diff = np.abs(g - e)
        both_nan = np.isnan(g) & np.isnan(e)
        diff = np.where(both_nan, 0.0, diff)
        mism = ~np.isclose(g, e, rtol=rtol, atol=atol, equal_nan=True)
        nz = np.abs(e) > 0
        rel = np.where(nz, diff / np.where(nz, np.abs(e), 1.0), 0.0)
        info["max_abs_err"] = num(np.max(diff)) if diff.size else 0.0
        info["max_rel_err"] = num(np.max(rel)) if rel.size else 0.0
        if diff.size and np.isnan(diff).any():
            info["max_abs_err"] = "nan"
    else:
        info["max_abs_err"] = "n/a"
        info["max_rel_err"] = "n/a"
    flat = np.flatnonzero(mism.reshape(-1))
    info["mismatches"] = int(flat.size)
    firsts = []
    gflat = got.reshape(-1)
    eflat = exp.reshape(-1)
    for i in flat[:max_listed]:
        idx = [int(x) for x in np.unravel_index(int(i), got.shape)] if got.ndim else []
        firsts.append({"index": idx, "flat_index": int(i), "got": num(gflat[i].item() if hasattr(gflat[i], "item") else gflat[i]),
                       "expected": num(eflat[i].item() if hasattr(eflat[i], "item") else eflat[i])})
    info["first_mismatches"] = firsts
    ranges, count = _ranges(flat)
    info["mismatch_ranges"] = ranges
    info["mismatch_range_count"] = count
    if notes:
        info["notes"] = notes
    return info


# ----------------------------------------------------------------------------------------------
# Actions
# ----------------------------------------------------------------------------------------------


def _load_all(req, need_case: bool, need_reference: bool = True):
    workdir = Path(req["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    kernel = load_user_file(req["kernel_file"], req.get("kernel_display") or "kernel.py", workdir, "kernel",
                            bool(req.get("kernel_is_source")))
    case = None
    if req.get("case_file"):
        case = load_user_file(req["case_file"], req.get("case_display") or "case.py", workdir, "case")
    k = select_kernel(kernel, req.get("kernel_name"))
    funcs = case_functions(kernel, case, need_reference) if need_case else (None, None)
    return kernel, case, k, funcs


def action_inspect(req):
    workdir = Path(req["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    kernel = load_user_file(req["kernel_file"], req.get("kernel_display") or "kernel.py", workdir, "kernel",
                            bool(req.get("kernel_is_source")))
    kernels = discover_kernels(kernel.module)
    if not kernels:
        raise Fail(
            "no_kernels",
            f"{kernel.display} defines no @txl.kernel function.",
            "Decorate the kernel with @txl.kernel(warps=..., arch=..., grid=...) after `import tirx_kernels.tirx_lite as txl`.",
        )
    wanted = req.get("kernel_name")
    if wanted:
        selected = [select_kernel(kernel, wanted)]
    else:
        selected = [k for _, k in kernels]
    outdir = Path(req["outdir"])
    facts = []
    files = []
    for k in selected:
        f = kernel_facts(k)
        text = str(k.func)
        path = outdir / f"{_safe_stem(k.name, 'kernel')}.tirx.txt"
        path.write_text(text, encoding="utf-8")
        f["tirx_lines"] = text.count("\n") + 1
        facts.append(f)
        files.append(str(path))
    mod = kernel.module
    has_make = callable(getattr(mod, "make_inputs", None))
    has_ref = callable(getattr(mod, "reference", None))
    case_check = None
    if has_make and len(selected) == 1:
        try:
            raw, _, extra = build_inputs(selected[0], mod.make_inputs)
            case_check = {"ok": True, "extra_keys": [str(x) for x in extra][:20],
                          "input_shapes": {n: list(getattr(v, "shape", [])) for n, v in raw.items() if hasattr(v, "shape")}}
        except Fail as e:
            case_check = {"ok": False, "problem": e.message}
    return {
        "file": kernel.display,
        "kernels": facts,
        "all_kernel_names": [k.name for _, k in kernels],
        "has_make_inputs": has_make,
        "has_reference": has_ref,
        "case_check": case_check,
        "files": files,
    }


def _group_diagnostics(diags, display, kpath, limit=20):
    """Simulator diagnostics grouped by status and kind, with a count and two example messages."""
    groups = {}
    for d in diags or []:
        if not isinstance(d, dict):
            d = {"message": str(d)}
        key = (str(d.get("status", "")), str(d.get("kind", "")))
        g = groups.setdefault(key, {"status": key[0], "kind": key[1], "count": 0, "examples": []})
        g["count"] += 1
        if len(g["examples"]) < 2:
            g["examples"].append(clip(str(d.get("message", "")).replace(str(kpath), display), 300))
    out = list(groups.values())
    order = {"error": 0, "review": 1}
    out.sort(key=lambda x: (order.get(x["status"], 2), -x["count"]))
    return out[:limit], len(out)


def action_simulate(req):
    import numpy as np

    kernel, case, k, (make_inputs, reference) = _load_all(req, need_case=True)
    facts = kernel_facts(k)
    if not facts["grid_static"]:
        raise Fail(
            "dynamic_launch",
            f"Kernel {k.name} has a launch grid that depends on a parameter ({facts['grid']}); the CPU simulator needs a constant grid.",
            "For simulation, declare a constant grid in @txl.kernel (e.g. grid=4) matching the shapes in make_inputs(), then call again.",
            {"grid": facts["grid"]},
        )
    raw, engine_inputs, extra = build_inputs(k, make_inputs)
    expected = expected_outputs(k, raw, reference)
    out_names = tuple(expected.keys())
    from tirx_harness import numsim
    from tirx_harness.numsim.errors import NumSimBuildError, NumSimExecutionError

    unsupported = tuple(
        getattr(numsim.errors, n) for n in ("UnsupportedTIRxError", "UnmodeledTIRxFormError") if hasattr(numsim.errors, n)
    )
    outdir = Path(req["outdir"])
    report = {"kernel": facts, "rtol": req["rtol"], "atol": req["atol"], "extra_input_keys": [str(x) for x in extra]}
    t0 = time.monotonic()
    try:
        module = numsim.transpile(k.func)
    except unsupported as e:
        msg = clip(str(e).replace(str(kernel.path), kernel.display), 900)
        report.update({"verdict": "incomplete", "reason": msg})
        (outdir / "numsim_report.json").write_text(json.dumps(jsonable(report), indent=1))
        return {"verdict": "incomplete", "kernel": k.name, "reason": msg, "unsupported": True,
                "outputs": [], "diagnostics": [], "sim_verdict": "not_run", "files": [str(outdir / "numsim_report.json")],
                "extra_input_keys": report["extra_input_keys"]}
    except NumSimBuildError as e:
        raise Fail("simulator_build_failed", clip(f"The CPU simulator could not be built for {k.name}: {e}", 900),
                   "Simplify the kernel or report the failure to the skill owner; check_kernel_safety may still work.") from None
    t1 = time.monotonic()
    try:
        result = numsim.Engine().run(module, inputs=engine_inputs, outputs=out_names)
    except NumSimExecutionError as e:
        msg = clip(str(e).replace(str(kernel.path), kernel.display), 900)
        low = msg.lower()
        if "missing required bindings" in low or "is unknown; expected" in low or "itemsize" in low:
            raise Fail("invalid_inputs", msg, "Make make_inputs() match the kernel's parameters and dtypes (see inspect_kernel).") from None
        fault = "deadlock" if "deadlock" in low else ("out_of_bounds" if "out-of-bounds" in low else "execution_fault")
        report.update({"verdict": "fail", "execution_error": msg, "fault": fault})
        (outdir / "numsim_report.json").write_text(json.dumps(jsonable(report), indent=1))
        return {"verdict": "fail", "kernel": k.name, "execution_error": msg, "fault": fault, "outputs": [],
                "diagnostics": [], "sim_verdict": "error", "files": [str(outdir / "numsim_report.json")],
                "extra_input_keys": report["extra_input_keys"]}
    t2 = time.monotonic()
    rtol, atol = float(req["rtol"]), float(req["atol"])
    per_output = []
    arrays = {}
    for name in out_names:
        got = result.outputs.get(name)
        if got is None:
            per_output.append({"name": name, "missing": True, "mismatches": int(expected[name].size)})
            continue
        got = np.asarray(got)
        arrays[name] = got
        arrays[f"{name}__expected"] = expected[name]
        per_output.append(compare_output(name, got, expected[name], rtol, atol))
    diags, n_groups = _group_diagnostics(getattr(result, "diagnostics", []), kernel.display, kernel.path)
    sim_verdict = str(getattr(result, "verdict", "") or "")
    any_mismatch = any(o.get("mismatches", 0) or o.get("missing") or o.get("shape_mismatch") for o in per_output)
    any_error_diag = any(d["status"] == "error" for d in diags)
    verdict = "fail" if (any_mismatch or any_error_diag) else "pass"
    np.savez(outdir / "simulated_outputs.npz", **arrays)
    full = [
        {kk: (clip(str(vv).replace(str(kernel.path), kernel.display), 400) if kk == "message" else vv) for kk, vv in d.items()}
        if isinstance(d, dict) else {"message": clip(str(d), 400)}
        for d in list(getattr(result, "diagnostics", []) or [])[:500]
    ]
    report.update({
        "verdict": verdict, "sim_verdict": sim_verdict, "outputs": per_output, "diagnostics": full,
        "stats": getattr(result, "stats", {}), "timing_s": {"transpile": round(t1 - t0, 3), "run": round(t2 - t1, 3)},
        "note": "Simulator time is CPU simulation time, not GPU latency.",
    })
    (outdir / "numsim_report.json").write_text(json.dumps(jsonable(report), indent=1))
    return {
        "verdict": verdict,
        "kernel": k.name,
        "outputs": per_output,
        "diagnostics": diags,
        "diagnostic_groups": n_groups,
        "sim_verdict": sim_verdict,
        "extra_input_keys": report["extra_input_keys"],
        "timing_s": report["timing_s"],
        "files": [str(outdir / "simulated_outputs.npz"), str(outdir / "numsim_report.json")],
    }


def _collect_lines(obj, kpath: str, acc: list):
    if isinstance(obj, dict):
        span = obj.get("source_span")
        if isinstance(span, dict) and span.get("line"):
            src = str(span.get("source_name", ""))
            if not src or Path(src).name == Path(kpath).name or src == kpath:
                acc.append(int(span["line"]))
        for v in obj.values():
            _collect_lines(v, kpath, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_lines(v, kpath, acc)


def _summarize_report(report_dict, kernel: Loaded, limit=10):
    findings = report_dict.get("findings", []) or []
    kinds = {}
    shown = []
    seen = set()
    order = {"error": 0, "review": 1}
    for f in sorted(findings, key=lambda f: order.get(str(f.get("status")), 2)):
        status, kind = str(f.get("status", "")), str(f.get("kind", ""))
        kinds[f"{status}:{kind}"] = kinds.get(f"{status}:{kind}", 0) + 1
        lines: list[int] = []
        _collect_lines(f.get("details", {}), str(kernel.path), lines)
        lines = sorted(set(lines))
        key = (status, kind, tuple(lines))
        if key in seen or len(shown) >= limit:
            continue
        seen.add(key)
        shown.append({
            "id": str(f.get("id", "")),
            "status": status,
            "kind": kind,
            "message": clip(str(f.get("message", "")).replace(str(kernel.path), kernel.display), 360),
            "lines": lines[:6],
            "code": {str(n): clip(kernel.line_text(n), 160) for n in lines[:6]},
        })
    return {
        "verdict": str(report_dict.get("verdict", "")),
        "finding_count": len(findings),
        "counts": kinds,
        "findings": shown,
    }


def action_safety(req):
    kernel, case, k, (make_inputs, _) = _load_all(req, need_case=True, need_reference=False)
    facts = kernel_facts(k)
    if not facts["grid_static"]:
        raise Fail(
            "dynamic_launch",
            f"Kernel {k.name} has a launch grid that depends on a parameter ({facts['grid']}); the checkers need a constant grid.",
            "Declare a constant grid in @txl.kernel (e.g. grid=4) matching the shapes in make_inputs(), then call again.",
            {"grid": facts["grid"]},
        )
    build_inputs(k, make_inputs)  # validates the case before running the checkers
    from tirx_harness import racecheck, synccheck

    wanted = req.get("checks", "both")
    which = [("synccheck", synccheck)] if wanted == "sync" else ([("racecheck", racecheck)] if wanted == "race" else
                                                                [("synccheck", synccheck), ("racecheck", racecheck)])
    outdir = Path(req["outdir"])
    summary, full, texts = {}, {}, []
    for label, fn in which:
        _, engine_inputs, _ = build_inputs(k, make_inputs)
        t0 = time.monotonic()
        try:
            rep = fn(k.func, inputs=engine_inputs)
            d = rep.to_dict()
            summary[label] = _summarize_report(d, kernel)
            summary[label]["seconds"] = round(time.monotonic() - t0, 2)
            full[label] = d
            try:
                texts.append(rep.format().replace(str(kernel.path), kernel.display))
            except Exception:  # noqa: BLE001
                texts.append(f"{label}: {d.get('summary', d.get('verdict'))}")
        except Exception as e:  # noqa: BLE001 - checker crashed: we could not check
            msg = clip(f"{type(e).__name__}: {e}".replace(str(kernel.path), kernel.display), 700)
            summary[label] = {"verdict": "incomplete", "finding_count": 0, "counts": {}, "findings": [], "reason": msg}
            full[label] = {"verdict": "incomplete", "reason": msg}
            texts.append(f"{label}: could not run: {msg}")
    rank = {"error": 3, "incomplete": 2, "review": 1, "clean": 0}
    overall = max((s["verdict"] for s in summary.values()), key=lambda v: rank.get(v, 2))
    (outdir / "safety_report.json").write_text(json.dumps(jsonable({"kernel": facts, "overall": overall, **full}), indent=1))
    (outdir / "safety_report.txt").write_text("\n\n".join(texts) + "\n")
    return {"kernel": k.name, "overall": overall, **summary,
            "files": [str(outdir / "safety_report.json"), str(outdir / "safety_report.txt")]}


def _target(arch: str, host: str | None = None):
    import tvm

    spec = {"kind": "cuda", "arch": arch}
    return tvm.target.Target(spec, host=host) if host else tvm.target.Target(spec)


def action_compile(req):
    kernel, case, k, _ = _load_all(req, need_case=False)
    arch = req.get("arch") or "kernel"
    arch = k.arch if arch == "kernel" else arch
    stages = req.get("stages", "sass")
    outdir = Path(req["outdir"])
    tmp = Path(req["workdir"]) / "dump"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        target = _target(arch)
        with target:
            compiled = k.compile(target=target)
    except Exception as e:  # noqa: BLE001
        msg = clip(f"{type(e).__name__}: {e}".replace(str(kernel.path), kernel.display), 900)
        raise Fail("compile_failed", f"Kernel {k.name} did not compile for {arch}: {msg}",
                   f"If the kernel uses instructions specific to {k.arch}, compile for that arch (arch='kernel'); "
                   "otherwise fix the reported problem.", {"arch": arch}) from None
    from tirx_harness.dump_kernel import dump_module

    want_ptx = stages in ("ptx", "sass")
    want_sass = stages == "sass"
    res = dump_module(compiled, arch=arch, ptx=want_ptx, sass=want_sass, outdir=str(tmp), name=_safe_stem(k.name, "kernel"))
    stem = _safe_stem(k.name, "kernel")
    files = []
    paths = dict(getattr(res, "paths", {}) or {})
    for key, ext in (("cuda", ".cu"), ("ptx", ".ptx"), ("sass", ".sass"), ("ptxas_log", ".ptxas.log")):
        p = paths.get(key)
        if p and Path(p).exists() and Path(p).stat().st_size > 0:
            dest = outdir / f"{stem}{ext}"
            shutil.copyfile(p, dest)
            files.append(str(dest))
    if not paths.get("cuda") and getattr(res, "cuda", None):
        dest = outdir / f"{stem}.cu"
        dest.write_text(res.cuda)
        files.append(str(dest))
    ptxas = dict(getattr(res, "ptxas", {}) or {})
    errors = [clip(str(e), 400) for e in (getattr(res, "errors", []) or [])]
    ptx_text = res.ptx or ""
    m = re.search(r"^\.target\s+(\S+)", ptx_text, re.M)
    sass_text = res.sass or ""
    data = {
        "kernel": k.name,
        "arch": arch,
        "kernel_arch": k.arch,
        "symbols": list(getattr(res, "symbols", []) or [])[:20],
        "registers": ptxas.get("registers"),
        "spill_stores": ptxas.get("spill_stores"),
        "spill_loads": ptxas.get("spill_loads"),
        "stack_bytes": ptxas.get("stack"),
        "static_smem_bytes": ptxas.get("smem") if ptxas.get("smem") is not None else (0 if ptxas.get("registers") is not None else None),
        "dynamic_smem_bytes": kernel_facts(k).get("dynamic_smem_bytes", 0),
        "barriers": ptxas.get("barriers"),
        "ptx_target": m.group(1) if m else None,
        "cuda_lines": (res.cuda or "").count("\n"),
        "ptx_lines": ptx_text.count("\n"),
        "sass_instructions": len(re.findall(r"/\*[0-9a-f]{4,}\*/", sass_text)),
        "stages": stages,
        "errors": errors,
        "compile_s": round(time.monotonic() - t0, 2),
        "files": files,
    }
    return data


_USAGE_TEMPLATE = """# {name}: standalone CUDA export

Generated from the TIRx-lite kernel `{name}` in `{file}` for `{arch}`.

`{stem}.cu` holds the device kernel and its host launch wrapper (FFI entry `{name}`).
Consumers need no TVM, tirx-kernels or tirx-harness; they need `apache-tvm-ffi[cpp]>=0.1.14.post0,<0.2`
and a CUDA toolkit whose `nvcc` supports `{arch}` to build, or only `apache-tvm-ffi>=0.1.14.post0,<0.2`
plus a compatible driver and the CUDA 13 runtime (`libcudart.so.13` on the library path) to load a prebuilt
`{stem}.so`.

## Argument contract (in this order)

{args}

Launch configuration baked into the wrapper: grid = {grid} CTA(s), {threads} threads per CTA ({warps} warps).
Pass tensors on the CUDA device with exactly these dtypes, shapes and contiguous layout; the kernel does
not check bounds beyond what it was written for.

## Build

```python
from pathlib import Path
import tvm_ffi.cpp

lib = tvm_ffi.cpp.build(name="{stem}", sources=["{stem}.cu"],
                        build_directory=str(Path("build").resolve()), backend="cuda")
```

Run with `TVM_FFI_CUDA_ARCH_LIST={ffi_arch}` set in the environment (the architecture used when exporting).

## Call

```python
import torch, tvm_ffi
mod = tvm_ffi.load_module("build/{stem}.so")
mod["{name}"]({call_args})
torch.cuda.synchronize()
```

Launches use the CUDA stream provided through tvm-ffi; synchronize before reading results on the CPU.
Validate the built artifact against the same inputs and reference as the original kernel before shipping it.
"""


def ffi_arch(arch: str) -> str:
    m = re.match(r"sm_(\d+)(a?)", arch)
    if not m:
        return arch
    digits, suffix = m.group(1), m.group(2)
    return f"{digits[:-1]}.{digits[-1]}{suffix}"


def action_export(req):
    kernel, case, k, _ = _load_all(req, need_case=False)
    arch = req.get("arch") or "kernel"
    arch = k.arch if arch == "kernel" else arch
    facts = kernel_facts(k)
    outdir = Path(req["outdir"])
    stem = _safe_stem(k.name, "kernel")
    try:
        from tvm.backend.cuda import export_cuda_host

        compiled = k.compile(target=_target(arch, host="cuda_host"))
        src = export_cuda_host(compiled.mod)
    except Exception as e:  # noqa: BLE001
        msg = clip(f"{type(e).__name__}: {e}".replace(str(kernel.path), kernel.display), 900)
        raise Fail("export_failed", f"Kernel {k.name} could not be exported for {arch}: {msg}",
                   "Check that the kernel compiles with compile_kernel for this arch; kernels that need runtime "
                   "workspace allocation or packed tensor-map encoders cannot be exported.", {"arch": arch}) from None
    # TVM guards its fp16/bf16 includes with __CUDA_ARCH__ (device pass only), but the exported file also
    # holds host code that names `half`/`nv_bfloat16`, so a single nvcc build needs them in the host pass too.
    prelude = []
    if re.search(r"\bhalf\b|__half", src):
        prelude.append("#include <cuda_fp16.h>")
    if re.search(r"nv_bfloat16|__nv_bfloat16", src):
        prelude.append("#include <cuda_bf16.h>")
    if prelude:
        src = ("// Added by export: half-precision types used by host-side helpers.\n#ifndef __CUDACC_RTC__\n"
               + "\n".join(prelude) + "\n#endif\n" + src)
    cu = outdir / f"{stem}.cu"
    cu.write_text(src, encoding="utf-8")
    entries = [e for e in re.findall(r"int32_t\s+__tvm_ffi_([A-Za-z0-9_]+)\s*\(", src)
               if e not in ("main", "_library_ctx")]
    entry = k.name if k.name in entries else (entries[0] if entries else None)
    kernels_in_src = list(dict.fromkeys(
        re.findall(r"__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?([A-Za-z0-9_]+)", src)))
    args_lines = []
    call_args = []
    for i, p in enumerate(facts["params"], 1):
        if p["kind"] == "pointer":
            args_lines.append(f"{i}. `{p['name']}`: CUDA tensor, dtype `{p['dtype']}`, shape `{tuple(p['shape'])}`")
        else:
            args_lines.append(f"{i}. `{p['name']}`: {p['kind']} `{p['dtype']}`")
        call_args.append(p["name"])
    usage = _USAGE_TEMPLATE.format(
        name=k.name, file=kernel.display, arch=arch, stem=stem, args="\n".join(args_lines) or "(no arguments)",
        grid=facts["grid"], threads=facts["threads_per_cta"], warps=facts["warps"], ffi_arch=ffi_arch(arch),
        call_args=", ".join(call_args),
    )
    usage_path = outdir / "USAGE.md"
    usage_path.write_text(usage, encoding="utf-8")
    files = [str(cu), str(usage_path)]
    built = False
    build_error = None
    if req.get("build_library"):
        try:
            import tvm_ffi.cpp

            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = ffi_arch(arch)
            bdir = Path(req["workdir"]) / "build"
            lib = tvm_ffi.cpp.build(name=stem, sources=[str(cu)], build_directory=str(bdir), backend="cuda")
            so = outdir / f"{stem}.so"
            shutil.copyfile(lib, so)
            files.append(str(so))
            built = True
        except Exception as e:  # noqa: BLE001
            text = f"{type(e).__name__}: {e}"
            # The compiler's real error is at the end of ninja's log; drop warnings and keep the tail.
            lines = [ln for ln in text.splitlines()
                     if ln.strip() and "warning #" not in ln and "expression has no effect" not in ln]
            text = "\n".join(lines)
            build_error = text if len(text) <= 900 else "…" + text[-899:]
    return {
        "kernel": k.name,
        "entry_symbol": entry or k.name,
        "device_kernels": kernels_in_src[:10],
        "arch": arch,
        "ffi_arch": ffi_arch(arch),
        "args": [{"name": p["name"], "kind": p["kind"], "dtype": p["dtype"]} for p in facts["params"]],
        "grid": facts["grid"],
        "threads_per_cta": facts["threads_per_cta"],
        "cu_lines": src.count("\n"),
        "built": built,
        "build_error": build_error,
        "files": files,
    }


# ----------------------------------------------------------------------------------------------
# Benchmark (needs an NVIDIA GPU)
# ----------------------------------------------------------------------------------------------


def _device_arch(dev) -> str:
    cv = str(dev.compute_version)  # e.g. "9.0"
    major, _, minor = cv.partition(".")
    base = f"sm_{major}{minor or '0'}"
    return base + ("a" if int(major) in (9, 10) else "")


def _arch_runs_on(kernel_arch: str, device_arch: str) -> bool:
    km, dm = arch_major(kernel_arch), arch_major(device_arch)
    if kernel_arch.endswith("a"):
        return kernel_arch.rstrip("a") == device_arch.rstrip("a")
    return km <= dm


_DEVICE_MODULES: dict = {}


def _run_on_device(k, arch, engine_inputs, out_names, dev, repeats, runs, number=1):
    import numpy as np
    import tvm

    key = (id(k), arch)
    rt = _DEVICE_MODULES.get(key)
    if rt is None:
        target = _target(arch)
        with target:
            ex = k.compile(target=target)
        rt = ex.jit() if hasattr(ex, "jit") else ex
        _DEVICE_MODULES[key] = rt
    name = k.name
    func = rt[name]
    params = kernel_params(k)
    args = []
    tensors = {}
    for p in params:
        v = engine_inputs[p["name"]]
        if p["kind"] == "pointer":
            t = tvm.runtime.tensor(np.ascontiguousarray(v), device=dev)
            tensors[p["name"]] = t
            args.append(t)
        else:
            args.append(v.item() if hasattr(v, "item") else v)
    func(*args)
    dev.sync()
    outputs = {n: tensors[n].numpy() for n in out_names if n in tensors}
    timer = rt.time_evaluator(name, dev, number=number, repeat=repeats, min_repeat_ms=0)
    for _ in range(3):  # warm-up launches
        func(*args)
    dev.sync()
    per_run = []
    all_us = []
    for _ in range(runs):
        res = timer(*args)
        us = [float(x) * 1e6 for x in res.results]
        all_us.extend(us)
        per_run.append(float(np.median(us)))
    return outputs, per_run, all_us


SKILL_GPU = "NVIDIA H100"
SKILL_GPU_ARCH = "sm_90a"


def _refuse_other_arch(k, role: str) -> None:
    """Static check, before any device work: the skill's GPU is an H100 (sm_90a)."""
    facts = kernel_facts(k)
    arch = str(k.arch)
    if arch_major(arch) >= 10 or facts["uses_tcgen05"]:
        why = []
        if arch_major(arch) >= 10:
            why.append(f"targets {arch}")
        if facts["uses_tcgen05"]:
            why.append(("uses" if why else f"declares {arch} but uses") + " tcgen05/TMEM instructions (Blackwell-only)")
        who = "Kernel" if role == "kernel" else "Baseline kernel"
        raise Fail(
            "arch_unsupported",
            f"{who} {k.name} {' and '.join(why)}, so it cannot run on this skill's GPU, an {SKILL_GPU} "
            f"({SKILL_GPU_ARCH}). Nothing was timed.",
            ("Tell the user this kernel can't be timed on the H100; it can still be checked on CPU with "
             "simulate_kernel, check_kernel_safety and compile_kernel. For H100 timing, retarget it: "
             "arch='sm_90a' in @txl.kernel and no tcgen05/TMEM instructions, then call benchmark_kernel again.")
            if role == "kernel" else
            ("Pass a baseline written for the H100 (arch='sm_90a', no tcgen05/TMEM), or benchmark the candidate "
             "alone without a baseline."),
            {"kernel_arch": arch, "role": role, "gpu": SKILL_GPU, "gpu_arch": SKILL_GPU_ARCH},
        )


def action_benchmark(req):
    import numpy as np

    # Trace first and refuse other architectures by name before touching the device, so the refusal is
    # the same whether or not a GPU is attached.
    kernel, case, k, (make_inputs, reference) = _load_all(req, need_case=True)
    _refuse_other_arch(k, "kernel")
    base = None
    if req.get("baseline_file"):
        bl = load_user_file(req["baseline_file"], req.get("baseline_display") or "baseline.py", Path(req["workdir"]),
                            "baseline", bool(req.get("baseline_is_source")))
        base = select_kernel(bl, req.get("kernel_name") or None)
        _refuse_other_arch(base, "baseline")
    try:
        import tvm

        dev = tvm.cuda(0)
        has_gpu = bool(dev.exist)
    except Exception:  # noqa: BLE001
        has_gpu = False
    if not has_gpu:
        raise Fail("no_gpu", "No NVIDIA GPU is visible to this run, so the kernel cannot be timed.",
                   "Use simulate_kernel and check_kernel_safety for correctness on CPU and compile_kernel for "
                   "register/spill numbers; tell the user GPU timing was not possible in this environment.",
                   {"kernel_arch": str(k.arch)})
    device_arch = _device_arch(dev)
    if not _arch_runs_on(k.arch, device_arch):
        raise Fail("arch_unsupported",
                   f"Kernel {k.name} targets {k.arch}, which cannot run on this GPU ({dev.device_name}, {device_arch}).",
                   "Simulate, check and compile it on CPU instead; for device timing, write a variant for "
                   f"{device_arch} (for H100 use arch='sm_90a' and no tcgen05/TMEM instructions).",
                   {"device": dev.device_name, "device_arch": device_arch, "kernel_arch": k.arch})
    if base is not None and not _arch_runs_on(base.arch, device_arch):
        raise Fail("arch_unsupported", f"Baseline kernel {base.name} targets {base.arch}, which cannot run on {device_arch}.",
                   "Pass a baseline written for this GPU's architecture.")
    raw, engine_inputs, extra = build_inputs(k, make_inputs)
    expected = expected_outputs(k, raw, reference)
    out_names = tuple(expected.keys())
    repeats, runs = int(req["repeats"]), int(req["runs"])
    rtol, atol = float(req["rtol"]), float(req["atol"])

    cand_runs, base_runs, cand_all, base_all = [], [], [], []
    outputs = None
    base_outputs = None
    # Interleave candidate and baseline runs so clock/thermal drift affects both alike.
    # Arch-specific targets (sm_90a) must match the device exactly; portable ones are rebuilt for the device.
    run_arch = k.arch if str(k.arch).endswith("a") else device_arch
    base_arch = None if base is None else (base.arch if str(base.arch).endswith("a") else device_arch)
    for _ in range(runs):
        o, pr, al = _run_on_device(k, run_arch, copy.deepcopy(engine_inputs), out_names, dev, repeats, 1)
        outputs = outputs or o
        cand_runs += pr
        cand_all += al
        if base is not None:
            _, b_inputs, _ = build_inputs(base, make_inputs)
            bo, bpr, bal = _run_on_device(base, base_arch, b_inputs, out_names, dev, repeats, 1)
            base_outputs = base_outputs or bo
            base_runs += bpr
            base_all += bal
    per_output = [compare_output(n, outputs[n], expected[n], rtol, atol, max_listed=5) for n in out_names if n in outputs]
    correct = all(o.get("mismatches", 1) == 0 for o in per_output) and len(per_output) == len(out_names)
    base_correct = None
    if base_outputs is not None:
        base_cmp = [compare_output(n, base_outputs[n], expected[n], rtol, atol, max_listed=3) for n in out_names if n in base_outputs]
        base_correct = all(o.get("mismatches", 1) == 0 for o in base_cmp)
    median_us = float(np.median(cand_all))
    data = {
        "kernel": k.name,
        "device": dev.device_name,
        "device_arch": device_arch,
        "arch": run_arch,
        "kernel_arch": str(k.arch),
        "correct": bool(correct),
        "max_abs_err": max((o.get("max_abs_err", 0) for o in per_output if isinstance(o.get("max_abs_err"), (int, float))), default=0.0),
        "outputs": per_output,
        "median_us": round(median_us, 3),
        "min_us": round(float(np.min(cand_all)), 3),
        "per_run_median_us": [round(x, 3) for x in cand_runs],
        "repeats": repeats,
        "runs": runs,
    }
    if base is not None:
        b_med = float(np.median(base_all))
        data.update({
            "baseline_kernel": base.name,
            "baseline_correct": base_correct,
            "baseline_median_us": round(b_med, 3),
            "baseline_per_run_median_us": [round(x, 3) for x in base_runs],
            "speedup": round(b_med / median_us, 4) if median_us > 0 else None,
            "faster_in_every_run": all(c < b for c, b in zip(cand_runs, base_runs)),
        })
    mod = (case or kernel).module
    bytes_moved = getattr(mod, "BYTES_MOVED", None)
    flops = getattr(mod, "FLOPS", None)
    if isinstance(bytes_moved, (int, float)) and bytes_moved > 0 and median_us > 0:
        data["bandwidth_gbs"] = round(bytes_moved / (median_us * 1e-6) / 1e9, 2)
    if isinstance(flops, (int, float)) and flops > 0 and median_us > 0:
        data["tflops"] = round(flops / (median_us * 1e-6) / 1e12, 3)
    outdir = Path(req["outdir"])
    (outdir / "benchmark_report.json").write_text(json.dumps(jsonable({**data, "all_samples_us": cand_all,
                                                                         "baseline_samples_us": base_all}), indent=1))
    data["files"] = [str(outdir / "benchmark_report.json")]
    return data


ACTIONS = {
    "inspect": action_inspect,
    "simulate": action_simulate,
    "safety": action_safety,
    "compile": action_compile,
    "export": action_export,
    "benchmark": action_benchmark,
}


def main() -> int:
    req = json.loads(Path(sys.argv[1]).read_text())
    resp_path = Path(sys.argv[2])
    limit_mb = req.get("memory_limit_mb")
    if limit_mb:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (int(limit_mb) << 20, int(limit_mb) << 20))
        except (ValueError, OSError):
            pass
    try:
        data = ACTIONS[req["action"]](req)
        resp = {"ok": True, "data": data}
    except Fail as f:
        resp = {"ok": False, "error": {"code": f.code, "message": f.message, "next_step": f.next_step,
                                       "details": jsonable(f.details)}}
    except MemoryError:
        resp = {"ok": False, "error": {"code": "out_of_memory", "message": "The kernel work ran out of memory.",
                                       "next_step": "Use smaller shapes in make_inputs().", "details": {}}}
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(limit=6)
        resp = {"ok": False, "error": {"code": "engine_error", "message": clip(f"{type(e).__name__}: {e}", 900),
                                       "next_step": "", "details": {"traceback": clip(tb, 3000)}}}
    resp_path.write_text(json.dumps(jsonable(resp)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
