"""Independent verifier for kernel tasks: drives TIRx Harness directly (not the skill's tools).

Usage: python kcheck.py <command> [args]   (see main)
"""
import importlib.util
import json
import pathlib
import re
import sys

import numpy as np


def kernel_files(root):
    found = []
    for p in sorted(pathlib.Path(root).rglob("*.py")):
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if "txl.kernel" in text:
            found.append(p)
    return found


def load_kernels(path):
    from tirx_kernels.tirx_lite.entry import Kernel

    spec = importlib.util.spec_from_file_location("candidate_%d" % abs(hash(str(path))), str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [v for v in vars(mod).values() if isinstance(v, Kernel)]


def param_names(k):
    return [str(getattr(p, "name", None) or p) for p in k.func.params]


def simulate(k, inputs, outputs):
    from tirx_harness import numsim

    r = numsim.Engine().run(numsim.transpile(k.func), inputs=inputs, outputs=tuple(outputs))
    return {n: np.asarray(r.outputs[n]) for n in outputs}


def checkers_clean(k, inputs):
    from tirx_harness import racecheck, synccheck

    s = synccheck(k.func, inputs=dict(inputs)).verdict
    r = racecheck(k.func, inputs=dict(inputs)).verdict
    return s, r


def first_matching(root, want_params, make_inputs, expected, rtol=1e-5, atol=1e-6, need_clean=False,
                   forbid_arch_prefix=None):
    """Return (path, kernel) of the first output kernel that computes `expected`, else (None, reasons)."""
    reasons = []
    for path in kernel_files(root):
        try:
            kernels = load_kernels(path)
        except Exception as e:  # noqa: BLE001
            reasons.append(f"{path.name}: does not load ({type(e).__name__}: {e})"[:300])
            continue
        for k in kernels:
            names = param_names(k)
            if names != want_params:
                reasons.append(f"{path.name}:{k.name}: parameters {names} != {want_params}")
                continue
            if forbid_arch_prefix and str(k.arch).startswith(forbid_arch_prefix):
                reasons.append(f"{path.name}:{k.name}: arch {k.arch} not allowed")
                continue
            try:
                got = simulate(k, make_inputs(), list(expected))
            except Exception as e:  # noqa: BLE001
                reasons.append(f"{path.name}:{k.name}: simulation failed ({type(e).__name__}: {e})"[:300])
                continue
            ok = all(np.allclose(got[n], expected[n], rtol=rtol, atol=atol) for n in expected)
            if not ok:
                bad = {n: int((~np.isclose(got[n], expected[n], rtol=rtol, atol=atol)).sum()) for n in expected}
                reasons.append(f"{path.name}:{k.name}: wrong values {bad}")
                continue
            if need_clean:
                s, r = checkers_clean(k, make_inputs())
                if s != "clean" or r != "clean":
                    reasons.append(f"{path.name}:{k.name}: synccheck={s} racecheck={r}")
                    continue
            return path, k
    return None, reasons


def answer_text(path="/task/answer.txt"):
    p = pathlib.Path(path)
    return p.read_text(errors="replace") if p.exists() else ""


def answer_json(path="/task/answer.txt"):
    text = answer_text(path)
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            return json.loads(m.group(0))
        except ValueError:
            continue
    return None


if __name__ == "__main__":
    print("helper module; import it from the task's check script")
    sys.exit(0)
