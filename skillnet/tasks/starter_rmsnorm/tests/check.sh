#!/bin/bash
# Pass when a kernel .py in /task/outputs (1) has no relative imports, (2) targets the H100 (sm_90/sm_90a),
# (3) computes RMSNorm on fresh 16 x 512 inputs in the simulator, (4) is synccheck/racecheck clean, and
# (5) its own reference() agrees with an independent NumPy RMSNorm.
set -uo pipefail
cd /task
. /task/numsim_env.sh
nspython - <<'PY'
import ast, importlib.util, sys
import numpy as np
sys.path.insert(0, "/task")
import kcheck

case = np.load("/task/truth/case.npz")
def make_inputs():
    return {"x": case["x"].copy(), "w": case["w"].copy(), "out": np.zeros(16 * 512, np.float32)}

def check(p):
    tree = ast.parse(p.read_text())
    rel = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level > 0]
    if rel:
        return f"relative import on line(s) {rel}"
    try:
        kernels = kcheck.load_kernels(p)
    except Exception as e:  # noqa: BLE001
        return f"does not load ({type(e).__name__}: {e})"[:300]
    why = []
    for k in kernels:
        if kcheck.param_names(k) != ["x", "w", "out"]:
            why.append(f"{k.name}: parameters {kcheck.param_names(k)}")
            continue
        if not str(k.arch).startswith("sm_90"):
            why.append(f"{k.name}: arch {k.arch} is not an H100 target")
            continue
        try:
            got = kcheck.simulate(k, make_inputs(), ["out"])["out"]
        except Exception as e:  # noqa: BLE001
            why.append(f"{k.name}: simulation failed ({type(e).__name__}: {e})"[:300])
            continue
        if not np.allclose(got, case["out"], rtol=1e-5, atol=1e-6):
            why.append(f"{k.name}: {int((~np.isclose(got, case['out'], rtol=1e-5, atol=1e-6)).sum())} wrong values")
            continue
        s, r = kcheck.checkers_clean(k, make_inputs())
        if s != "clean" or r != "clean":
            why.append(f"{k.name}: synccheck={s} racecheck={r}")
            continue
        spec = importlib.util.spec_from_file_location("cand_ref_%d" % abs(hash(str(p))), str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not callable(getattr(mod, "reference", None)):
            why.append("no reference() in the file")
            continue
        ref = np.asarray(mod.reference(make_inputs())["out"], dtype=np.float64).reshape(-1)
        if not np.allclose(ref, case["out"], rtol=1e-5, atol=1e-6):
            why.append("its reference() disagrees with an independent NumPy RMSNorm")
            continue
        return None, k
    return "; ".join(why) or "no @txl.kernel found"

reasons = []
for p in kcheck.kernel_files("/task/outputs"):
    res = check(p)
    if isinstance(res, tuple):
        print(f"PASS: {p.name}:{res[1].name} ({res[1].arch}) computes RMSNorm, checks clean, reference agrees")
        sys.exit(0)
    reasons.append(f"{p.name}: {res}")
print("FAIL: no output kernel qualifies:", *reasons, sep="\n  ")
sys.exit(1)
PY
