#!/bin/bash
set -e
# Oracle: simulate the original with TIRx Harness to find the wrong elements, fix the CTA stride, re-verify.
set -euo pipefail
mkdir -p /task/outputs
. /task/numsim_env.sh
nspython - <<'PY'
import importlib.util, sys
import numpy as np
sys.path.insert(0, "/task")
import kcheck

def load(path):
    spec = importlib.util.spec_from_file_location("m%d" % abs(hash(path)), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

src = "/task/inputs/saxpy_kernel.py"
m = load(src)
got = kcheck.simulate(m.saxpy, m.make_inputs(), ["out"])["out"]
bad = np.flatnonzero(~np.isclose(got, m.reference(m.make_inputs())["out"], rtol=1e-5, atol=1e-8))
fixed = open(src).read().replace("txl.cta_id() * 64 + txl.thread_id()", "txl.cta_id() * THREADS + txl.thread_id()")
open("/task/outputs/saxpy_kernel.py", "w").write(fixed)
f = load("/task/outputs/saxpy_kernel.py")
got2 = kcheck.simulate(f.saxpy, f.make_inputs(), ["out"])["out"]
assert np.allclose(got2, f.reference(f.make_inputs())["out"], rtol=1e-5, atol=1e-8)
open("/task/answer.txt", "w").write(
    f"Elements {bad[0]}..{bad[-1]} ({bad.size} values) were never written: each CTA advanced by 64 instead of "
    "128 threads. Fixed the index to cta_id * THREADS + thread_id; the simulator now matches NumPy exactly.\n")
PY
