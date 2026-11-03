#!/bin/bash
set -e
# Oracle: simulate the pasted kernel with TIRx Harness directly, find the wrong elements, fix the w index.
set -euo pipefail
mkdir -p /task/outputs /tmp/oracle
cat > /tmp/oracle/bug.py <<'KERNEL'
import numpy as np
import tirx_kernels.tirx_lite as txl

ROWS = 8
COLS = 64
WARPS = 4
GRID = ROWS // WARPS


@txl.kernel(warps=WARPS, arch="sm_90a", grid=GRID)
def scale_bias(x: txl.gptr(txl.f32), w: txl.gptr(txl.f32), bias: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    row = txl.cta_id() * WARPS + txl.thread_id() // 32
    v = txl.local_scalar("float32")
    s = txl.local_scalar("float32")
    b = txl.local_scalar("float32")
    with txl.unroll(COLS // 32) as k:
        col = k * 32 + lane
        txl.ptx.ld.global_.f32(v, x.ptr_to([row * COLS + col]))
        txl.ptx.ld.global_.f32(s, w.ptr_to([lane]))
        txl.ptx.ld.global_.f32(b, bias.ptr_to([col]))
        txl.ptx.st.global_.f32(out.ptr_to([row * COLS + col]), v * s + b)


def make_inputs():
    rng = np.random.default_rng(3)
    return {
        "x": rng.uniform(0.5, 1.5, ROWS * COLS).astype(np.float32),
        "w": rng.uniform(0.5, 1.5, COLS).astype(np.float32),
        "bias": rng.uniform(0.5, 1.5, COLS).astype(np.float32),
        "out": np.zeros(ROWS * COLS, np.float32),
    }


def reference(inputs):
    x = inputs["x"].astype(np.float64).reshape(ROWS, COLS)
    y = x * inputs["w"].astype(np.float64) + inputs["bias"].astype(np.float64)
    return {"out": y.reshape(-1).astype(np.float32)}
KERNEL
. /task/numsim_env.sh
nspython - <<'PY'
import importlib.util, sys
import numpy as np
sys.path.insert(0, "/task")
import kcheck

def load(path):
    spec = importlib.util.spec_from_file_location("m%d" % abs(hash(path)), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

m = load("/tmp/oracle/bug.py")
got = kcheck.simulate(m.scale_bias, m.make_inputs(), ["out"])["out"]
bad = np.flatnonzero(~np.isclose(got, m.reference(m.make_inputs())["out"], rtol=1e-5, atol=1e-8))
cols = sorted(set((bad % 64).tolist()))
src = open("/tmp/oracle/bug.py").read().replace("w.ptr_to([lane])", "w.ptr_to([col])")
open("/task/outputs/scale_bias.py", "w").write(src)
f = load("/task/outputs/scale_bias.py")
got2 = kcheck.simulate(f.scale_bias, f.make_inputs(), ["out"])["out"]
assert np.allclose(got2, f.reference(f.make_inputs())["out"], rtol=1e-5, atol=1e-8)
open("/task/answer.txt", "w").write(
    f"{bad.size} of 512 outputs were wrong: columns {cols[0]}..{cols[-1]} of every row (flat 32-63, 96-127, ...). "
    "The weight was loaded with w[lane] instead of w[col], so the second 32-column pass reused w[0..31]. "
    "Fixed it to w.ptr_to([col]); the simulator now matches NumPy for all 512 values.\n")
PY
