#!/bin/bash
set -e
# Oracle: run Racecheck on the original with TIRx Harness, add the CTA barrier, re-check and re-simulate.
set -euo pipefail
mkdir -p /task/outputs
. /task/numsim_env.sh
nspython - <<'PY'
import sys
import numpy as np
sys.path.insert(0, "/task")
import kcheck
from tirx_harness import racecheck

src = "/task/inputs/tile_reverse.py"
(k,) = kcheck.load_kernels(src)
rng = np.random.default_rng(5)
inputs = {"a": rng.standard_normal(256).astype(np.float32), "out": np.zeros(256, np.float32),
          "scale": np.float32(1.5)}
assert racecheck(k.func, inputs=dict(inputs)).verdict == "error"
text = open(src).read()
load = "    txl.ptx.ld.shared.f32(y, tile.ptr_to([THREADS - 1 - tid]))"
fixed = text.replace(load, "    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))\n" + load)
open("/task/outputs/tile_reverse.py", "w").write(fixed)
(f,) = kcheck.load_kernels("/task/outputs/tile_reverse.py")
assert kcheck.checkers_clean(f, dict(inputs)) == ("clean", "clean")
got = kcheck.simulate(f, dict(inputs), ["out"])["out"]
assert np.allclose(got, (inputs["a"].reshape(2, 128)[:, ::-1] * 1.5).reshape(-1), rtol=1e-6, atol=1e-7)
open("/task/answer.txt", "w").write(
    "Racecheck reports a data race: each thread stores its value to shared memory (line 22) and then reads "
    "another warp's slot (line 23) with no barrier in between. Added bar.sync over all 128 threads between "
    "the store and the load; Synccheck and Racecheck are now clean and NumSim matches NumPy.\n")
PY
