#!/bin/bash
set -e
# Input: a tile-reverse kernel whose shared-memory store and load have no CTA barrier between them.
# Truth: the checkers' verdicts on the original (computed with TIRx Harness directly) and a fresh case.
set -euo pipefail
mkdir -p /task/inputs /task/truth
cp /task/tile_reverse.py /task/inputs/tile_reverse.py
. /task/numsim_env.sh
nspython - <<'PY'
import json, sys
import numpy as np
sys.path.insert(0, "/task")
import kcheck

(k,) = kcheck.load_kernels("/task/inputs/tile_reverse.py")
rng = np.random.default_rng(5)
inputs = {"a": rng.standard_normal(256).astype(np.float32), "out": np.zeros(256, np.float32),
          "scale": np.float32(1.5)}
sync, race = kcheck.checkers_clean(k, inputs)
if race != "error":
    sys.exit(f"setup: expected the original kernel to race, racecheck says {race}")
json.dump({"synccheck": sync, "racecheck": race}, open("/task/truth/verdict.json", "w"))
rng = np.random.default_rng(99)
a = rng.standard_normal(256).astype(np.float32)
out = (a.reshape(2, 128)[:, ::-1] * np.float32(-0.5)).reshape(-1).astype(np.float32)
np.savez("/task/truth/case.npz", a=a, out=out)
PY
