#!/bin/bash
set -e
# No input files: the buggy kernel is pasted in the instruction. Truth: a fresh test case for the fixed
# kernel and the wrong columns of the original (w is indexed by lane, so columns 32..63 use w[c - 32]).
set -euo pipefail
mkdir -p /task/inputs /task/truth
python3 - <<'PY'
import json
import numpy as np
rows, cols = 8, 64
rng = np.random.default_rng(4242)
x = rng.uniform(0.5, 1.5, rows * cols).astype(np.float32)
w = rng.uniform(0.5, 1.5, cols).astype(np.float32)
bias = rng.uniform(0.5, 1.5, cols).astype(np.float32)
out = (x.astype(np.float64).reshape(rows, cols) * w.astype(np.float64) + bias.astype(np.float64))
np.savez("/task/truth/case.npz", x=x, w=w, bias=bias, out=out.reshape(-1).astype(np.float32))
json.dump({"wrong_cols_first": 32, "wrong_cols_last": 63}, open("/task/truth/wrong.json", "w"))
PY
