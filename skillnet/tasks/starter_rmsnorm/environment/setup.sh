#!/bin/bash
set -e
# No input files: the user describes the math. Truth: a fresh 16 x 512 test case with an independent
# float64 NumPy RMSNorm (eps 1e-5) cast to float32.
set -euo pipefail
mkdir -p /task/inputs /task/truth
python3 - <<'PY'
import numpy as np
rows, cols, eps = 16, 512, 1e-5
rng = np.random.default_rng(9090)
x = rng.standard_normal(rows * cols).astype(np.float32)
w = rng.uniform(0.5, 1.5, cols).astype(np.float32)
x64 = x.astype(np.float64).reshape(rows, cols)
out = x64 / np.sqrt((x64 ** 2).mean(axis=1, keepdims=True) + eps) * w.astype(np.float64)
np.savez("/task/truth/case.npz", x=x, w=w, out=out.reshape(-1).astype(np.float32))
PY
