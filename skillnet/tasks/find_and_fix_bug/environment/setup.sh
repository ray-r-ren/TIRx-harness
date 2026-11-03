#!/bin/bash
set -e
# Input: the buggy kernel (CTA stride 64 instead of 128). Truth: which elements the bug leaves unwritten,
# computed by replaying the kernel's index math in NumPy.
set -euo pipefail
mkdir -p /task/inputs /task/truth
cp /task/saxpy_kernel.py /task/inputs/saxpy_kernel.py
python - <<'PY'
import json
import numpy as np
n, threads, grid = 300, 128, 3
written = np.zeros(n, bool)
for cta in range(grid):
    for tid in range(threads):
        i = cta * 64 + tid          # the buggy index
        if i < n:
            written[i] = True
wrong = np.flatnonzero(~written)
rng = np.random.default_rng(777)
x = rng.standard_normal(n).astype(np.float32)
y = rng.standard_normal(n).astype(np.float32)
np.savez("/task/truth/case.npz", x=x, y=y, out=(np.float32(-0.75) * x + y).astype(np.float32))
json.dump({"wrong_first": int(wrong[0]), "wrong_last": int(wrong[-1]), "wrong_count": int(wrong.size)},
          open("/task/truth/wrong.json", "w"))
PY
