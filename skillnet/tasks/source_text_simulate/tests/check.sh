#!/bin/bash
# Pass when (1) a kernel in /task/outputs with parameters (x, w, bias, out) computes x*w + bias for fresh
# inputs in the simulator (independent NumPy expectation), and (2) the answer names the wrong range 32..63.
set -uo pipefail
cd /task
. /task/numsim_env.sh
nspython - <<'PY'
import json, re, sys
import numpy as np
sys.path.insert(0, "/task")
import kcheck

case = np.load("/task/truth/case.npz")
truth = json.load(open("/task/truth/wrong.json"))
def make_inputs():
    return {"x": case["x"].copy(), "w": case["w"].copy(), "bias": case["bias"].copy(),
            "out": np.zeros(512, np.float32)}
path, k = kcheck.first_matching("/task/outputs", ["x", "w", "bias", "out"], make_inputs, {"out": case["out"]},
                                rtol=1e-5, atol=1e-6, forbid_arch_prefix="sm_10")
if path is None:
    print("FAIL: no fixed kernel computes x*w + bias:", *k, sep="\n  ")
    sys.exit(1)
answer = kcheck.answer_text()
nums = set(re.findall(r"\d+", answer))
if not {str(truth["wrong_cols_first"]), str(truth["wrong_cols_last"])} <= nums:
    print("FAIL: the answer does not name the wrong elements (columns 32..63 of every row)")
    sys.exit(1)
print(f"PASS: {path.name}:{k.name} fixed; answer names 32..63")
PY
