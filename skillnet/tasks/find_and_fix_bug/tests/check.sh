#!/bin/bash
# Pass when (1) a kernel in /task/outputs with parameters (x, y, out, alpha, n) computes alpha*x + y for
# fresh inputs in the simulator, and (2) the answer names the first wrong index of the original kernel.
set -uo pipefail
cd /task
. /task/numsim_env.sh
nspython - <<'PY'
import json, sys
import numpy as np
sys.path.insert(0, "/task")
import kcheck

case = np.load("/task/truth/case.npz")
truth = json.load(open("/task/truth/wrong.json"))
def make_inputs():
    return {"x": case["x"].copy(), "y": case["y"].copy(), "out": np.zeros(300, np.float32),
            "alpha": np.float32(-0.75), "n": 300}
path, k = kcheck.first_matching("/task/outputs", ["x", "y", "out", "alpha", "n"], make_inputs,
                                {"out": case["out"]}, rtol=1e-5, atol=1e-6, forbid_arch_prefix="sm_10")
if path is None:
    print("FAIL: no fixed kernel computes alpha*x + y:", *k, sep="\n  ")
    sys.exit(1)
answer = kcheck.answer_text()
if str(truth["wrong_first"]) not in answer:
    print(f"FAIL: the answer does not name the wrong elements (expected {truth['wrong_first']}..{truth['wrong_last']})")
    sys.exit(1)
print(f"PASS: {path.name} fixed; answer names index {truth['wrong_first']}")
PY
