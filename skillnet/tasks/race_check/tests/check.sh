#!/bin/bash
# Pass when (1) the answer reports the race / missing barrier and (2) a kernel in /task/outputs with the
# same parameters is clean under Synccheck and Racecheck and reverses fresh tiles correctly in NumSim.
set -uo pipefail
cd /task
. /task/numsim_env.sh
nspython - <<'PY'
import json, re, sys
import numpy as np
sys.path.insert(0, "/task")
import kcheck

truth = json.load(open("/task/truth/verdict.json"))
case = np.load("/task/truth/case.npz")
answer = kcheck.answer_text()
if not re.search(r"rac(e|es|ing|y)\b|barrier|bar\.sync|__syncthreads", answer, re.I):
    print(f"FAIL: the answer does not report the race (racecheck on the original: {truth['racecheck']})")
    sys.exit(1)
def make_inputs():
    return {"a": case["a"].copy(), "out": np.zeros(256, np.float32), "scale": np.float32(-0.5)}
path, k = kcheck.first_matching("/task/outputs", ["a", "out", "scale"], make_inputs, {"out": case["out"]},
                                rtol=1e-6, atol=1e-7, need_clean=True, forbid_arch_prefix="sm_10")
if path is None:
    print("FAIL: no fixed kernel that is clean and correct:", *k, sep="\n  ")
    sys.exit(1)
print(f"PASS: {path.name} is race-free and correct; answer reports the race")
PY
