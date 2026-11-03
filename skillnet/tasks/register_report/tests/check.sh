#!/bin/bash
# Pass when the answer's JSON gives the exact sm_90a register count, spill bytes and static shared memory.
set -uo pipefail
cd /task
python - <<'PY'
import json, sys
sys.path.insert(0, "/task")
import kcheck

truth = json.load(open("/task/truth/resources.json"))
ans = kcheck.answer_json()
if not isinstance(ans, dict):
    print("FAIL: no JSON object with the numbers in the answer")
    sys.exit(1)
bad = []
for key, want in truth.items():
    try:
        got = int(ans.get(key))
    except (TypeError, ValueError):
        got = None
    if got != want:
        bad.append(f"{key}: answer {ans.get(key)!r}, ptxas {want}")
if bad:
    print("FAIL:", *bad, sep="\n  ")
    sys.exit(1)
print(f"PASS: {truth}")
PY
