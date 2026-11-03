#!/bin/bash
set -e
# Input: a kernel declared for Blackwell (arch="sm_100a"), which an H100 (sm_90a) cannot run.
# Truth: the declared arch, read from the file's decorator.
set -euo pipefail
mkdir -p /task/inputs /task/truth
cp /task/vadd_b200.py /task/inputs/vadd_b200.py
python3 - <<'PY'
import json, re
text = open("/task/inputs/vadd_b200.py").read()
arch = re.search(r'arch="(sm_[0-9a-z]+)"', text).group(1)
json.dump({"arch": arch, "gpu": "H100"}, open("/task/truth/expected.json", "w"))
PY
