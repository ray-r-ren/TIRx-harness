#!/bin/bash
set -e
# Input: an unrolled row-dot kernel for sm_90a. Truth: ptxas resource usage for sm_90a, computed with
# TIRx Harness's compile + dump_module directly (the NVIDIA pip toolkit provides ptxas).
set -euo pipefail
mkdir -p /task/inputs /task/truth
cp /task/row_dot.py /task/inputs/row_dot.py
cd /tmp
python - <<'PY'
import importlib.util, json, os, pathlib, sys, tempfile
home = next((pathlib.Path(b) / "nvidia" / "cu13" for b in sys.path
             if (pathlib.Path(b) / "nvidia" / "cu13" / "bin").is_dir()), None)
if home is None:
    sys.exit("setup: the NVIDIA pip toolkit (ptxas) is missing")
os.environ["CUDA_HOME"] = str(home)
import tvm
from tirx_harness.dump_kernel import dump_module

spec = importlib.util.spec_from_file_location("row_dot_task", "/task/inputs/row_dot.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
with target:
    compiled = mod.row_dot.compile(target=target)
res = dump_module(compiled, arch="sm_90a", ptx=True, sass=True, outdir=tempfile.mkdtemp(), name="row_dot")
p = dict(res.ptxas or {})
if p.get("registers") is None:
    sys.exit(f"setup: ptxas produced no resource report: {res.errors}")
truth = {"registers": int(p["registers"]),
         "spill_bytes": int(p.get("spill_stores") or 0) + int(p.get("spill_loads") or 0),
         "static_smem_bytes": int(p.get("smem") or 0)}
json.dump(truth, open("/task/truth/resources.json", "w"))
PY
