#!/bin/bash
set -e
# Oracle: compile for sm_90a with TIRx Harness and read ptxas's resource report.
set -euo pipefail
cd /tmp
python - <<'PY'
import importlib.util, json, os, pathlib, sys, tempfile
home = next(pathlib.Path(b) / "nvidia" / "cu13" for b in sys.path
            if (pathlib.Path(b) / "nvidia" / "cu13" / "bin").is_dir())
os.environ["CUDA_HOME"] = str(home)
import tvm
from tirx_harness.dump_kernel import dump_module

spec = importlib.util.spec_from_file_location("row_dot_solve", "/task/inputs/row_dot.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
with target:
    compiled = mod.row_dot.compile(target=target)
p = dump_module(compiled, arch="sm_90a", ptx=True, sass=True, outdir=tempfile.mkdtemp(), name="row_dot").ptxas
spill = (p.get("spill_stores") or 0) + (p.get("spill_loads") or 0)
out = {"registers": p["registers"], "spill_bytes": spill, "static_smem_bytes": p.get("smem") or 0}
open("/task/answer.txt", "w").write(
    f"On sm_90a ptxas reports {p['registers']} registers per thread, {spill} bytes of spills and "
    f"{out['static_smem_bytes']} bytes of static shared memory.\n{json.dumps(out)}\n")
PY
