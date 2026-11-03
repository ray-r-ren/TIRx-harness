#!/bin/bash
# Needs the skill's H100. Pass when /task/outputs holds the benchmark report (JSON) from a run on an H100
# with correct device outputs and a positive median, and the answer names the H100 and quotes that median
# (within 10%) in microseconds.
set -uo pipefail
cd /task
python3 - <<'PY'
import json, pathlib, re, sys
report = None
for p in sorted(pathlib.Path("/task/outputs").rglob("*.json")):
    try:
        d = json.loads(p.read_text())
    except ValueError:
        continue
    if isinstance(d, dict) and "median_us" in d and "device" in d:
        report = d
        break
if report is None:
    print("FAIL: no benchmark report (JSON with median_us and device) in /task/outputs")
    sys.exit(1)
if "H100" not in str(report.get("device")):
    print(f"FAIL: the report's device is {report.get('device')!r}, not an H100")
    sys.exit(1)
if report.get("correct") is not True:
    print("FAIL: the report does not show correct device outputs")
    sys.exit(1)
med = float(report["median_us"])
if not med > 0:
    print("FAIL: no positive median latency in the report")
    sys.exit(1)
p = pathlib.Path("/task/answer.txt")
answer = p.read_text(errors="replace") if p.exists() else ""
if "h100" not in answer.lower():
    print("FAIL: the answer does not name the H100")
    sys.exit(1)
quoted = [float(x.replace(",", ".")) for x in
          re.findall(r"(\d+(?:[.,]\d+)?)\s*(?:µs|μs|us\b|usec|microsec)", answer, re.I)]
if not any(abs(q - med) <= 0.1 * med + 0.05 for q in quoted):
    print(f"FAIL: the answer does not quote the measured median ({med} us); found {quoted}")
    sys.exit(1)
print(f"PASS: H100 report with median {med} us, quoted in the answer")
PY
