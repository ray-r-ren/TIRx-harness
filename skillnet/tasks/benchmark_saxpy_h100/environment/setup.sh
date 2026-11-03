#!/bin/bash
set -e
# Input: an H100 (sm_90a) SAXPY kernel over 16384 values. Truth: its parameters and size.
set -euo pipefail
mkdir -p /task/inputs /task/truth
cp /task/saxpy.py /task/inputs/saxpy.py
printf '{"n": 16384, "gpu": "H100"}\n' > /task/truth/expected.json
