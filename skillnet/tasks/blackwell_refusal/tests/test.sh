#!/bin/bash
# SkillNet task verifier (Harbor): put the ground truth in place, run the check, write the reward.
# The agent's /task/outputs and /task/answer.txt arrive as artifacts.
mkdir -p /logs/verifier /task/outputs
touch /task/answer.txt
rm -rf /task/truth && mkdir -p /task/truth
if [ -d /tests/truth ]; then cp -r /tests/truth/. /task/truth/ && rm -f /task/truth/.keep; fi
cd /task
# SkillNet's final checks run the code checks here and call the judge themselves (SKILLNET_CODE_CHECKS=1).
if [ -f /tests/judge.py ] && [ -z "$SKILLNET_CODE_CHECKS" ]; then
  python3 /tests/judge.py > /logs/verifier/check.log 2>&1
else
  if bash /tests/check.sh > /logs/verifier/check.log 2>&1; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
fi
