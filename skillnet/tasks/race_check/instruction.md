Before I run my kernel `tile_reverse.py` on our H100, I want to know whether it can race or
deadlock. Each block of 128 threads should reverse its 128-element tile of `a` into `out`,
multiplied by `scale`.

Check it. If there is a synchronization or memory-race problem, tell me what it is and where,
fix it without changing the kernel's parameters or what it computes, and give me the fixed
kernel file. Confirm the fixed version is both safe and numerically correct.

---
Your files are in `/task/inputs`. Save every file you produce in `/task/outputs`, and write your final answer to the user in `/task/answer.txt`.
