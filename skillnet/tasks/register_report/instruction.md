I'm tuning my kernel `row_dot.py` for an H100. Before I change anything, tell me what the
compiler actually does with it on that GPU: how many registers per thread it uses, whether it
spills to local memory, and how much static shared memory it needs per block.

End your answer with a one-line JSON object like
`{"registers": 0, "spill_bytes": 0, "static_smem_bytes": 0}` with the real numbers
(`spill_bytes` = spill stores + spill loads in bytes).

---
Your files are in `/task/inputs`. Save every file you produce in `/task/outputs`, and write your final answer to the user in `/task/answer.txt`.
