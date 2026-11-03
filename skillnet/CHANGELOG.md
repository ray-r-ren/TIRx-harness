## v2
- Paste kernel code straight into any check, compile, export or benchmark step; you no longer need to upload a file first. Reported line numbers match the code exactly as you pasted it.
- Seven ready-made H100 starter kernels: vector add, SAXPY, row sums, softmax, RMSNorm, LayerNorm and a tiled matrix multiply. Each one is already checked correct and race-free, and you can ask for one by name.
- Plain words now find the right starter: "vector add", "sum", "matrix multiply" or "gemm", "layer norm", "rms norm".
- Benchmarks run on the skill's NVIDIA H100. Kernels written only for Blackwell GPUs are refused up front with a clear explanation, so you never get a made-up timing.
- Every example kernel the skill hands back now loads on its own. Kernels that import helpers from other files are flagged with the exact line to fix.
- Re-checking the same kernel is recognised, so you can confirm your final code is exactly the code that passed.

## v1
- First version
