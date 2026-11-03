# GPU Kernel Forge

Describe the math. Get a GPU kernel that's verified correct and tuned for speed.

Kernels are written as one TIRx-lite Python file (NVIDIA GPUs). The skill loads it, checks its
numbers on a CPU simulator against an independent NumPy reference, checks it for deadlocks and data
races, shows the generated PTX/SASS and register report, exports standalone CUDA, and, on the skill's
H100 GPU, times it against a baseline.

## Where to start

Call `find_kernel_examples` with the math the user described (`query="rmsnorm"`, or `starter="rmsnorm"` for that starter exactly), then pass the starter's `source` text, adapted to the user's math, as `source=` to `inspect_kernel`.

- Seven verified H100 (`sm_90a`) starters ship with the skill: `elementwise` (vector add),
  `saxpy`, `reduction` (row sums), `softmax`, `rmsnorm`, `layernorm` and `matmul` (tiled, shared
  memory, tail-guarded; a correctness starter, not a fast GEMM). Each simulates `pass` and
  checks `clean`. Plain words find them: "vector add", "sum", "gemm", "matrix multiply",
  "layer norm", "rms norm".
- Collection kernels (`tirx_kernels/...`) are mostly Blackwell (`sm_100a`) ports with PyTorch host
  code: read them for idioms only, don't pass them to the other tools.
- Every kernel tool takes the kernel as `source=` (the Python code as text, up to 200,000
  characters) **or** `kernel_file=` (an uploaded `file_id` or https URL), never both. Passing both
  is `ambiguous_kernel_input`; neither is `missing_kernel`. For a file on the user's computer,
  `upload_file` it, or read it and pass the text. Results come back as download links in
  `outputs`; download the files the user needs.
- Idioms that are verified to work (loops, branches, warp shuffles, shared memory, fp16 loads) are in
  [references/writing-kernels.md](references/writing-kernels.md). Read it before writing a kernel.

## Kernel file contract

One standalone `.py` file (or the same text passed as `source`) holding:

1. `import numpy as np` and `import tirx_kernels.tirx_lite as txl`.
2. One or more `@txl.kernel(warps=W, arch="sm_90a", grid=G)` functions. `grid` must be a constant
   (int or tuple) for the simulator and checkers. Parameter annotations are live expressions:
   `txl.gptr(txl.f32)` for global arrays, `txl.f32` / `txl.i32` for scalars. Never add
   `from __future__ import annotations` (it turns annotations into strings and tracing fails).
3. `make_inputs()` returning a dict keyed by the kernel's parameter names: every input array with the
   exact dtype (`np.float32` for `f32`, `np.float16` for `f16`), every output buffer zero-filled, and
   every scalar (`np.float32(1.5)`, a plain int for `i32`).
4. `reference(inputs)` returning a dict `{output_name: expected array}` computed with plain NumPy,
   independently of the kernel (never by re-implementing the kernel's indexing).
5. Optional module constants `BYTES_MOVED` and `FLOPS` (per launch) so `benchmark_kernel` can report
   GB/s and TFLOP/s.
6. No relative imports (`from .helpers import x`) and no imports of the user's other files: the
   tools refuse them with `relative_import` naming the line. Inline the helper or import it from
   its absolute `tirx_kernels.<package>` path.

```python
import numpy as np
import tirx_kernels.tirx_lite as txl

N = 32

@txl.kernel(warps=1, arch="sm_90a", grid=1)
def vector_add(a: txl.gptr(txl.f32), b: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32")
    y = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([lane]))
    txl.ptx.ld.global_.f32(y, b.ptr_to([lane]))
    txl.ptx.st.global_.f32(out.ptr_to([lane]), x + y)

def make_inputs():
    rng = np.random.default_rng(0)
    return {"a": rng.standard_normal(N).astype(np.float32),
            "b": rng.standard_normal(N).astype(np.float32),
            "out": np.zeros(N, np.float32)}

def reference(inputs):
    return {"out": inputs["a"] + inputs["b"]}
```

To test the same kernel at another size without editing it, pass a separate `case_file` holding only
`make_inputs()`/`reference()` (and `BYTES_MOVED`/`FLOPS`). The kernel's `grid` must still cover that size.

## Stages

1. **Find** — `find_kernel_examples(query=<the math>)`, or `starter=<name>` to fetch one starter
   exactly. Start from the top `starter/` match: its `source` field holds the full text (top 2
   starters) and the same file is in `outputs`. For a specific instruction (`"shfl"`,
   `"mbarrier"`, `"ld.global"`), use `scope="reference"`; the simulator's supported-operations
   table is among the reference results.
2. **Write & inspect** — adapt the starter (or write your own), then call
   `inspect_kernel(source=<the code>)` (or `kernel_file=` for an uploaded file). Check `params`
   (names, dtypes), `grid`, `warps`, `arch`, `blackwell_only` and `case_check` (are
   `make_inputs()`/`reference()` present and matching the parameters). On `kernel_trace_failed`, fix
   the reported line and call again. If the file has several kernels, pass `kernel_name` from then on.
3. **Safety** — `check_kernel_safety` (both checkers). `overall`:
   - `clean` → go on.
   - `error` → a deadlock or data race was found; fix it before anything else (a racy kernel's
     numbers mean nothing). Findings name the kind (`deadlock`, `data_race`, …) and kernel lines.
   - `review` → read each finding; e.g. `uninitialized_read` usually means a missing barrier or a read
     of memory nobody wrote.
   - `incomplete` (status `partial`) → an operation the checkers can't model (`reason` names it);
     the kernel is not verified. Replace the operation (see Pitfalls) or tell the user.
4. **Correctness** — `simulate_kernel` (default `rtol=1e-5`, `atol=1e-8`). `verdict`:
   - `pass` → every output matched the reference.
   - `fail` → read `outputs[].mismatches`, `first_mismatches` (index, got, expected) and
     `mismatch_ranges`; the index pattern locates the bug (see examples). A `fault` such as
     `out_of_bounds` means the kernel touched memory outside a buffer.
   - `incomplete` (status `partial`) → the simulator stopped on an unsupported operation.
   Pass the returned `state_id` to the next `simulate_kernel` / `check_kernel_safety` call so
   compiled simulator work is reused. After every edit, repeat stages 2–4.
5. **Inspect generated code** (optional) — `compile_kernel(arch=...)` when the user asks about
   registers, spills, shared memory, PTX or SASS, or before tuning. `stages="sass"` (default) gives
   the register/spill report and all three files.
6. **Tune & benchmark** (only when speed is asked) — `benchmark_kernel(source=<new>,
   baseline_source=<previous verified version>)` (or `kernel_file`/`baseline_file`). It runs only
   on the skill's NVIDIA H100 (`sm_90a`). Before any device work it refuses with
   `arch_unsupported`, naming the kernel's arch and the H100's, any kernel or baseline that targets
   `sm_100a`+ or uses tcgen05/TMEM; then tell the user it can't be timed on the H100 (it still
   simulates, checks and compiles), or retarget to `arch="sm_90a"` without tcgen05 and try again.
   `sm_80`/`sm_89`/`sm_90` kernels run, compiled for `sm_90a` (noted in `uncertain`). It re-checks
   correctness on the GPU first. Keep a change only if `correct` is true and
   `faster_in_every_run` is true (speedup > 1 in every run). On `no_gpu`, tell the user the kernel
   was verified but not timed; never estimate speed from the simulator.
7. **Export** (optional) — `export_kernel` gives `<name>.cu` + `USAGE.md` and the verified kernel
   as `<name>.py`; with `build_library=true` also a `.so` for the chosen arch. Give the user the
   argument order from `args`.

## Examples that teach a decision

- **Lane-range bug.** `simulate_kernel` → `fail`, `mismatches: 16`, `mismatch_ranges: [[16, 31]]`,
  first mismatch index 16 with got 15.0, expected 17.0 on a 32-element add. Elements 0–15 are right
  and 16–31 are wrong, so look for a lane-dependent branch (`txl.Select(lane < 16, …)`,
  `txl.If(lane < 16)`) or an index that differs for the upper half-warp; got 15 vs 17 for an add
  hints at subtraction. Fix, then re-run `simulate_kernel` with the `state_id`.
- **Missing barrier.** `check_kernel_safety` → `overall: error`, racecheck `data_race` between a
  shared-memory store and a load on two lines of a 2-warp kernel. Add
  `txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))` between the store and the load, where
  `THREADS` is every thread of the CTA (warps × 32), then run the check again; it should be `clean`.
  A barrier whose count is larger than the threads that reach it is a `deadlock` instead.
- **Speed claim.** Two versions both pass stages 3–4. `benchmark_kernel(source=v2,
  baseline_source=v1, runs=3)` → `speedup: 1.18`, `faster_in_every_run: true`. Report "v2 is 1.18×
  faster (median X µs vs Y µs over 3 runs)". With `faster_in_every_run: false`, report no reliable gain.

## Pitfalls

- **Constant grid.** The simulator and checkers launch exactly `grid` CTAs; a grid computed at
  run time returns `dynamic_launch`. Fix it with a module constant sized for the test inputs
  (`GRID = (N + THREADS - 1) // THREADS`), and guard the tail with `txl.If(i < n)`.
- **Small shapes for simulation.** NumSim runs every thread on the CPU. Use tens to a few thousand
  elements in `make_inputs()`; time realistic sizes with `benchmark_kernel` and a `case_file`.
- **Exact dtypes.** `make_inputs()` arrays must have the parameter's dtype (float64 for an `f32`
  parameter → `invalid_inputs`). Use `inspect_kernel`'s `case_check` to catch this early.
- **No `.f16` loads.** `txl.ptx.ld.global_` takes `b16/u16/s16/b32/u32/s32/f32/f64/…` but not
  `f16`; load fp16 as `b16` into a `uint16` scalar and `txl.reinterpret("float16", h)`
  (see references/writing-kernels.md).
- **Unsimulated operations.** `txl.sqrt` and `txl.tanh` make `simulate_kernel` return
  `incomplete`; use `a * txl.rsqrt(a)` and an `exp`-based tanh (references/writing-kernels.md).
  `txl.cuda.func_call` with your own CUDA body isn't modeled either: the checkers return
  `incomplete`. Write the math with supported TIRx-lite expressions.
- **Hopper tensor cores.** The simulator models the SM100 target: WGMMA (`wgmma_*`) operations are
  rejected. sm_90a kernels that use WGMMA can still go through `compile_kernel` and
  `benchmark_kernel`, but not the simulator or checkers; say so.
- **Blackwell on H100.** `sm_100a` kernels (tcgen05, TMEM) simulate, check and compile on CPU but
  `benchmark_kernel` returns `arch_unsupported` before timing. Never report a latency for them.
- **Line numbers follow the text you sent.** With `source=`, findings, trace errors and
  `relative_import` lines count lines of that exact text (nothing is prepended). Edit the same
  text and resend it.
- **Relative imports are refused.** A kernel that does `from .x import y` returns
  `relative_import`: inline `y` into the kernel text. Kernel copies from `find_kernel_examples`
  already have such imports rewritten to absolute `tirx_kernels.…` ones.
- **Tolerances.** The starters (elementwise, reductions, `exp`, `rsqrt`) pass at the defaults with a
  float64 reference cast to float32 at the end. fp16/bf16 outputs keep only about 3 (fp16) or 2
  (bf16) significant digits, so loosen `rtol`/`atol` on purpose and tell the user the values used.
  A tiny `max_rel_err` (around 1e-6) on a reduction is rounding order, not a bug.
- **Simulator time is not GPU time.** `simulate_seconds` is CPU cost only. Speed comes only from
  `benchmark_kernel`; compare numbers only from the same call (same GPU, same settings).
- **Clean means these inputs.** Checker and simulator verdicts cover the given launch and inputs; a
  different shape needs its own run.
- **Collection kernels need PyTorch.** Files under `tirx_kernels/` import torch for host code; the
  skill doesn't run them. Use their kernel bodies as patterns inside your own file.

## Evidence

- Correctness: `verdict`, per-output `max_abs_err`, `max_rel_err`, `mismatches` and the
  `rtol`/`atol` used (`simulate_kernel`); attach `numsim_report.json` when the user wants detail.
- Safety: `overall` plus each checker's `verdict` and finding kinds/lines (`check_kernel_safety`);
  attach `safety_report.txt`.
- Resources: `registers`, `spill_stores`/`spill_loads`, static/dynamic shared memory from
  `compile_kernel`; attach the `.ptx`/`.sass` only when asked.
- Speed: `device`, `median_us`, `min_us`, `per_run_median_us`, `speedup`, `faster_in_every_run`,
  `bandwidth_gbs`/`tflops` (`benchmark_kernel`); attach `benchmark_report.json`.
- Always give the user the final kernel `.py` (the code you verified: the `source` text you sent, or
  `export_kernel`'s `<name>.py`).
- State the limits: verdicts hold for the tested inputs; timing exists only if `benchmark_kernel` ran.

## Done when

- `inspect_kernel` succeeds on the final kernel, `check_kernel_safety` returns `overall: clean` and
  `simulate_kernel` returns `verdict: pass`, all on that same final text (`same_kernel_as_before`
  confirms the same text when you pass the `state_id`);
- if speed was asked: `benchmark_kernel` returned `correct: true` with `median_us` (and `speedup`
  against the baseline), or you told the user why it couldn't be timed (`no_gpu`, `arch_unsupported`);
- the user has the kernel file and, if asked, the export or the register report.

## Can't do

- Time or run Blackwell-only kernels (sm_100a, tcgen05/TMEM) on the H100 → simulate, check and
  compile them on CPU; retarget to sm_90a for timing.
- Simulate opaque CUDA code, WGMMA, or operations missing from the simulator's table → rewrite them
  with TIRx-lite expressions; check coverage with `find_kernel_examples(scope="reference")`.
- Treat simulator time as GPU latency → only `benchmark_kernel` measures speed.
- Multi-GPU or distributed kernels → one kernel launch on one GPU only.
- Kernels written in CUDA C++, Triton or other languages → port the logic to TIRx-lite, starting
  from the closest starter.
- Run on remote GPUs (the user's machines, other GPU services) → timing uses only the skill's own GPU.
- Prove correctness for all inputs → test several shapes, including a tail that isn't a multiple
  of the block size, with separate `case_file`s.
- Run kernels that use relative imports or import other user files → put everything in one
  standalone kernel text.
