# Writing TIRx-lite kernels: verified idioms

Every snippet below comes from a kernel that passed `simulate_kernel` and `check_kernel_safety`
and compiled for `sm_90a`. Copy the shapes, change the math. Full files: the `starter/` matches
of `find_kernel_examples` (`starter=` one of `elementwise`, `saxpy`, `reduction`, `softmax`,
`rmsnorm`, `layernorm`, `matmul`).

## Kernel header and parameters

```python
import numpy as np
import tirx_kernels.tirx_lite as txl

WARPS = 4
THREADS = WARPS * 32
N = 300
GRID = (N + THREADS - 1) // THREADS        # constant: the simulator launches exactly this

@txl.kernel(warps=WARPS, arch="sm_90a", grid=GRID)
def saxpy(x: txl.gptr(txl.f32), y: txl.gptr(txl.f32), out: txl.gptr(txl.f32),
          alpha: txl.f32, n: txl.i32):
    ...
```

- Arrays: `txl.gptr(txl.f32)`, `txl.gptr(txl.f16)`, `txl.gptr(txl.i32)`, … (flat, 1-D indexing).
- Scalars: `txl.f32`, `txl.i32`. In `make_inputs()` give `np.float32(1.5)` / a plain `int`.
- A 2-D problem is a flat array plus `row * COLS + col` indexing.

## Thread indices

| Expression | Meaning |
| --- | --- |
| `txl.thread_id()` | thread index inside the CTA (0 … warps×32−1) |
| `txl.lane_id()` | lane inside the warp (0 … 31) |
| `txl.cta_id()` | CTA (block) index in the grid |
| `txl.cta_id() * THREADS + txl.thread_id()` | global element index for a 1-D grid |

## Registers

```python
v = txl.local_scalar("float32")          # destination register for loads
acc = txl.alloc_local([1], "float32")    # register you update in loops
txl.assign(acc[0], txl.float32(0.0))
txl.assign(acc[0], acc[0] + v)
```

Constants inside expressions need explicit types: `txl.float32(0.5)`, `txl.uint32(31)`.

## Global loads and stores

```python
txl.ptx.ld.global_.f32(v, x.ptr_to([i]))            # v = x[i]
txl.ptx.st.global_.f32(out.ptr_to([i]), alpha * v)  # out[i] = alpha * v
```

`ld.global_` accepts the types `b8 u8 s8 b16 u16 s16 b32 u32 s32 b64 u64 s64 f32 f64`. There
is no `.f16`: load half precision as 16 raw bits and reinterpret:

```python
h = txl.local_scalar("uint16")
txl.ptx.ld.global_.b16(h, x.ptr_to([i]))
val = txl.cast(txl.reinterpret("float16", h), "float32")
```

## Branches

```python
with txl.If(i < n), txl.Then():                  # tail guard: no access past n
    ...

with txl.If(v > txl.float32(0.0)):               # if / else
    with txl.Then():
        txl.ptx.st.global_.f32(out.ptr_to([i]), txl.exp(v))
    with txl.Else():
        txl.ptx.st.global_.f32(out.ptr_to([i]), txl.max(v, txl.float32(-0.5)))

r = txl.Select(cond, a, b)                       # branch-free choice
```

## Loops

```python
with txl.serial(COLS // 32) as k:   # run-time loop
    txl.ptx.ld.global_.f32(v, x.ptr_to([row * COLS + k * 32 + lane]))
    txl.assign(acc[0], acc[0] + v)

with txl.unroll(N // 32) as k:      # unrolled by the compiler
    ...

for delta in (16, 8, 4, 2, 1):      # plain Python loop: unrolled while tracing
    ...
```

Strided access `k * 32 + lane` keeps a warp's 32 loads on consecutive addresses (coalesced).

## Math

Simulated and compiled: `+ - * /`, comparisons, `txl.exp`, `txl.log`, `txl.rsqrt`, `txl.max`,
`txl.min`, `txl.fabs`, `txl.cast(x, "float32")`, `txl.reinterpret("uint32", f)` (same bits,
other type).

Not simulated (`simulate_kernel` returns `incomplete` naming `tirx.sqrt` / `tirx.tanh`): use
these verified rewrites instead.

| Instead of | Write | Note |
| --- | --- | --- |
| `txl.sqrt(a)` | `a * txl.rsqrt(a)` | needs `a > 0` (add a tiny epsilon if `a` can be 0) |
| `txl.tanh(v)` | `e = txl.exp(txl.float32(-2.0) * v)`; `(1 - e) / (1 + e)` | relative error ~1e-4 near 0: use `atol=1e-6` |

Check anything else against the simulator's supported-operations table
(`find_kernel_examples(query=..., scope="reference")`).

## Warp reduction (butterfly shuffle)

Every lane ends with the sum of all 32 lanes:

```python
peer = txl.local_scalar("uint32")
for delta in (16, 8, 4, 2, 1):
    txl.ptx.shfl_sync.bfly.b32(
        peer, txl.reinterpret("uint32", acc[0]), txl.uint32(delta), txl.uint32(31), txl.uint32(0xFFFFFFFF)
    )
    txl.assign(acc[0], acc[0] + txl.reinterpret("float32", peer))
with txl.If(lane == 0), txl.Then():
    txl.ptx.st.global_.f32(out.ptr_to([row]), acc[0])
```

The shuffle moves 32-bit raw values, hence the `reinterpret` on the way in and out.

## Shared memory and the CTA barrier

```python
buf = txl.smem_pool().alloc([THREADS], txl.f32)
txl.ptx.st.shared.f32(buf.ptr_to([tid]), x)
txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))   # every thread of the CTA, once
txl.ptx.ld.shared.f32(y, buf.ptr_to([(tid + 32) % THREADS]))
```

- Without the barrier, `check_kernel_safety` reports a `data_race` between the store and the load.
- The count must equal the threads that reach the barrier (warps × 32). A larger count is a
  `deadlock`: the barrier waits for threads that don't exist.
- Barrier inside a branch that only some warps take → those warps never arrive: also a deadlock.
- `compile_kernel` reports the allocation as `dynamic_smem_bytes` (256 for 64 floats).

## Patterns by job

| Job | Start from | Shape |
| --- | --- | --- |
| elementwise (add, scale, activation) | `elementwise`, `saxpy` | one element per thread, tail guard |
| row reduction (sum, max, mean) | `reduction` | one warp per row, strided loop + butterfly |
| softmax | `softmax` | max-reduce, `exp(x - max)`, sum-reduce, divide |
| normalization (RMSNorm, LayerNorm) | `rmsnorm`, `layernorm` | reduction pass(es), then rescale pass |
| matrix multiply (GEMM) | `matmul` | 16×16 tiles in shared memory, `bar.sync` before and after each K tile, zero-filled edges |
| data exchange between warps | `matmul` | shared memory + `bar.sync` |

All seven pass `simulate_kernel` at the default tolerances (float64 reference cast to float32) and
`check_kernel_safety` clean. `matmul` uses no tensor cores (WGMMA is not simulated): it shows the
tiling and barriers, not peak speed.
