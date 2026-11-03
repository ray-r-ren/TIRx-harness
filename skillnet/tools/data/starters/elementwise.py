"""Elementwise starter: out = a + b over float32 vectors (vector add).

One element per thread, a 1-D grid of 4-warp CTAs and a tail guard, so N need not be a multiple
of the block size. Change the expression in the store for any other elementwise math
(scale, ReLU via txl.max, exp, ...). Verified on CPU: simulate_kernel pass, check_kernel_safety clean.
"""
import numpy as np
import tirx_kernels.tirx_lite as txl

WARPS = 4
THREADS = WARPS * 32
N = 300
GRID = (N + THREADS - 1) // THREADS  # constant: the simulator launches exactly this many CTAs

BYTES_MOVED = 3 * N * 4
FLOPS = N


@txl.kernel(warps=WARPS, arch="sm_90a", grid=GRID)
def elementwise_add(a: txl.gptr(txl.f32), b: txl.gptr(txl.f32), out: txl.gptr(txl.f32), n: txl.i32):
    i = txl.cta_id() * THREADS + txl.thread_id()
    x = txl.local_scalar("float32")
    y = txl.local_scalar("float32")
    with txl.If(i < n), txl.Then():
        txl.ptx.ld.global_.f32(x, a.ptr_to([i]))
        txl.ptx.ld.global_.f32(y, b.ptr_to([i]))
        txl.ptx.st.global_.f32(out.ptr_to([i]), x + y)


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "a": rng.standard_normal(N).astype(np.float32),
        "b": rng.standard_normal(N).astype(np.float32),
        "out": np.zeros(N, np.float32),
        "n": N,
    }


def reference(inputs):
    a = inputs["a"].astype(np.float64)
    b = inputs["b"].astype(np.float64)
    return {"out": (a + b).astype(np.float32)}
