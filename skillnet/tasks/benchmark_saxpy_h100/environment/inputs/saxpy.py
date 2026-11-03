"""SAXPY: out = alpha * x + y over 16384 float32 values."""
import numpy as np
import tirx_kernels.tirx_lite as txl

WARPS = 4
THREADS = WARPS * 32
N = 16384
GRID = (N + THREADS - 1) // THREADS

BYTES_MOVED = 3 * N * 4
FLOPS = 2 * N


@txl.kernel(warps=WARPS, arch="sm_90a", grid=GRID)
def saxpy(x: txl.gptr(txl.f32), y: txl.gptr(txl.f32), out: txl.gptr(txl.f32), alpha: txl.f32, n: txl.i32):
    i = txl.cta_id() * THREADS + txl.thread_id()
    xv = txl.local_scalar("float32")
    yv = txl.local_scalar("float32")
    with txl.If(i < n), txl.Then():
        txl.ptx.ld.global_.f32(xv, x.ptr_to([i]))
        txl.ptx.ld.global_.f32(yv, y.ptr_to([i]))
        txl.ptx.st.global_.f32(out.ptr_to([i]), alpha * xv + yv)


def make_inputs():
    rng = np.random.default_rng(1)
    return {
        # Positive values keep alpha*x + y away from zero, so the default rtol holds whether or not the
        # compiler fuses the multiply-add (an FMA rounds once, the reference rounds once at the end).
        "x": rng.uniform(0.5, 1.5, N).astype(np.float32),
        "y": rng.uniform(0.5, 1.5, N).astype(np.float32),
        "out": np.zeros(N, np.float32),
        "alpha": np.float32(2.5),
        "n": N,
    }


def reference(inputs):
    x = inputs["x"].astype(np.float64)
    y = inputs["y"].astype(np.float64)
    return {"out": (float(inputs["alpha"]) * x + y).astype(np.float32)}
