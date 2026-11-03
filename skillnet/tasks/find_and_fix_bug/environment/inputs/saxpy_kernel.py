"""out[i] = alpha * x[i] + y[i] for n = 300 float32 values on an H100."""
import numpy as np
import tirx_kernels.tirx_lite as txl

N = 300
WARPS = 4
THREADS = WARPS * 32
GRID = (N + THREADS - 1) // THREADS


@txl.kernel(warps=WARPS, arch="sm_90a", grid=GRID)
def saxpy(x: txl.gptr(txl.f32), y: txl.gptr(txl.f32), out: txl.gptr(txl.f32), alpha: txl.f32, n: txl.i32):
    i = txl.cta_id() * 64 + txl.thread_id()
    xv = txl.local_scalar("float32")
    yv = txl.local_scalar("float32")
    with txl.If(i < n), txl.Then():
        txl.ptx.ld.global_.f32(xv, x.ptr_to([i]))
        txl.ptx.ld.global_.f32(yv, y.ptr_to([i]))
        txl.ptx.st.global_.f32(out.ptr_to([i]), alpha * xv + yv)


def make_inputs():
    rng = np.random.default_rng(1)
    return {
        "x": rng.standard_normal(N).astype(np.float32),
        "y": rng.standard_normal(N).astype(np.float32),
        "out": np.zeros(N, np.float32),
        "alpha": np.float32(1.5),
        "n": N,
    }


def reference(inputs):
    return {"out": inputs["alpha"] * inputs["x"] + inputs["y"]}
