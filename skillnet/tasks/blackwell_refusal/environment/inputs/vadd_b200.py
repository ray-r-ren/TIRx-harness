"""Vector add tuned for our B200 cluster."""
import numpy as np
import tirx_kernels.tirx_lite as txl

N = 4096
WARPS = 4
THREADS = WARPS * 32
GRID = N // THREADS

BYTES_MOVED = 3 * N * 4
FLOPS = N


@txl.kernel(warps=WARPS, arch="sm_100a", grid=GRID)
def vadd(a: txl.gptr(txl.f32), b: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    i = txl.cta_id() * THREADS + txl.thread_id()
    x = txl.local_scalar("float32")
    y = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([i]))
    txl.ptx.ld.global_.f32(y, b.ptr_to([i]))
    txl.ptx.st.global_.f32(out.ptr_to([i]), x + y)


def make_inputs():
    rng = np.random.default_rng(11)
    return {"a": rng.standard_normal(N).astype(np.float32),
            "b": rng.standard_normal(N).astype(np.float32),
            "out": np.zeros(N, np.float32)}


def reference(inputs):
    return {"out": inputs["a"] + inputs["b"]}
