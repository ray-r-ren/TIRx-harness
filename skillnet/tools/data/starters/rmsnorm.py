"""RMSNorm starter: out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w over float32 rows.

One warp per row: a strided sum of squares, a butterfly shuffle to share it across the warp, then
a second pass that rescales and applies the weight. Uses txl.rsqrt (simulated) rather than sqrt.
Verified on CPU: simulate_kernel pass, check_kernel_safety clean.
"""
import numpy as np
import tirx_kernels.tirx_lite as txl

WARPS = 4
ROWS = 8
COLS = 256
GRID = ROWS // WARPS
EPS = 1e-6

BYTES_MOVED = (2 * ROWS * COLS + COLS) * 4
FLOPS = 4 * ROWS * COLS


@txl.kernel(warps=WARPS, arch="sm_90a", grid=GRID)
def rmsnorm(x: txl.gptr(txl.f32), w: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    row = txl.cta_id() * WARPS + txl.thread_id() // 32
    base = row * COLS
    v = txl.local_scalar("float32")
    g = txl.local_scalar("float32")
    peer = txl.local_scalar("uint32")
    acc = txl.alloc_local([1], "float32")
    txl.assign(acc[0], txl.float32(0.0))
    with txl.serial(COLS // 32) as k:
        txl.ptx.ld.global_.f32(v, x.ptr_to([base + k * 32 + lane]))
        txl.assign(acc[0], acc[0] + v * v)
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(
            peer, txl.reinterpret("uint32", acc[0]), txl.uint32(delta), txl.uint32(31), txl.uint32(0xFFFFFFFF)
        )
        txl.assign(acc[0], acc[0] + txl.reinterpret("float32", peer))
    inv = txl.rsqrt(acc[0] * txl.float32(1.0 / COLS) + txl.float32(EPS))
    with txl.serial(COLS // 32) as k:
        txl.ptx.ld.global_.f32(v, x.ptr_to([base + k * 32 + lane]))
        txl.ptx.ld.global_.f32(g, w.ptr_to([k * 32 + lane]))
        txl.ptx.st.global_.f32(out.ptr_to([base + k * 32 + lane]), v * inv * g)


def make_inputs():
    rng = np.random.default_rng(4)
    return {
        "x": rng.standard_normal(ROWS * COLS).astype(np.float32),
        "w": rng.uniform(0.5, 1.5, COLS).astype(np.float32),
        "out": np.zeros(ROWS * COLS, np.float32),
    }


def reference(inputs):
    x = inputs["x"].astype(np.float64).reshape(ROWS, COLS)
    w = inputs["w"].astype(np.float64)
    inv = 1.0 / np.sqrt((x * x).mean(axis=1, keepdims=True) + EPS)
    return {"out": (x * inv * w).reshape(-1).astype(np.float32)}
