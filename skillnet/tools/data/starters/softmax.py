"""Softmax starter: row-wise softmax of a ROWS x COLS float32 matrix.

One warp per row, three passes over the row: max-reduce (butterfly shuffle), sum of exp(x - max)
(butterfly shuffle), then out = exp(x - max) / sum. Subtracting the row max keeps exp() from
overflowing. Verified on CPU: simulate_kernel pass, check_kernel_safety clean.
"""
import numpy as np
import tirx_kernels.tirx_lite as txl

WARPS = 4
ROWS = 8
COLS = 128
GRID = ROWS // WARPS

BYTES_MOVED = 2 * ROWS * COLS * 4
FLOPS = 5 * ROWS * COLS


def _warp_allreduce(acc, peer, op):
    """Butterfly shuffle: every lane ends with op() over all 32 lanes' acc[0]."""
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(
            peer, txl.reinterpret("uint32", acc[0]), txl.uint32(delta), txl.uint32(31), txl.uint32(0xFFFFFFFF)
        )
        txl.assign(acc[0], op(acc[0], txl.reinterpret("float32", peer)))


@txl.kernel(warps=WARPS, arch="sm_90a", grid=GRID)
def softmax(x: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    row = txl.cta_id() * WARPS + txl.thread_id() // 32
    base = row * COLS
    v = txl.local_scalar("float32")
    peer = txl.local_scalar("uint32")
    mx = txl.alloc_local([1], "float32")
    sm = txl.alloc_local([1], "float32")
    # Pass 1: row max.
    txl.assign(mx[0], txl.float32(-3.0e38))
    with txl.serial(COLS // 32) as k:
        txl.ptx.ld.global_.f32(v, x.ptr_to([base + k * 32 + lane]))
        txl.assign(mx[0], txl.max(mx[0], v))
    _warp_allreduce(mx, peer, txl.max)
    # Pass 2: sum of exp(x - max).
    txl.assign(sm[0], txl.float32(0.0))
    with txl.serial(COLS // 32) as k:
        txl.ptx.ld.global_.f32(v, x.ptr_to([base + k * 32 + lane]))
        txl.assign(sm[0], sm[0] + txl.exp(v - mx[0]))
    _warp_allreduce(sm, peer, lambda a, b: a + b)
    # Pass 3: normalize.
    with txl.serial(COLS // 32) as k:
        txl.ptx.ld.global_.f32(v, x.ptr_to([base + k * 32 + lane]))
        txl.ptx.st.global_.f32(out.ptr_to([base + k * 32 + lane]), txl.exp(v - mx[0]) / sm[0])


def make_inputs():
    rng = np.random.default_rng(3)
    return {
        "x": (2.0 * rng.standard_normal(ROWS * COLS)).astype(np.float32),
        "out": np.zeros(ROWS * COLS, np.float32),
    }


def reference(inputs):
    x = inputs["x"].astype(np.float64).reshape(ROWS, COLS)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return {"out": (e / e.sum(axis=1, keepdims=True)).reshape(-1).astype(np.float32)}
