"""Reduction starter: row sums of a ROWS x COLS float32 matrix (out[r] = sum(x[r, :])).

One warp per row: each lane adds a strided slice (coalesced loads), then a butterfly warp
shuffle combines the 32 partial sums; lane 0 stores the row's total. Adapt for max/min/mean by
changing the combine. Verified on CPU: simulate_kernel pass, check_kernel_safety clean.
"""
import numpy as np
import tirx_kernels.tirx_lite as txl

WARPS = 4
ROWS = 8
COLS = 256
GRID = ROWS // WARPS  # each CTA handles WARPS rows

BYTES_MOVED = (ROWS * COLS + ROWS) * 4
FLOPS = ROWS * COLS


@txl.kernel(warps=WARPS, arch="sm_90a", grid=GRID)
def row_sum(x: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    row = txl.cta_id() * WARPS + txl.thread_id() // 32
    v = txl.local_scalar("float32")
    acc = txl.alloc_local([1], "float32")
    txl.assign(acc[0], txl.float32(0.0))
    with txl.serial(COLS // 32) as k:
        txl.ptx.ld.global_.f32(v, x.ptr_to([row * COLS + k * 32 + lane]))
        txl.assign(acc[0], acc[0] + v)
    peer = txl.local_scalar("uint32")
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(
            peer, txl.reinterpret("uint32", acc[0]), txl.uint32(delta), txl.uint32(31), txl.uint32(0xFFFFFFFF)
        )
        txl.assign(acc[0], acc[0] + txl.reinterpret("float32", peer))
    with txl.If(lane == 0), txl.Then():
        txl.ptx.st.global_.f32(out.ptr_to([row]), acc[0])


def make_inputs():
    rng = np.random.default_rng(2)
    return {
        "x": rng.standard_normal(ROWS * COLS).astype(np.float32),
        "out": np.zeros(ROWS, np.float32),
    }


def reference(inputs):
    x = inputs["x"].astype(np.float64).reshape(ROWS, COLS)
    return {"out": x.sum(axis=1).astype(np.float32)}
