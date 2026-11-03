"""Row-wise dot product: y[r] = sum_j A[r, j] * v[j] for a 64 x 512 float32 matrix (H100).

One warp per row. Each lane loads 16 values of the row and of v into registers (unrolled), multiplies,
then the warp reduces the partial sums with butterfly shuffles. Lane 0 writes the result.
"""
import numpy as np
import tirx_kernels.tirx_lite as txl

ROWS = 64
COLS = 512
PER_LANE = COLS // 32
FULL_MASK = 0xFFFFFFFF


@txl.kernel(warps=1, arch="sm_90a", grid=ROWS)
def row_dot(A: txl.gptr(txl.f32), v: txl.gptr(txl.f32), y: txl.gptr(txl.f32)):
    row = txl.cta_id()
    lane = txl.lane_id()
    a_reg = txl.alloc_local([PER_LANE], "float32")
    v_reg = txl.alloc_local([PER_LANE], "float32")
    acc = txl.alloc_local([1], "float32")
    peer = txl.local_scalar("uint32")
    with txl.unroll(PER_LANE) as k:
        txl.ptx.ld.global_.f32(a_reg[k], A.ptr_to([row * COLS + k * 32 + lane]))
        txl.ptx.ld.global_.f32(v_reg[k], v.ptr_to([k * 32 + lane]))
    txl.assign(acc[0], txl.float32(0.0))
    with txl.unroll(PER_LANE) as k:
        txl.assign(acc[0], acc[0] + a_reg[k] * v_reg[k])
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(
            peer, txl.reinterpret("uint32", acc[0]), txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK)
        )
        txl.assign(acc[0], acc[0] + txl.reinterpret("float32", peer))
    with txl.If(lane == 0):
        with txl.Then():
            txl.ptx.st.global_.f32(y.ptr_to([row]), acc[0])


def make_inputs():
    rng = np.random.default_rng(11)
    return {"A": rng.standard_normal(ROWS * COLS).astype(np.float32),
            "v": rng.standard_normal(COLS).astype(np.float32),
            "y": np.zeros(ROWS, np.float32)}


def reference(inputs):
    A = inputs["A"].reshape(ROWS, COLS).astype(np.float64)
    return {"y": (A @ inputs["v"].astype(np.float64)).astype(np.float32)}
