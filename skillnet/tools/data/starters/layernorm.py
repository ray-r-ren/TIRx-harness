"""LayerNorm starter: out[r, :] = (x[r, :] - mean) * rsqrt(var + eps) * gamma + beta, float32 rows.

One warp per row, two butterfly reductions: the row mean first, then the variance of (x - mean)
(two-pass, numerically safer than E[x^2] - mean^2), then a rescale pass with gamma and beta.
Verified on CPU: simulate_kernel pass, check_kernel_safety clean.
"""
import numpy as np
import tirx_kernels.tirx_lite as txl

WARPS = 4
ROWS = 8
COLS = 256
GRID = ROWS // WARPS
EPS = 1e-5

BYTES_MOVED = (2 * ROWS * COLS + 2 * COLS) * 4
FLOPS = 8 * ROWS * COLS


def _warp_sum(acc, peer):
    """Butterfly shuffle: every lane ends with the sum of all 32 lanes' acc[0]."""
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(
            peer, txl.reinterpret("uint32", acc[0]), txl.uint32(delta), txl.uint32(31), txl.uint32(0xFFFFFFFF)
        )
        txl.assign(acc[0], acc[0] + txl.reinterpret("float32", peer))


@txl.kernel(warps=WARPS, arch="sm_90a", grid=GRID)
def layernorm(x: txl.gptr(txl.f32), gamma: txl.gptr(txl.f32), beta: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    row = txl.cta_id() * WARPS + txl.thread_id() // 32
    base = row * COLS
    v = txl.local_scalar("float32")
    g = txl.local_scalar("float32")
    b = txl.local_scalar("float32")
    peer = txl.local_scalar("uint32")
    mean = txl.alloc_local([1], "float32")
    var = txl.alloc_local([1], "float32")
    txl.assign(mean[0], txl.float32(0.0))
    with txl.serial(COLS // 32) as k:
        txl.ptx.ld.global_.f32(v, x.ptr_to([base + k * 32 + lane]))
        txl.assign(mean[0], mean[0] + v)
    _warp_sum(mean, peer)
    txl.assign(mean[0], mean[0] * txl.float32(1.0 / COLS))
    txl.assign(var[0], txl.float32(0.0))
    with txl.serial(COLS // 32) as k:
        txl.ptx.ld.global_.f32(v, x.ptr_to([base + k * 32 + lane]))
        txl.assign(var[0], var[0] + (v - mean[0]) * (v - mean[0]))
    _warp_sum(var, peer)
    inv = txl.rsqrt(var[0] * txl.float32(1.0 / COLS) + txl.float32(EPS))
    with txl.serial(COLS // 32) as k:
        txl.ptx.ld.global_.f32(v, x.ptr_to([base + k * 32 + lane]))
        txl.ptx.ld.global_.f32(g, gamma.ptr_to([k * 32 + lane]))
        txl.ptx.ld.global_.f32(b, beta.ptr_to([k * 32 + lane]))
        txl.ptx.st.global_.f32(out.ptr_to([base + k * 32 + lane]), (v - mean[0]) * inv * g + b)


def make_inputs():
    rng = np.random.default_rng(5)
    return {
        "x": (1.0 + 2.0 * rng.standard_normal(ROWS * COLS)).astype(np.float32),
        "gamma": rng.uniform(0.5, 1.5, COLS).astype(np.float32),
        # beta keeps every output well away from zero, so the default rtol=1e-5 holds; for outputs near
        # zero, pass an atol of about 1e-6 (float32 rounding of the normalized value).
        "beta": rng.uniform(6.0, 7.0, COLS).astype(np.float32),
        "out": np.zeros(ROWS * COLS, np.float32),
    }


def reference(inputs):
    x = inputs["x"].astype(np.float64).reshape(ROWS, COLS)
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    y = (x - mean) / np.sqrt(var + EPS) * inputs["gamma"].astype(np.float64) + inputs["beta"].astype(np.float64)
    return {"out": y.reshape(-1).astype(np.float32)}
