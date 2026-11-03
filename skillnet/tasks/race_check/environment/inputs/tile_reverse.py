"""Block-wise reverse: each CTA of 128 threads reverses its 128-element tile of `a` into `out`.

out[c*128 + t] = a[c*128 + (127 - t)] * scale
"""
import numpy as np
import tirx_kernels.tirx_lite as txl

WARPS = 4
THREADS = WARPS * 32
TILES = 2
N = THREADS * TILES


@txl.kernel(warps=WARPS, arch="sm_90a", grid=TILES)
def tile_reverse(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32), scale: txl.f32):
    tid = txl.thread_id()
    base = txl.cta_id() * THREADS
    tile = txl.smem_pool().alloc([THREADS], txl.f32)
    x = txl.local_scalar("float32")
    y = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([base + tid]))
    txl.ptx.st.shared.f32(tile.ptr_to([tid]), x * scale)
    txl.ptx.ld.shared.f32(y, tile.ptr_to([THREADS - 1 - tid]))
    txl.ptx.st.global_.f32(out.ptr_to([base + tid]), y)


def make_inputs():
    rng = np.random.default_rng(5)
    return {"a": rng.standard_normal(N).astype(np.float32), "out": np.zeros(N, np.float32),
            "scale": np.float32(1.5)}


def reference(inputs):
    a = inputs["a"].reshape(TILES, THREADS)
    return {"out": (a[:, ::-1] * inputs["scale"]).reshape(-1).astype(np.float32)}
