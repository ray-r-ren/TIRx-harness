"""Matmul starter: C = A @ B for float32 matrices (A is M x K, B is K x N, all row-major).

A tiled SIMT GEMM: each CTA computes one TILE x TILE block of C, one element per thread. For each
K tile, the CTA stages a tile of A and a tile of B in shared memory (zero-filled past the matrix
edges), waits on a CTA barrier, accumulates the dot products, and waits again before the next
tile overwrites shared memory. M, N and K need not be multiples of TILE (tail guards).
This is a correctness starter, not a fast GEMM: it uses no tensor cores (WGMMA is not simulated).
Verified on CPU: simulate_kernel pass, check_kernel_safety clean.
"""
import numpy as np
import tirx_kernels.tirx_lite as txl

TILE = 16
THREADS = TILE * TILE  # 256 threads = 8 warps, one per C element of the tile
WARPS = THREADS // 32
M, N, K = 40, 24, 36
TILES_M = (M + TILE - 1) // TILE
TILES_N = (N + TILE - 1) // TILE
TILES_K = (K + TILE - 1) // TILE
GRID = TILES_M * TILES_N  # constant 1-D grid; the CTA index is split into (tile row, tile col)

BYTES_MOVED = (M * K + K * N + M * N) * 4
FLOPS = 2 * M * N * K


@txl.kernel(warps=WARPS, arch="sm_90a", grid=GRID)
def matmul(a: txl.gptr(txl.f32), b: txl.gptr(txl.f32), c: txl.gptr(txl.f32)):
    tid = txl.thread_id()
    ty = tid // TILE
    tx = tid % TILE
    row = (txl.cta_id() // TILES_N) * TILE + ty
    col = (txl.cta_id() % TILES_N) * TILE + tx
    smem = txl.smem_pool()
    a_tile = smem.alloc([THREADS], txl.f32)
    b_tile = smem.alloc([THREADS], txl.f32)
    v = txl.local_scalar("float32")
    av = txl.local_scalar("float32")
    bv = txl.local_scalar("float32")
    stage = txl.alloc_local([1], "float32")
    acc = txl.alloc_local([1], "float32")
    txl.assign(acc[0], txl.float32(0.0))
    with txl.serial(TILES_K) as kt:
        # Stage A[row, kt*TILE + tx] (zero past the edges).
        txl.assign(stage[0], txl.float32(0.0))
        with txl.If(row < M), txl.Then():
            with txl.If(kt * TILE + tx < K), txl.Then():
                txl.ptx.ld.global_.f32(v, a.ptr_to([row * K + kt * TILE + tx]))
                txl.assign(stage[0], v)
        txl.ptx.st.shared.f32(a_tile.ptr_to([ty * TILE + tx]), stage[0])
        # Stage B[kt*TILE + ty, col] (zero past the edges).
        txl.assign(stage[0], txl.float32(0.0))
        with txl.If(kt * TILE + ty < K), txl.Then():
            with txl.If(col < N), txl.Then():
                txl.ptx.ld.global_.f32(v, b.ptr_to([(kt * TILE + ty) * N + col]))
                txl.assign(stage[0], v)
        txl.ptx.st.shared.f32(b_tile.ptr_to([ty * TILE + tx]), stage[0])
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))  # tiles fully written before anyone reads
        with txl.unroll(TILE) as kk:
            txl.ptx.ld.shared.f32(av, a_tile.ptr_to([ty * TILE + kk]))
            txl.ptx.ld.shared.f32(bv, b_tile.ptr_to([kk * TILE + tx]))
            txl.assign(acc[0], acc[0] + av * bv)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))  # everyone done reading before the next tile
    with txl.If(row < M), txl.Then():
        with txl.If(col < N), txl.Then():
            txl.ptx.st.global_.f32(c.ptr_to([row * N + col]), acc[0])


def make_inputs():
    rng = np.random.default_rng(7)
    # Positive entries keep every dot product away from zero, so the default rtol=1e-5 holds.
    return {
        "a": rng.uniform(0.0, 1.0, M * K).astype(np.float32),
        "b": rng.uniform(0.0, 1.0, K * N).astype(np.float32),
        "c": np.zeros(M * N, np.float32),
    }


def reference(inputs):
    a = inputs["a"].astype(np.float64).reshape(M, K)
    b = inputs["b"].astype(np.float64).reshape(K, N)
    return {"c": (a @ b).reshape(-1).astype(np.float32)}
