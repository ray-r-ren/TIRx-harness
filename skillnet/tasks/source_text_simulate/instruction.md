I don't have this kernel as a file, only the code below. It's meant to compute
`out[r, c] = x[r, c] * w[c] + bias[c]` for an 8 × 64 float32 matrix `x` (row-major) on our H100,
but the numbers come out wrong.

Tell me exactly which output elements are wrong (rows/columns or flat index ranges), fix the kernel
without changing its parameters (`x`, `w`, `bias`, `out`), and save the fixed kernel as a `.py` file.
Make sure the fix actually works.

```python
import numpy as np
import tirx_kernels.tirx_lite as txl

ROWS = 8
COLS = 64
WARPS = 4
GRID = ROWS // WARPS


@txl.kernel(warps=WARPS, arch="sm_90a", grid=GRID)
def scale_bias(x: txl.gptr(txl.f32), w: txl.gptr(txl.f32), bias: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    row = txl.cta_id() * WARPS + txl.thread_id() // 32
    v = txl.local_scalar("float32")
    s = txl.local_scalar("float32")
    b = txl.local_scalar("float32")
    with txl.unroll(COLS // 32) as k:
        col = k * 32 + lane
        txl.ptx.ld.global_.f32(v, x.ptr_to([row * COLS + col]))
        txl.ptx.ld.global_.f32(s, w.ptr_to([lane]))
        txl.ptx.ld.global_.f32(b, bias.ptr_to([col]))
        txl.ptx.st.global_.f32(out.ptr_to([row * COLS + col]), v * s + b)


def make_inputs():
    rng = np.random.default_rng(3)
    return {
        "x": rng.uniform(0.5, 1.5, ROWS * COLS).astype(np.float32),
        "w": rng.uniform(0.5, 1.5, COLS).astype(np.float32),
        "bias": rng.uniform(0.5, 1.5, COLS).astype(np.float32),
        "out": np.zeros(ROWS * COLS, np.float32),
    }


def reference(inputs):
    x = inputs["x"].astype(np.float64).reshape(ROWS, COLS)
    y = x * inputs["w"].astype(np.float64) + inputs["bias"].astype(np.float64)
    return {"out": y.reshape(-1).astype(np.float32)}
```

---
Your files are in `/task/inputs`. Save every file you produce in `/task/outputs`, and write your final answer to the user in `/task/answer.txt`.
