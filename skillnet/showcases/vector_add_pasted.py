import numpy as np
import tirx_kernels.tirx_lite as txl


@txl.kernel(warps=1, arch="sm_90a", grid=1)
def vector_add(a: txl.gptr(txl.f32), b: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32")
    y = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([lane]))
    txl.ptx.ld.global_.f32(y, b.ptr_to([lane]))
    txl.ptx.st.global_.f32(out.ptr_to([lane]), txl.Select(lane < 16, x + y, x - y))

def make_inputs():
    a = np.arange(32, dtype=np.float32)
    return {"a": a, "b": np.ones(32, np.float32), "out": np.zeros(32, np.float32)}

def reference(inputs):
    return {"out": inputs["a"] + inputs["b"]}
