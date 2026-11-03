"""Kernel fixtures for the real-engine tests, written as small TIRx-lite files at test time."""
import shutil
import subprocess
import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STARTERS = ROOT / "tools" / "data" / "starters"

HEADER = "import numpy as np\nimport tirx_kernels.tirx_lite as txl\n\n"

VECTOR_ADD_BODY = '''N = 32

@txl.kernel(warps=1, arch="sm_90a", grid=1)
def vector_add(a: txl.gptr(txl.f32), b: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32")
    y = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([lane]))
    txl.ptx.ld.global_.f32(y, b.ptr_to([lane]))
    txl.ptx.st.global_.f32(out.ptr_to([lane]), x + y)

def make_inputs():
    rng = np.random.default_rng(0)
    return {"a": rng.standard_normal(N).astype(np.float32),
            "b": rng.standard_normal(N).astype(np.float32),
            "out": np.zeros(N, np.float32)}

def reference(inputs):
    return {"out": inputs["a"] + inputs["b"]}
'''

VECTOR_ADD = HEADER + VECTOR_ADD_BODY

SOURCES = {
    # Correct 32-element add, one warp (the playbook's example).
    "vector_add": VECTOR_ADD,
    # The racy rotate below, fixed with a CTA barrier between the shared store and load.
    "smem_exchange": HEADER + '''
@txl.kernel(warps=2, arch="sm_90a", grid=1)
def rotate(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    tid = txl.thread_id()
    smem = txl.smem_pool()
    buf = smem.alloc([64], txl.f32)
    x = txl.local_scalar("float32")
    y = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([tid]))
    txl.ptx.st.shared.f32(buf.ptr_to([tid]), x)
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(64))
    txl.ptx.ld.shared.f32(y, buf.ptr_to([(tid + 32) % 64]))
    txl.ptx.st.global_.f32(out.ptr_to([tid]), y)

def make_inputs():
    return {"a": np.arange(64, dtype=np.float32), "out": np.zeros(64, np.float32)}

def reference(inputs):
    return {"out": np.roll(inputs["a"], -32)}
''',
    # Lanes 16..31 compute a - b instead of a + b.
    "vector_add_bug": HEADER + '''
@txl.kernel(warps=1, arch="sm_90a", grid=1)
def kernel(a: txl.gptr(txl.f32), b: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
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
''',
    # Two CTAs of 2 warps, scalar parameter.
    "scale_add": HEADER + '''
@txl.kernel(warps=2, arch="sm_90a", grid=2)
def scale_add(a: txl.gptr(txl.f32), b: txl.gptr(txl.f32), out: txl.gptr(txl.f32), alpha: txl.f32):
    i = txl.cta_id() * 64 + txl.thread_id()
    x = txl.local_scalar("float32")
    y = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([i]))
    txl.ptx.ld.global_.f32(y, b.ptr_to([i]))
    txl.ptx.st.global_.f32(out.ptr_to([i]), alpha * x + y)

def make_inputs():
    n = 128
    return {"a": np.arange(n, dtype=np.float32), "b": np.ones(n, np.float32),
            "out": np.zeros(n, np.float32), "alpha": np.float32(2.0)}

def reference(inputs):
    return {"out": inputs["alpha"] * inputs["a"] + inputs["b"]}
''',
    # Two warps exchange through shared memory with no barrier: a data race.
    "racy": HEADER + '''
@txl.kernel(warps=2, arch="sm_90a", grid=1)
def rotate(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    tid = txl.thread_id()
    smem = txl.smem_pool()
    buf = smem.alloc([64], txl.f32)
    x = txl.local_scalar("float32")
    y = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([tid]))
    txl.ptx.st.shared.f32(buf.ptr_to([tid]), x)
    txl.ptx.ld.shared.f32(y, buf.ptr_to([(tid + 32) % 64]))
    txl.ptx.st.global_.f32(out.ptr_to([tid]), y)

def make_inputs():
    return {"a": np.arange(64, dtype=np.float32), "out": np.zeros(64, np.float32)}

def reference(inputs):
    return {"out": np.roll(inputs["a"], -32)}
''',
    # Barrier waits for 64 threads but the CTA has 32: a deadlock.
    "hang": HEADER + '''
@txl.kernel(warps=1, arch="sm_90a", grid=1)
def kernel(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([lane]))
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(64))
    txl.ptx.st.global_.f32(out.ptr_to([lane]), x)

def make_inputs():
    return {"a": np.arange(32, dtype=np.float32), "out": np.zeros(32, np.float32)}

def reference(inputs):
    return {"out": inputs["a"]}
''',
    # An opaque CUDA helper the simulator and checkers cannot model.
    "opaque": HEADER + '''
HELPER = """
__device__ __forceinline__ float my_square(float v) { return v * v; }
"""

@txl.kernel(warps=1, arch="sm_90a", grid=1, allowed_func_calls=("my_square",))
def square(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([lane]))
    y = txl.cuda.func_call("my_square", x, source_code=HELPER, return_type="float32")
    txl.ptx.st.global_.f32(out.ptr_to([lane]), y)

def make_inputs():
    return {"a": np.arange(32, dtype=np.float32), "out": np.zeros(32, np.float32)}

def reference(inputs):
    return {"out": inputs["a"] ** 2}
''',
    # Reads 32 elements of a 16-element input.
    "oob": HEADER + '''
@txl.kernel(warps=1, arch="sm_90a", grid=1)
def kernel(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([lane]))
    txl.ptx.st.global_.f32(out.ptr_to([lane]), x)

def make_inputs():
    return {"a": np.arange(16, dtype=np.float32), "out": np.zeros(32, np.float32)}

def reference(inputs):
    return {"out": np.concatenate([inputs["a"], inputs["a"]])}
''',
    # Syntax error on line 8 (unclosed parenthesis).
    "broken": '''import numpy as np
import tirx_kernels.tirx_lite as txl


@txl.kernel(warps=1, arch="sm_90a", grid=1)
def kernel(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32"
    txl.ptx.st.global_.f32(out.ptr_to([lane]), x)
''',
    # Two kernels, make_inputs only (no reference).
    "two": HEADER + '''
@txl.kernel(warps=1, arch="sm_90a", grid=1)
def add_one(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([lane]))
    txl.ptx.st.global_.f32(out.ptr_to([lane]), x + 1.0)

@txl.kernel(warps=2, arch="sm_80", grid=4)
def double(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    i = txl.cta_id() * 64 + txl.thread_id()
    x = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([i]))
    txl.ptx.st.global_.f32(out.ptr_to([i]), x * 2.0)

def make_inputs():
    return {"a": np.arange(256, dtype=np.float32), "out": np.zeros(256, np.float32)}
''',
    # Postponed annotations break tracing.
    "future": "from __future__ import annotations\n" + VECTOR_ADD,
    # Same add, but for Blackwell.
    "va_blackwell": VECTOR_ADD.replace('arch="sm_90a"', 'arch="sm_100a"'),
    # Uses txl.sqrt, which the simulator lacks.
    "sqrt_kernel": HEADER + '''
@txl.kernel(warps=1, arch="sm_90a", grid=1)
def root(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([lane]))
    txl.ptx.st.global_.f32(out.ptr_to([lane]), txl.sqrt(x))

def make_inputs():
    return {"a": np.arange(1, 33, dtype=np.float32), "out": np.zeros(32, np.float32)}

def reference(inputs):
    return {"out": np.sqrt(inputs["a"])}
''',
    # fp16 input loaded as raw 16-bit values.
    "half_scale": HEADER + '''
N = 128

@txl.kernel(warps=1, arch="sm_90a", grid=1)
def half_scale(x: txl.gptr(txl.f16), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    h = txl.local_scalar("uint16")
    with txl.unroll(N // 32) as k:
        txl.ptx.ld.global_.b16(h, x.ptr_to([k * 32 + lane]))
        txl.ptx.st.global_.f32(out.ptr_to([k * 32 + lane]), txl.cast(txl.reinterpret("float16", h), "float32") * txl.float32(2.0))

def make_inputs():
    rng = np.random.default_rng(6)
    return {"x": rng.standard_normal(N).astype(np.float16), "out": np.zeros(N, np.float32)}

def reference(inputs):
    return {"out": inputs["x"].astype(np.float32) * 2}
''',
    # A case file: vector_add at 32 elements but with a wrong-dtype input.
    "bad_dtype_case": '''import numpy as np

def make_inputs():
    return {"a": np.arange(32, dtype=np.float64), "b": np.ones(32, np.float32), "out": np.zeros(32, np.float32)}

def reference(inputs):
    return {"out": inputs["a"] + inputs["b"]}
''',
    # A case file for vector_add with different values.
    "va_case": '''import numpy as np

BYTES_MOVED = 3 * 32 * 4

def make_inputs():
    a = np.linspace(-1, 1, 32, dtype=np.float32)
    return {"a": a, "b": a[::-1].copy(), "out": np.zeros(32, np.float32)}

def reference(inputs):
    return {"out": inputs["a"] + inputs["b"]}
''',
}


@pytest.fixture(scope="session")
def kernels(tmp_path_factory) -> dict[str, pathlib.Path]:
    """name -> path of each fixture kernel, plus the shipped starters as starter_<stem>."""
    d = tmp_path_factory.mktemp("kernels")
    paths = {}
    for name, src in SOURCES.items():
        p = d / f"{name}.py"
        p.write_text(src)
        paths[name] = p
    for f in sorted(STARTERS.glob("*.py")):
        p = d / f"starter_{f.name}"
        shutil.copyfile(f, p)
        paths[f"starter_{f.stem}"] = p
    return paths


def has_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-L"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def has_nvcc() -> bool:
    try:
        from tools._common import cuda_home
        home = cuda_home()
        return bool(home and (home / "bin" / "nvcc").exists())
    except Exception:
        return False
