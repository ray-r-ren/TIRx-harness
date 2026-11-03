"""Shared helpers: the engine environment and running the engine worker in a subprocess."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from skillnet_sdk import ToolContext, ToolError

TOOLS_DIR = Path(__file__).resolve().parent
ENGINE = TOOLS_DIR / "_engine.py"
REFDOCS = TOOLS_DIR / "refdocs"

# The CPU simulator caches its compiled Rust engine in a build cache keyed on the toolchain and
# this environment; the image pre-builds it under exactly these settings (see the environment's
# warm step), so every engine call must use the same values.
IMAGE_BUILD_CACHE = Path("/opt/numsim-cache")
CANONICAL_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "CARGO_HOME": "/opt/cargo",
    "RUSTUP_HOME": "/opt/rustup",
    "NUMSIM_CARGO": "/usr/local/bin/cargo",
    "NUMSIM_RUSTC": "/usr/local/bin/rustc",
    "NUMSIM_GENERATED_OPT_LEVEL": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}
MEMORY_LIMIT_MB = 6144
PASS_THROUGH = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "LD_LIBRARY_PATH")


def cuda_home() -> Path | None:
    """The CUDA toolkit installed from NVIDIA's pip wheels (ptxas, nvcc, cuobjdump)."""
    for base in sys.path:
        cand = Path(base) / "nvidia" / "cu13"
        if (cand / "bin").is_dir():
            return cand
    return None


def _cache_usable(path: Path) -> bool:
    try:
        st = path.stat()
    except OSError:
        return False
    return (
        st.st_uid == os.getuid()
        and not (st.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
        and os.access(path, os.W_OK)
    )


def build_cache_dir() -> tuple[Path, bool]:
    """A NumSim build cache this process may use, and whether it is pre-built.

    NumSim requires its build cache to be owned by the current user and not group/world
    writable. When the image's cache belongs to another user, copy it once to /tmp.
    """
    if _cache_usable(IMAGE_BUILD_CACHE):
        return IMAGE_BUILD_CACHE, True
    private = Path(tempfile.gettempdir()) / f"gkf-numsim-build-{os.getuid()}"
    marker = private / ".gkf-ready"
    if marker.exists():
        return private, True
    lock_path = Path(tempfile.gettempdir()) / f"gkf-numsim-build-{os.getuid()}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            return private, True
        warm = False
        if private.exists():
            shutil.rmtree(private, ignore_errors=True)
        if IMAGE_BUILD_CACHE.is_dir():
            try:
                shutil.copytree(IMAGE_BUILD_CACHE, private, symlinks=True)
                warm = True
            except OSError:
                shutil.rmtree(private, ignore_errors=True)
        private.mkdir(parents=True, exist_ok=True)
        os.chmod(private, 0o700)
        if warm:
            marker.write_text("ok")
        return private, warm


def engine_env(numsim_dir: Path | None) -> tuple[dict[str, str], bool]:
    env = dict(CANONICAL_ENV)
    cache, warm = build_cache_dir()
    env["NUMSIM_BUILD_CACHE_DIR"] = str(cache)
    artifacts = numsim_dir or Path(tempfile.gettempdir()) / f"gkf-numsim-artifacts-{os.getuid()}"
    artifacts.mkdir(parents=True, exist_ok=True)
    os.chmod(artifacts, 0o700)
    env["NUMSIM_CACHE_DIR"] = str(artifacts)
    if Path("/opt/cargo/registry").is_dir():
        env["CARGO_NET_OFFLINE"] = "true"
    home = cuda_home()
    if home is not None:
        env["CUDA_HOME"] = str(home)
        env["CUDA_PATH"] = str(home)
        # PATH stays canonical: NumSim's build-cache key includes it.
    env["HOME"] = os.environ.get("HOME") or tempfile.gettempdir()
    env["TMPDIR"] = tempfile.gettempdir()
    env["LANG"] = "C.UTF-8"
    return env, warm


def ffi_cuda_home(workdir: Path | None = None) -> Path | None:
    """A CUDA home layout tvm-ffi's builder accepts, made of links into the pip toolkit."""
    home = cuda_home()
    if home is None:
        return None
    if workdir is None:
        workdir = Path(tempfile.gettempdir()) / f"gkf-cuda-{os.getuid()}"
    root = workdir / "cuda-home"
    lib64 = root / "lib64"
    lib64.mkdir(parents=True, exist_ok=True)
    for sub in ("bin", "include", "nvvm"):
        if (home / sub).exists() and not (root / sub).exists():
            (root / sub).symlink_to(home / sub)
    libdir = home / "lib"
    if libdir.is_dir():
        for f in libdir.iterdir():
            dest = lib64 / f.name
            if not dest.exists():
                dest.symlink_to(f)
        cudart = sorted(libdir.glob("libcudart.so.*"))
        if cudart and not (lib64 / "libcudart.so").exists():
            (lib64 / "libcudart.so").symlink_to(cudart[0])
    if not (root / "lib").exists():
        (root / "lib").symlink_to(lib64)
    return root


def run_engine(
    ctx: ToolContext,
    action: str,
    request: dict[str, Any],
    *,
    timeout_s: float,
    numsim_dir: Path | None = None,
    memory_limit: bool = True,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one engine action in a subprocess and return its data, or raise ToolError."""
    workdir = Path(tempfile.mkdtemp(prefix=f"gkf-{action}-"))
    try:
        env, warm = engine_env(numsim_dir)
        if extra_env:
            env.update(extra_env)
        req = {k: v for k, v in request.items() if not k.startswith("_")}
        req.update({
            "action": action,
            "workdir": str(workdir / "work"),
            "outdir": str(ctx.output_dir),
            "memory_limit_mb": MEMORY_LIMIT_MB if memory_limit else None,
        })
        req_path = workdir / "request.json"
        resp_path = workdir / "response.json"
        req_path.write_text(json.dumps(req))
        # A clean environment (env -i): NumSim keys its build cache on PATH, CC, CFLAGS and similar
        # variables, so inherited values would force a rebuild of the simulator.
        for name in PASS_THROUGH:
            if name in os.environ and name not in env:
                env[name] = os.environ[name]
        cmd = ["env", "-i", *[f"{k}={v}" for k, v in sorted(env.items())],
               sys.executable, "-I", str(ENGINE), str(req_path), str(resp_path)]
        proc = ctx.run(
            cmd,
            engine="tirx-harness",
            timeout_s=timeout_s,
            cwd=str(workdir),
            check=False,
        )
        if not resp_path.exists():
            tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-4:]
            ctx.mark_untrusted("engine output")
            raise ToolError(
                "engine_crashed",
                "The kernel engine stopped without a result (exit "
                f"{proc.returncode}): " + (" | ".join(tail)[-400:] or "no output"),
                "Run inspect_kernel on the kernel first; if it loads, reduce the shapes in make_inputs() and retry, "
                "then report the failure to the skill owner if it persists.",
            )
        resp = json.loads(resp_path.read_text())
        if resp.get("ok"):
            data = resp["data"]
            data["_warm_cache"] = warm
            return data
        err = resp.get("error", {})
        code = err.get("code") or "engine_error"
        message = err.get("message") or "The kernel engine failed."
        next_step = err.get("next_step") or (
            "Check the kernel with inspect_kernel and compare it with a working example from find_kernel_examples; "
            "report the failure to the skill owner if the kernel is valid."
        )
        details = err.get("details") or {}
        extra = []
        if details.get("line"):
            extra.append(f"Failing line {details['line']}: {details.get('source_line', '')}"[:300])
        if code == "engine_error" and details.get("traceback"):
            extra.append("Engine traceback tail: " + details["traceback"].strip().splitlines()[-1][:300])
        ctx.mark_untrusted("kernel file / engine output")
        raise ToolError(code, message, next_step, uncertain=extra)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        for d in request.get("_tmpdirs") or []:
            shutil.rmtree(d, ignore_errors=True)


def output_files(data: dict, descriptions: dict[str, str]) -> list:
    """OutputFile entries for the files the engine wrote, described by suffix."""
    from skillnet_sdk import OutputFile

    outs = []
    for f in data.get("files", []) or []:
        p = Path(f)
        if not p.exists():
            continue
        desc = next((d for suffix, d in descriptions.items() if p.name.endswith(suffix)), p.name)
        outs.append(OutputFile(p, desc))
    return outs


def seen_before(state_dir: Path, sha256: str) -> bool:
    """Whether this exact kernel text was already processed in this state."""
    try:
        return sha256 in json.loads((state_dir / "kernels_seen.json").read_text())
    except (OSError, ValueError, TypeError):
        return False


def remember(state_dir: Path, sha256: str) -> None:
    """Record a kernel text's hash in the state (the compiled simulator work is keyed on content)."""
    path = state_dir / "kernels_seen.json"
    try:
        seen = json.loads(path.read_text())
        if not isinstance(seen, list):
            seen = []
    except (OSError, ValueError):
        seen = []
    if sha256 not in seen:
        seen = (seen + [sha256])[-200:]
    state_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen))
