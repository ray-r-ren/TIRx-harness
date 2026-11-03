FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git xz-utils unzip file procps xvfb xauth ffmpeg libgl1 libglib2.0-0 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz | tar xJ -C /usr/local --strip-components=1
RUN pip install --no-cache-dir pydantic==2.10.3 pytest==8.3.4 pyyaml==6.0.2 requests==2.32.3 pypdf==5.1.0
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DISPLAY=:99 MPLBACKEND=Agg
RUN mkdir -p /workspace/inputs /workspace/outputs /skill
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ make cmake pkg-config libc6-dev git ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir tirx-harness==0.1.2 nvidia-cuda-nvcc==13.4.92 nvidia-cuda-cuobjdump==13.4.92 nvidia-cuda-cccl==13.3.4.3.1 ninja==1.13.0
RUN export RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo && curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.89.0 && /opt/cargo/bin/rustup default 1.89.0 && TC=$(ls -d /opt/rustup/toolchains/1.89.0-*) && for b in cargo rustc rustdoc; do ln -sf "$TC/bin/$b" /usr/local/bin/$b; done && chmod -R a+rX /opt/rustup /opt/cargo
RUN mkdir -p /opt/numsim-cache /opt/skill-warm && printf '%s\n' 'import numpy as np' 'import tirx_kernels.tirx_lite as txl' 'from tirx_harness import numsim, synccheck, racecheck' '@txl.kernel(warps=1, arch="sm_90a", grid=1)' 'def warm(a: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):' '    x = txl.local_scalar("float32")' '    txl.ptx.ld.global_.f32(x, a.ptr_to([txl.lane_id()]))' '    txl.ptx.st.global_.f32(out.ptr_to([txl.lane_id()]), x + x)' 'def inp():' '    return {"a": np.arange(32, dtype=np.float32), "out": np.zeros(32, np.float32)}' 'r = numsim.Engine().run(numsim.transpile(warm.func), inputs=inp(), outputs=("out",))' 'assert np.allclose(r.outputs["out"], 2 * np.arange(32, dtype=np.float32))' 'assert synccheck(warm.func, inputs=inp()).verdict == "clean"' 'assert racecheck(warm.func, inputs=inp()).verdict == "clean"' 'print("numsim warm OK")' > /opt/skill-warm/warm_numsim.py && env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup NUMSIM_CARGO=/usr/local/bin/cargo NUMSIM_RUSTC=/usr/local/bin/rustc NUMSIM_BUILD_CACHE_DIR=/opt/numsim-cache NUMSIM_GENERATED_OPT_LEVEL=1 HOME=/tmp LANG=C.UTF-8 NUMSIM_CACHE_DIR=/tmp/numsim-warm-artifacts python /opt/skill-warm/warm_numsim.py && rm -rf /tmp/numsim-warm-artifacts && chmod -R go-w,a+rX /opt/numsim-cache && chmod -R a+rwX /opt/cargo
RUN rm -rf /root/.cache/pip /var/lib/apt/lists/*
COPY files/ /task/
COPY inputs/ /task/inputs/
RUN rm -f /task/.keep /task/inputs/.keep && mkdir -p /task/outputs && useradd -m -u 1000 agent && chown -R agent /task
WORKDIR /task
