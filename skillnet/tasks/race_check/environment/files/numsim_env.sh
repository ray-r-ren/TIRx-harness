# Source this to run Python with the simulator's pre-built cache (same environment the image warmed it in).
C=/opt/numsim-cache
if [ ! -O "$C" ]; then
  if [ ! -f /tmp/nsc-task/.ready ]; then rm -rf /tmp/nsc-task && cp -r "$C" /tmp/nsc-task && chmod 700 /tmp/nsc-task && touch /tmp/nsc-task/.ready; fi
  C=/tmp/nsc-task
fi
mkdir -p /tmp/numsim-task && chmod 700 /tmp/numsim-task
nspython() {
  env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup \
    NUMSIM_CARGO=/usr/local/bin/cargo NUMSIM_RUSTC=/usr/local/bin/rustc NUMSIM_GENERATED_OPT_LEVEL=1 \
    CARGO_NET_OFFLINE=true HOME=/tmp LANG=C.UTF-8 NUMSIM_BUILD_CACHE_DIR="$C" NUMSIM_CACHE_DIR=/tmp/numsim-task \
    PYTHONPATH=/task python "$@"
}
