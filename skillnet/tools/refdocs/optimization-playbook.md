# Optimization Tricks

The bundled canonical package uses registered raw PTX, explicit pointers,
TensorMaps, descriptors, and barriers in public `get_kernel` IR. The recipes
below follow that low-level style. Do not introduce the legacy `Tx.*` tile
authoring surface merely to express a copy, TMA, or MMA operation.

## Pattern-backed Moves

The empirically strong patterns and mechanical cleanup items live in
`perf_checklist/section_*.md`. A playbook entry exists only
when there is implementation detail beyond the checklist text itself —
load/store API forms, performance numbers, saturation thresholds, recipe
sketches:

| Playbook # | Checklist item |
|------------|----------------|
| #5  | Minimize SMEM access overhead |
| #10 | `exp2`, not `exp` |
| #12 | Iterate the actual tile count |
| #13 | TMA for bulk GMEM <-> SMEM |
| #14 | Persistent grid + single-role warp partitioning |

FMA has no playbook entry — the pattern is mechanical and there's
nothing to add beyond it.

The remaining entries (#1, #2, #3, #4, #6, #7, #8, #9) are conditional —
they have a precondition or a tradeoff worth weighing per workload.

---

### Technique #1: Pipeline TMA Stores with wait_group(1)

Allocate two D SMEM buffers and ping-pong between them. Each iteration writes one
buffer and issues a TMA store + `T.ptx.cp.async_.bulk.commit_group()` from that
buffer. `T.ptx.cp.async_.bulk.wait_group.read(1)` allows at most one recent group
to remain pending at its source-read milestone, so an older buffer whose source
read has completed may be reused while destination writes can still be in flight.
`.wait_group.read(0)` drains all source reads, not all destination writes. Use the
unqualified `.wait_group(0)` only when full destination completion and visibility
to the executing thread are required.

### Technique #2: K-Splitting for SM Utilization

**When to apply:** num_m_blocks × CTA_GROUP < SM_COUNT (148 on B200). E.g., M=4096/BLK_M=64 = 64 CTAs → 43% utilization.

Split K across multiple CTAs. Each handles `K/kNumSplits` iterations. Grid = `num_m_blocks * kNumSplits`. Host-side reduction after kernel: `D_out = D_splits.sum(dim=0)`.

Target `num_m_blocks * kNumSplits ≈ SM_COUNT`.

### Technique #3: Offload Scalar Loops to MMA Phases

Identify matmul hidden in consumer scalar loops → add SMEM staging buffers (f16, swizzled) → consumer prepares operands → MMA phases execute the matmul.

### Technique #4: use_a_tmem

When A would otherwise be staged reg/TMEM → SMEM → MMA, set `use_a_tmem=True` so MMA reads A directly out of TMEM and skip the SMEM round-trip (writer SMEM write + MMA SMEM read both gone). Saves SMEM and consumer cycles, costs extra TMEM columns.

**Constraint:** A must be K-major in TMEM — M maps to lanes, K to columns. transA is not supported with this path; if you need `A^T @ B`, stage A through SMEM the normal way.

### Technique #5: SMEM is a Cross-Thread Interface

The rule and audit method live in
[`section_3_scalar_dataflow.md`](perf_checklist/section_3_scalar_dataflow.md).
This entry is the implementation handbook for the hot-path fixes.

**Vectorize at the low-level memory-operation boundary.** Replace scalar
register↔SMEM loops with a packed low-level load/store or `stmatrix` sequence
that matches the alignment and cross-thread ownership contract. Keep the
address calculation explicit; do not hide it behind a tile-copy helper.

One 16-byte transfer (= 8×bf16 / 4×fp32 / 16×fp8). On a 32-element strided
epilogue this single rewrite is typically ~5-8× faster (the scalar STS loop is
usually the dominant cost in such blocks). Alignment: `c × elem_size` must be
16-byte aligned. For swizzled SMEM, compute the physical address using the
chosen swizzle and issue the matching low-level operation; there is no
layout-aware tile-copy dispatch to rely on.

**Readback-compute fusion.** Pattern `compute → SMEM → compute → SMEM` on the same thread→data partition collapses to `compute → compute → SMEM` — chain both computes on registers, store once. Different-thread partitions (one warp produces, WG consumes) genuinely need the SMEM — fix the partitioning, not the fusion.

### Technique #6: Swizzle SMEM Buffers to Eliminate Bank Conflicts

Trigger: the perf checklist's SMEM audit says swizzle when per-lane offset
analysis shows multiple lanes in a warp landing on the same bank. Pick the
swizzle mode whose row width matches the access width: 32B / 64B / 128B row →
`SWIZZLE_32B_ATOM` / `SWIZZLE_64B_ATOM` / `SWIZZLE_128B_ATOM`. The buffer base
must be aligned to the swizzle row width.

**Accessing a swizzled SMEM buffer.** Derive the physical SMEM address from the
chosen swizzle and issue explicit `stmatrix`/store PTX or another packed
low-level operation. The current
`tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py` epilogue is the reference
pattern: it owns the address calculation instead of delegating it to a tile
copy helper.

### Technique #7: Cross-Lane Sharing via Warp Shuffle

**Goal:** raise thread participation — and thus utilization — by getting idle lanes involved, instead of leaving values stuck on one lane and routing them through SMEM. Use `__shfl_*_sync` to move values directly between lanes of the same warp.

Common shapes: one lane holds the value and broadcasts to the rest (`__shfl_sync`); only some lanes are active and the others idle, redistribute work via `__shfl_xor_sync(mask, val, stride)` (e.g., stride=16 to fill the upper half of a warp); butterfly reductions across the warp.

### Technique #8: Skip the Barrier Between tcgen05.cp and tcgen05.mma

`tcgen05.cp` (SMEM→TMEM) and a following `tcgen05.mma` from the same thread are pipelined by the hardware in issue order — no barrier or fence is needed between them. Don't insert one; it just stalls the issuing warp without buying any safety.

### Technique #9: Cross-Tile Barrier for Aliased SMEM in Persistent Kernels

When you alias SMEM (e.g., `pool.move_base_to(offset)`), watch for cross-tile races: in a persistent kernel the TMA warp for tile T+1 may start loading into the aliased region while the epilogue for tile T is still writing to it. Add a barrier so TMA for T+1 waits for the epilogue (or the last reader) of T to finish before the load is issued.

### Technique #10: Prefer `exp2` over `exp`

Rule and code pattern in
[`section_2_static_source_patterns.md`](perf_checklist/section_2_static_source_patterns.md).

**Saturation thresholds.** Skipping the fp32 cast shrinks input range fast: fp16 saturates around `|x| ≈ 11`, bf16 around `|x| ≈ 88`. Always cast the source to fp32 before `T.ptx.ex2.approx.ftz.f32(dst, src)` even when the surrounding compute is bf16.

### Technique #12: Workload Compaction

Rule lives in [`section_4_dead_work.md`](perf_checklist/section_4_dead_work.md).

**Prefer on-GPU compaction when inputs are already device-resident.** Do not assume host-side preprocessing is available unless the caller's integration contract explicitly permits and measures it.

When the variable-length shape arrives as a cumulative-count (indptr-style) array, the per-item extent is one subtraction `indptr[i+1] - indptr[i]`. The flat tile-coordinate list is cheap — emit it in a one-off preprocess kernel, or recompute it inside each consuming kernel; either is fine. Pick whichever shape is simpler.

### Technique #13: TMA for Bulk Global Memory Transfers

Rule and skip-conditions live in
[`section_2_static_source_patterns.md`](perf_checklist/section_2_static_source_patterns.md).

The canonical kernels build TensorMaps in the host prologue and invoke
registered raw `cp.async.bulk.tensor...` forms with explicit SMEM and barrier
pointers. Keep descriptor construction, pointer conversion, and completion
barriers visible in the kernel; do not replace them with the legacy tile/TMA
helper surface. For atomic accumulation into GMEM, use the corresponding raw
TMA reduce form and preserve its completion contract.

When scalar `ld.global` / `st.global` loops were the bottleneck, this single rewrite typically cuts the affected stage by ~40%.

### Technique #14: Persistent Kernel by Default; Scale Out via Warpgroups, not CTAs

Rules live in [`section_1_persistent_kernel_structure.md`](perf_checklist/section_1_persistent_kernel_structure.md)
under persistent grid and role-major warp partitioning.

**Soaking up idle SM capacity.** When an SM has spare CUDA-core time between MMA ops, the answer is *another warpgroup inside the same CTA* — scalar epilogue, softmax, address-arithmetic WGs are typical fits. Don't increase grid size to fill the gap; that breaks the deterministic issue order the rule protects (and loses intra-CTA SMEM operand reuse).
