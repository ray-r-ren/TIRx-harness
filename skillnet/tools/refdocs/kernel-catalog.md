# TIRx Kernels — Optimization Techniques Reference

A cross-cutting catalog of **distinctive** optimization techniques in selected
canonical exemplars under `tirx-kernels/tirx_kernels/`. Use it as a
lookup table when designing or optimizing a kernel: find where a technique is
already implemented, then inspect the environment's source. This page is
deliberately not a second inventory of every registry entry. The recursively discovered
`KERNEL_META` registry is the authority for the complete kernel set; the
checkout's `README.md` renders its family-level
overview.

Resolve `tirx-kernels/` paths under the wiki's `references/repos/` directory.
Fetch that checkout with the wiki reference fetcher if absent. The installed TIRx source and canonical
package own the exact `tirx_kernels.low_level_ir` boundary, including the `tirx.address_of`
pointer-syntax carve-out; it is not restated here.

The doc has two views:

- **[§1 By exemplar/family](#1-by-kernel)** — quick "what makes this schedule special"
- **[§2 By technique](#2-by-technique-reverse-index)** — reverse index from
  optimization name → kernels that use it

What's intentionally **NOT** listed: recurring patterns captured in
`perf_checklist/section_*.md`. They have performed well across prior kernel
work, but each kernel should apply only the patterns that fit its workload:

- `T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})` + 1 CTA / SM + tile scheduler
- runtime TensorMap encoding plus registered raw TMA for bulk GMEM ↔ SMEM
- registered raw `tcgen05.mma` with explicit matrix/instruction descriptors
- exact registered PTX for ex2, packed FMA/arithmetic, CVT, LD/ST, and register-resident scratch
- Warpgroup-level role specialization

Everything below highlights techniques more distinctive than these recurring
patterns.

**Anchors, not line numbers.** Every bullet names a function, constant, PTX
mnemonic, or unique string you can grep for in the cited file. Line numbers
drift on every edit; symbol names survive everything except a rename. If you
add a bullet, give it a grep-able anchor — if there isn't one, the bullet is
probably too vague.

---

## §1 By kernel

### `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py` — FA4 forward (fp16/bf16, sm_100)

- **Loop peeling for regional per-iter cost** — Peel into compile-time
  phases when per-iter cost is regional (init / mask / boundary).
  *FA4:* KV loop high-index-first in 3 phases: `is_first` cold-start /
  `apply_mask=True` causal / `apply_mask=False` bulk; reverse order also
  stabilizes `row_max` for later `rescale_threshold` skips.
- **Scalar bitmask + bit-extract for monotonic gates** — For gates
  monotonic in the index, build the mask once as `(1 << bound) - 1`
  then bit-extract per element in an unrolled loop — ptxas fuses to
  1 build + N predicated ops vs N compare+selects. *FA4:* `mask_r2p`
  via `apply_causal_mask`; 24-elem chunks dodge ptxas's shift-by-31+
  quirk.
- **Dual-issue transcendentals on MUFU + CUDA cores** — Hot
  transcendentals saturate MUFU. Route some lanes to a CUDA-core
  polynomial (fractional-bits Horner + integer-part bit-trick recombine)
  instead — both units issue in parallel, ~2× throughput. *FA4:*
  `ex2_emulation_2` + `combine_int_frac_ex2`; `EMU_PAIRS_CAUSAL`,
  `EMU_PAIRS_NC`, and `EMU_START_CAUSAL` select the emulated pairs, while the
  rest use exact `T.ptx.ex2.approx.ftz.f32`.
- **Pack TMEM for concurrent MMAs** — Coexisting TMEM tensors take
  disjoint columns so independent MMAs issue in parallel;
  different-precision tensors can alias. *FA4:* `S_region` /
  `O_region` / `P_region` in `tmem[N_COLS_TMEM=512]`; `tmem_as_f16`
  aliases P over S's upper half (saves 256 fp32 cols), so P→V MMA
  overlaps next-tile S write.
- **Keep the running accumulator in TMEM across the reduction** — When
  a streaming reduction repeatedly updates a large accumulator with a
  correction rescale (`acc = acc*scale + new`), keep the accumulator
  resident in TMEM and do both halves of the update in place: the
  correction warp read-modify-writes it (mul by scale), the MMA
  accumulates the new term with `accum=True` into the same columns. The
  accumulator never round-trips to registers/SMEM between iterations.
  *FA4:* the online-softmax output update `O = O*acc_scale + P@V` lives
  entirely in `O_region` — the correction warp uses `_tmem_load`,
  `mul_f32x2`, and `_tmem_store`, then `gemm_pv_part1` forwards
  `should_accumulate` to the raw `_MMA_F16` predicate while part 2
  accumulates into the same region.
- **Pack single-warp roles into one warpgroup** — Multiple roles whose
  work each fits in one warp (TMA load/store, MMA issue, single-warp
  compute like SF transpose) co-inhabit one WG's 4 warps, freeing
  other WGs entirely for full-WG roles. *FA4:* WG3 hosts TMA-load +
  TMA-store + MMA-issue.
- **Stream producer fragments via consumer mid-barriers** — Holding a
  large producer output live across a long consumer is a register-budget
  wall for the producer WG. Mid-stream `partial-ready` barriers on the
  consumer's K-loop let the producer write-and-free in chunks. *FA4:*
  softmax releases P chunk-by-chunk via V-MMA's `p_ready_2`, instead of
  holding all of P live through PV-MMA.
- **Workload-shape scheduler swap (LPT for triangular work)** —
  Non-uniform per-tile work (causal triangle, varlen packing) calls for
  LPT + L2-locality grouping over the linear scheduler. *FA4:*
  `FlashAttentionLPTScheduler` + `L2_SWIZZLE` on `is_causal=True`.
- **Predicate-elide fp updates below precision floor** —
  Multiplicative corrections rounding to ≈1 (or additive to ≈0) within
  accumulator precision are fp no-ops; guard with a cheap scalar
  predicate to skip the entire path including its memory R/W and
  cross-thread syncs. Convergent processes (online stats, EMA,
  iterative refinement) mostly qualify in steady state.
  **Measured 33% FA forward improvement in the recorded comparison.** *FA4:* skip
  rescale when `(max_old − max_new) * scale ≥ −rescale_threshold`
  (`= 8.0`).
- **`any_sync` to prevent SIMT divergence on data-dependent branches** —
  Per-lane `if pred[tid]` serializes both paths via warp divergence;
  `any_sync` collapses `pred` to warp-uniform so all lanes take/skip
  together. Branch must be side-effect-safe on non-needing lanes.
  *FA4:* `O *= 1/row_sum` + TMEM rescale gated on warp vote.
- **Fold single-side dims into matmul M/N** — Any dim appearing in only
  one operand can fold into that operand's M (LHS) or N (RHS) — the
  MMA sees a bigger tile, not an extra einsum dim; the shared operand
  loads once, broadcast is implicit. *FA4:* GQA's `GQA_RATIO` (Q-side
  only) folds into M alongside seq-pos so one K-head serves all G
  Q-heads; `apply_causal_mask` decomposes packed M back to coords.

---

### `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention_backward.py` — FA4 backward (fp16 D=128, sm_100)

- **Static two-CTA ownership with causal work trimming** — Each physical
  cluster owns one KV tile for its lifetime; `n_tile_idx` selects that tile,
  while `m_tile_start` removes query tiles that are identically zero in the
  causal schedule.
- **Pack infrastructure roles into WG3** — `wg_id == 3` places leader-CTA MMA
  issue, per-CTA TMA production, DSMEM-completion relay, and one idle warp in
  one warpgroup. WG1/WG2 form P and dS; WG0 reduces dQ.
- **Interleave five dependent MMAs across query tiles** — `mma_n_tile` peels
  `S`, `dP`, and `dV` as a prologue, then orders each steady-state iteration as
  next-S, prior-dK, next-dP, prior-dQ, next-dV.
- **Cut dK release away from DSMEM exchange** — `wg02mma_tmem` publishes dS as
  soon as its TMEM image is ready for dK, while `ds_exch_mbar` independently
  publishes the local-plus-peer SMEM image consumed by dQ.
- **Alias storage at explicit lifetime boundaries** — `TMEM_OFF_DQ` overlaps
  the drained upper half of S/P, dP becomes dS in place, and `dV_epi`/`dK_epi`
  reuse the upstream V/K SMEM allocations. `dq_tmem_free` guards the S/dQ
  alias boundary.
- **Overlap split epilogues with the MMA tail** — `dv_done` releases dV before
  the last dK/dQ work; `dk_done` releases dK before the final dQ. `DQ_STAGES`
  pipelines TMEM readback through a four-stage SMEM-to-TMA-add ring.
- **Compact schedule reference** — see
  `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention_backward_sm100_sketch.md`.

---

### `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp4.py` — MQA-logits with packed FP4 (e2m1fn)

- **Sub-byte operand packing with exponent-only per-token scale** — Pack
  two FP4 nibbles per uint8 byte and carry a per-token e8m0 exponent
  scale; 4× operand-bandwidth reduction vs fp32 with one shared scale
  per block. *fp4_logits:* `per_token_cast_to_fp4` +
  `use_packed_ue8m0=True`.
- **SMEM transpose fixup for `tcgen05.cp` consumption** — Some TMA
  layouts deliver SF tiles that the raw `tcgen05.cp` descriptor cannot consume
  directly. *fp4_logits:* `emit_sf_transpose` performs explicit
  `ld.shared.u32` + `st.shared.v4.u32` into `smem_sf_q_t` / `smem_sf_kv_t`,
  whose addresses are installed into the raw CP descriptor.
- **No-FTZ paired-float FMA in accumulation** — IEEE-754 paired-float FMA
  (no flush-to-zero) preserves denormals when accumulation must match a
  reference; otherwise FTZ can drop bits.
  *fp4_logits:* `_mqa_fp4_wrelu_reduce_src` emits `__ffma2_rn` and
  `__fadd2_rn`.
- **Variable-length per-row KV range via min/max-of-tile** — When each
  row has its own valid K range, take `min(ks)`/`max(ke)` over the
  Q-tile rows and floor the start to the vector alignment so one
  shared per-tile loop covers all rows. *fp4_logits:* `load_schedule`
  with 4-aligned start.
- **Paired-float reduction tree (`f32x2`)** — Reductions over even-pair
  data (e.g. (max-with-0, weight) for ReLU-weighted heads) use
  `__ffma2_rn` / `__fadd2_rn` on two values at once, halving the FMA count
  vs scalar. *fp4_logits:* `_mqa_fp4_wrelu_reduce_src`.

---

### `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp8.py` — MQA-logits with native FP8 e4m3fn

- **Non-block-scale FP8 MMA with post-MMA scale apply** — When operands
  fit a native MMA dtype with no per-block scale fusion needed, use the
  non-block-scale descriptor and apply per-token scales after the MMA
  with one SMEM load per thread. *fp8_logits:* `smem_q_fp8` and
  `smem_kv_fp8` feed explicit matrix descriptors and the raw dense
  `tcgen05_mma`; `smem_kv_scales` supplies the post-MMA scale.
- **Co-bundle two TMAs under one `expect_tx`** — When two TMA loads
  retire under the same downstream barrier, combine their byte counts
  into a single `expect_tx` so only one arrive is needed; saves a
  barrier-arrive without growing pipeline state. *fp8_logits:* Q +
  weights bundled on one full-Q barrier.
- **Fused packed reduction tree** — Accumulate `(relu(x), weight)` products
  into a four-value register buffer with `emit_wrelu_reduce`, using raw
  `fma.rn.f32x2` for head groups and an `add.rn.f32x2`/`add.rn.f32` tree
  instead of a separate multiply pass followed by a serial reduction.

---

### `tirx-kernels/tirx_kernels/ported/deepgemm/tf32_hc_prenorm_gemm.py` — TF32 GEMM with fused sum-of-squares ("Hadamard Cast prenorm")

- **Fold pre-MMA cast pass into operand sum-of-squares** — When a kernel
  already streams an operand through a cast warp before MMA, accumulate
  the per-row sum-of-squares in the same pass; saves a downstream
  normalization kernel reading A again. *tf32_gemm:* cast warps issue raw
  `ldmatrix`, CVT, packed `fma.rz.ftz.f32x2`, and `tcgen05.st` while
  side-folding the two `sqr*` accumulators.
- **TMEM A operand driven by cast-warp-written TMEM A** — When
  cast warps produce A in registers anyway, write A directly to TMEM with raw
  `tcgen05.st`, then pass `a_col` to the raw `tcgen05_mma_tf32` A operand
  rather than an SMEM descriptor; this eliminates an SMEM round-trip for A.
- **Two-shuffle butterfly reduction for per-N-lane sums** — Reduction
  scoped to N lanes per row (not full warp) needs only `log2(N)`
  shuffles; for N=4, two xor-shuffles suffice. *tf32_gemm:* two
  `T.tvm_warp_shuffle_xor` steps (distances 1 and 2) reduce each `sqr_part`.
- **Split-K with deterministic output axis** — A `num_splits` outer
  axis distributes K across CTAs; results land in a 3D output, summed
  post-kernel — deterministic vs atomic split-K. *tf32_gemm:*
  `num_splits` axis.

---

### `tirx-kernels/tirx_kernels/gemm/fp16_bf16_gemm.py` — fp16/bf16 reference GEMM (cluster launch-control baseline)

- **2-CTA cluster with L2-grouped launch-control scheduler** — Bind each tile
  to a 2×1 cluster so two CTAs share operand TMAs and L2 lines;
  `ClusterLaunchControlScheduler` with `cta_group=2` and
  `l2_group_size=L2_GROUP_SIZE` groups adjacent clusters for L2 locality.
- **Multi-consumer WGs to scale tile M while sharing B** — N consumer
  WGs each own their own A slice + D output but share one B SMEM tile,
  scaling effective BLK_M by N without growing B's SMEM or per-output
  GMEM cost (B amortizes 1/N). *fp16_gemm:* `NUM_CONSUMER` is one for the
  `OVERLAP_EPILOGUE` shapes and two otherwise; A and D are per-consumer,
  while B is shared.
- **Chunked TMEM→register epilogue readback** — Read TMEM into a small
  register slice in `NOL`-wide chunks in the non-overlap path, so the consumer
  WG never holds the full fp32 D tile in registers. *fp16_gemm:* `NOL=16`;
  the overlap path reads one `EPI_N` chunk at a time.

---

### DeepGEMM `sm100_fp8_fp4_gemm_1d1d` family — one device template, five public entries

The public registry modules
`tirx-kernels/tirx_kernels/ported/deepgemm/fp8_gemm_1d1d.py`,
`tirx-kernels/tirx_kernels/ported/deepgemm/fp8_bmm.py`,
`tirx-kernels/tirx_kernels/ported/deepgemm/m_grouped_fp8_gemm_contiguous.py`,
`tirx-kernels/tirx_kernels/ported/deepgemm/m_grouped_fp8_gemm_masked.py`, and
`tirx-kernels/tirx_kernels/ported/deepgemm/k_grouped_fp8_gemm_contiguous.py` bind host
descriptors and workload-specific data. They all instantiate the shared device
body in
`tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`
through `GemmType`, `get_best_config`, `make_spec`, and the public
`build_kernel` bridge in
`tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/spec.py`.

- **One heuristic schedule across seven GEMM modes.** `GemmType` covers normal,
  batched, M-grouped contiguous/psum/masked, and K-grouped contiguous/psum.
  `get_best_config` chooses block, stage, cluster, and `swap_ab` geometry once;
  the device body specializes compile-time branches rather than maintaining
  parallel handwritten kernels.
- **Swap-AB as a layout decision.** The `swap_ab` layout flips operand/SF
  routing and selects the `tcgen05.ld.16x256b` plus
  `stmatrix.sync.aligned.m8n8.x4.trans.shared.b16` epilogue. The non-swapped
  path uses `.32x32b` TMEM loads and direct swizzled shared stores. Both remain
  in the same source and share scheduling/pipeline state.
- **Scale granularity is descriptor data, not a second kernel.** `gran_k_for`,
  `sfa_stages_per_load`, `sfb_stages_per_load`, and `utccp_chain` cover
  FP8×FP8 and FP8×FP4 scale staging, including independent A/B load cadence.
- **Grouped traversal stays in the shared scheduler.** The
  `is_m_grouped_*`, `is_k_grouped`, and `grouped_layout` branches derive each
  role's group cursor and effective M/K while leaving the TMA, UMMA, transposer,
  and epilogue roles structurally shared.
- **Converged elected issue.** A single raw `T.ptx.elect_sync` produces
  `mma_elected`; UTCCP, MMA, and matching commits use the same instruction
  predicate. The raw MMA's accumulator-input operand
  `T.Or(ki > 0, mma_k > 0)` is an independent semantic control.
- **Persistent TMA-store source ring.** The epilogue uses
  `cp.async.bulk.wait_group(NUM_TMA_STORE_STAGES - 1)` before reusing a source
  stage, then `commit_group` after the elected TMA store.
- **Persistent scale-stage handoff.** The M-grouped contiguous transposer waits
  for the prior task's stage generation, then executes
  `fence.proxy.async.shared::cta` before its generic loads reuse the shared
  scale bytes. The full-launch Synccheck and Racecheck corpus cases complete;
  only the independently source-backed scale-padding reads remain `review`.

---

### DeepGEMM `sm100_fp8_fp4_mega_moe` — routed plus rank-local shared experts

- **One scheduler, three execution regions.** `block_phase_shared_l1` and
  `block_phase_shared_l2` bracket the routed L1/L2 task stream. SharedLinear1
  runs before `fetch_expert_recv_count`, filling the EP-dispatch bubble;
  SharedLinear2 consumes the fused rank-local intermediate afterward.
- **Four-way descriptor selection without a second device body.** The active
  block phase selects routed L1/L2 or shared L1/L2 A/B and scale descriptors.
  Shared weights use plain E4M3 and therefore select `desc_i_shared`; routed
  FP4 weights retain the original instruction descriptor.
- **Shared experts bypass dispatch communication.** `num_shared_experts > 0`
  reserves full-size non-ring shared buffers and one extra combine slot. Their
  unweighted output is accumulated into every local token while the routed
  experts keep their existing dispatch/combine protocol. At zero shared
  experts, the same constant 18-TensorMap ABI binds routed descriptors into the
  unused shared slots.
- **Proxy-safe combine-stage reuse.** The two TMA load stages overlap the next
  combine-slot fetch with the current reduction. Before a later slot reuses a
  stage, `fence.proxy.async.shared::cta` plus `warp_sync` publishes every
  lane's completed generic-proxy reads to the elected async-proxy issuer.

---

### `tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py` — NVFP4 block-scale GEMM (e2m1 + ue4m3 SF)

- **NVFP4 nibble-packed operand view** — Two e2m1 nibbles per `uint8`
  byte, allocated as `uint8[..., K//2]` and viewed as
  `float4_e2m1fn` at the MMA call. *nvfp4_gemm:* (also
  `mqa_logits_fp4`).
- **Cluster-pair MMA: split A+B, multicast SFB** — In a 2-CTA cluster,
  both A and B are split by `id_in_pair` (each CTA reads its own
  half — no duplicated DRAM); SFB needs full visibility on both CTAs
  for block-scale MMA, so multicast it via `cta_mask=pair_mask`.
  *nvfp4_gemm:* `CLUSTER_M=2`, `id_in_pair` / `pair_leader_rank`; A
  and B per-CTA TMA, SFB multicast.
- **SMEM→TMEM copy** — Raw cluster-collective `tcgen05.cp` transfer.
  *nvfp4_gemm:* `_TCGEN05_CP_2SM` stages SFA/SFB descriptors into TMEM each
  K-stage.
- **Scalar broadcast value baked into register-side epilogue mul** —
  Read a single-element scalar (e.g. alpha) once per CTA into a
  register, then fold it into the readback's mul; saves N reads and
  defers the apply out of the MMA path. *nvfp4_gemm:* raw
  `ld.global.nc.f32` fills `alpha_local`; `_mul_f32x2_inplace` applies it to
  TMEM-readback registers before CVT.
- **L2 evict-normal hint on operand TMAs** — For persistent kernels
  whose K-stream dominates L2 footprint, hinting operand TMAs as
  `evict_normal` keeps them out of MRU so the K-stream isn't
  evicted; opposite of the default. *nvfp4_gemm:* `_tma_g2c_args` sets
  `cache_hint="evict_normal"` on operand loads.

---

### `tirx-kernels/tirx_kernels/ported/flashinfer/gdn_decode/gdn_decode_fp32_mtp_warp.py` — Gated Delta Net decode (FP32 state, MTP)

- **One CTA per batch/head/value-row tile.** `linear_cta`, `v_tile`, `hv`, and
  `n` decompose the launch so four warps partition each `TILE_V` value-row
  tile. `ILP_ROWS` controls the register-resident state rows updated by each
  warp.
- **Publish shared recurrence inputs once per CTA.** Warp 0 normalizes Q/K and
  writes Q, K, decay, and beta to `s_q`, `s_k`, `s_g`, and `s_beta`; the other
  warps can prefetch FP32 state rows before the single `T.cuda.cta_sync()` that
  releases those inputs. `USE_SMEM_V` selects shared or direct BF16 V loads.
- **Frozen, fail-closed source dispatch.** `_source_config` owns the
  `(TILE_V, ILP_ROWS, USE_SMEM_V)` schedule choice. `_require_supported_config`
  rejects unsupported sequence lengths, head configurations, or callers whose
  explicit schedule does not match that source picker.

---

### `tirx-kernels/tirx_kernels/ported/flashinfer/gdn_prefill/gdn_prefill_sm100.py` — Gated Delta Net prefill (FP16, SM100)

- **Sixteen explicit pipeline rings.** `PIPELINE_SPECS` records each full/empty
  ring's depth, byte offsets, producer count, and consumer count; the source
  advances stage/phase explicitly through `_pipe_*` helpers.
- **CTA-private mutable TensorMaps.** Four 128-byte workspace slots copy typed
  Q/K/V/O descriptors via `_descriptor_copy_payload`, update them through
  `_replace_descriptor`, publish with `_tensormap_release`, and acquire before
  exact raw TMA issue. Descriptor identity and lifecycle remain separate from
  the PTX call payload.
- **TMEM-column and warp-role partition.** `TMEM_*_COL`, the CG0/CG1 compute
  branches, issuer branches, descriptor/load role, and output-store role divide
  the 12-warp CTA while exact `setmaxnreg` calls expose each role's budget.
- **Synchronization disposition.** The current canonical source acquires the
  single-slot `state_inp_ready.empty` ring before reuse and synchronizes the
  four CG1 compute warps before TMEM deallocation. The canonical full-launch
  Synccheck case is clean. Racecheck remains `review`: three same-warp TMEM
  `read_write` findings require register-dependency review. The former
  physical-alias stale-read advisories were checker-created TMEM owner splits
  and are no longer reported.

### `tirx-kernels/tirx_kernels/ported/flashinfer/gdn_prefill/gdn_cp_prefill_sm100.py` — chunk-parallel Gated Delta Net prefill

- **Four explicit launch phases.** `t_precompute`, `mn_precompute`, the
  shape-selected fixup, and `prefill` preserve the source algorithm's chunk
  transfer/local-state workspaces and make every inter-launch dependency part
  of the public host contract.
- **M/N affine recurrences share one 512-column TMEM allocation.** WG0 and WG1
  retain M/N matrices while warp 8 and warp 11 issue their independent MMA
  streams. `MN_OPT_PIPELINES` owns all seventeen full/empty rings, including
  the two-consumer `x_ready` handoff.
- **Completion precedes cross-warp TMEM reuse.** `_mn_opt_materialize_x`
  executes `tcgen05.wait::ld` before publishing `x_ready`, so warp 11 cannot
  overwrite `MN_OPT_TMEM_XY_COL` while WG0's asynchronous read is outstanding.
  The valid producer warpgroup also uses one uniform `setmaxnreg.dec(72)`
  contract.
- **State rings remain backpressured.** The prefill consumer acquires the
  single-stage `state_input` empty generation before republishing it. The
  four-phase Synccheck result is clean; Racecheck has no error/incomplete and
  retains only five same-warp true-register-dependency TMEM reviews.

---

### FlashInfer activation + quantization ports — vector ownership and bit-exact packing

- **One activation template, three functions.** In
  `tirx-kernels/tirx_kernels/ported/flashinfer/activation/act_and_mul.py`, `_ACTS` and
  `VEC_BYTES=16` keep SiLU, GELU, and tanh-GELU behind one 16-byte vectorized
  grid plus scalar remainder schedule; the `griddepcontrol` pair is preserved.
- **Fuse gate math into expert-local packing.** In
  `tirx-kernels/tirx_kernels/ported/flashinfer/activation/silu_and_mul_nvfp4_experts_quantize.py`,
  `_launch_shape` partitions a persistent grid by expert, `SF_VEC_SIZE=16`
  computes one E4M3 scale per block, and `_fp32_vec_to_e2m1_16` packs the fused
  SiLU×mul result directly into NVFP4 without an intermediate tensor.
- **Share instruction-level quantization, not launch policy.** The four public
  modules under `tirx-kernels/tirx_kernels/ported/flashinfer/quantization/` import
  packing, absmax, scale, shuffle, and SF-offset helpers from
  `flashinfer/utils/fp_quant.py`. Each module still owns its source dispatch:
  `tirx-kernels/tirx_kernels/ported/flashinfer/quantization/mxfp4_quantize.py` uses one
  thread per 32-element SF block,
  `tirx-kernels/tirx_kernels/ported/flashinfer/quantization/mxfp8_quantize.py` selects
  two or four threads with `_use_2t`,
  `tirx-kernels/tirx_kernels/ported/flashinfer/quantization/nvfp4_quantize.py` owns the
  optional fused SiLU×mul path, and
  `tirx-kernels/tirx_kernels/ported/flashinfer/quantization/nvfp4_quantize_per_token.py`
  owns the per-row reduction.
- **Keep the source SF layout as an output contract.** `sf_offset_128x4` and
  `sf_offset_8x4` calculate the swizzled scale byte, while the data packers emit
  exact E2M1/E4M3/UE8M0 carrier bits. There is no post-kernel layout conversion.

---

### FlashInfer selective-state-update family — six schedules, shared recurrence semantics

The STP/MTP × simple/vertical/horizontal registry entries live under
`tirx-kernels/tirx_kernels/ported/flashinfer/mamba/`. Their module split is the
schedule choice; the state-update algorithm and dtype conversions stay shared
through imports from
`tirx-kernels/tirx_kernels/ported/flashinfer/mamba/selective_state_update_stp_simple.py`
and
`tirx-kernels/tirx_kernels/ported/flashinfer/mamba/selective_state_update_mtp_simple.py`.

- **Simple is the scalar semantic anchor.** `_state_bits_to_f32`,
  `_f32_to_state_bits`, `_load_two_byte_vector`, and the explicit PTX arithmetic
  define BF16/FP16/FP32/int16 state handling, optional state indices, and
  update/no-update behavior once per STP or MTP family.
- **Vertical separates producer and consumer over one staged tile.**
  `tirx-kernels/tirx_kernels/ported/flashinfer/mamba/selective_state_update_stp_vertical.py`
  and
  `tirx-kernels/tirx_kernels/ported/flashinfer/mamba/selective_state_update_mtp_vertical.py`
  use `_TMA_G2S_4D`, `_BULK_G2S`, and explicit full/empty mbarriers to overlap
  state/input movement with the recurrence.
- **Horizontal pipelines state columns.**
  `tirx-kernels/tirx_kernels/ported/flashinfer/mamba/selective_state_update_stp_horizontal.py`
  and
  `tirx-kernels/tirx_kernels/ported/flashinfer/mamba/selective_state_update_mtp_horizontal.py`
  implement fill/steady/drain stages in `_producer_horizontal`, so the next
  state column can move while consumers update the current one.

---

### FlashInfer KDA decode family — dispatch-shaped specializations

- **One-warp vs grouped CTA is a real schedule boundary.**
  `tirx-kernels/tirx_kernels/ported/flashinfer/kda/recurrent_kda_decode_one_warp.py`
  handles single-token calls with at least `ONE_WARP_MIN_SEQUENCE_HEADS`; state
  stays in registers and `_reduce_k_group` uses shuffles. The sibling
  `tirx-kernels/tirx_kernels/ported/flashinfer/kda/recurrent_kda_decode_grouped.py`
  handles smaller single-token batches and all speculative T>1 calls, with
  `_grouped_tiling` and one CTA barrier between preprocessing and the sequential
  recurrence.
- **One-warp output publication is ordered.** The source initializes its output
  tile, executes `T.cuda.warp_sync()`, and only then performs the cross-lane
  final stores. The full-launch numerical oracle, Synccheck, and Racecheck cases
  are ordinary clean corpus entries.

---

## §2 By technique (reverse index)

When exploring an optimization direction, find the kernels that already use
it and read them as a reference. Each entry names the symbol(s) to grep for
in the cited file.

### MMA

- **Cluster-pair MMA (`cta_group=2`, one MMA across both CTAs)** —
  `tirx-kernels/tirx_kernels/gemm/fp16_bf16_gemm.py`,
  `tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py`, and the cluster-shaped
  specializations of
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`.
- **Cluster-pair MMA: split A+B, multicast SFB (`pair_mask`)** —
  `tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py`.
- **Multi-consumer WGs sharing one B SMEM to scale tile M
  (`NUM_CONSUMER`)** — `tirx-kernels/tirx_kernels/gemm/fp16_bf16_gemm.py`.
- **Pack TMEM for concurrent MMAs (disjoint regions + fp-width
  aliasing)** — `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py`.
- **Keep the running accumulator in TMEM (in-place rescale +
  `accum=True` MMA)** — `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py`.
- **A direct from TMEM, written by cast warps** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/tf32_hc_prenorm_gemm.py`.
- **Fold single-side dims into matmul M/N (GQA `GQA_RATIO`)** —
  `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py`.
- **Swap matmul LHS/RHS operands (`swap_ab`)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`.
- **Raw descriptor/TMEM recurrent-update sequences** —
  `tirx-kernels/tirx_kernels/ported/flashinfer/gdn_prefill/gdn_prefill_sm100.py`.

### Data movement

- **Persistent L2-grouped tile order** — `ClusterPersistentScheduler2D` in
  `tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py`; the in-source
  `get_swizzled_block_idx` transcription in
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`.
- **`ClusterLaunchControlScheduler` + `l2_group_size`** —
  `tirx-kernels/tirx_kernels/gemm/fp16_bf16_gemm.py`.
- **Cluster-collective SMEM→TMEM (raw `_TCGEN05_CP_2SM`)**
  — `tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py`.
- **SMEM scale transpose before raw `tcgen05.cp` (`emit_sf_transpose` or
  explicit post-layout address mapping)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp4.py`,
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`.
- **Co-bundled TMAs under one `expect_tx`** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp8.py`.
- **Chunked TMEM→register epilogue readback (`NOL`)** —
  `tirx-kernels/tirx_kernels/gemm/fp16_bf16_gemm.py`.
- **`tcgen05.ld.16x256b` + `stmatrix.x4.trans` TMEM→SMEM transpose** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`,
  `tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py`.
- **Swizzle choice gated on per-tile row width** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`.
- **L2 evict-normal hint on operand TMAs** — `tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py`.
- **Physical TensorMap slot lifecycle** — typed-payload copy, replace,
  release, and acquire in
  `tirx-kernels/tirx_kernels/ported/flashinfer/gdn_prefill/gdn_prefill_sm100.py`.
- **Direct low-precision packing with source-layout SF stores** — the exact
  data packers plus `sf_offset_128x4` / `sf_offset_8x4` in
  `tirx-kernels/tirx_kernels/ported/flashinfer/utils/fp_quant.py`, consumed by the
  activation and quantization entry points under `flashinfer/`.
- **TMA-staged selective-state update** — `_TMA_G2S_4D` and `_BULK_G2S` in
  the vertical schedules under
  `tirx-kernels/tirx_kernels/ported/flashinfer/mamba/`.

### Warp specialization + pipelining

- **`cta_group=2` clusters** —
  `tirx-kernels/tirx_kernels/gemm/fp16_bf16_gemm.py`,
  `tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py`, and cluster-shaped
  DeepGEMM 1d1d specializations.
- **`FlashAttentionLPTScheduler` + `L2_SWIZZLE` (workload-shape swap)** —
  `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py`.
- **Variable per-row KV range (`load_schedule`)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp4.py`.
- **Split-K with deterministic output axis (`num_splits`)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/tf32_hc_prenorm_gemm.py`.
- **Pack single-warp roles into one WG** —
  `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py` (WG3: TMA-load / TMA-store / MMA-issue),
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py` (WG0: TMA / SF-transpose / MMA),
  `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp4.py`, `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp8.py` (spec WG:
  Q-TMA / KV-TMA / MMA+SF).
- **Cast-and-reduce warps alongside MMA warps** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/tf32_hc_prenorm_gemm.py`.
- **Asymmetric `setmaxnreg` (low for issue-only, high for math)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp4.py`, `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp8.py`.
- **Persistent TMA-store ring (`wait_group(N-1)`)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`.
- **Stream producer fragments via consumer mid-barriers** —
  `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py`.
- **Loop peeling for regional per-iter cost (3-phase KV loop)** —
  `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py`.
- **Runtime grouped traversal (`grouped_layout`, `is_m_grouped_*`,
  `is_k_grouped`)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`.
- **Large fixed role transcription with exact register budgets** — 12 warps split into
  CG0/CG1, issuer, descriptor/load, and store roles in
  `tirx-kernels/tirx_kernels/ported/flashinfer/gdn_prefill/gdn_prefill_sm100.py`.
- **Dispatch-specific recurrent schedules** — `_grouped_tiling` and
  `_reduce_k_group` in the recurrent KDA decode pair under
  `tirx-kernels/tirx_kernels/ported/flashinfer/kda/`; the frozen
  `_source_config` picker in
  `tirx-kernels/tirx_kernels/ported/flashinfer/gdn_decode/gdn_decode_fp32_mtp_warp.py`.

### Synchronization

- **Remote-view barrier for in-cluster signal (`remote_view(0)`)** —
  `tirx-kernels/tirx_kernels/gemm/fp16_bf16_gemm.py`.
- **Mid-stream `partial-ready` barrier to stream producer fragments
  (`p_ready_2`)** — `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py`.
- **`griddepcontrol.wait()` PDL handshake** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`, `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp4.py`,
  `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp8.py`, and
  `tirx-kernels/tirx_kernels/ported/deepgemm/tf32_hc_prenorm_gemm.py`.
- **`any_sync` to prevent SIMT divergence on data-dependent branches** —
  `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py`.
- **Warp-uniform state with one elected instruction predicate
  (`mma_elected`)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`.
- **Explicit many-ring recurrent pipeline (`PIPELINE_SPECS`)** —
  `tirx-kernels/tirx_kernels/ported/flashinfer/gdn_prefill/gdn_prefill_sm100.py`.
- **Full/empty rings around selective-state recurrence** — the vertical and
  horizontal schedules under
  `tirx-kernels/tirx_kernels/ported/flashinfer/mamba/`.

### Math

- **Dual-issue transcendentals on MUFU + CUDA cores
  (`ex2_emulation_2`)** — `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py`.
- **Predicate-elide fp updates below precision floor
  (`rescale_threshold`)** — `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py`.
- **Fold pre-MMA cast pass into operand sum-of-squares** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/tf32_hc_prenorm_gemm.py`.
- **Packed reduction trees (`_mqa_fp4_wrelu_reduce_src`,
  `emit_wrelu_reduce`)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp4.py` and
  `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp8.py`.
- **Two-shuffle butterfly reduction (per-N-lane sum)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/tf32_hc_prenorm_gemm.py`.
- **Scalar broadcast baked into epilogue mul (`alpha_local`)** —
  `tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py`.
- **Scalar bitmask + bit-extract for monotonic gates (`mask_r2p`,
  `apply_causal_mask`)** — `tirx-kernels/tirx_kernels/ported/flashattention/flash_attention4.py`.
- **No-FTZ paired-float reduction (`__ffma2_rn`, `__fadd2_rn`)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp4.py`.
- **Thread-group absmax and bit-exact low-precision conversion** — the
  reduction, shuffle, and packing helpers in
  `tirx-kernels/tirx_kernels/ported/flashinfer/utils/fp_quant.py`.

### Low-precision (fp4 / fp8)

- **NVFP4 nibble-packed `uint8[..., K//2]`** — `tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py`,
  `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp4.py`.
- **Block-scale `tcgen05.mma.block_scale` (sfa/sfb fused in MMA)** —
  `tirx-kernels/tirx_kernels/ported/deepgemm/mqa_logits_fp4.py`, `tirx-kernels/tirx_kernels/gemm/nvfp4_gemm.py`,
  `tirx-kernels/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py`.
- **E2M1/E4M3/UE8M0 carrier packing with swizzled SF output** —
  `sf_offset_128x4`, `sf_offset_8x4`, and the packers in
  `tirx-kernels/tirx_kernels/ported/flashinfer/utils/fp_quant.py`.

---

## How to maintain this doc

When a new technique lands in a kernel:

1. Add a bullet to the kernel's section in §1, **anchored on a function /
   constant / PTX mnemonic / unique string** that you can grep for in the
   cited file. No anchor means the bullet is too vague — rewrite.
2. Add (or reuse) an entry under the matching technique category in §2.

When a new canonical source is added under `tirx-kernels/tirx_kernels/`,
do not copy the registry inventory here. Add a subsection only if it contributes
a distinctive, reusable technique beyond the recurring baseline, then index
that technique in §2. The package registry and README remain the owners of
membership and family names.

Do **not** add line numbers — they drift on every edit and create false
confidence. The audit history of this doc proved it: line refs rotted within
weeks while symbol anchors held up.
