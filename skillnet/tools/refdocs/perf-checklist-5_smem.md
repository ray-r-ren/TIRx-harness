# Section 5 - Shared Memory (SMEM)

- **SMEM elimination** — for each SMEM buffer, decide whether data can stay in
  registers or the consumer can fold onto its producer. Revalidate
  synchronization and numerical correctness after eliminating a buffer because
  buffer and barrier lifetimes change. SMEM is necessary only to carry data
  across a boundary registers cannot:
  - (a) an MMA/TMA operand the engine reads from SMEM;
  - (b) a cross-warp/warpgroup handoff;
  - (c) a cross-lane exchange you cannot avoid by re-choosing the lane partition.

  Most of the judgement is in (c). A lane partition is a CHOICE that PROPAGATES: keep a consumer in its producer's partition instead of re-laying the producer's result out to a different one, and every SMEM operand it reads is just indexed at that partition's coordinate — SMEM is random-access, so an operand PRODUCED in another partition imposes nothing on a reader. So a cross-lane exchange is NECESSARY only when the consumer needs a value a DIFFERENT lane produced no matter which partition you pick (a reduction, an all-to-all, a shared-matrix read); re-laying a producer's own result out merely to line up with where another SMEM operand was written is AVOIDABLE — propagate the producer's partition forward and fold the consumer onto the live fragment.

- **SMEM swizzling**:
  - MMA operands (a buffer kept under elimination-(a)) must be swizzled — `alloc_mma`/`mma_shared_layout`; a plain operand is rejected ("no MMA SMEM descriptor matches"). Confirm it.
  - Bank conflicts: analyze per-lane access patterns; where lanes hit the same bank, propose a SwizzleLayout/SwizzleMode rather than padding; match swizzle row width to access width and ensure base alignment; couple swizzle with vectorized access.

- **SMEM transpose** — for a transposed elementwise MMA-operand store
  (`dst[a,b]=f(src[b,a])`), the operand is already swizzled and the scattered
  transposed store becomes the cost. Prefer reading the natural orientation via
  the consumer's `transA`/`transB`, or use an `stmatrix`/`ldmatrix` register
  transpose. CUDA-core, operand-producer, and GMEM-output stores require
  separate access-pattern analysis.
