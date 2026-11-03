# Section 2 - Static Source Patterns

- Never write `unroll=False` on a hot inner loop — it lowers to `#pragma unroll 1`, forbidding ptxas from unrolling/software-pipelining it: rolled loops pay loop control every iteration and cannot batch their loads, whether the body is independent (elementwise) or a dependence chain. Default `T.serial` emits no pragma and lets ptxas decide (it also handles runtime trip counts); `unroll=True` fully unrolls at compile time (constant-trip small loops only); write `unroll=False` only when ptxas's own unrolling hurts (large-body loop blowing I-cache/registers).
- Prefer `for` over `while`: if a loop can be rewritten as a `for i in range(...):` (optionally with an if-tail when the count isn't divisible), use `for`; only keep a `while` when it genuinely cannot be expressed as a counted loop.
- Use the exact destination-passing `exp2` form for exponentials: `T.ptx.ex2.approx.ftz.f32(dst, T.cast(x, "float32") * 1.4426950408889634)` on fp32.
- Use explicit FMA for multiply-add or multiply-subtract where the API supports it.
- Vectorize register/shared-memory copies with packed low-level operations
  where possible; avoid elementwise copy loops and implicit tile-copy helpers.
- Pack narrow output stores; avoid per-element narrow writes where a packed/vector store is legal.
- Bulk contiguous GMEM<->SMEM movement should use TMA. Skip when transfer is small, non-contiguous, or fixed-shape TMA would overrun externally visible packed/varlen output.
- Cap each TMEM<->REG readback chunk at <=32 registers per thread; split wider readbacks into chunks and wait once.
