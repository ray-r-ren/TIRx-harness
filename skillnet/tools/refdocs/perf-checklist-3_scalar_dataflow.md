# Section 3 - Scalar Dataflow

- Scalar CSE: compute transcendentals, reciprocals, scales, and row factors once and reuse them.
- Hoist loop invariants into real storage when needed; `T.let` may inline and not reduce generated work.
- Check block-level work done by only one thread and identical block-level work
  redundantly done by every thread; both patterns indicate poor thread
  utilization.
