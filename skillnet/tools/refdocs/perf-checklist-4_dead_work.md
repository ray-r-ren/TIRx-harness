# Section 4 - Dead Work

- Iterate and launch over actual tile count; avoid structurally empty tiles.
- Avoid full-tile serial scans that iterate the whole logical tile and elect work
  with `idx % thread_count == thread_id` or equivalent. Even when the produced
  values are live, this is dead control/predicate work; each thread should
  iterate only its owned elements, or use tile/warpgroup primitives.
- Bound loops to live ranges, especially triangular or suffix-only regions.
- Skip expensive terms that will be masked, discarded, or never read.
- Drop redundant initializations that are immediately overwritten or outside the consuming guard.
- Remove always-true guards and dead branches.
- Remove provably dead or duplicate fences/syncs; rerun synccheck/racecheck if changed.
- Shrink over-broad sync scope: if a barrier only orders one warpgroup, use
  `T.cuda.warpgroup_sync(id)` (128 threads), or `T.ptx.bar.sync(id, count)` for an
  exact participant count, instead of a full-block `T.cuda.cta_sync()` that stalls
  uninvolved warps. Scope must still cover all producers+consumers (too narrow =
  race); pick a free named-barrier id (0–15; framework cooperative ops use 8);
  rerun synccheck/racecheck.
