# Section 6 - Register Pressure

- **`setmaxnreg` under-commits the register file (source-visible).** The
  persistent 1-CTA/SM kernel owns the 64K-register SM file (`512 regs/thread x 128
  threads`). `setmaxnreg` only REDISTRIBUTES the conserved total that the launch
  splits evenly into per-thread caps of `floor8(512 / num_wg)`, so two limits bind:
  `sum_wg(cap) <= floor8(512 / num_wg) * num_wg` and **each per-WG cap <= 256** (the
  per-thread register ceiling). The `/8` rounding can waste a little — the max total
  is `256+256 = 512` at 2 WGs, `168 x 3 = 504` at 3, `128 x 4 = 512` at 4. If the
  actual `sum_wg(cap)` falls below that, registers sit idle while threads may
  spill. Raise the caps, favoring a warpgroup that spills, or drop a cap that
  does not redistribute registers. `setmaxnreg` is useful only to skew the
  budget toward a heavier WG; explicit counts are multiples of 8, while an
  omitted WG retains the exact `floor8(512 / num_wg)` default.

- **Spill / pressure from the compiled artifact.** Read `ptxas -v` spill bytes,
  SASS local-memory
  traffic (`STL`/`LDL`), schedule shape, or NCU register stalls to surface
  register pressure and spill sites; do not infer these from source.
  - The front-loadable fix for a genuine heterogeneous-WG imbalance is
    `setmaxnreg` (the executing warpgroup calls collectively, total budget
    conserved, explicit counts multiples of 8). A warpgroup may omit it and
    retain the topology default.
  - Deeper fixes reduce live ranges or move storage to a more appropriate
    memory space. Validate them with compiler resource reports, SASS, and NCU.
