# Software Pipelining via Graph Cutting

A concrete methodology for attacking loop-carried recurrence bubbles in
warp-specialized kernels. Apply it when source inspection identifies a
per-iteration DAG across multiple lanes (MMA / WG compute / TMA / DSMEM) and
benchmark plus NCU evidence indicates that pipeline serialization is worth
investigating.

## The unifying idea: PD = edge cut

**Every time you set `PD > 1` on a buffer, you are cutting the producer→consumer edge that flows through that buffer.** PD value = cut distance + 1.

The existing kernel conventions — `Q_smem PD=2`, `dO_smem PD=2`, TMA-load-then-GEMM patterns — are already applied SWP cuts. They just weren't named as such.

This methodology is the **generalized form**: you don't just decide PDs locally per buffer, you reason about **which edges in the full DAG to cut**, then realize each cut by setting PD on the buffers it crosses (or by region overlap + anti-deps).

Three views of the same action:

| Layer | Operation |
|-------|-----------|
| Topology | Cut a forward edge (or graph cut) in the DAG |
| Schedule | Let the producer run N iters ahead of the consumer |
| Implementation | Buffer PD = N+1 (physical) or PD = N+1 realized via region overlap + anti-dep (logical) |

Adding PD is a **specialization** of this method — it's the case where you realize one cut at one edge. The general method lets you reason about many cuts jointly: where they land in the DAG, which buffers (including registers) they multiply, which anti-deps they introduce, and whether the resulting schedule hits the lane lower bound.

## Phase 1 — Diagnose

Start from the **original per-iteration DAG**: the dependency graph where one
iteration holds one instance of every operation. Build it from the kernel
source and generated code. Do not infer hardware execution order or exact
operation latency from program order alone.

What each artifact gives you:

| What you need | Where to get it |
|---------------|-----------------|
| List of ops and issuing warps/lanes | Program source plus generated CUDA/PTX/SASS |
| Original per-iter DAG (forward data deps, pre-cut) | Program source — trace which op writes each buffer and which op reads it within one logical iter |
| Currently-applied cuts (existing PDs) | Buffer declarations in source (`PD=2`, `Pipe.tma(...)`, etc.) — each is an applied cut on the DAG edge through that buffer |
| Runtime, stalls, utilization, occupancy, and memory traffic | Stable benchmark plus NCU |
| Compiler resource usage and instruction mix | Generated-code dump and compiler output |

For candidate ranking, you may construct an explicitly labeled static model:
```
estimated_lane_floor = max over lanes: estimated serial work per iteration
estimated_period     = max(estimated_lane_floor, estimated_carried_cycle)
estimated_headroom   = estimated_period - estimated_lane_floor
```

These values are hypotheses, not profiler measurements. Use them to choose an
experiment, then accept or reject the cut with like-for-like benchmark and NCU
evidence. Small estimated headroom suggests looking elsewhere; large estimated
headroom justifies continuing the analysis.

**Important caveats**:
- **Source-visible mbarrier handshakes are only a subset of the real data
  dependencies.** They show explicit inter-warp synchronization. Same-lane or
  same-warp sequencing, implicit engine serialization, and register-resident
  handoffs are dependencies too. Build the DAG from every buffer producer and
  consumer, not only from mbarrier calls.
- Existing SWP cuts reshape which logical iterations overlap. Annotate every
  node and carried edge with a logical iteration offset before proposing an
  additional cut.
- NCU counters are aggregate hardware evidence; they do not provide exact
  per-operation durations. Keep uncertain latency and scheduling assumptions
  marked `[VERIFY]` until an independent source establishes them.

## Phase 2 — Graph cuts (not just single edges)

A **cut** is a partition of the DAG's vertices into `H` (head) and `T` (tail). **Every forward edge crossing from H to T is cut simultaneously.** If an op has multiple producers, a cut may sever multiple dep edges in one go.

### Cut distance

- **Distance 1**: tail[i] runs concurrent with head[i+1]. Every buffer live across a cut edge needs **PD = 2** (realized as physical doubling or as region overlap with anti-dep).
- **Distance 2**: tail[i] runs concurrent with head[i+2] (head[i+1] running in the middle). Buffers need **PD = 3**.
- **Distance N**: buffers need **PD = N+1**. "Buffer" here includes SMEM, TMEM, and register-resident tensors.

PD and cut distance are in **bijection**: choosing PD = k is the same as choosing cut distance k−1. Over-provisioning PD (e.g., PD=3 when you only need distance 1) means you've implicitly chosen a larger cut distance — you just haven't exploited the extra iter of slack in the schedule.

### Existing PDs are applied cuts

Every `PD>1` in the current kernel is an edge cut already in place. Before adding new cuts, list them. Typical patterns:
- `<input>_smem PD=2` → cut on `TMA_load → GEMM_reads` (distance 1, lets the next iter's load run while the current iter's GEMM reads).
- `<output>_smem PD=2` (reverse direction) → cut on `compute_writes → TMA_store_reads` (distance 1, lets the next iter's compute run while the current iter's epilogue store drains).
- A `Pipe.tma(...)` declaration packages a producer/consumer barrier pair around such a cut.

For each existing PD, write down which DAG edge it cuts. Those are the cuts you've already paid for; new cuts compose with them.

New cuts compose with these. When evaluating where to add a cut, the existing cuts are part of the DAG; the question is which additional cuts are needed to eliminate the remaining carried cycles.

### After cutting, H and T are independent

**Critical**: once the cut edges are severed, H-ops and T-ops have no forward deps between them (only anti-deps, if region-overlap introduces any). **They can be arbitrarily interleaved** subject only to:
- Each lane's serial constraint (MMA lane is serialized by tcgen05 proxy, etc.)
- Anti-deps introduced by region overlap
- Original intra-H and intra-T forward deps

This means you don't "fit T inside H's bubbles" — you freely schedule both sets of ops across all lanes. Schedule and allocation are **coupled**, not sequential: a schedule choice can make a region-overlap free (its anti-dep is already satisfied by wall-time order) or costly (forces a timestamp shift). Iterate between schedule candidates and allocation plans.

## Phase 3 — Evaluate each cut

### A. Multiplied storage on cut edges

**Every buffer live across a cut edge needs `PD = cut_distance + 1`. Including registers.**

Register-resident tensors are easy to miss. If you cut through an edge where a register-resident fp32 tile lives, the register count per thread doubles. Once a kernel is already near the per-thread reg ceiling, this pushes it into spill territory fast.

Do the full accounting:
- **SMEM**: each SMEM buffer on cut edges × new PD. Check against 227 KB ceiling.
- **TMEM**: each TMEM column range × new PD. Check against 512 cols.
- **Registers**: each register-resident tensor × new PD. Check per-thread reg budget (`setmaxnreg`, spill threshold).

### B. Region overlap to reduce physical cost

If the physical cost of PD=N+1 exceeds budgets, see if two buffer slots can have their **memory regions overlap** (fully — equivalent to classical aliasing — or partially, sharing only a subset of bytes).

**Region overlap between two buffers is legal iff there is a dependency edge (forward or anti-dep, direct or transitive) in the DAG that orders every access pair which would race on the shared region.** Concretely: every "old-reader → new-writer" pair must have a dep path ensuring the old read completes before the new write.

If such an ordering edge already exists in the current DAG (e.g., through natural forward deps or lane serialization), overlap is free. Otherwise, add an **anti-dep**:
```
consumer of slot_old must finish → before producer of slot_new writes
```

The anti-dep may be within the same iter or loop-carried, depending on where the racing pair sits. Either way it becomes a new edge in the DAG that may affect the new period. Account for it.

**Overlapping regions doesn't save storage for free — it trades storage for a dependency (barrier).** The anti-dep may close the gap back up if it creates a new long carried cycle.

This tool is not limited to cut-edge buffers. Any two buffers — whether they cross the cut or not — can have overlapping regions, subject to the same condition: dep edges ordering all race-prone access pairs. The allocation plan is thus a joint decision over the whole buffer set.

### C. Scope conversion — choose where each buffer lives

Allocation isn't just "how many bytes / cols / regs". A logical tensor can live in different physical scopes at different points in its lifetime, and even simultaneously in different scopes for different consumers. Treat scope as a design variable, not a fixed property of the tensor.

**Available scopes** (Blackwell SM100):
- **SMEM** (~227 KB/CTA): required for TMA targets and MMA B operand.
- **TMEM** (512 fp32 cols × 128 lanes): required for MMA C; allowed for MMA A subject to layout constraints (typically K-major, no transpose flag available).
- **Register** (~64K regs/CTA, distributed across threads): per-thread, fastest; not addressable by MMA. Per-thread budget tightens with `setmaxnreg`.

**Rank the scopes for each cut-edge tensor — do not first-fit.** When a buffer
needs to move, the trap is to take the *first* scope that frees the bytes and
stop. That is how a register hoist gets chosen when TMEM was the better home.
Among the scopes it can legally live in, pick by the *effect of the move*, not
by which scope happens to have room:
1. **Schedule / dependencies** — does relocating ADD ordering or sync (a scope
   round-trip, an extra staging copy, a new anti-dep) or REMOVE it (an engine
   can now read/write the tensor directly, dropping an explicit RMW or staging
   step off the critical path / carried cycle)?
2. **Data movement** — does it cut reads/writes and cross-scope traffic? (A
   per-iter SMEM read-modify-write, its bank conflicts, or a staging copy can
   disappear.)

Fitting the budget (SMEM bytes / TMEM cols / regs) is a *feasibility* gate, not
the criterion — a move into a scope with room is worthless if it adds a
dependency or moves no data.

#### Scope conversion as a budget release valve

When one scope is full but another has room, moving a tensor's residence unlocks cuts the natural scope blocked. A move into a scope is available in **two** forms — don't assume only the first:
- **Reusing a dead window** (zero cost): host the tensor in a region another tensor occupies only during a disjoint phase.
- **Taking dedicated capacity** (costs cols/bytes/regs, but may still be the best-scored placement): if the scope simply has free room, put the tensor there outright. A tensor being **persistent** (alive across all iters) rules out the dead-window form, but does **not** rule out the dedicated form — a persistent tensor can live in TMEM or registers just fine if that scope has spare dedicated capacity. Don't let "it's persistent" reflexively pin a tensor to SMEM.

- **SMEM → TMEM** (free if a TMEM region is dead in the right window; otherwise costs dedicated cols). Dead-window pattern: tensor X occupies a TMEM region only during one phase (e.g. read by a compute warp early in the iter, dead afterward). A later-phase scratch tensor Y can host its own data in that region until the next iter's writer of X overwrites it. Zero TMEM cols added if the dead window contains Y's lifetime. Dedicated form: when TMEM has free cols (check the col budget — MMA outputs often leave half the 512 unused), a tensor — *including a persistent one* — can simply take its own cols.
- **TMEM → SMEM** (less common — SMEM usually tighter): useful when freeing TMEM cols is worth more than the SMEM bytes it costs.
- **SMEM/TMEM → register** ("register hoist"): buffer a transient between phases in registers to break a barrier wait — e.g. compute the data while a needed memory-clear barrier hasn't fired yet, hold the result register-resident, write to the destination scope only after the barrier. Limited by per-thread reg count; may require `setmaxnreg` to redistribute the per-CTA pool.

#### The loop-carried accumulator may already be an MMA accumulator

When the loop-carried tensor `S` is an **accumulator** — updated each iter as
`S = (decay·)S + A@B` (a recurrent state, running KV cache, chunk-recurrent
hidden state) — check whether `S` can simply *be* the TMEM accumulator of the
GEMM that already computes `A@B`, persisted across iters with `accum=True` so the
tensor core does `S += A@B` in place. That single move is both a **scope
conversion** (S leaves SMEM, freeing it — often the release valve for a cut
elsewhere) and a **fusion** (the add moves off the CUDA-core lane onto the
usually-idle TC lane), collapsing the explicit read-modify-write that would
otherwise be a large op on the busiest lane. It's easy to miss precisely because
it only surfaces when you rank scopes (above) instead of defaulting `S` to where
it already lives.

#### Multi-scope residence — one tensor, two homes

A tensor can be stored in 2+ scopes at once if different consumers have different layout/scope requirements. The producer writes to all scopes in one pass.

Pattern: tensor T is read by multiple consumers whose scope and layout constraints can't all be satisfied by a single physical placement — different consumers may need different axes pinned to different hardware dimensions, or one may require a scope the other cannot read from. Rather than picking one consumer to favor and forcing the others through an expensive layout fix (cross-lane shuffle, scope round-trip, operand-role swap), the producer materializes T into multiple scopes during one pass. Each consumer reads from the placement that satisfies its own constraint.

Cost: one extra write per producer pass per additional scope. Benefit: unblocks the cut that scope-mismatch alone would otherwise force open.

#### Layout constraints couple scope and operand role

Hardware rules decide whether a tensor *can* live in a given scope for a given consumer:

- For a GEMM (MMA), the B operand must be in SMEM.
- For a GEMM, an A operand held in TMEM is K-major and forbids `transA`.
- For a GEMM, an SMEM operand (A or B) must tile a swizzle atom.
- TMEM physical model — 128 lanes × N TCols. Each lane is privately owned by
  one thread. A TMEM readback gives each thread the full set of TCol cells in
  *its own lane* (i.e., one row along the lane-axis). Within that row a thread
  can access any TCol freely. Cross-lane access — one thread reading another
  lane's row — requires explicit cross-lane shuffle.

  So the rule for whether a producer suffers cross-lane cost is: **does each thread's computation stay within its own lane's row, or does it need data from other lanes?** If the latter, the cross-lane shuffle (log-N steps per element) is on the critical path.

**Layout/scope choices propagate — trace through every downstream consumer.**

A tensor's layout (which axis on TLane vs TCol) and scope (SMEM/TMEM/register) aren't local choices. Each chosen layout L pins the consumer-side constraints for *every* downstream op that reads this tensor — and for any tensor those ops produce (whose layout is in turn determined by their operand assignment + L), and so on. A choice that satisfies one consumer can force a mismatch several ops downstream.

Before committing a layout/scope decision, walk the full operator chain:

1. **Producer**: pick a candidate layout/scope L (driven by the cut you want, or the constraint you're trying to satisfy locally).
2. **For each consumer in the chain** (immediate, and transitive — consumers of this op's output, etc.):
   - Does L match the consumer's constraint (required scope, axis-on-K direction, swizzle/atom alignment, available trans-flag, etc.)?
   - If yes → free.
   - If no → identify the fix: cross-lane shuffle, scope round-trip, operand-role swap (which itself propagates a *new* layout downstream), or multi-scope residence (producer writes to two scopes). Price it.
3. **Sum the cascading costs**. The cut's net benefit = original gain − all downstream fix costs.

A producer-side choice that initially looks cheap can become expensive once a downstream consumer's mismatch fix is added in. Equally, a choice that looks costly at the producer can pay off if it unblocks a long chain of cheaper downstream consumers. Evaluate the whole propagation, not the producer-consumer pair in isolation.

#### Composing with PD and region overlap

Scope conversion is orthogonal to where the cut lands and orthogonal to region overlap. The full allocation plan jointly chooses, for each tensor:
- Which scope(s) it lives in, in each phase of its lifetime.
- Whether its region overlaps with other tensors (within each scope).
- What PD multiplies it (per scope).

When SMEM looks ceiling-blocked, ask "can this move to TMEM" or "can this be register-hoisted between phases" *before* declaring the cut infeasible. Often a free dead region in another scope unlocks the same cut at zero cost.

### D. New carried cycle

With the cut applied and any anti-deps introduced, trace the new tightest loop-carried cycle:
- If pure PD=N+1 without any region overlap: the cut edge is gone, new cycle may be much shorter or go through a different path.
- If region overlap introduced: new cycle goes through the anti-dep.

### E. Expected period

```
new_period = max(lane_lower_bound, new_carried_cycle_length)
```

If `new_carried_cycle_length ≤ lane_lower_bound`, you're at the floor — no more SWP helps; further gains need shorter ops (smaller tile shape, more registers per WG, task split across additional warps, etc.).

## Phase 4 — Rank and pick

The criterion is simple: **shortest achievable period among feasible candidates**. Everything else is a constraint check or tiebreaker.

For each candidate cut (or combination of cuts), determine:

1. **Feasibility** — does any valid allocation plan fit all budgets, *under the chosen scope assignment for each buffer*?
   - SMEM ≤ 227 KB after PD, region-overlap, and scope-conversion decisions
   - TMEM ≤ 512 fp32 cols (count dead-region reuse if a tensor is hosted there temporarily)
   - Per-thread register count ≤ spill threshold (consider `setmaxnreg` redistribution if a register hoist is on the table)
   - Hardware layout constraints (K-major for A-in-TMEM, swizzle atoms, etc.) — couples scope choice to operand role
   
   If no allocation plan is feasible *across all scopes*, discard the candidate.

2. **Achievable period** — under the feasible schedule, what's the new carried cycle? `new_period = max(lane_lower_bound, carried_cycle_length)`.

3. **Tiebreakers** (only if multiple candidates hit the same period):
   - Implementation complexity (mbarrier count, prologue/epilogue depth)
   - Robustness to implementation error (fewer subtle anti-dep phases = safer)

**There is no simple heuristic like "pick the latest cut"** — cut cost depends on (a) which specific buffers cross the cut edges, (b) what region-overlap opportunities the resulting schedule enables or blocks, and (c) register pressure from any live-across-edge tensor. A small-tail cut can still be expensive if the single crossing edge carries a large register-resident tensor or forces an unfavorable schedule.

Evaluate each candidate concretely.

## Phase 5 — Implement

1. **Iterate schedule and allocation together**. They're coupled (see Phase 2 note). If allocation can't satisfy schedule, either:
   - Relax schedule (pick a different cut or a different time-slot placement)
   - Restructure allocation (overlap different buffer pairs, or move a tensor to a different scope — see Phase 3.C)

2. **Add mbarriers for anti-deps**. Usually one per region-overlap pair. Single-slot MBarrier with phase alternation is enough for distance-1 cuts; distance-N needs phase tracking.

3. **Loop structure**:
   - **Prologue**: iter 0 runs H only (no prior tail data).
   - **Steady state**: iter k runs H[k] + T[k−d] where d is cut distance.
   - **Epilogue**: after loop, flush T for the last d iters.

4. **Check first, compile second**:
   - synccheck (CPU-side barrier protocol)
   - racecheck (data race detection)
   - **Do not skip these before GPU compile.**

## Common pitfalls

1. **"Intra-iter program order == period"**: wrong. Sum of ops on a serial lane gives a lower bound for that lane; total program-order sum is just the current schedule, not a limit. Only `max(max-per-lane-sum, carried-cycle-length)` constrains period — they're parallel lower bounds, not additive.

2. **Optimize the period, not one dependency in isolation.** A local cut can
   shorten one chain while shifting the critical path elsewhere, leaving the
   period unchanged.

   The modeling target is the **steady-state period** (= `max(lane_lower_bound,
   carried_cycle_length)` — the max of two parallel lower bounds, not their
   sum). Look at the whole proposed schedule: where is the carried cycle, what
   edges does it traverse, and which cut would actually shorten it? Re-estimate
   the full period after the proposed cut, then validate the hypothesis with
   measurement.

3. **Forgetting register PD**: a cut that looks cheap in SMEM/TMEM may explode register usage. Check `setmaxnreg` budgets + register-resident tile sizes × PD.

4. **Hardware layout constraints**: tcgen05 A-TMEM requires K-major. If a buffer is read by two GEMMs with different K-axes, you cannot make both read from TMEM simultaneously. SMEM operands have `transA`/`transB` as a runtime swap; TMEM does not.

5. **Persistent buffers**: buffers alive across all outer iters (cross-iter accumulators, carried state) cannot *alias/overlap* their region with per-iter buffers. But "can't overlap" ≠ "can't relocate": a persistent buffer can still move to a different scope that has free **dedicated** capacity (TMEM cols, registers) — and for a carried *accumulator* that relocation is frequently the best move in the whole kernel, because its update GEMM's TMEM accumulator can host it via `accum=True` (see Phase 3.C "the loop-carried accumulator is often an MMA accumulator"). Do not treat persistence as a reason to pin the tensor to its current scope.

6. **Treating scope as fixed**: defaulting to "this tensor was in SMEM in v_prev, so it stays in SMEM" misses cuts that scope conversion enables. Scope is a design variable, choosable per-phase of a tensor's lifetime and even per-consumer (multi-scope residence). When SMEM ceiling blocks a buffer split, before declaring the cut infeasible, **rank all three scopes** (Phase 3.C) for the blocking tensor rather than first-fitting one: can it move to TMEM (a dead per-iter window, *or its own dedicated cols* if TMEM has room)? Can it be register-hoisted between phases? If it's a carried accumulator, can it *become* its update GEMM's TMEM accumulator (`accum=True`)? Can the same logical tensor live in two scopes serving different consumers? Pick by what the move does to the schedule (added/removed dependencies) and to data movement — fitting the budget is only the feasibility gate.

## Toy example: A→B→C→D→E on 3 lanes

### Setup

Per iter: 5 ops with serial data deps `A → B → C → D → E`. Lanes:

| Op | Lane |
|----|------|
| A  | 0 |
| B  | 1 |
| C  | 2 |
| D  | 1 |
| E  | 0 |

**Assumption**: every op takes exactly 1 unit of time. This lets us reason about
period in abstract units. For a real kernel, use conservative latency estimates
only to rank experiments; report them as estimates and validate each candidate
with the stable benchmark and NCU.

### Current period (no cuts)

Data deps force serial execution: 5 units per iter. Every lane is idle most of the time:
- Lane 0 runs 2 ops (A, E), busy 2/5 = 40%
- Lane 1 runs 2 ops (B, D), busy 2/5 = 40%
- Lane 2 runs 1 op (C), busy 1/5 = 20%

Lane floor = max(2, 2, 1) = **2 units/iter**. Headroom = 5 − 2 = 3 units.

### Apply cut on edge C → D (distance 1)

- **Head**(i+1) = {A, B, C}
- **Tail**(i) = {D, E}
- No forward deps between head(i+1) and tail(i) → they can interleave arbitrarily subject only to lane serial + intra-head/intra-tail deps.

Steady-state schedule (time on X, lanes on Y):

| Lane   | t₁     | t₂     | t₃     |
|--------|--------|--------|--------|
| Lane 0 | A[i+1] | E[i]   | —      |
| Lane 1 | D[i]   | B[i+1] | —      |
| Lane 2 | —      | —      | C[i+1] |

Check: A[i+1] before B[i+1] before C[i+1] ✓. D[i] before E[i] ✓. A on lane 0, E on lane 0 — serialized ✓. B and D both on lane 1 — serialized ✓.

**New period = 3 units/iter**, down from 5. Still 1 above floor because lane 2 has no second op to coexist with C — lane 2 is sparsely loaded, not the bottleneck.

### Cost of the cut

- Intermediate buffer on cut edge (C's output, consumed by D) needs **PD ≥ 2**. At t₁, D[i] reads C[i]'s output while C[i+1] hasn't fired yet (t₃); but at t₃ of iter i+1's cycle — which is concurrent with the next iter's t₁ — C[i+1] is writing and D[i+1] is reading the other slot. Two slots alive simultaneously.
- If this intermediate lives in a register-resident tensor, register pressure ×2.
- If we want the two slots' regions to overlap (one physical buffer): add anti-dep D[i] → C[i+1]. This anti-dep creates a new loop-carried edge: C[i+1] can't start until D[i] has finished. In wall-time terms, that's from t₁ (D[i]) to t₃ (C[i+1]) of the same timestamp cycle — 2 units. Since the full iter is 3 units, the anti-dep is not tighter than the schedule itself. Safe.

### What the cut achieved

- Shortened period 5 → 3 units (40% reduction)
- Gap to floor: 3 − 2 = 1 unit, caused by lane 0 and lane 1 each carrying 2 ops that must serialize within a lane

### Reaching the floor: add a second cut

To reach period 2 (the lane floor), add a second cut on B → C. Now all three lanes can run concurrent ops from different iters.

But **where to place C** becomes a free variable. Two valid schedules reach period 2:

**Option "C at t1"** (on lane 2's t₁ slot):

| Lane   | t₁       | t₂     |
|--------|----------|--------|
| Lane 0 | A[k]     | E[k-2] |
| Lane 1 | D[k-2]   | B[k]   |
| Lane 2 | C[k-1]   | —      |

**Option "C at t2"** (on lane 2's t₂ slot):

| Lane   | t₁     | t₂     |
|--------|--------|--------|
| Lane 0 | A[k]   | E[k-2] |
| Lane 1 | D[k-2] | B[k]   |
| Lane 2 | —      | C[k-1] |

Both cut the same edges (B→C and C→D), both hit period 2. But they differ in **which intermediate buffer can be region-overlapped back to PD=1 physical for free**:

| Option | B→C buffer | C→D buffer |
|--------|-----------|-----------|
| "C at t1" | PD=1 (region overlap OK) — anti-dep C[k-1]→B[k] within same wall: t₁→t₂, free | PD=2 physical (or pay schedule overhead) — D[k-1] read and C[k] write land on same wall t₁ |
| "C at t2" | PD=2 physical (or pay schedule overhead) — B[k] write and C[k-1] read land on same wall t₂ | PD=1 (region overlap OK) — anti-dep D[k-1]→C[k] within same wall: t₁→t₂, free |

The rule: **overlapping two regions always requires adding an anti-dep edge** (old-reader → new-writer). Whether this is free or costly depends on the existing schedule:

- **Zero overhead**: the schedule already places old-reader strictly before new-writer. The anti-dep is redundant with wall-time ordering; the scheduler doesn't have to move anything.
- **Non-zero overhead**: the schedule places them concurrently (same timestamp) or in the wrong order (new-writer before old-reader). Adding the anti-dep forces rescheduling — at best a timestamp shift, at worst a period extension.

So "C at t2" lets the C and D region overlap for free because D[k−1] at (I+1, t₁) naturally precedes C[k] at (I+1, t₂) — the anti-dep D[k−1] → C[k] is already satisfied. In "C at t1", D[k−1] and C[k] both land on (I+1, t₁); forcing non-overlap via anti-dep would move one to t₂ (or a later wall), breaking the period-2 schedule.

The check: trace the physical region across time, find each (reader_old, writer_new) pair that would share it, ask "does the current schedule already order them correctly?". If yes for all pairs → region overlap is free. If any pair is simultaneous or reversed → overlap is still legal but costs a schedule change.

### Scheduling/allocation tradeoff

Both schedules have the same period but different footprint costs. The pick depends on which intermediate is cheaper to double-buffer. This is the classic schedule-allocation dance: a specific schedule choice makes one region-overlap free (halving storage on that edge) while forcing the other to pay PD=2 or a schedule change. Try both and price them.

### Takeaways generalized

1. **Cut location matters more than cut count.** C→D happens to be the ideal cut because it splits the 5-op chain into two halves that load lanes 0 and 1 with one op each per sub-chain. Cutting a different edge (e.g., A→B) would leave an imbalanced split.

2. **Reaching lane floor may require multiple cuts.** Each cut removes one "layer" of serial dependency. If any lane has K ops per iter, you need K−1 cuts through the chain to parallelize them all.

3. **Allocation is decoupled from the cut location.** The cut only dictates that buffers on the cut edge need PD = cut_distance + 1 in principle. Beyond that, region overlap is a general tool — any two buffers (edge-crossing or not) can share physical bytes if their lifetimes don't overlap, and you can *force* non-overlap by adding anti-deps. So the allocation plan is a joint decision over all buffers: which ones to physically double, which pairs to place in overlapping memory regions (possibly using anti-deps that the existing schedule already satisfies, or deliberately paying schedule overhead to introduce new ones). Evaluate the whole plan, not just the cut edge.
