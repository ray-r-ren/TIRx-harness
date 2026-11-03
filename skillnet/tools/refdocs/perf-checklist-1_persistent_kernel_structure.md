# Section 1 - Persistent Kernel Structure

Persistent kernels use a persistent grid and role-major warp specialization to
keep engine issue streams regular and to preserve overlap across work tiles.

- Persistent grid, 1 CTA per SM: use launch bounds min blocks per SM = 1, `T.cta_id([SM_COUNT])`, and a tile scheduler loop. Audit every `T.cta_id([N])`; `SM_COUNT * k` means k CTAs per SM.
- Role-major warp specialization: split warps and warpgroups into single-role branches, each owning its own tile-stride loop. The audit is loop nesting, not only `wg_id`/`warp_id` guards. TMA issue, MMA issue, and CUDA-core compute need disjoint owner warp/warpgroup roles.

```python
# WRONG - one shared tile loop with role-gated steps scatters each issue stream.
for tile in ...:
    if wg == COMPUTE: ...
    if wg == MMA:     ...  # gemm issue 1
    if wg == COMPUTE: ...
    if wg == MMA:     ...  # gemm issue 2 in a different block

# RIGHT - each role owns one branch and one tile loop holding all its issues.
if wg_id == 3:
    T.ptx.setmaxnreg.dec.sync.aligned.u32(48)
    if warp_id == 1:        # TMA load role
        for tile in ...: ...
    elif warp_id == 2:      # TMA store role
        for tile in ...: ...
    elif warp_id == 0:      # MMA issue role
        for tile in ...: ...
elif wg_id < 2:             # CUDA-core compute role
    T.ptx.setmaxnreg.inc.sync.aligned.u32(200)
    for tile in ...: ...
elif wg_id == 2:            # correction / epilogue role
    T.ptx.setmaxnreg.dec.sync.aligned.u32(64)
    for tile in ...: ...
```

  Raw TMA and MMA issue sequences are async-engine work; softmax, correction,
  and epilogue rescale are synchronous CUDA-core compute. Give each engine a
  dedicated owner role, keep all issues of one role in that role's branch, and
  flag any convenience TMA/MMA/CUDA-core issue placed inside another role.
- Express cross-role dependencies with the right primitive. A **directional
  handoff** — role 0's A produces what role 1's B consumes — is an mbarrier
  (producer `.arrive()`, consumer `.wait()`): the producer keeps running, so the
  pipeline stays overlapped. Reserve `bar.sync`/`T.cuda.cta_sync()`/`warpgroup_sync`
  for a **collective rendezvous** where every role must reach one line before any
  proceeds (e.g. all roles' A_i done before any role's B_i starts). A barrier used
  for a one-way handoff over-synchronizes and serializes the roles, defeating warp
  specialization.
- Use one mbarrier per distinct sync event and one logical buffer per purpose; overlap storage through pool-base aliasing only for disjoint lifetimes.
