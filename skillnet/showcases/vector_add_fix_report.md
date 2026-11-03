# Pasted kernel code: locating and fixing a lane-range bug

**Before** (`simulate_kernel`, code pasted as text): FAIL: kernel vector_add output differs from the reference. out: 16 of 32 wrong (flat 16-31; max_abs_err 2.0)

- wrong outputs: 16 of 32, flat range [[16, 31]]
- first mismatch: index 16, got 15, expected 17 (a - b instead of a + b)

Lanes 16–31 took the other branch of a lane-dependent select. Fix:

```diff
-    txl.ptx.st.global_.f32(out.ptr_to([lane]), txl.Select(lane < 16, x + y, x - y))
+    txl.ptx.st.global_.f32(out.ptr_to([lane]), x + y)
```

**After**: PASS: kernel vector_add matches the reference. out: all 32 match (max_abs_err 0.0)

**Safety**: Safety CLEAN for vector_add: Synccheck clean; Racecheck clean
