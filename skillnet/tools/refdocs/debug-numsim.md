# NumSim

NumSim executes supported TIRx kernels deterministically on CPU, without a GPU.
It transpiles a specialized kernel, runs concrete inputs, and returns outputs
and diagnostics. Use it to reproduce numerical behavior, compare outputs with
an independent reference, and locate mismatches by output name, logical index,
actual and expected values, and static writer/source sites.

NumSim is not a reference oracle and does not detect races or synchronization
errors. Use Racecheck and Synccheck for those properties.
