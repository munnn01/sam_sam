# Parallel task P0.5 implementation plan

> Use superpowers:executing-plans with TDD and one final independent review.

Goal: run task-specific AR, OD, IC diagnostics alongside repaired OT, using separate data, checkpoints and reports.

Spec: user authorized parallel deployment after the task-status explanation; video uses VirtualCodec, images use Minnen/CompressAI+BPG. Never apply the192-DCT zero reference formula to image proxies. QPs30/35/40/45, no identity fallback.

1. Fix the live OT prior-report duplicate-LR error with a failing integration regression; make AR cohort selection reject short/corrupt videos; deploy both using their own owned private checkpoint capsules.
2. Add image_proxy_audit.py and tests: task/source/hash/counter binding, actual BPG raw/blur/perturb paired rates, reconstruction quality and eval/noisy-gradient probes; never evaluate the untrained NPP as a trained model.
3. Add private image cloud payload builder and tests. Mount immutable owned R6 kernel outputs and verify checkpoint SHA; materialize the exact original5e277 source bundle separately for proxy-only continuation. Keep optimizer/RNG/config and global phase counters. Audit source and continuation source stay distinct.
4. Run full tests, final read-only review, publish source and launch OD/IC in parallel on linhowi05. Each image job has audit<=5400s, worker<=18000s and Kaggle timeout21600s; no NPP update before a separate trained-task audit.

Review focus: input dataset versions/geometry, entropy-buffer restoration, no validation leakage, stochastic-gradient averaging, byte-measured rate units, immutable legacy execution, truthful budget/progress and no untrained-NPP score claims.
