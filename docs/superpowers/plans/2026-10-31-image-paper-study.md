# Image Paper Study Implementation Plan

> For agentic workers: use superpowers:executing-plans inline, with one final independent review.

Goal: deploy a bounded, reproducible image-paper parameter study from verified existing weights.
Architecture: separate image_study runner and cloud payload module reuse native training/evaluation under recorded scoped adaptations; original core sources remain immutable.
Tech Stack: PyTorch/Torchvision, CompressAI1.2.8, BPG, private Kaggle notebooks, connected GitHub.
Spec: docs/IMAGE_PAPER_STUDY.md

## Global constraints

QP30/35/40/45. No identity fallback. Real BPG forward/proxy backward. No validation leakage. Explicit warmstart history and shortened-screening labels. Nine initial jobs, total worker cap28h. Frozen core42 files. Exact source/input hashes and private owned data. No untrained-NPP metric.

## Task1: study protocol, adapters and runner

Create reproduce/image_study.py and tests/test_image_study.py. Interfaces: study_config(task,profile,lambda_scale), study_objective, study_context(cfg), load_proxy_input(path,sha,task), StudyAnalyzer, BatchCodec, main.

- [ ] Write RED tests for paper/screen schedules and scales, Algorithm1 real-valued forward/input gradients, scoped restoration of native functions, checkpoint task/hash binding, batch byte accounting, resume schedule compatibility and source-disjoint Flickr split.
- [ ] Implement minimally using released ImageNPP and native train/evaluate; fixed/adaptive random-QP sampler, explicit frozen proxy import, FCOS family, OD geometry, partial trained model evaluation/checkpoint and filter baseline.
- [ ] GREEN targeted tests. Runtime CUDA imports are lazy; CPU tests use real Torch algebra and tiny fixtures.

## Task2: frozen private cloud builder

Create reproduce/image_study_cloud.py and tests/test_image_study_cloud.py. Interface prepare_study(root,directory,account,slug,cfg,input_spec), verify_study(directory,root=None).

- [ ] RED tests: task/version/owner/path/SHA binding; separate Flickr data; exact study config and budgets; no canonical legacy runner fallthrough; notebook tamper rejection.
- [ ] Implement immutable source bundle, pinned data, safely scoped input selection and streaming SHA, new runner invocation; private GPU metadata. GREEN tests.

## Task3: operational preparation and publication

- [ ] Verify latest full checkpoint source/config/counters/optimizer/RNG; extract proxy-only capsule with provenance. Transfer only private owned bundles for cross-account jobs and verify readiness/version/privacy.
- [ ] Run full test suite and one independent whole-change review. Fix Important findings RED→GREEN. Verify original42 source hashes.
- [ ] Publish reviewed source under the existing feature branch using a lease. Build frozen jobs, submit up to two jobs per account, verify server privacy/source and actual runtime events. Fix launch errors within scope.

## Review focus

Warmstarts must not silently migrate NPP/Adam/counters. Scoped native overrides must restore even on errors. Real forward values and proxy gradients must both survive. FCOS label/geometry/AP semantics must match COCO IDs. Budget/source hashing must include setup and evaluation limitations honestly. New checkpoints must be resumable with matching config and RNG, and probes must match saved weights. Flickr crop train/dev IDs must be disjoint. No CUDA compatibility or empirical gain can be inferred from CPU tests.
