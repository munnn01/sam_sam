# Two-paper reproduction implementation plan

> For agentic workers: execute inline with superpowers:executing-plans; use test-driven-development and a final independent review when available. User authorized implementation and Kaggle runs; do not repeat approval gates.

Goal: implement and launch independent image OD/IC and video AR/OT runs, using original paper task definitions and user-mandated QP30/35/40/45.

Architecture: vendor the pinned image NPP; use a Minnen autoregressive learned-image proxy with real BPG forward/straight-through backward. Reconstruct the under-specified video preprocessor/virtual codec explicitly, freeze analyzers, evaluate using real x264/x265. Keep source and scientific-fidelity receipts. No identity fallback or teacher distillation.

Tech stack: Python, PyTorch/Torchvision, CompressAI, FFmpeg/libbpg, pytracking DiMP50, Kaggle private GPU notebooks.

Spec: latest user request supersedes joint AR+OD: original tasks are image detection/classification and video action recognition/tracking. QPs are explicitly retained at30/35/40/45. Earlier source analysis is output/ar_od_paper_reproduction_20261023/DESIGN_VI.md; its joint-task sections are superseded.

## Constraints and review focus

- Each task has its own trained NPP; do not claim a shared joint-task model.
- Image paper uses BPG, video paper uses H.264/H.265 medium; full stream bytes determine reported bitrate.
- Learned proxy values must not be reported as actual compression results.
- Source-group TRAIN/validation disjointness and original label mapping are mandatory.
- Preserve partial-session checkpoints and optimizer/RNG state; a budget stop must not say scientific_complete.
- No random-head, teacher-label or template-only tracking result may substitute for original pretrained analyzer metrics.
- Missing author details, altered task implementation, resized input or subset data must be declared.
- Per-QP metric tables must be complete; invalid BD-rate stays invalid.

## Task1: protocol, pinned NPP, virtual codec

- [ ] Write tests for task/QP contract, paper proxy lambda interpolation, bridge derivatives and no-fallback checkpoint validation; watch RED.
- [ ] Vendor image fixed/QA NPP + license at official commit6a77c4633cd5bb6750c77ea9d213d38ffc88ee7a. Implement wrappers and explicit video reconstruction.
- [ ] Implement factorized-CDF residual entropy and intra/inter/DCT/quant/dequant virtual codec. Tests check DCT inversion, finite input gradients, eight real-codec conditions and reference reset.
- [ ] Run task tests GREEN and record deviations.

## Task2: real datasets, analyzers, real codecs, training and evaluation

- [ ] Tests first for source disjointness, image/box coordinate transforms, multi-frame video loading, rate units and temporal crop labels.
- [ ] Implement actual COCO, ImageNet, K400 and GOT10k loaders; frozen ResNet50/FasterRCNN/Swin3D-T/DiMP50 adapters.
- [ ] Implement BPG and FFmpeg standard-codec roundtrips, task-specific paper objectives, phase schedules and resumable state.
- [ ] Implement real-codec validation, task metric aggregation and conservative BD-rate. No step0 fallback.

## Task3: package, run, inspect

- [ ] Tests first for immutable payload hashes, exact requested task/config binding, source verification and incomplete run receipts.
- [ ] Create private GPU payloads for four original tasks. Build/check libbpg and pytracking dependencies in preflight.
- [ ] Run local suite and source/fidelity audit. Publish source on a separate GitHub branch if needed by deployment.
- [ ] Launch one run per task in configured Kaggle pool; inspect bootstrap, assets, task weights and actual finite training steps. Fix failures and relaunch without rewriting successful job provenance.
- [ ] Report URLs and exact running/complete/blocked state, target versus completed steps and material reproduction limits.
