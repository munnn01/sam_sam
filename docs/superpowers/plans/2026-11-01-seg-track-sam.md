# Segmentation + Tracking SAM Study Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with one final independent whole-change review.

**Goal:** Deploy two matched mask-loss arms for each of COCO instance segmentation, COCO detection and GOT10k tracking.

**Architecture:** Add task/data adapters and a bounded runner reusing released ImageNPP, ImageProxyBank, real codecs, SAM2 mask cache primitives and native checkpoint identities. An additive immutable cloud bootstrap chooses the new runner; original42 core sources remain unchanged.

**Tech Stack:** PyTorch, Torchvision Mask R-CNN, DiMP50, SAM2.1 Tiny, CompressAI, BPG/ffmpeg, Kaggle.

**Spec:** docs/SEG_TRACK_SAM_STUDY.md

## Global Constraints

QP 30/35/40/45; real forward/proxy backward; no identity fallback; automatic TRAIN-only masks; only initial box at tracking inference; no GT prompts at segmentation inference. COCO retry R2: 4 jobs, 8 worker-hours /10 timeout-hours; pilot usage recorded separately. Tracking stays in a later separate batch.

## Review Focus

COCO mask polygons/RLE/crowd and resizing must align; tracking frame/mask identities must bind exact clips; hard clipping forward and STE backward must agree; warm-up gate failure cannot be called task training; evaluation budget/cache identity must preserve partial honest results.

### Task1: Task/data contracts

- [x] Add failing tests for configs, COCO masks/geometry, fixed tracking clips/cache identities, frozen task calls and box-only tracking inference.
- [x] Implement focused adapters, observe green tests.

### Task2: Optimization and runtime

- [x] Add failing tests for framewise proxy, positive rate calibration, STE clipping, fidelity gates, real-forward gradient chain, checkpoint phase accounting and immutable cloud inputs.
- [x] Implement warm-up/task stages, mask preprocessing, evaluation and private payloads; run full suite.
- [x] Independent whole-change review; Important findings RED-to-GREEN, full suite green (158 passed). Cloud-driven convergence schedule revision has a fresh regression and full suite green (159 passed); four third-party deprecation warnings remain.

### Task3: Deploy

- [ ] Verify original42 core files unchanged; publish to authorized feature branch with lease.
- [ ] Prioritize4COCOjobs on tranthihongdieu/hoangminhhuy123 after fresh quota checks; use public proxy and public version-pinned COCO dataset. Tracking remains a separate later batch with its own direction check/GPU preflight.
- [ ] Verify server privacy/source, startup and task-gradient/fidelity stages; record real statuses and limitations.

Ruling: user clarified “Image Segmentation Object Tracking”. No DAVIS/VOS job is implemented. Existing delegation authorizes architecture-preserving implementation and Kaggle trials; no additional approval gate. No private checkpoint transfer is needed.
