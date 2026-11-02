# SAM 2 Image Study Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy matched control/foreground/background-penalty image-paper screens on private Kaggle accounts.

**Architecture:** Add a SAM mask/cache/loss module and small optional hooks in the existing image study. Reuse immutable cloud packaging and real-codec evaluation; no changes to the original paper core.

**Tech Stack:** PyTorch, pinned official SAM 2.1 Tiny, BPG, CompressAI, Torchvision, Kaggle API.

**Spec:** docs/SAM2_IMAGE_STUDY.md

## Global Constraints

QPs 30/35/40/45; no identity fallback; SAM frozen/offline; TRAIN-only automatic prompts; six bounded jobs; do not interrupt existing jobs.

## Review Focus

Mask/pixel crop mismatch must fail; source/checkpoint/cache tampering must fail; empty masks remain finite; control preserves original objective; actual training must not load SAM.

### Task 1: Mask, cache and loss contracts

- [x] Write tests for matched configs, weighted-loss gradients, empty masks, confidence/area filtering, cache alignment and invalidation.
- [x] Run and observe failures, implement reproduce/sam2_masks.py, rerun tests.
- [x] Pin upstream code/checkpoint and document runtime provenance.

### Task 2: Runtime and immutable packaging

- [x] Add tests for training-context mask binding/restoration and SAM cloud config binding.
- [x] Observe failures, implement optional hooks and bounded preprocessing, rerun full project suite.
- [x] Fresh independent final review, fix findings with regressions.

### Task 3: Publish and deploy

- [ ] Confirm original core unchanged; publish additive source to authorized feature branch with a lease.
- [x] Inspect account quotas/active jobs; publish shared proxy input after explicit public-sharing instruction.
- [ ] Freeze six payloads, submit within slots/budget, verify server source/privacy and observe runtime.
- [ ] Record actual status, limitations and reproducible handles.

Ruling: execute within the user's existing delegation rather than adding another approval gate; the request explicitly asks to run this extension and supplies its architecture. Inline execution with one final independent review.
