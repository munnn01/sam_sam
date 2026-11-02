# Reviewed Antigravity follow-up: keep author NPP
User requests evaluation/improvement/implementation of external implementation_plan.md and explicitly selects keeping original NPP. All code/output stays D:/STUDY/LAB/SAM. Use Superpowers TDD, one final independent review, verification before deployment.

Evidence: all4content-v1 jobs stopped with task0 because mixed rounded/noise scale fails rate gate. Offline TRAIN-only positive-power fits assessed on cached TRAIN-dev have content p90~13-19.5%; these are development diagnostics, not GPU/BD-rate evidence.

Corrections to external plan:
- ReLU/nonresidual architecture is an architectural constraint and optimization hypothesis; it does not prove all bitrate reduction is blur or guarantee a residual variant fixes it.
- Zero-init output makes y=x initially; earlier layers can initially have zero parameter gradients. Identity initialization is not a fallback, but user selects original architecture, so residual is out of current scope.
- A Python separate_modes flag is not checkpoint state. Persist mode selection as a tensor buffer; verify save/load and gradient mode dispatch.
- Validate positive finite calibration data and numerical bounds, both modes/all QPs, stable low-variance fallback; no train/dev leakage.
- Fix structured-map shape in draft: concatenate two N,T,1,H,W maps on channel axis2. Cache both channels with exact identity/content hashes; derive each instance edge before union. GT mask remains only native supervised task target.
- Native ROI requires AQ and verified encoder support. CQP and CRF/AQ are distinct declared protocols; no blind addroi label or cross-protocol QP claims.
- COCO AP is not average per-image AP. Proper paired bootstrap must resample images with multiplicity and recompute corpus AP (clone IDs), reporting invalid BD-rate resample fraction. Defer final CI until sufficient VAL data exist.

Current implementation:
1. Keep legacy ContentProxy behavior by default, add mode-specific positive scale/power buffers and checkpoint-persistent enable flag. Fit log-linear power per(mode,QP) on TRAIN only, bounds power[.25,4], scale(1e-4,1e4); use variance fallback power1. Content/non-VCL accounting and gates unchanged.
2. Add structured offline SAM foreground/instance edge cache and native task wrapper without GT edge extraction. Three arms: control, foreground(interior), sam_boundary(interior+edge). gamma0 for all, foreground_extra1 in both SAM arms, boundary_weight2 only final arm. Same original NPP/native loss weights and qps768updates/LR3e-5, warmup/proxy/optimizer/RNG snapshot restored for both SAM arms.
3. Add Gaussian sigma.5/.8 real-codec baselines on identical64VAL images after control primary evaluation. Retain paid bytes and whole-cohort AP; invalid BD-rate remains invalid.
4. New four Kaggle workers task x codec, three sequential arms each,6000s/arm plus1200s service margin (19200s/job), four prior accounts after fresh quota. No cancellation or private-data transfer. Stage counters take precedence over worker status.
5. TDD mode recovery/checkpoint/gradient, real cached audit regression, SAM edge shape/seams/cache corruption/no-GT edge path, Gaussian invariants, immutable cloud binding. Run complete suite and CPU codec smoke; one independent review, fix Important/Critical findings with regressions. Publish on authorized feature branch with lease and verify blobs/remote notebook/privacy/startup.

Deferred: residual architecture (user choice), online spatial gating/ROI (changes inference costs/protocol), SAM feature distillation/task-agnostic training (new experiment), full500/5000VAL, three seeds and bootstrap CI (after successful pilot, not silently substituted by per-image AP).
