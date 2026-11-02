# Content-rate follow-up ledger

User approved the proposed next step and moved all code/outputs to D:/STUDY/LAB/SAM. Copied source and metadata without copying old outputs/checkpoints. Prior code remains untouched. GitHub uses the existing paper_reproduction layout; local workspace is SAM.

Using Superpowers writing-plans, TDD, executing-plans and verification-before-completion. Plan: docs/superpowers/plans/2026-11-01-codec-content-rate.md. No new approval required for previously authorized public-input private Kaggle runs.

TDD: first5tests failed on missing codec_content_rate module, then5passed. Full-geometry train/dev/RNG paired calibration test passed. Fixed-cost hiding a10x content error reproduced RED; screening now checks both total and content errors, then targeted8tests passed. New binder test passed after module was written; its collection was delayed, so no missing-module RED claim for that test. Existing strict binder tests cover source/config tampering.

Real codec smoke32points passed: H264/H265 x4QP x128/512 xflat/random. All bytes accounted exactly by Annex-B NAL partition; non-VCL byte counts stable across content in these tests. Definitions checked against official FFmpeg H264/H265 NAL tables. This is not a task accuracy or temporal-video benchmark.

Entropy/analysis weights remain frozen while only reconstruction decoders are refit on full TRAIN geometry. Four positive content scales fitted after NPP warm-up on16TRAIN sources x4QP x{x,NPP(x)} in rounded and relaxed modes. Eight disjoint TRAIN-dev sources audit total/content/reconstruction and paired deltas;2devsources also finite direction probes on both variants. Fixed overhead is measured non-VCL bytes, not flat-image total stream size. Screening requires relaxed total AND content median<=35%,p90<=80%,direction alignment>=75%. Screening failures block task updates. Gate thresholds are declared screening rules, not evidence of generalization.

Full suite and final independent review /root/review_content_rate are in progress. No Kaggle run submitted before they complete. Four quota checks requested; all logs/results stored under SAM/outputs. No tracking job planned.

Full suite173passed,4third-partywarnings280.85s. Final independent review found no Critical/Important issues. Source published88d26cfdedbc9e388ccfdbad90eafd4eb39e3228,10exactblobsverified,90sourcefiles,original42coreunchanged. Fourv1workers acknowledged on the same four authorized accounts; sourceprivacy/startup checks underway. No fresh BD-rate claim.

Final launch verification: four RUNNING, remote notebook/privacy exact all4, payload verification stage only; no screening/task/BD-rate claim yet. Registry published67fd17124c9759ff641e088596408052d89f38cb and exact blob checked. All new code and outputs remain in SAM.

Result check2026-11-01: freshERRORall4; warm-up25.41/25.89/27.31/27.34dB passed, proxy updates256each, task_updates0, no task model/VALmetrics/BD-rate. Initial proxy gate blocked all4. Content median errors35.75/33.10/36.94/34.25percent; p90132.09/136.84/128.52/140.62percent. Direction16/16each does not establish rate magnitude accuracy. Paid byte partition validated on64paired audit rows/job, fit16TRAIN and8TRAIN-dev disjoint. Diagnosed single scale mixing rounded and relaxed rate features; median relaxed/rounded ratios1.4xQP30 to~3xQP45. Offline TRAIN-only separate positive-power fits on cached features reduce content p90 to13.1-19.5percent on the same TRAIN-dev; not a new GPU/task result, development tuning only. No ML source changes, gate relaxation or extra GPU submission this check. Results/registry/report committed5e00c2fdee9f381df81d8528a8428608805c4fa9;3blobsverified. All outputs remain SAM.
