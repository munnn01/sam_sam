# COCO H264/H265 follow-up

User authorizes analysis, parameter selection and deployment across Kaggle pool accounts. Keep NPP/real codec/frozen task/proxy backward/offline SAM topology. No tracking launch.

Evidence: SAM IS BD-rate +7.69 vs control -9.81; OD SAM -12.98 vs control +2.74 on 32 images. IS SAM AP at QP30/35/40 is below control despite comparable/higher rates. Mask weighting emphasizes merged interiors rather than instance boundaries; native loss sums five detection/mask terms. These are hypotheses, not isolated causal findings. Different warm-up states confound previous comparisons.

Plan: add tested cross-codec profile, explicit shared warm-up import/export with configuration/source/cohort identity validation, codec-specific reconstruction/rate proxy calibration, weighted mask-task loss and boundary reconstruction. Pair control/improved sequentially within each worker to share exact warm-up/proxy/RNG. Four workers (IS/OD x H264/H265) on four available pool accounts; two arms each, 768 task updates, LR3e-5, 64 VAL images, four QPs. Each arm cap6000s; worker pair service timeout13200s. Check quotas before submission, never cancel unrelated jobs.

COCO frames are encoded independently with yuv420p and actual elementary stream bytes including headers. This is an intra-image codec experiment, not temporal video coding. Log component losses and codec/proxy probes. Preserve invalid BD-rate for nonmonotone/no-overlap curves. Original BPG profile remains unchanged. CPU pytest and actual codec smoke checks precede cloud deployment; verify cloud source and startup, and report failures honestly.
