# Image-paper reproduction study, 2026-10-31

User scope: focus on the local IEEE TCSVT paper Preprocessing Enhanced Image Compression for Machine Vision, inspect munnn01/test, and deploy multiple parameter settings across the authorized private Kaggle pool. Prior constraints remain QP30/35/40/45 and no identity fallback. Parameter/platform choices are delegated by this request; existing publication/private-transfer authorization remains valid.

## Evidence and fidelity

The local13-page IEEE paper, DOI10.1109/TCSVT.2024.3441049, states Algorithm1 real BPG forward values and learned-proxy backward gradients; Minnen proxy fitting on256px Flickr crops; NPP pixel1x1+U-Net with QA layers; loss R+lambda_task*task+0.5*MSE(input,NPP); FCOS and ResNet50 default downstream models; Adam1e-4 with stage targets400k/120k/100k and LR decay at320k/80k/60k. QPs and task lambdas in the paper are28/31/34/37/41 and0.5/1/2/4/8. Released NPP code conditions QA on task-lambda/8, which differs from the paper's raw-QP description.

GitHub branch paper-proxy-p05-20261028 at55bc8de2a432310e25b7ad75c4a2005f1863b360 matches the current core implementation. It already implements Algorithm1 correctly, but uses FasterRCNN for OD, task-specific COCO/ImageNet proxy data, global400k bank accounting, and incomplete image schedules. Latest verified proxy counters are315254 OD and283760 IC; neither NPP is trained. The paper's Flickr citation is Young et al.2014 (Flickr30k). Kaggle eeshawn/flickr30k/version1 has raw JPEGs; no author proxy/NPP checkpoint is present in the pinned official repository tree6a77c4633cd5bb6750c77ea9d213d38ffc88ee7a.

## Chosen approach

Continue the existing reproduction machinery in separate new modules; retain the42 original core files unchanged. Use released image NPP code and real BPG Algorithm1. Change the OD backbone family to FCOS; use Torchvision pretrained FCOS/ResNet50 with fingerprints and explicitly disclose that these are not the exact OpenMMLab checkpoints. Use standard800/1333 aspect-preserving OD geometry, even dimensions for BPG, and224px classification crops. Correct proxy data in a separate Flickr branch rather than silently relabel old proxy history.

NPP branches warmstart only frozen proxy tensors from the latest verified checkpoint. NPP/Adam start fresh with the same seed within a task; original NPP, proxy optimizer and phase counters are not migrated into a new experiment. Record original proxy data/source/counters/SHA. Thus these are paper-recipe warmstarts with declared deviations, not exact original-paper equivalence.

Fair screening: per task lambda multipliers0.5/1/2, beta0.5, LR1e-4, fixedQP35,1000 fixed+1000 adaptive updates. Adaptive QPs are sampled randomly with saved RNG. Full-schedule branches retain120000 fixed+100000 adaptive targets and paper LR milestones; per-session caps can stop them earlier and preserve continuation state. Do not compare unequal-stage budgets as a lambda ablation.

Flickr branch: pretrained CompressAI Minnen2018 bank, four QPs, actual BPG targets, batch4,256px deterministic crop sampling,400000 global bank updates with per-QP counts reported. Exact original400k-per-model interpretation and unpublished proxy initialization remain unresolved, so this accounting stays explicit. A heldout Flickr source split is excluded from training; no COCO/ImageNet validation is used for optimization. Full proxy targets may require continuation.

## Budget and outputs

Nine initial private jobs across five authorized pool accounts: two full-schedule sessions capped6h each; six fair screenings capped2h each; one Flickr proxy session capped4h. Total worker cap28h, service timeouts with setup reserve at most38h. Check fresh quota and occupied slots, never cancel unrelated jobs. All jobs stop on nonfinite/missing gradients, save optimizer/RNG/phase counters on regular budget stops, record actual learning rate and parameter drift, and never replace a poor NPP with identity.

Screen validation uses identical source-disjoint COCO128/ImageNet1024 validation identities within each task; full sessions use COCO512/ImageNet2048 subsets in this first pass. Reserve evaluation time. Evaluate trained current NPP, codec-only and Gaussian0.5 against actual bytes and whole-subset AP/Top1. Four-QP BD-rate is valid only for strictly monotone overlapping curves; no extrapolation or curve repair. These subsets cannot establish full5k/50k benchmark gains. Save the trained checkpoint whose metrics are evaluated, including partial-stage checkpoints.

Progress receipts must distinguish bootstrap, real NPP updates, proxy-only updates, partial curriculum completion, evaluation completion, and scientific completion. Source and private input SHA are verified before training; server notebooks/privacy are checked after submission. No secrets enter source, logs or notebooks.

## References

- Local PDF: D:/STUDY/LAB/News/Preprocessing_Enhanced_Image_Compression_for_Machine_Vision.pdf, pages4–6.
- https://github.com/XingtongGe/PreprocessingICM/tree/6a77c4633cd5bb6750c77ea9d213d38ffc88ee7a
- https://github.com/munnn01/test/tree/paper-proxy-p05-20261028/paper_reproduction
