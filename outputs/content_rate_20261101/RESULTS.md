# Content-rate screening results: 2026-11-01

All four workers stopped before task optimization. No new AP/BD-rate exists. Gate limits: total AND content median error<=35%, p90<=80%, finite direction alignment>=75%.

| Worker | Warm-up PSNR | Total median error | Content median error | Content p90 error | Direction | Task updates |
|---|---:|---:|---:|---:|---:|---:|
| [is-h264](https://www.kaggle.com/code/tranthihongdieu/sam-content-is-h264-20261101) | 25.41 dB | 34.01% | 35.75% | 132.09% | 16/16 | 0 |
| [is-h265](https://www.kaggle.com/code/dngbolm/sam-content-is-h265-20261101) | 25.89 dB | 26.68% | 33.10% | 136.84% | 16/16 | 0 |
| [od-h264](https://www.kaggle.com/code/hoangminhhuy123/sam-content-od-h264-20261101) | 27.31 dB | 35.71% | 36.94% | 128.52% | 16/16 | 0 |
| [od-h265](https://www.kaggle.com/code/ngynanhthuw/sam-content-od-h265-20261101) | 27.34 dB | 28.87% | 34.25% | 140.62% | 16/16 | 0 |

## Cached-feature diagnosis

A single scale fitted jointly to rounded and relaxed rates compromises between two different feature distributions. The relaxed path is the path supplying NPP gradients. Separate heads per quantization mode are an immediate candidate; a positive power mapping is a second candidate. Both remain unverified by a new GPU task run.

| Worker | Separate-scale content median / p90 | Rate gate met offline | Separate-power content median / p90 | Rate gate met offline |
|---|---:|---|---:|---|
| is-h264 | 10.45% / 63.37% | True | 6.91% / 13.93% | True |
| is-h265 | 12.29% / 77.06% | True | 8.53% / 19.45% | True |
| od-h264 | 10.74% / 64.00% | True | 6.84% / 13.09% | True |
| od-h265 | 12.99% / 82.49% | False | 7.94% / 16.87% | True |

## is-h264: error by QP

| QP | Content median / p90 | Median relaxed/rounded content ratio |
|---|---:|---:|
| 30 | 17.13% / 46.90% | 1.43x |
| 35 | 28.13% / 84.77% | 1.57x |
| 40 | 43.71% / 138.49% | 1.96x |
| 45 | 75.00% / 228.80% | 2.95x |

## is-h265: error by QP

| QP | Content median / p90 | Median relaxed/rounded content ratio |
|---|---:|---:|
| 30 | 14.65% / 48.63% | 1.43x |
| 35 | 20.74% / 98.53% | 1.57x |
| 40 | 39.06% / 153.51% | 1.96x |
| 45 | 73.57% / 255.50% | 3.00x |

## od-h264: error by QP

| QP | Content median / p90 | Median relaxed/rounded content ratio |
|---|---:|---:|
| 30 | 18.33% / 43.85% | 1.44x |
| 35 | 28.76% / 85.67% | 1.57x |
| 40 | 45.20% / 134.13% | 1.96x |
| 45 | 77.07% / 226.63% | 2.97x |

## od-h265: error by QP

| QP | Content median / p90 | Median relaxed/rounded content ratio |
|---|---:|---:|
| 30 | 15.67% / 36.79% | 1.43x |
| 35 | 32.58% / 86.88% | 1.56x |
| 40 | 46.39% / 148.49% | 1.92x |
| 45 | 76.04% / 238.93% | 2.89x |

## Checks

Paid bytes/content/non-VCL accounting checked against downloaded audit. Training fit sources do not intersect TRAIN-dev. Screening statistics recomputed. No model/recipe change, gate relaxation or additional job submission during this check.

Candidate fitting uses the original16TRAIN sources and is assessed on the same8TRAIN-dev sources; using this assessment for choosing the next model is development tuning, not an independent final benchmark. COCO VAL was not evaluated because screening failed.
