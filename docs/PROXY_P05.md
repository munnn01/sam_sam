# Proxy P0.5 diagnostics and recovery

The original reproduction runner, virtual codec, objectives and checkpoints are unchanged. Separate runners load SHA-bound task checkpoints and report TRAIN-heldout diagnostics for video AR/OT and image OD/IC. No full benchmark gain is implied.

## Separate image tasks

`python -m reproduce.image_proxy_audit --task od --checkpoint last.pt --checkpoint-sha256 <SHA256> --training-source-sha256 <SHA256> --legacy-root /kaggle/working/legacy_image_reproduction --continue-proxy --out /kaggle/working/paper_results`

OD and IC each use their own R6 checkpoint and four CompressAI/Minnen proxies. The audit restores entropy buffers through each network's public loader, freezes weights, and measures actual BPG bytes on16 heldout TRAIN images. Controlled inputs are raw, two Gaussian blurs and a bounded perturbation; an untrained NPP is never scored. Original-pixel bpp and reconstruction error against BPG are reported at256-scale geometry. Eight noisy input-gradient vectors are averaged before computing their norm, followed by finite-step BPG direction checks. The video192-channel zero-reference formula does not apply here.

Each private image payload mounts its owned version-pinned R6 kernel output, verifies checkpoint SHA, and extracts the exact original training bundle into a separate source root. After an audit budget of at most5400seconds, the remaining worker budget (total at most18000seconds) can continue only the original400k proxy phase. Original config, optimizer moments, RNG and counters are retained. A recorded operational wrapper blocks fixed/adaptive NPP phases, including when the proxy target is already complete. A parent watchdog bounds setup and training and terminates the entire worker process group. Completing the proxy alone does not mark the scientific experiment complete.

## What runs by default

`python -m reproduce.proxy_recovery --task ot --checkpoint last.pt --checkpoint-sha256 <SHA256> --training-source-sha256 <SHA256> --out /kaggle/working/paper_results`

Default `--stage diagnose --max-seconds 7200`:

1. Verify checkpoint SHA, legacy source, completed phase counts, dataset versions and exact TRAIN inventory. Build a source-disjoint calibration/dev manifest with a nested48-source dev reserve.
2. Measure actual x264/x265 bytes for the same16 dev clips, QP30/35/40/45 and raw/NPP inputs. Record total/entropy/zero-reference/excess rates, actual zero/nonzero symbol costs, zero fraction and residual spectra. A large static rate offset alone never fails the sign gate.
3. Cache bounded TRAIN/dev transform symbols, at most512MiB. Screen prior LR1e-3/3e-3/1e-2, then fit shared and per-QP priors. One5400-second prior deadline includes TRAIN cache construction, LR screening, fitting and paired reevaluation, with an explicit evaluation reserve. Count updates globally across the four QPs, log heldout NLL, select the best dev state and report convergence separately from budget completion.
4. Reuse actual-codec measurements only when model/cohort/clip/settings identities match. Recompute paired proxy responses after fitting. A failed or insufficient gate blocks lambda recovery and records `supervised_rate_head_required`; P1 is a next action, not an already-trained model.
5. Measure Gaussian/bilateral filter rates and actual frozen-analyzer gradient components for STE versus uniform-noise rate paths. Average gradient vectors from eight antithetic draws before computing norms, and report variance/RMS separately for both NPP parameters and NPP outputs. Hard quantized reconstruction stays identical. Residual energy uses the full2D FFT so spatial orientation does not bias frequency fractions.

The scalar zero-bin probe uses deterministic midpoint integration and reports the absolute expected derivative separately from the average absolute noisy derivative. Subtracting the frozen zero reference changes no NPP gradient.

## Explicit supplementary stages

`--stage recover --max-seconds 18000` permits additional NPP steps only after the paired gate passes and prior dev convergence is verified. It uses fresh Adam at a fixed LR1e-4 and preserves the original100k-step provenance. Lambda0.001/0.01/0.1/1.0 branches start from the same NPP and RNG state, each capped at1000 supplementary updates. Telemetry includes actual LR, parameter drift and fixed-probe output drift.

Task operating points use matched64-frame clips. Their rates are never combined with16-frame rate-only probes. Baselines include codec-only, Gaussian sigma0.5/1.0 and a fixed bilateral filter. Source-paired bootstrap compares candidate quality with the baseline interpolated at the same log-rate, with no extrapolation. Before quality-based stopping, E0 measures three paired tracker seeds and calibrates a tolerance from the observed seed range (minimum0.5pp). If E0 is unavailable, quality-based early stopping stays disabled. Time/numerical limits remain active. This observed-range calibration is diagnostic, not a formal stochastic confidence bound.

Final ambiguous candidates can expand from16 to32/48 sources, reusing individual-source measurements in the same budget. Recovery checkpoints preserve optimizer/RNG state and exact supplementary update counts on exceptional exits. Existing progress is accepted only for the verified invocation; a new recovery invocation requires a new output directory.

`--stage evaluate` evaluates the original trained NPP and simple filters on matched64-frame clips without NPP updates. Its outputs remain a diagnostic subset, not GOT10k hidden TEST or full-paper reproduction.

## Outputs and limitations

- `run.json`: truthful completion, budget/error, source/checkpoint and NPP-update counts.
- `cohort.json`, `binding/identity.json`: disjoint source manifest and cache identity.
- `paired_*.json`: true bytes, zero-reference decomposition and gate decisions.
- `prior_lr_screen.json`, `prior_fits.json`, `prior_*.pt`: bounded fit evidence and selected weights.
- `symbol_cache.pt`: deterministic bounded cache with identities; TRAIN and dev remain separate.
- `gradient_audit.json`, `filter_rates.json`: gradient component evidence and rate-only filter results.
- Optional `lambda_*.pt`, `lambda_runs.json`, `recovery_progress.json`, `dev_*.json`: invocation-bound supplementary checkpoints/counters and matched task RD diagnostics.

The code does not assume that90–97% of coefficients are zero, that512 steps converge, or that a fitted prior guarantees valid real-codec gradients. Rate gates require sufficient non-flat paired examples for every codec/QP; unresolved results remain unresolved. No identity fallback is selected. A successful Kaggle session does not establish negative BD-rate.
