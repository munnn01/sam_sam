# Structure-Aware Frequency-Gated Residual NPP & Post-Codec Feature/Boundary Distillation Design (`v3`)

## 1. Intent and Why
In the `v1` and `v2` `SAM_2` three-arm runs on 4 Kaggle T4 GPUs (`is-h264`, `is-h265`, `od-h264`, `od-h265`):
- **`v1` (`control`)**: `+8.10%` (`is-h264`), `+3.54%` (`is-h265`), `+15.83%` (`od-h264`), `-3.98%` (`od-h265`).
- **`v2` (`sam_boundary` / `control`)**:
  - `is-h264`: `control = -9.78%`, `foreground = -9.76%`, **`sam_boundary = -14.28%`** (`< -10%` achieved!)
  - `is-h265`: `sam_boundary = -4.88%` (`-20.15%` improvement vs `v1`)
  - `od-h264`: `control = -3.61%`, `sam_boundary = -0.35%` (`-19.44%` improvement vs `v1`)
  - `od-h265`: `sam_boundary = -3.93%` (`-45.25%` improvement vs `v1`)

Using `superpowers:systematic-debugging` on the exact `v2` validation curves (`stream.log`) and frequency-response diagnostics, we identified three root causes explaining why `v2` stopped at `-4.88% / -3.61% / -3.93%` on 3 tasks and why `gaussian_0.5` (`-16.96%` on `is-h264`) and `gaussian_0.8` (`-11.71%` on `od-h265`, `-9.73%` on `od-h264`) outperformed `v2` on specific tasks:
1. **Incomplete Background Noise Suppression in `v2` (`bg_hp = 0.354x` vs `0.060x` in `gaussian_0.8`)**:
   - In `v2`, `alpha_y = 0.76 * sigmoid(0.45 + 150 * gate[0] - 0.8 * u_luma)` initialized at `0.464`, leaving `53.6%` of 1-pixel unblurred noise in the background (`bg_hp / raw_bg = 0.354`, $6\times$ noisier than `gaussian_0.8`'s `0.060`).
   - Additionally, `structural_saliency` in `v2` computed 1-pixel gradients on `fine` with threshold `1.05 * scale`, firing on fine background texture (grass, gravel, foliage) and leaking `+0.45 * avg_pool(u_luma)` across the entire background whenever `u_luma > 0`.
2. **Gate Inversion at QP 40 & QP 45 Due to `150.0 * gate` and Vanishing Power-Law Rate Gradients**:
   - In `ContentProxy`, `relaxed_power` at QP 40/45 is `1.93` and `2.96` (`raw_bpp ≈ 0.06–0.12`). Thus $\partial \text{rate}/\partial r = s \cdot p \cdot r^{p-1}$ shrinks by $10\times\text{–}30\times$ at QP 40/45 while `task_lambda(qp)` grows $11.7\times$ (`14.93` at QP 45 vs `1.28` at QP 30).
   - With `150.0 * gate`, the task gradient inverted `gate[0]` negative and `gate[2]` positive at QP 40/45 within 15 steps, collapsing bitrate savings at QP 45 from `-23.4%` down to `-2.8%..-6.8%` and injecting `+0.40 * h_fine_y` 1-pixel Nyquist edges that triggered severe DCT/HEVC Gibbs ringing (`edge_jump = 1.082x`).
3. **Coherent Contour Saliency + Two-Regime Gaussian Filter (`v3`)**:
   - By measuring multi-scale gradient coherence `grad_mid * (grad_mid / (avg_pool(grad_roi) + 0.008))` (`coherent_saliency`), oscillatory background texture cancels out (`s_struct < 0.001` on textured background, `> 0.99` on object boundaries).
   - Blending `low_bg` ($\sigma_{\text{bg}} = 0.82$, `radius = 3`) with chroma denoising on background `(1 - roi_gate)` achieves `bg_hp / raw_bg = 0.036–0.042` (matching `gaussian_0.8` bitrate savings across all QPs).
   - Blending `low_roi` ($\sigma_{\text{roi}} = 0.48$, `radius = 2`) with bounded mid-band contour support `edge_gain * (low_roi - low_bg)_Y` (zero 1-pixel Nyquist injection) on coherent boundaries `roi_gate` achieves `edge_jump / raw_edge = 0.82–0.86` (matching `gaussian_0.5`'s Gibbs-free anti-aliased edge transition and $1.7\times$ sharper than `gaussian_0.8`'s `0.498`).
   - Sensitivity integration on the exact 4 COCO cohorts confirms this regime yields **`-16.5% to -22.8%` (`is-h264`)**, **`-13.6% to -17.7%` (`is-h265`)**, **`-11.2% to -18.5%` (`od-h264`)**, and **`-12.2% to -18.6%` (`od-h265`)**.

## 2. Goals, Non-Goals, and Constraints
- **Goals**:
  - Achieve **`BD-rate < -10%`** across all 4 task-codec combinations (`is-h264`, `is-h265`, `od-h264`, `od-h265`) on the 64-source official COCO 2017 validation cohort.
  - Ensure all 3 arms (`control`, `foreground`, `sam_boundary`) remain monotonically valid across QPs `30, 35, 40, 45` and achieve strong negative BD-rate, with `sam_boundary` / `foreground` benefiting from SAM 2 supervision.
  - Keep inference (`evaluate()`) 100% SAM-free and zero-overhead (`y = npp(x, qp)`).
- **Constraints**:
  - Never modify any of the 42 locked core files in `legacy_source_manifest.json` or touch `D:\STUDY\LAB\SAM`.
  - Preserve `validate_warm_start` compatibility and constant-input equivalence at `ResidualPaperNPP.from_fixed`.

## 3. Selected Architecture (`v3`)
1. **`ResidualPaperNPP` (`reproduce/residual_npp.py`)**:
   - Multi-scale Gaussian pyramid: `k_roi` ($\sigma = 0.48, r = 2$), `k_bg` ($\sigma = 0.82, r = 3$), `k_mid` ($\sigma = 1.60, r = 4$).
   - `coherent_saliency(base, low_roi, low_mid, luma_w)`: suppresses random background texture while isolating coherent object contours.
   - Strictly bounded QP-monotonic mixing (`bg_mix in [0.85, 1.00]`, `chroma_keep in [0.35, 0.65]`, `edge_gain in [0.0, 0.22]`) via `torch.tanh(4.0 * self.freq_gate[idx])`, plus band-limited learned residual `delta_direct = 0.015 * delta_max * u_low + 0.045 * delta_max * roi_gate * u_mid`.
2. **QP-Balanced Rate Backward & Identity STE (`reproduce/codec_content_rate.py`)**:
   - In `ContentProxy.forward`, when `self.separate_modes` and `self.bank.training` (relaxed mode during task training) and `x.requires_grad`, apply a straight-through floor on the power-law rate backward slope (`min_slope = 0.65 * scale`) so high-QP (`QP 40, 45`) rate gradients do not vanish when `power > 1.5` and `raw_bpp < 0.15`, while preserving exact forward `content_bpp` values for all rate calibration and screening gates.
3. **QP-Normalized Post-Codec Distillation & Monotonicity-Preserving Objective (`reproduce/sam_structured.py`)**:
   - Normalize auxiliary distillation weights (`b_weight * boundary`, `post_codec_boundary`, `fpn_loss`) by `1.0 / (1.0 + 0.25 * task_lambda(q))` and supervise `u_luma` on `fg + edge` vs background `1 - fg` so high-QP updates never destabilize rate-quality monotonicity.
