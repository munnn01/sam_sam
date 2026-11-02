# BD-Rate < -10% Coherent-Saliency Two-Regime Residual NPP (`v3`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `SAM_2` from `v2` to `v3` so that all four COCO task-codec evaluations (`is-h264`, `is-h265`, `od-h264`, `od-h265`) achieve **`BD-rate < -10%`** with monotonically valid RD curves across all three arms (`control`, `foreground`, `sam_boundary`).

**Architecture (`v3`):**
1. **Coherent-Saliency Two-Regime `ResidualPaperNPP` ([residual_npp.py](file:///D:/STUDY/LAB/SAM_2/reproduce/residual_npp.py))**:
   - Multi-scale separable Gaussian pyramid: `k_roi` ($\sigma = 0.48, r = 2$), `k_bg` ($\sigma = 0.82, r = 3$), `k_mid` ($\sigma = 1.60, r = 4$).
   - `coherent_saliency(base, low_roi, low_mid, luma_w)`: weights mid-scale luma gradient $\|\nabla \text{low\_mid}_Y\|$ by multi-scale coherence ratio $\|\nabla \text{low\_mid}_Y\| / (\text{AvgPool}_{5\times 5}(\|\nabla \text{low\_roi}_Y\|) + 0.008)$ with threshold `1.55 * scale`, suppressing background texture activation (`< 0.001`) while isolating coherent object contours (`> 0.99`).
   - Bounded QP-monotonic two-regime mixing (`bg_mix in [0.85, 1.00]`, `chroma_keep in [0.35, 0.65]`, `edge_gain in [0.0, 0.22]`) using `torch.tanh(4.0 * (0.85 * self.freq_gate.mean(dim=0) + 0.15 * self.freq_gate[idx]))` and band-limited learned residual `delta_direct = 0.015 * delta_max * u_low + 0.045 * delta_max * roi_gate * u_mid`.
2. **QP-Balanced Power-Law Rate Backward & Identity STE in `ContentProxy` ([codec_content_rate.py](file:///D:/STUDY/LAB/SAM_2/reproduce/codec_content_rate.py))**:
   - Preserves exact forward `content_bpp = scale * raw.pow(power)` and gradient sign while flooring the relaxed backward slope at `0.70 * scale` (`eff_slope = torch.maximum(scale * power * raw.detach().pow(power - 1.0), 0.70 * scale)`) so high-QP (`QP 40, 45`) rate gradients do not vanish during task training.
3. **Anti-Aliased Structural Reference & QP-Normalized Objective in `structured_objective` ([sam_structured.py](file:///D:/STUDY/LAB/SAM_2/reproduce/sam_structured.py))**:
   - Computes `boundary` and `post_codec_boundary` against the anti-aliased structural reference `x_struct` (3x3 Gaussian-smoothed `x`) instead of noisy raw 1-pixel differences, adds SAM background high-frequency suppression `(1 - max(fg, edge)) * highpass(y)^2`, and normalizes task/distillation weights across QPs via `q_norm = 2.0 / (1.0 + 0.35 * task_lambda(q))`.

**Tech Stack:** PyTorch, Torchvision, CompressAI, FFmpeg (`libx264`, `libx265`), `pytest`.

**Spec:** `D:\STUDY\LAB\SAM_2\docs\superpowers\specs\2026-11-01-bdrate-under-minus-10-design.md`

---

### Task 1: Coherent-Saliency Two-Regime `ResidualPaperNPP` (`v3`)

**Files:**
- Modify: `D:\STUDY\LAB\SAM_2\reproduce\residual_npp.py`
- Modify: `D:\STUDY\LAB\SAM_2\tests\test_residual_npp.py`

- [ ] **Step 1: Update/add failing tests in `D:\STUDY\LAB\SAM_2\tests\test_residual_npp.py`**
  - Verify `bg_hf_after < 0.12 * bg_hf_before` (matching `gaussian_0.8`-level background suppression), anti-aliased step-edge preservation (`0.75 * edge_before <= edge_after <= 0.95 * edge_before` and 3-pixel transition `>= 0.95`), QP monotonicity of background suppression across `(30, 35, 40, 45)` even under adversarial `freq_gate` perturbations, and non-zero gradients to all branches (`conv1`, `conv2`, `UNet`, `Scaling_Net`, `freq_gate`).
- [ ] **Step 2: Run `pytest D:\STUDY\LAB\SAM_2\tests\test_residual_npp.py` to confirm RED**
- [ ] **Step 3: Implement `coherent_saliency` and `v3` `ResidualPaperNPP.forward` in `D:\STUDY\LAB\SAM_2\reproduce\residual_npp.py`**
- [ ] **Step 4: Run `pytest D:\STUDY\LAB\SAM_2\tests\test_residual_npp.py` to confirm GREEN**

---

### Task 2: QP-Balanced Power-Law Rate Backward in `ContentProxy`

**Files:**
- Modify: `D:\STUDY\LAB\SAM_2\reproduce\codec_content_rate.py`
- Modify: `D:\STUDY\LAB\SAM_2\tests\test_codec_content_rate.py`

- [ ] **Step 1: Add failing test in `D:\STUDY\LAB\SAM_2\tests\test_codec_content_rate.py`**
  - Verify that when `separate_modes=True`, `bank.training=True`, `relaxed_power=[1.0, 1.4, 2.0, 3.0]`, and `raw_bpp=0.08` at `qp=45`, `out.content_bpp` matches exact `scale * (0.08 ** 3.0)` in forward while `torch.autograd.grad(out.bpp, x)[0]` retains at least `0.70 * scale * d(raw_bpp)/dx` instead of vanishing (`3 * 0.08^2 = 0.0192 * scale`).
- [ ] **Step 2: Run `pytest D:\STUDY\LAB\SAM_2\tests\test_codec_content_rate.py` to confirm RED**
- [ ] **Step 3: Implement straight-through slope floor in `ContentProxy.forward` in `D:\STUDY\LAB\SAM_2\reproduce\codec_content_rate.py`**
- [ ] **Step 4: Run `pytest D:\STUDY\LAB\SAM_2\tests\test_codec_content_rate.py` to confirm GREEN**

---

### Task 3: Anti-Aliased Structural Reference & QP-Normalized `structured_objective`

**Files:**
- Modify: `D:\STUDY\LAB\SAM_2\reproduce\sam_structured.py`
- Modify: `D:\STUDY\LAB\SAM_2\tests\test_structured_sam.py`

- [ ] **Step 1: Add failing test in `D:\STUDY\LAB\SAM_2\tests\test_structured_sam.py`**
  - Verify that `structured_objective` computes `bg_highpass_y` and `qp_balance`, penalizes high-frequency noise on SAM background `(1 - fg)`, and keeps the task/distillation gradient ratio across `qp=45` vs `qp=30` within `< 3.5x` (instead of `> 11x`).
- [ ] **Step 2: Run `pytest D:\STUDY\LAB\SAM_2\tests\test_structured_sam.py` to confirm RED**
- [ ] **Step 3: Implement `v3` `structured_objective` in `D:\STUDY\LAB\SAM_2\reproduce\sam_structured.py`**
- [ ] **Step 4: Run full `pytest` suite (`187+` tests) and verify 42 locked core hashes**
