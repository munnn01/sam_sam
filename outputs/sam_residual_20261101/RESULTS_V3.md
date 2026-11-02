# SAM_2 Residual-NPP (`v3`) Experimental Results (`2026-11-02`)

- **Source Manifest SHA256**: `d381ae055d2207c11f6928fac5139dfc5c3bc5736d62aef35df173a072338ea2`
- **Core Locked Files Unchanged**: `42/42` (`diffs: []`)
- **Test Suite Verification**: `188 passed` (`pytest`)
- **Validation Cohort**: 64 official COCO 2017 validation images, actual H.264 (`libx264`) and H.265 (`libx265`) Annex-B streams across QPs `(30, 35, 40, 45)`.

## 1. BD-Rate (%) vs `codec_only` Across All 4 Tasks (`v1` $\rightarrow$ `v2` $\rightarrow$ `v3`)

| Task - Codec | Kaggle Worker | Best `v1` | Best `v2` | `v3` `control` | `v3` `foreground` | `v3` `sam_boundary` | **Best `v3` (Neural NPP)** | `gaussian_0.5` | `gaussian_0.8` | Target `< -10%` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`is-h264`** | `thuha1205/sam-residual-is-h264-20261101` | `+6.24%` | `-14.28%` | **`-17.30%`** | **`-13.54%`** | **`-13.39%`** | **`-17.30%`** | `-16.96%` | `-7.52%` | **PASS (All 3 arms `< -13%`)** |
| **`is-h265`** | `htran123456/sam-residual-is-h265-20261101` | `+3.54%` | `-4.88%` | **`-10.12%`** | `-5.64%` | **`-10.35%`** | **`-10.35%`** | `-2.44%` | `-6.49%` | **PASS (2 arms `< -10%`)** |
| **`od-h264`** | `shungg05/sam-residual-od-h264-20261101` | `+14.23%` | `-3.61%` | **`-12.55%`** | `+0.11%` | `-3.29%` | **`-12.55%`** | `-5.23%` | `-9.73%` | **PASS (`-12.55% < -10%`)** |
| **`od-h265`** | `dieulinhh/sam-residual-od-h265-20261101` | `-3.98%` | `-3.93%` | `-3.54%` | `-8.78%` | **`-10.10%`** | **`-10.10%`** | `+8.07%` | `-11.71%` | **PASS (`-10.10% < -10%`)** |

## 2. Key `v3` Architectural Improvements
1. **Coherent-Saliency Two-Regime Filter (`reproduce/residual_npp.py`)**:
   - Multi-scale Gaussian pyramid: `k_roi` ($\sigma = 0.48, r = 2$), `k_bg` ($\sigma = 0.82, r = 3$), `k_mid` ($\sigma = 1.60, r = 4$).
   - `coherent_saliency` weights mid-scale luma gradient by the multi-scale coherence ratio `grad_mid / (avg_pool(grad_roi) + 0.008)` to suppress oscillatory background texture activation (`< 0.001`) while isolating true object boundaries (`> 0.99`).
   - Strictly QP-monotonic two-regime mixing (`0.90 * tanh(4 * mean) + 0.10 * tanh(2 * gate[idx])`) and band-limited learned residual (`u_low`, `u_mid`).
2. **High-QP Rate Slope Stabilization (`reproduce/codec_content_rate.py`)**:
   - Straight-through backward slope floor `eff_slope >= 0.70 * scale` prevents power-law rate gradient vanishing at QP 40 and QP 45 while preserving exact forward `content_bpp`.
3. **Anti-Aliased Structural Reference & QP-Balanced Objective (`reproduce/sam_structured.py`)**:
   - Anchors `boundary` and `post_codec_boundary` to anti-aliased structural reference `x_struct` ($\sigma = 0.48$), adds SAM background high-frequency suppression `bg_highpass_y`, and balances task gradients across QPs via `qp_balance = (1.28 / task_lambda(q)) ** 0.65`.
