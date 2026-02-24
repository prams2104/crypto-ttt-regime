# Test-Time Training for Robust Crypto Volatility Regime Classification

**Pramesh Singhavi**  
February 2026

---

## 1. Problem and Research Question

Cryptocurrency markets are highly non-stationary, with frequent regime shifts in volatility and participant behavior. Models trained on historical data often fail to generalize when the test environment differs from training.

**Research question:** Can self-supervised Test-Time Training (TTT) adapt a vision-based volatility classifier to non-stationary crypto market regimes without access to ground-truth labels during inference?

**Hypothesis:** Market regimes correspond to domain shifts in chart geometry and volatility structure; solving a temporally-aware self-supervised task on test samples can align the feature extractor to the active regime before prediction.

---

## 2. Approach

- **Data:** Hourly OHLCV (BTCUSDT) → rolling 168h windows → candlestick + volume chart images. Binary labels from next-24h realised volatility (threshold from training set).
- **Model:** Y-shaped ResNet18-GN: shared encoder (layers 1–3) → main head (classification) and aux head (self-supervised). GroupNorm throughout for single-sample TTT.
- **Auxiliary tasks:** (1) **Rotation** (0°/90°/180°/270°) — baseline from Sun et al. (2020). (2) **Temporal masking** — mask random middle columns (interpolation) or rightmost columns (extrapolation); reconstruct with foreground-weighted MSE so gradients focus on chart pixels.
- **Training:** Joint optimization of main (cross-entropy) + aux loss. Checkpoint by validation loss.
- **Test time:** Adapt encoder with gradient steps on aux loss (no labels), then predict. Compared: **Baseline** (no adaptation), **TTT (standard)** (adapt per sample, reset encoder), **TTT (online)** (adapt sequentially, keep encoder).
- **Entropy-adaptive LR:** Scale TTT step size by prediction entropy (high uncertainty → larger step).

---

## 3. Results

### 3.1 Baseline and TTT (Mask Aux, Exp 01)

| Mode           | Accuracy | F1    | ECE  | Brier | IC     |
|----------------|----------|-------|------|-------|--------|
| Baseline       | 0.76     | 0.08  | 0.06 | 0.17  | 0.09   |
| TTT (standard) | 0.47     | 0.32  | 0.15 | 0.19  | 0.06   |
| TTT (online)   | 0.34     | **0.35** | 0.20 | 0.21 | -0.03 |

TTT increases F1 (better recall on high-vol) at the cost of accuracy and calibration. Visualization of prediction changes shows TTT generally shifts P(high_vol) upward.

### 3.2 Rotation vs Mask Aux (Exp 02)

| Aux task  | Mode           | Accuracy | F1    | IC     |
|-----------|----------------|----------|-------|--------|
| **Mask**  | Baseline       | 0.76     | 0.08  | 0.09   |
| Mask      | TTT (standard) | 0.47     | 0.32  | 0.06   |
| Mask      | TTT (online)   | 0.34     | 0.35  | -0.03  |
| **Rotation** | Baseline    | 0.50     | **0.37** | **0.19** |
| Rotation  | TTT (standard) | 0.46     | 0.36  | 0.14   |
| Rotation  | TTT (online)   | 0.77     | 0.02  | ~0     |

Rotation yields a stronger baseline (higher F1 and IC). Mask TTT (online) gives more balanced predictions (higher F1); rotation TTT (online) collapses to near-zero F1.

### 3.3 Regime-Stratified Evaluation (Exp 03)

| RV bin        | n   | Baseline acc | Baseline F1 | TTT acc | TTT F1 |
|---------------|-----|--------------|------------|---------|--------|
| low (0–25%)   | 194 | 0.96         | 0.00       | 0.31    | 0.00   |
| mid-low       | 194 | 0.94         | 0.00       | 0.33    | 0.00   |
| mid-high      | 193 | 0.97         | 0.00       | 0.39    | 0.00   |
| **high (75–100%)** | 194 | 0.18   | 0.09       | **0.58** | **0.72** |

TTT improves performance mainly in the high-volatility regime; in low/mid-vol regimes the baseline dominates and TTT hurts by over-adapting toward high-vol.

---

## 4. Conclusion

- **TTT works:** Aux loss decreases during adaptation (foreground-weighted MSE + interpolation masking). Predictions change in a meaningful way.
- **Regime-specific benefit:** TTT improves accuracy and F1 in high-volatility regimes, as hypothesized. In low-vol regimes, baseline is already well-calibrated; TTT degrades accuracy.
- **Aux task choice:** Rotation gives a better baseline; temporal masking yields more useful online TTT behavior (higher F1, no collapse). Finance-specific masking is a viable alternative to generic rotation.
- **Limitations:** Single asset (BTCUSDT); online TTT sensitive to order and hyperparameters; calibration (ECE, IC) often worse under TTT.

---

## 5. Reproducibility

- **Data:** `data/raw/btcusdt_1h.parquet` (Binance hourly).
- **Run order:** Experiment 01 (data prep + mask training + eval) → 02 (rotation training + eval) → 03 (regime stratification). See README and notebook headers.
- **Checkpoints:** `checkpoints/joint/best.pt` (mask), `checkpoints/rotation/best.pt` (rotation).
