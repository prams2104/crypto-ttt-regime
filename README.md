# crypto-ttt-regime

**Test-Time Training for Robust Crypto Volatility Regime Classification**

Re-implementation of [TTT (Sun et al., 2020)](https://arxiv.org/abs/1909.13231) from scratch,
applied to crypto market volatility regime detection using Binance OHLCV chart images.

## Research Question

> Can self-supervised Test-Time Training adapt a vision-based volatility classifier to
> non-stationary crypto market regimes without access to ground-truth labels during inference?

## Novel Contributions

1. **Temporal Masking Auxiliary Task** — mask the rightmost 20% of chart images (most recent hours)
   and train the auxiliary head to reconstruct the missing region.  Unlike rotation prediction, this
   preserves temporal semantics and forces the model to learn regime-specific momentum and volatility
   structure.

2. **Entropy-Adaptive TTT Learning Rate** — scale the test-time adaptation step size by prediction
   entropy.  High uncertainty triggers stronger adaptation; low uncertainty preserves stability.

## Architecture

```
                 ┌──────────────┐
  x (chart) ──► │ Shared Encoder│──► features (256, 14, 14)
                 │ (ResNet18-GN) │        │
                 │ layers 1–3    │        ├──► Main Head (layer4 → FC → 2)  → P(high vol)
                 └──────────────┘        │
  x̃ (masked) ──► │ Shared Encoder│──────► Aux Head  (decoder → 3,224,224) → reconstruction
```

- **GroupNorm** everywhere (BatchNorm fails for single-sample TTT).
- Encoder split after 3rd residual group (deeper shared features).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Sanity check: synthetic data, 3 epochs (~2 min)
python -m src.train --synthetic --epochs 3 --batch_size 16

# Evaluate with TTT
python -m src.eval --checkpoint checkpoints/best.pt --ttt_steps 10 --entropy_adaptive

# Full training on real data
python -m src.train --parquet data/raw/btcusdt_1h.parquet --epochs 50 --aux_task mask
```

## Project Structure

```
crypto-ttt-regime/
├── data/
│   ├── raw/             # OHLCV parquet files (gitignored)
│   └── processed/       # Pre-rendered chart tensors (gitignored)
├── src/
│   ├── __init__.py
│   ├── dataset.py       # Data loading, windowing, chart rendering, splits
│   ├── models.py        # ResNet18-GN with Y-structure dual heads
│   ├── ttt_learner.py   # Joint trainer + TTT adaptation loops
│   ├── train.py         # CLI: training
│   ├── eval.py          # CLI: evaluation + stress tests
│   └── utils.py         # Financial metrics, calibration, seeding
├── experiments/
│   ├── 01_baseline_benchmark.ipynb
│   ├── 02_ttt_masked_patch.ipynb
│   └── 03_regime_stress_test.ipynb
├── dashboard/
│   └── app.py           # Streamlit visualisation (TODO)
├── requirements.txt
└── README.md
```

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| Accuracy | Overall classification rate |
| F1 | Harmonic mean of precision & recall |
| ECE | Expected Calibration Error (reliability) |
| Brier | Brier score (proper scoring rule) |
| IC | Information Coefficient (rank correlation with realised vol) |

## License

MIT
