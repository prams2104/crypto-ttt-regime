"""crypto-ttt-regime: Test-Time Training for Crypto Volatility Regime Classification.

Re-implementation of TTT (Sun et al., 2020) from scratch, applied to
crypto market volatility regime detection using Binance OHLCV chart images.

Novelty:
    1. Temporal masking auxiliary task (chart-aware masked-patch reconstruction).
    2. Entropy-adaptive test-time learning rate.
"""

__version__ = "0.1.0"
