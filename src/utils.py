"""Utility functions for the crypto-ttt-regime project.

Contains:
    - Reproducibility: seed_everything, get_device
    - Financial metrics: information_coefficient, sharpe_ratio
    - Calibration metrics: expected_calibration_error, brier_score
    - Entropy: prediction_entropy (used by entropy-adaptive TTT LR)
    - Logging: MetricLogger
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
from scipy import stats


# ── Reproducibility ──────────────────────────────────────────────────────


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for full reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Financial Metrics ────────────────────────────────────────────────────


def information_coefficient(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Rank Information Coefficient (Spearman correlation).

    Measures monotonic relationship between predicted probabilities
    and realised volatility values.  Higher |IC| = better signal.

    Args:
        predictions: model outputs (e.g. P(high-vol)).
        targets: realised volatility values (continuous).

    Returns:
        Spearman rank-correlation coefficient in [-1, 1].
    """
    if len(predictions) < 3:
        return 0.0
    corr, _ = stats.spearmanr(predictions, targets)
    return float(corr) if np.isfinite(corr) else 0.0


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualisation_factor: float = np.sqrt(365),
) -> float:
    """Annualised Sharpe ratio for a strategy's daily returns.

    Args:
        returns: array of daily (or per-period) returns.
        risk_free_rate: annual risk-free rate (default 0).
        annualisation_factor: sqrt(periods_per_year).

    Returns:
        Annualised Sharpe ratio (float).
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / 365.0
    std = excess.std()
    if std < 1e-12:
        return 0.0
    return float(excess.mean() / std * annualisation_factor)


# ── Calibration Metrics ──────────────────────────────────────────────────


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Partitions predictions into equal-width bins and computes the
    weighted-average gap between mean confidence and mean accuracy.

    Args:
        probs: predicted P(positive class), shape (N,).
        labels: ground-truth binary labels {0, 1}, shape (N,).
        n_bins: number of bins.

    Returns:
        ECE in [0, 1].  Lower is better.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Brier score for binary classification.

    BS = mean((p - y)^2).  Lower is better.  Range [0, 1].
    """
    return float(np.mean((probs - labels) ** 2))


# ── Entropy ──────────────────────────────────────────────────────────────


def prediction_entropy(
    probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Shannon entropy of prediction distribution.

    Args:
        probs: (B, C) softmax probabilities.

    Returns:
        (B,) entropy values.  Max for binary = ln(2) ≈ 0.693.
    """
    return -(probs * (probs + eps).log()).sum(dim=-1)


# ── Logging helper ───────────────────────────────────────────────────────


class MetricLogger:
    """Simple running-average metric accumulator for training loops."""

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self._data.setdefault(key, []).append(value)

    def average(self, key: str) -> float:
        vals = self._data.get(key, [])
        return sum(vals) / max(len(vals), 1)

    def reset(self) -> None:
        self._data.clear()

    def summary(self) -> dict[str, float]:
        return {k: self.average(k) for k in self._data}
