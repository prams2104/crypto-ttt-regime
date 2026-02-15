"""Crypto OHLCV Dataset for TTT Regime Classification.

Pipeline:
    1. Load hourly OHLCV parquet from ``data/raw/``.
    2. Create 168 h (7-day) rolling windows with configurable stride.
    3. Render each window as a candlestick + volume chart image tensor.
    4. Compute binary vol label from next-24 h realised volatility (no look-ahead).
    5. Time-based train / val / test split with embargo ≥ window length.
    6. Save everything to ``data/processed/dataset.pt``.

For quick pipeline validation without real Binance data, use
``generate_synthetic_ohlcv()`` to create regime-switching synthetic OHLCV.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Synthetic data generation (for pipeline validation)
# ═══════════════════════════════════════════════════════════════════════════


def generate_synthetic_ohlcv(
    n_hours: int = 4320,
    start_price: float = 30_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic BTC-like OHLCV with regime-switching volatility.

    Two regimes:
        - *low-vol*  (σ = 0.008 / hour) — ~70 % of the time
        - *high-vol* (σ = 0.035 / hour) — ~30 % of the time

    Uses a simple Markov chain to switch between regimes so that the
    resulting data has realistic-ish clustering of calm / turbulent periods.

    Args:
        n_hours: total number of hourly bars.
        start_price: initial close price.
        seed: random seed for reproducibility.

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume].
    """
    rng = np.random.RandomState(seed)

    # ── regime sequence (Markov chain) ──
    regime = np.zeros(n_hours, dtype=int)
    for t in range(1, n_hours):
        if regime[t - 1] == 0:                       # low → high
            regime[t] = 1 if rng.random() < 0.003 else 0
        else:                                         # high → low
            regime[t] = 0 if rng.random() < 0.008 else 1

    vol = np.where(regime == 0, 0.008, 0.035)

    # ── price process ──
    log_ret = rng.normal(0, vol)
    close = start_price * np.exp(np.cumsum(log_ret))

    # ── OHLC from close ──
    noise_hi = rng.uniform(0.001, 0.008, n_hours)
    noise_lo = rng.uniform(0.001, 0.008, n_hours)
    high = close * (1 + noise_hi)
    low  = close * (1 - noise_lo)
    open_ = np.roll(close, 1)
    open_[0] = start_price

    # ── volume (correlated with absolute returns) ──
    base_vol = rng.lognormal(mean=10, sigma=0.8, size=n_hours)
    vol_mult = 1 + np.abs(log_ret) * 50
    volume = base_vol * vol_mult

    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n_hours, freq="h"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 2. Chart rendering
# ═══════════════════════════════════════════════════════════════════════════


def render_candlestick_chart(
    ohlcv: pd.DataFrame,
    img_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Render OHLCV rows as a candlestick + volume chart image.

    Renders at 3× resolution then down-samples with Lanczos for quality.
    Deterministic: no random state involved.

    Args:
        ohlcv: DataFrame with [open, high, low, close, volume].
        img_size: (height, width) of output image.

    Returns:
        (3, H, W) float32 array in [0, 1].
    """
    h, w = img_size
    n = len(ohlcv)

    # render at 3× then down-sample
    render_w, render_h = w * 3, h * 3
    dpi = 100
    fig, (ax_p, ax_v) = plt.subplots(
        2, 1,
        figsize=(render_w / dpi, render_h / dpi),
        dpi=dpi,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("black")
    for ax in (ax_p, ax_v):
        ax.set_facecolor("black")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    opens  = ohlcv["open"].values
    highs  = ohlcv["high"].values
    lows   = ohlcv["low"].values
    closes = ohlcv["close"].values
    vols   = ohlcv["volume"].values

    for i in range(n):
        color = "#26a69a" if closes[i] >= opens[i] else "#ef5350"
        # wick
        ax_p.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.6)
        # body
        body_lo = min(opens[i], closes[i])
        body_h  = max(abs(closes[i] - opens[i]), (highs[i] - lows[i]) * 0.01)
        ax_p.add_patch(
            Rectangle((i - 0.35, body_lo), 0.7, body_h,
                       facecolor=color, edgecolor=color, linewidth=0.3),
        )

    # volume bars
    v_colors = [
        "#26a69a" if closes[i] >= opens[i] else "#ef5350" for i in range(n)
    ]
    ax_v.bar(range(n), vols, color=v_colors, width=0.7)

    ax_p.set_xlim(-1, n)
    ax_v.set_xlim(-1, n)
    fig.tight_layout(pad=0.0)

    # ── rasterise ──
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB").resize((w, h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # HWC → CHW


# ═══════════════════════════════════════════════════════════════════════════
# 3. Label computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_forward_rv(
    close: np.ndarray,
    forward_hours: int = 24,
) -> np.ndarray:
    """Compute *forward-looking* realised volatility for each bar.

    RV_{t→t+H} = sqrt( Σ_{i=1}^{H} r²_{t+i} )

    Returns NaN where the forward window is incomplete.
    """
    log_ret = np.log(close[1:] / close[:-1])
    rv = np.full(len(close), np.nan)
    for t in range(len(close) - forward_hours):
        rv[t] = np.sqrt(np.sum(log_ret[t : t + forward_hours] ** 2))
    return rv


# ═══════════════════════════════════════════════════════════════════════════
# 4. Rolling windows
# ═══════════════════════════════════════════════════════════════════════════


def create_windows(
    df: pd.DataFrame,
    window_hours: int = 168,
    stride_hours: int = 24,
    forward_hours: int = 24,
) -> list[dict]:
    """Create rolling windows with forward RV labels.

    Each element contains:
        ohlcv      – DataFrame slice for the window
        rv         – scalar realised vol over the *next* ``forward_hours``
        start_ts   – first timestamp in window
        end_ts     – last timestamp in window

    No look-ahead: label is computed from bars *after* the window.
    """
    n = len(df)
    close_all = df["close"].values
    log_ret_all = np.log(close_all[1:] / close_all[:-1])

    windows: list[dict] = []
    for start in range(0, n - window_hours - forward_hours + 1, stride_hours):
        end = start + window_hours
        fwd_rets = log_ret_all[end - 1 : end - 1 + forward_hours]
        if len(fwd_rets) < forward_hours:
            continue
        rv = float(np.sqrt(np.sum(fwd_rets ** 2)))
        windows.append({
            "ohlcv": df.iloc[start:end],
            "rv": rv,
            "start_ts": df.iloc[start]["timestamp"],
            "end_ts": df.iloc[end - 1]["timestamp"],
        })
    return windows


# ═══════════════════════════════════════════════════════════════════════════
# 5. Time-based splits with embargo
# ═══════════════════════════════════════════════════════════════════════════


def time_based_split(
    windows: list[dict],
    train_end: str,
    val_end: str,
    embargo_hours: int = 168,
) -> tuple[list[int], list[int], list[int]]:
    """Strict time-based split with embargo gap.

    Embargo ≥ window_hours prevents any data leakage from overlapping
    windows.  Windows whose ``end_ts`` falls in an embargo zone are
    dropped entirely (not assigned to either split).

    Returns:
        (train_idx, val_idx, test_idx) — lists of integer indices.
    """
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt   = pd.Timestamp(val_end)
    embargo      = pd.Timedelta(hours=embargo_hours)

    train_idx: list[int] = []
    val_idx:   list[int] = []
    test_idx:  list[int] = []

    for i, w in enumerate(windows):
        t = w["end_ts"]
        if t <= train_end_dt:
            train_idx.append(i)
        elif t > train_end_dt + embargo and t <= val_end_dt:
            val_idx.append(i)
        elif t > val_end_dt + embargo:
            test_idx.append(i)
        # else: falls in embargo gap → excluded
    return train_idx, val_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════
# 6. Dataset preparation pipeline
# ═══════════════════════════════════════════════════════════════════════════


def prepare_dataset(
    parquet_path: Optional[str | Path] = None,
    output_dir: str | Path = "data/processed",
    *,
    synthetic: bool = False,
    synthetic_hours: int = 4320,
    window_hours: int = 168,
    stride_hours: int = 24,
    forward_hours: int = 24,
    img_size: tuple[int, int] = (224, 224),
    train_end: Optional[str] = None,
    val_end: Optional[str] = None,
    embargo_hours: int = 168,
    vol_percentile: float = 50.0,
    seed: int = 42,
) -> Path:
    """Full data-preparation pipeline: parquet → processed tensors.

    Steps:
        1. Load (or generate) OHLCV data.
        2. Create rolling windows.
        3. Render chart images (slow — one-time cost).
        4. Compute labels from training-set vol threshold.
        5. Create splits with embargo.
        6. Save ``dataset.pt`` to *output_dir*.

    Args:
        parquet_path: path to raw OHLCV parquet (ignored if *synthetic*).
        output_dir: where to save the processed dataset.
        synthetic: if True, generate synthetic OHLCV for pipeline testing.
        synthetic_hours: number of synthetic bars to generate.
        window_hours: rolling-window length (default 168 = 7 days).
        stride_hours: step between consecutive windows.
        forward_hours: horizon for realised-vol label.
        img_size: chart image size (H, W).
        train_end: cutoff timestamp for training set.
        val_end: cutoff timestamp for validation set.
        embargo_hours: gap between splits (≥ window_hours).
        vol_percentile: percentile for high/low vol threshold.
        seed: random seed (for synthetic data).

    Returns:
        Path to the saved ``dataset.pt`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dataset.pt"

    # 1. Load data ─────────────────────────────────────────────────────
    if synthetic:
        logger.info("Generating %d hours of synthetic OHLCV data …", synthetic_hours)
        df = generate_synthetic_ohlcv(n_hours=synthetic_hours, seed=seed)
    else:
        assert parquet_path is not None, "Provide parquet_path or set synthetic=True"
        logger.info("Loading OHLCV from %s …", parquet_path)
        df = pd.read_parquet(parquet_path)

    # ensure column order
    for col in ("timestamp", "open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"

    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d hourly bars (%s → %s).",
                len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    # Auto-compute split dates if not provided (70/15/15 split by time)
    if train_end is None or val_end is None:
        t_min = df["timestamp"].iloc[0]
        t_max = df["timestamp"].iloc[-1]
        total_span = t_max - t_min
        if train_end is None:
            train_end = str((t_min + total_span * 0.70).floor("D"))
        if val_end is None:
            val_end = str((t_min + total_span * 0.85).floor("D"))
        logger.info("Auto-computed split dates: train_end=%s, val_end=%s", train_end, val_end)

    # 2. Windows ───────────────────────────────────────────────────────
    windows = create_windows(df, window_hours, stride_hours, forward_hours)
    logger.info("Created %d windows (stride=%dh, fwd=%dh).", len(windows), stride_hours, forward_hours)

    # 3. Render chart images ───────────────────────────────────────────
    images: list[np.ndarray] = []
    for w in tqdm(windows, desc="Rendering charts"):
        images.append(render_candlestick_chart(w["ohlcv"], img_size=img_size))
    images_t = torch.from_numpy(np.stack(images))          # (N, 3, H, W)

    # 4. Labels ────────────────────────────────────────────────────────
    rv_values = np.array([w["rv"] for w in windows])
    timestamps = [w["end_ts"] for w in windows]

    # Threshold from TRAINING set only (no look-ahead)
    train_idx_tmp, _, _ = time_based_split(windows, train_end, val_end, embargo_hours)
    train_rv = rv_values[train_idx_tmp]
    threshold = float(np.percentile(train_rv, vol_percentile))
    labels = (rv_values > threshold).astype(np.int64)
    labels_t = torch.from_numpy(labels)
    rv_t = torch.from_numpy(rv_values.astype(np.float32))

    # 5. Splits ────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = time_based_split(windows, train_end, val_end, embargo_hours)
    logger.info("Split sizes — train: %d, val: %d, test: %d (embargo=%dh).",
                len(train_idx), len(val_idx), len(test_idx), embargo_hours)

    # 6. Save ──────────────────────────────────────────────────────────
    torch.save({
        "images": images_t,
        "labels": labels_t,
        "rv_values": rv_t,
        "timestamps": timestamps,
        "threshold": threshold,
        "split_indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        "metadata": {
            "window_hours": window_hours,
            "stride_hours": stride_hours,
            "forward_hours": forward_hours,
            "img_size": img_size,
            "train_end": train_end,
            "val_end": val_end,
            "embargo_hours": embargo_hours,
            "vol_percentile": vol_percentile,
            "threshold": threshold,
        },
    }, out_path)
    logger.info("Saved dataset to %s (%.1f MB).", out_path, out_path.stat().st_size / 1e6)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# 7. PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════


class CryptoRegimeDataset(Dataset):
    """PyTorch dataset backed by a ``dataset.pt`` file from :func:`prepare_dataset`.

    Each sample returns ``(image, label, rv_value)`` where:
        - image: (3, H, W) float32 chart tensor in [0, 1].
        - label: int64 binary class (0 = low vol, 1 = high vol).
        - rv_value: float32 realised volatility (continuous, for IC metric).

    Use :meth:`get_splits` to obtain train / val / test ``Subset`` objects
    with the correct embargo-aware indices.
    """

    def __init__(self, data_dir: str | Path = "data/processed") -> None:
        data_dir = Path(data_dir)
        pt_path = data_dir / "dataset.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"{pt_path} not found. Run prepare_dataset() first or "
                f"use `python -m src.train --synthetic` to generate it."
            )
        blob = torch.load(pt_path, weights_only=False)
        self.images:    torch.Tensor = blob["images"]
        self.labels:    torch.Tensor = blob["labels"]
        self.rv_values: torch.Tensor = blob["rv_values"]
        self.timestamps: list        = blob["timestamps"]
        self.threshold: float        = blob["threshold"]
        self._splits: dict           = blob["split_indices"]
        self.metadata: dict          = blob["metadata"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx], self.rv_values[idx]

    def get_splits(self) -> tuple[Subset, Subset, Subset]:
        """Return (train_subset, val_subset, test_subset)."""
        return (
            Subset(self, self._splits["train"]),
            Subset(self, self._splits["val"]),
            Subset(self, self._splits["test"]),
        )
