"""Fetch hourly OHLCV data from Binance and save as parquet.

Uses the same python-binance approach as the stat-arb project but adapted
for hourly frequency and full OHLCV columns (needed for chart rendering).

Usage:
    # Both BTC and ETH (default)
    python -m src.fetch_data

    # Just BTC, custom date range
    python -m src.fetch_data --symbols BTCUSDT --start 2022-01-01

    # Check what you have
    python -m src.fetch_data --check
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_START = "2020-01-01"
OUTPUT_DIR = Path("data/raw")


def format_binance_klines(data: list) -> pd.DataFrame:
    """Convert raw Binance kline response to a clean DataFrame.

    Same approach as the stat-arb project's format_binance(), but keeps
    all OHLCV columns and converts timestamps to proper datetimes.
    """
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_base_volume", "taker_quote_volume", "ignore",
    ]
    df = pd.DataFrame(data, columns=columns)

    df["timestamp"] = df["open_time"].map(
        lambda x: datetime.fromtimestamp(x / 1000)
    )

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_symbol(
    client,
    symbol: str,
    interval: str = "1h",
    start: str = DEFAULT_START,
) -> pd.DataFrame:
    """Fetch historical klines for a single symbol.

    The Binance client handles pagination automatically for large requests.
    Hourly data since 2020 is ~50k rows — takes about 30-60 seconds.
    """
    logger.info("Fetching %s %s data from %s …", symbol, interval, start)
    raw = client.get_historical_klines(symbol, interval, start)
    df = format_binance_klines(raw)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("  → %d bars (%s to %s)", len(df),
                df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
    return df


def check_existing() -> None:
    """Print summary of any existing parquet files in data/raw/."""
    parquets = list(OUTPUT_DIR.glob("*.parquet"))
    if not parquets:
        logger.info("No parquet files found in %s", OUTPUT_DIR)
        return
    for p in sorted(parquets):
        df = pd.read_parquet(p)
        logger.info(
            "  %s — %d bars, %s → %s",
            p.name, len(df),
            df["timestamp"].iloc[0], df["timestamp"].iloc[-1],
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Binance OHLCV data")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                   help="Trading pairs to fetch (default: BTCUSDT ETHUSDT)")
    p.add_argument("--start", type=str, default=DEFAULT_START,
                   help="Start date (default: 2020-01-01)")
    p.add_argument("--interval", type=str, default="1h",
                   help="Candle interval (default: 1h)")
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--check", action="store_true",
                   help="Just check existing data, don't fetch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        check_existing()
        return

    # Lazy import so --check works without python-binance installed
    try:
        from binance.client import Client as BinanceClient
    except ImportError:
        logger.error(
            "python-binance not installed. Run: pip install python-binance"
        )
        return

    # No API key needed for public historical klines
    client = BinanceClient(tld="US")

    for symbol in args.symbols:
        df = fetch_symbol(client, symbol, args.interval, args.start)

        filename = f"{symbol.lower()}_{args.interval}.parquet"
        out_path = output_dir / filename
        df.to_parquet(out_path, index=False)
        logger.info("  → Saved to %s (%.1f MB)", out_path,
                     out_path.stat().st_size / 1e6)

    logger.info("Done. To train on this data:")
    logger.info("  python -m src.train --parquet %s",
                output_dir / f"{args.symbols[0].lower()}_{args.interval}.parquet")


if __name__ == "__main__":
    main()
