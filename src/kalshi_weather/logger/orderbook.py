"""
Logs Kalshi orderbook snapshots to parquet.

Why this matters: Kalshi's public API gives OHLC candles and trade prints,
but never exposes historical orderbook depth. Once a minute has passed, the
depth at that moment is gone forever. This logger captures it from today forward.

Storage layout:
  data/logger/orderbook/{YYYY-MM-DD}.parquet   (one file per day, appended)

Schema:
  snapshot_utc  datetime64[ns, UTC]  — when the snapshot was taken
  ticker        str                  — e.g. "KXHIGHNY-26JUN04-T89"
  side          str                  — "yes" or "no"
  price_cents   int16                — price level in cents (1–99)
  quantity      int32                — contracts available at this price

One row per (snapshot_utc, ticker, side, price_cents). The full depth of book
at each snapshot can be reconstructed by grouping on snapshot_utc + ticker.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from kalshi_weather.tz import UTC
from pathlib import Path

import pandas as pd

from kalshi_weather.ingest.kalshi import _get

log = logging.getLogger(__name__)


def _open_tickers(stations_cfg: dict) -> list[str]:
    """Return all open weather market tickers across all configured cities."""
    tickers: list[str] = []
    seen_series: set[str] = set()

    for cfg in stations_cfg.values():
        series = cfg.get("kalshi_series", "")
        if not series or series in seen_series:
            continue
        seen_series.add(series)

        try:
            data = _get("/markets", {"series_ticker": series, "status": "open", "limit": 50})
            for m in data.get("markets", []):
                t = m.get("ticker", "")
                if t:
                    tickers.append(t)
        except Exception as exc:
            log.warning("failed to list open markets for %s: %s", series, exc)

        time.sleep(0.15)

    return tickers


def snapshot_all(stations_cfg: dict) -> list[dict]:
    """
    Fetch the full orderbook for every open weather market.
    Returns a list of row dicts ready for DataFrame construction.
    """
    tickers = _open_tickers(stations_cfg)
    snap_utc = datetime.now(UTC).replace(microsecond=0)
    rows: list[dict] = []

    for ticker in tickers:
        try:
            data = _get(f"/markets/{ticker}/orderbook")
            # API returns orderbook_fp with yes_dollars / no_dollars:
            # each level is [price_dollar_str, quantity_str]
            ob = data.get("orderbook_fp", {})

            for api_key, side in (("yes_dollars", "yes"), ("no_dollars", "no")):
                for level in ob.get(api_key, []):
                    if len(level) < 2:
                        continue
                    price_cents = round(float(level[0]) * 100)
                    quantity    = float(level[1])   # fractional contracts allowed
                    rows.append({
                        "snapshot_utc": snap_utc,
                        "ticker":       ticker,
                        "side":         side,
                        "price_cents":  price_cents,
                        "quantity":     quantity,
                    })
        except Exception as exc:
            log.warning("orderbook fetch failed %s: %s", ticker, exc)

        time.sleep(0.1)

    log.debug("snapshot: %d tickers, %d orderbook levels", len(tickers), len(rows))
    return rows


def append_snapshot(rows: list[dict], log_dir: Path) -> None:
    """
    Append snapshot rows to today's parquet file.
    Creates the file if it doesn't exist; appends if it does.
    """
    if not rows:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    path  = log_dir / f"{today}.parquet"

    df = pd.DataFrame(rows)
    df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"], utc=True)
    df["price_cents"]  = df["price_cents"].astype("int16")
    df["quantity"]     = df["quantity"].astype("float32")   # fractional contracts

    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(path, index=False)


def load_day(log_dir: Path, date_str: str) -> pd.DataFrame:
    """Load all orderbook snapshots for a given date (YYYY-MM-DD)."""
    path = log_dir / f"{date_str}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
