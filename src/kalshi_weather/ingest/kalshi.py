"""
Fetch and cache Kalshi market data for the KXHIGHNY (NYC daily high temp) series.

Cache layout:
  data/raw/kalshi/markets.parquet       — all settled market metadata
  data/raw/kalshi/candles/{ticker}.parquet — 60-min OHLC for each market

API notes (as of 2026):
  - Base URL: https://api.elections.kalshi.com/trade-api/v2
  - Auth: Authorization: Bearer {key}
  - start_ts / end_ts: Unix SECONDS (not ms)
  - period_interval: minutes (60 = hourly candles; max 5000 candles per request)
  - Historical cutoff ~2026-04-04; all NYC weather markets are in /historical/
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from kalshi_weather.tz import utc_now

load_dotenv()

_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_SERIES = "KXHIGHNY"
_CACHE_ROOT = Path(__file__).parents[3] / "data" / "raw" / "kalshi"
_CANDLE_DIR = _CACHE_ROOT / "candles"
_PERIOD = 60          # 60-minute candles
_MAX_CANDLES = 4500   # stay safely under the 5000 limit
_PAGE_SIZE = 200


def _headers() -> dict:
    key = os.getenv("KALSHI_API_KEY")
    if not key:
        raise EnvironmentError("KALSHI_API_KEY not set. Check your .env file.")
    return {"Authorization": f"Bearer {key}"}


def _get(path: str, params: dict | None = None, max_retries: int = 5) -> dict:
    """GET with exponential backoff on 429 (rate limit) responses."""
    delay = 2.0
    for attempt in range(max_retries):
        resp = requests.get(f"{_BASE}{path}", headers=_headers(), params=params, timeout=30)
        if resp.status_code == 429 and attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Market metadata
# ---------------------------------------------------------------------------

def fetch_all_markets(
    series: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download all settled market metadata for a series and cache to parquet.

    series defaults to the primary NYC series (KXHIGHNY). Pass a different
    series ticker (e.g. "KXHIGHCHI") to fetch another city. Each series is
    cached in its own file: markets_{series}.parquet.
    """
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    series = series or _SERIES
    cache_name = "markets.parquet" if series == _SERIES else f"markets_{series}.parquet"
    path = _CACHE_ROOT / cache_name

    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    markets: list[dict] = []
    cursor: str | None = None

    while True:
        params: dict = {"series_ticker": series, "limit": _PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor

        data = _get("/historical/markets", params)
        batch = data.get("markets", [])
        markets.extend(batch)

        cursor = data.get("cursor")
        if not cursor or len(batch) < _PAGE_SIZE:
            break
        time.sleep(0.2)

    df = pd.DataFrame(markets)
    df["retrieved_utc"] = utc_now().isoformat()
    df.to_parquet(path, index=False)
    print(f"Fetched {len(df)} markets for {series}")
    return df


# ---------------------------------------------------------------------------
# Candlestick data
# ---------------------------------------------------------------------------

def _candle_cache_path(ticker: str) -> Path:
    return _CANDLE_DIR / f"{ticker}.parquet"


def _fetch_candles_for_market(ticker: str, open_time: str, close_time: str) -> pd.DataFrame:
    """Fetch all 60-min candles for one market, chunking if needed."""
    import datetime

    start_dt = pd.Timestamp(open_time).to_pydatetime()
    end_dt   = pd.Timestamp(close_time).to_pydatetime()

    # Max window per request: _MAX_CANDLES * _PERIOD minutes
    chunk_seconds = _MAX_CANDLES * _PERIOD * 60
    all_candles: list[dict] = []

    cursor_dt = start_dt
    while cursor_dt < end_dt:
        chunk_end = min(cursor_dt + datetime.timedelta(seconds=chunk_seconds), end_dt)

        params = {
            "period_interval": _PERIOD,
            "start_ts": int(cursor_dt.timestamp()),
            "end_ts":   int(chunk_end.timestamp()),
        }
        data = _get(f"/historical/markets/{ticker}/candlesticks", params)
        batch = data.get("candlesticks", [])
        all_candles.extend(batch)
        cursor_dt = chunk_end
        if batch:
            time.sleep(0.05)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df["ticker"] = ticker

    # Flatten nested price dict
    if "price" in df.columns:
        price_df = pd.json_normalize(df["price"].tolist()).add_prefix("price_")
        df = pd.concat([df.drop(columns=["price"]), price_df], axis=1)

    for nested in ("yes_bid", "yes_ask"):
        if nested in df.columns:
            sub = pd.json_normalize(df[nested].tolist()).add_prefix(f"{nested}_")
            df = pd.concat([df.drop(columns=[nested]), sub], axis=1)

    df["end_period_ts"] = pd.to_datetime(df["end_period_ts"], unit="s", utc=True)
    df["retrieved_utc"] = utc_now().isoformat()

    for col in df.columns:
        if col not in ("ticker", "end_period_ts", "retrieved_utc"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_decision_prices(
    markets_df: pd.DataFrame,
    station: str = "KNYC",
    fetch_if_missing: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    For each market, return the price of the last candle before the settlement
    window opens (i.e. before LST midnight on the settlement day).

    This is the overnight market price that a trader would observe after the
    00Z model run (~03:30 UTC) but before the settlement day begins.  It has
    much tighter spreads than the first-candle-at-market-open proxy that was
    used previously.

    Decision candle selection (in priority order):
      1. Last candle with end_period_ts < settlement_window_start
      2. First candle within the first 12 h of the settlement window
      3. First candle in the file (last-resort fallback only)

    Returns a DataFrame with columns:
        ticker, decision_mid, decision_bid, decision_ask, decision_ts,
        decision_candle_type  ("pre_settlement" | "early_window" | "first_candle")

    Markets with no candle data are omitted (caller must handle NaNs after join).
    """
    from kalshi_weather.tz import lst_date_for_utc, settlement_window_utc

    _CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def _f(v: object) -> float:
        try:
            return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")

    for _, mrow in markets_df.iterrows():
        ticker     = mrow["ticker"]
        open_time  = mrow.get("open_time") or mrow.get("created_time")
        close_time = mrow.get("close_time") or mrow.get("expiration_time")
        path = _candle_cache_path(ticker)

        if not path.exists() or force_refresh:
            if not fetch_if_missing:
                continue
            try:
                df = _fetch_candles_for_market(ticker, open_time, close_time)
                if not df.empty:
                    df.to_parquet(path, index=False)
            except requests.HTTPError as e:
                print(f"  Warning: candles unavailable for {ticker}: {e}")
            time.sleep(0.3)

        if not path.exists():
            continue

        candles = pd.read_parquet(path)
        if candles.empty:
            continue

        candles = candles.sort_values("end_period_ts").reset_index(drop=True)

        # Compute settlement window start for this contract
        try:
            ct = pd.Timestamp(close_time)
            ct_utc = ct.tz_localize("UTC") if ct.tzinfo is None else ct.tz_convert("UTC")
            sdate = lst_date_for_utc(ct_utc, station)
            window_start, _ = settlement_window_utc(sdate, station)
        except Exception:
            # Station not in LST config (multi-city expansion pending): use first candle
            candle_row = candles.iloc[0]
            candle_type = "first_candle"
            bid   = _f(candle_row.get("yes_bid_close"))
            ask   = _f(candle_row.get("yes_ask_close"))
            close = _f(candle_row.get("price_close"))
            rows.append(_price_row(ticker, bid, ask, close, candle_row["end_period_ts"], candle_type))
            continue

        # Priority 1: last candle before settlement window opens
        pre = candles[candles["end_period_ts"] < window_start]
        if not pre.empty:
            candle_row  = pre.iloc[-1]
            candle_type = "pre_settlement"
        else:
            # Priority 2: first candle within first 12 h of settlement window
            early = candles[
                (candles["end_period_ts"] >= window_start) &
                (candles["end_period_ts"] <  window_start + pd.Timedelta(hours=12))
            ]
            if not early.empty:
                candle_row  = early.iloc[0]
                candle_type = "early_window"
            else:
                candle_row  = candles.iloc[0]
                candle_type = "first_candle"

        bid   = _f(candle_row.get("yes_bid_close"))
        ask   = _f(candle_row.get("yes_ask_close"))
        close = _f(candle_row.get("price_close"))
        rows.append(_price_row(ticker, bid, ask, close, candle_row["end_period_ts"], candle_type))

    return pd.DataFrame(rows)


def _price_row(
    ticker: str,
    bid: float,
    ask: float,
    close: float,
    ts: object,
    candle_type: str,
) -> dict:
    if np.isfinite(bid) and np.isfinite(ask) and bid < ask:
        mid = (bid + ask) / 2
    elif np.isfinite(ask):
        mid = ask
    elif np.isfinite(close):
        mid = close
    else:
        mid = float("nan")
    return {
        "ticker":               ticker,
        "decision_mid":         mid,
        "decision_bid":         bid,
        "decision_ask":         ask,
        "decision_ts":          ts,
        "decision_candle_type": candle_type,
    }


def fetch_candles(
    markets_df: pd.DataFrame | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch 60-min candlestick data for all markets in markets_df.
    Caches per-ticker to parquet; skips already-cached tickers.
    Returns a single concatenated DataFrame.
    """
    _CANDLE_DIR.mkdir(parents=True, exist_ok=True)

    if markets_df is None:
        markets_df = fetch_all_markets()

    frames: list[pd.DataFrame] = []
    total = len(markets_df)

    for i, row in markets_df.iterrows():
        ticker     = row["ticker"]
        open_time  = row.get("open_time", row.get("created_time"))
        close_time = row.get("close_time", row.get("expiration_time"))
        path = _candle_cache_path(ticker)

        if path.exists() and not force_refresh:
            frames.append(pd.read_parquet(path))
            continue

        try:
            df = _fetch_candles_for_market(ticker, open_time, close_time)
            if not df.empty:
                df.to_parquet(path, index=False)
                frames.append(df)
        except requests.HTTPError as e:
            print(f"  Warning: candles unavailable for {ticker}: {e}")

        if i % 50 == 0:
            print(f"  Candles: {i}/{total} markets processed")
        time.sleep(0.1)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
