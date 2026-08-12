"""
Fetch market metadata for all configured cities except NYC (already cached).
Also re-fetches missing 2025 NYC candles now that retry logic is in place.

Run: .venv/bin/python scripts/fetch_city_markets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import time
import yaml
import pandas as pd

from kalshi_weather.ingest.kalshi import fetch_all_markets, get_decision_prices

CONFIG_PATH = Path(__file__).parents[1] / "config" / "stations.yaml"
CACHE_ROOT  = Path(__file__).parents[1] / "data" / "raw" / "kalshi"

with open(CONFIG_PATH) as fh:
    stations = yaml.safe_load(fh)

# ── Step 1: Fetch market metadata for all cities ──────────────────────────────
print("=== Fetching market metadata for all configured cities ===\n")

all_markets_frames: list[pd.DataFrame] = []

for key, cfg in stations.items():
    series = cfg.get("kalshi_series")
    city   = cfg.get("city", key)
    if not series:
        continue
    print(f"  {city} ({series}) …", end=" ", flush=True)
    try:
        df = fetch_all_markets(series=series)
        print(f"{len(df)} markets")
        all_markets_frames.append(df)
    except Exception as e:
        print(f"FAILED: {e}")

if all_markets_frames:
    combined = pd.concat(all_markets_frames, ignore_index=True)
    out_path = CACHE_ROOT / "markets_all_cities.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nCombined: {len(combined)} markets across all cities → {out_path.name}")

# ── Step 2: Re-fetch missing 2025 NYC T-type candles ─────────────────────────
print("\n=== Re-fetching missing 2025 NYC T-type candles ===\n")

nyc_markets = pd.read_parquet(CACHE_ROOT / "markets.parquet")
nyc_markets["close_dt"] = pd.to_datetime(nyc_markets["close_time"], utc=True, errors="coerce")

# Only T-type (threshold) 2025 contracts
m2025 = nyc_markets[
    (nyc_markets["close_dt"].dt.year == 2025) &
    (nyc_markets["ticker"].str.contains(r"-T\d+$", regex=True, na=False))
].reset_index(drop=True)

candle_dir = CACHE_ROOT / "candles"
candle_dir.mkdir(parents=True, exist_ok=True)
cached = {f.stem for f in candle_dir.glob("*.parquet")}
missing = m2025[~m2025["ticker"].isin(cached)].reset_index(drop=True)

print(f"2025 T-type contracts: {len(m2025)}")
print(f"Already cached:        {len(m2025) - len(missing)}")
print(f"Still missing:         {len(missing)}")

if len(missing) > 0:
    print("Fetching missing candles (retry-with-backoff active) …")
    prices = get_decision_prices(missing, fetch_if_missing=True)
    still_missing = missing[~missing["ticker"].isin({f.stem for f in candle_dir.glob("*.parquet")})]
    print(f"After fetch: {len(still_missing)} still missing")
else:
    print("All 2025 T-type candles already cached.")

print("\nDone.")
