"""
Milestone 1 — end-to-end vertical slice backtest.

City: New York City (KNYC, Central Park)
Rule: Climatological baseline — P(tmax >= threshold) from historical CLI records.
Test: 2023-2024 T-contracts (threshold contracts only, settled).
Train: 2010-2022 historical CLI records (for climo distribution).

Market price: first-candle bid/ask after market open (decision-time proxy).
              This avoids using the post-settlement price (0.99/0.01) in
              market metadata, which encodes the answer.

Run: .venv/bin/python scripts/run_m1_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd

from kalshi_weather.align import alignment_summary, build_aligned_dataset, threshold_contracts
from kalshi_weather.backtest import run_backtest
from kalshi_weather.config import get_station
from kalshi_weather.ingest.forecasts import fetch_forecasts
from kalshi_weather.ingest.kalshi import fetch_all_markets, get_decision_prices
from kalshi_weather.ingest.labels import fetch_labels
from kalshi_weather.rules import make_climo_rule

# ── Config ────────────────────────────────────────────────────────────────────
STATION_KEY  = "NYC"
TRAIN_YEARS  = list(range(2010, 2023))   # 13 years for climo baseline
TEST_YEARS   = [2023, 2024]              # vertical slice test window
FCST_MODELS  = ["gfs_seamless"]

# ── Ingest ────────────────────────────────────────────────────────────────────
station = get_station(STATION_KEY)
nws_station = station["nws_station"]
lat         = station["lat"]
lon         = station["lon"]

print("=== Milestone 1 Backtest — NYC Climo Baseline ===\n")

print("1/5  Loading Kalshi market metadata …")
markets = fetch_all_markets()
print(f"     {len(markets)} markets loaded.\n")

print("2/5  Fetching IEM labels …")
all_label_years = sorted(set(TRAIN_YEARS + TEST_YEARS))
labels = fetch_labels(nws_station, all_label_years)
print(f"     {len(labels)} daily records ({labels['date'].min()} → {labels['date'].max()}).\n")

print("3/5  Fetching Open-Meteo GFS forecasts …")
forecasts = fetch_forecasts(nws_station, lat, lon, TEST_YEARS, models=FCST_MODELS)
print(f"     {len(forecasts)} forecast rows.\n")

# ── Align ─────────────────────────────────────────────────────────────────────
print("4/5  Aligning data sources …")
aligned = build_aligned_dataset(markets, labels, forecasts, station=nws_station, model=FCST_MODELS[0])
aligned = threshold_contracts(aligned)

# Restrict to test window
test_dates = {
    d for d in aligned["settlement_date"]
    if hasattr(d, "year") and d.year in TEST_YEARS
}
aligned = aligned[aligned["settlement_date"].apply(lambda d: d in test_dates)].reset_index(drop=True)
print()
alignment_summary(aligned)

# ── Decision-time prices ──────────────────────────────────────────────────────
# The market metadata price fields (last_price_dollars etc.) are POST-SETTLEMENT
# values (0.99 for YES, 0.01 for NO). We need the pre-settlement candle price.
# get_decision_prices() fetches candles per market and returns the FIRST candle
# bid/ask (market-open price), which reflects genuine pre-settlement uncertainty.
print(f"\n5/5  Fetching decision-time prices for {len(aligned)} contracts …")
test_markets = markets[markets["ticker"].isin(aligned["ticker"])].reset_index(drop=True)
decision_prices = get_decision_prices(test_markets)
n_fetched = len(decision_prices)
print(f"     Got prices for {n_fetched}/{len(aligned)} contracts.\n")

# Join decision prices into aligned; fall back to NaN (backtest will skip)
aligned = aligned.merge(decision_prices, on="ticker", how="left")

# Replace price columns with decision-time values
# backtest.py reads: last_price_dollars (market_price), yes_bid_dollars, yes_ask_dollars
aligned["last_price_dollars"] = aligned["decision_mid"]
aligned["yes_bid_dollars"]    = aligned["decision_bid"]
aligned["yes_ask_dollars"]    = aligned["decision_ask"]

n_priced = aligned["last_price_dollars"].notna().sum()
print(f"Decision prices available: {n_priced}/{len(aligned)}")
print(f"Sample prices: {aligned['last_price_dollars'].describe().round(3).to_dict()}\n")

# ── Baseline rule ─────────────────────────────────────────────────────────────
rule = make_climo_rule(labels, TRAIN_YEARS, rule_name="climo_2010_2022")

# ── Backtest ──────────────────────────────────────────────────────────────────
print("Running backtest …")
result = run_backtest(aligned, rule, rule_name="climo_2010_2022")
