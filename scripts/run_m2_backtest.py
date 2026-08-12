"""
Milestone 2 — GFS forecast + bias-correction backtest.

City: New York City (KNYC, Central Park)
Rule: Bias-corrected GFS forecast — P(actual_high ≥ T | GFS_tmax) via
      empirical residual distribution from 2021-2022 training window.
Test: 2023-2024 T-contracts (same test window as M1).

Bias correction training uses the Open-Meteo Historical Forecast API
(GFS starts 2021), paired with NWS CLI actuals. The residual distribution
is grouped by calendar month to capture seasonal GFS bias patterns.

Run: .venv/bin/python scripts/run_m2_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from kalshi_weather.align import alignment_summary, build_aligned_dataset, threshold_contracts
from kalshi_weather.backtest import run_backtest
from kalshi_weather.config import get_station
from kalshi_weather.ingest.forecasts import fetch_forecasts
from kalshi_weather.ingest.kalshi import fetch_all_markets, get_decision_prices
from kalshi_weather.ingest.labels import fetch_labels
from kalshi_weather.rules import make_climo_rule, make_forecast_rule

# ── Config ────────────────────────────────────────────────────────────────────
STATION_KEY = "NYC"
# Bias-correction training: 2021-2022 (earliest GFS Historical Forecast data).
# M1 climo training: 2010-2022 (same as M1 run, for fair comparison).
# Test window: 2023-2024 — same as M1 so results are directly comparable.
BIAS_TRAIN_YEARS = [2021, 2022]
CLIMO_TRAIN_YEARS = list(range(2010, 2023))
TEST_YEARS        = [2023, 2024]
FCST_MODELS       = ["gfs_seamless"]
# ATM filter: only trade contracts where the overnight market price is in
# this range. Removes deep OOT tails where spreads are widest and liquidity
# is lowest. Set to (0.0, 1.0) to disable.
ATM_MIN, ATM_MAX  = 0.15, 0.85

# ── Setup ─────────────────────────────────────────────────────────────────────
station    = get_station(STATION_KEY)
nws_station = station["nws_station"]
lat         = station["lat"]
lon         = station["lon"]

print("=== Milestone 2 Backtest — GFS Bias-Corrected Rule ===\n")

# ── Ingest ────────────────────────────────────────────────────────────────────
print("1/5  Loading Kalshi market metadata …")
markets = fetch_all_markets()
print(f"     {len(markets)} markets loaded.\n")

print("2/5  Fetching IEM labels …")
all_label_years = sorted(set(CLIMO_TRAIN_YEARS + BIAS_TRAIN_YEARS + TEST_YEARS))
labels = fetch_labels(nws_station, all_label_years)
print(f"     {len(labels)} daily records ({labels['date'].min()} → {labels['date'].max()}).\n")

print("3/5  Fetching GFS forecasts …")
all_fcst_years = sorted(set(BIAS_TRAIN_YEARS + TEST_YEARS))
forecasts = fetch_forecasts(nws_station, lat, lon, all_fcst_years, models=FCST_MODELS)
print(f"     {len(forecasts)} forecast rows.\n")

# ── Align test window ─────────────────────────────────────────────────────────
print("4/5  Aligning data sources (test window: 2023-2024) …")
aligned = build_aligned_dataset(markets, labels, forecasts, station=nws_station, model=FCST_MODELS[0])
aligned = threshold_contracts(aligned)

test_dates = {
    d for d in aligned["settlement_date"]
    if hasattr(d, "year") and d.year in TEST_YEARS
}
aligned = aligned[aligned["settlement_date"].apply(lambda d: d in test_dates)].reset_index(drop=True)
print()
alignment_summary(aligned)

# ── Decision-time prices ──────────────────────────────────────────────────────
print(f"\n5/5  Loading decision-time prices for {len(aligned)} contracts …")
test_markets  = markets[markets["ticker"].isin(aligned["ticker"])].reset_index(drop=True)
decision_prices = get_decision_prices(test_markets, fetch_if_missing=True)
print(f"     Got prices for {len(decision_prices)}/{len(aligned)} contracts.\n")

aligned = aligned.merge(decision_prices, on="ticker", how="left")
aligned["last_price_dollars"] = aligned["decision_mid"]
aligned["yes_bid_dollars"]    = aligned["decision_bid"]
aligned["yes_ask_dollars"]    = aligned["decision_ask"]

n_priced = aligned["last_price_dollars"].notna().sum()
print(f"Decision prices available: {n_priced}/{len(aligned)}\n")

# ── Build rules ───────────────────────────────────────────────────────────────
print("Building rules …")

# M1 climo baseline (for comparison)
climo_rule = make_climo_rule(labels, train_years=CLIMO_TRAIN_YEARS,
                              rule_name="climo_2010_2022")

# M2a: empirical CDF (previous run)
fcst_rule_ecdf = make_forecast_rule(
    labels, forecasts,
    train_years=BIAS_TRAIN_YEARS,
    station=nws_station,
    model=FCST_MODELS[0],
    distribution="empirical",
    rule_name="gfs_ecdf_2021_2022",
)

# M2b: Gaussian CDF (proper tail behaviour)
fcst_rule_gauss = make_forecast_rule(
    labels, forecasts,
    train_years=BIAS_TRAIN_YEARS,
    station=nws_station,
    model=FCST_MODELS[0],
    distribution="gaussian",
    rule_name="gfs_gaussian_2021_2022",
)
print()

# ── Backtests ─────────────────────────────────────────────────────────────────
print("Running M1 climo backtest (comparison baseline) …")
climo_result = run_backtest(aligned, climo_rule, rule_name="climo_2010_2022",
                            atm_min=ATM_MIN, atm_max=ATM_MAX)

print("Running M2a empirical CDF backtest …")
ecdf_result  = run_backtest(aligned, fcst_rule_ecdf, rule_name="gfs_ecdf_2021_2022",
                             atm_min=ATM_MIN, atm_max=ATM_MAX)

print("Running M2b Gaussian CDF backtest …")
gauss_result = run_backtest(aligned, fcst_rule_gauss, rule_name="gfs_gaussian_2021_2022",
                             atm_min=ATM_MIN, atm_max=ATM_MAX)

# ── Side-by-side summary ──────────────────────────────────────────────────────
print("\n" + "═" * 72)
print("  Comparison — pessimistic fill model")
print("═" * 72)
metrics = [
    ("n_estimates",  "Estimates"),
    ("n_trades",     "Trades"),
    ("brier_score",  "Brier score"),
    ("win_rate",     "Win rate"),
    ("total_return", "Total return ($)"),
    ("avg_return",   "Avg return/trade ($)"),
    ("edge_mean",    "Mean edge"),
    ("edge_ci_low",  "Edge CI low (95%)"),
    ("edge_ci_high", "Edge CI high (95%)"),
    ("effective_n",  "Effective N"),
]
print(f"  {'Metric':<26} {'M1 Climo':>12} {'M2a ECDF':>12} {'M2b Gaussian':>14}")
print(f"  {'-'*26} {'-'*12} {'-'*12} {'-'*14}")
for key, label in metrics:
    cv = climo_result.metrics_pess.get(key, "—")
    ev = ecdf_result.metrics_pess.get(key, "—")
    gv = gauss_result.metrics_pess.get(key, "—")
    print(f"  {label:<26} {str(cv):>12} {str(ev):>12} {str(gv):>14}")
print("═" * 72)
