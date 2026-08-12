"""
Milestone 2c — Isotonic-calibrated GFS forecast rule.

Calibration split:
  Bias-correction training: 2021-2022 GFS residuals → Gaussian rule
  Isotonic calibration:     2023 predicted probs vs. actual outcomes
  Test (Brier only):        2024

Decision-time prices: now uses the last candle before the settlement window
opens (LST midnight). For NYC T-contracts this is typically the overnight
price after the 00Z model run — a more realistic decision-time proxy than
the original first-candle-at-market-open approach. 99.4% of contracts have
a pre-settlement candle; the rest fall back to the first candle within the
first 12h of the settlement window.

Run: .venv/bin/python scripts/run_m2c_isotonic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd

from kalshi_weather.align import build_aligned_dataset, threshold_contracts
from kalshi_weather.backtest import reliability_curve, run_backtest
from kalshi_weather.config import get_station
from kalshi_weather.ingest.forecasts import fetch_forecasts
from kalshi_weather.ingest.kalshi import fetch_all_markets, get_decision_prices
from kalshi_weather.ingest.labels import fetch_labels
from kalshi_weather.rules import (
    make_forecast_rule,
    make_isotonic_calibrated_rule,
)

# ── Config ────────────────────────────────────────────────────────────────────
STATION_KEY       = "NYC"
BIAS_TRAIN_YEARS  = [2021, 2022]   # GFS residuals for Gaussian rule
CAL_YEARS         = [2023]         # isotonic calibration hold-out
TEST_YEARS        = [2024]         # final test
ALL_YEARS         = sorted(set(BIAS_TRAIN_YEARS + CAL_YEARS + TEST_YEARS))
FCST_MODELS       = ["gfs_seamless"]
# ATM filter: only evaluate contracts where the overnight market price is in
# this range. Removes deep OOT tails with wide spreads and no liquidity.
ATM_MIN, ATM_MAX  = 0.15, 0.85

# ── Setup ─────────────────────────────────────────────────────────────────────
station     = get_station(STATION_KEY)
nws_station = station["nws_station"]
lat, lon    = station["lat"], station["lon"]

print("=== Milestone 2c — Isotonic-Calibrated GFS Rule ===\n")
print(f"Bias train: {BIAS_TRAIN_YEARS}  |  Cal: {CAL_YEARS}  |  Test: {TEST_YEARS}\n")

# ── Ingest ────────────────────────────────────────────────────────────────────
print("1/4  Loading data …")
markets   = fetch_all_markets()
labels    = fetch_labels(nws_station, ALL_YEARS)
forecasts = fetch_forecasts(nws_station, lat, lon, ALL_YEARS, models=FCST_MODELS)
print(f"     {len(markets)} markets | {len(labels)} label days | {len(forecasts)} forecast rows\n")

# ── Align (cal + test together so prices/labels join cleanly) ─────────────────
print("2/4  Aligning data …")
aligned_full = build_aligned_dataset(
    markets, labels, forecasts, station=nws_station, model=FCST_MODELS[0]
)
aligned_full = threshold_contracts(aligned_full)

def year_filter(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    mask = df["settlement_date"].apply(lambda d: hasattr(d, "year") and d.year in set(years))
    return df[mask].reset_index(drop=True)

cal_aligned  = year_filter(aligned_full, CAL_YEARS)
test_aligned = year_filter(aligned_full, TEST_YEARS)
print(f"     Cal: {len(cal_aligned)} contracts  |  Test: {len(test_aligned)} contracts\n")

# ── Decision prices (informational; not used for edge, known to be noisy) ─────
print("3/4  Loading decision prices (for backtest loop; see caveat in module doc) …")
test_markets    = markets[markets["ticker"].isin(test_aligned["ticker"])].reset_index(drop=True)
decision_prices = get_decision_prices(test_markets, fetch_if_missing=False)
test_aligned    = test_aligned.merge(decision_prices, on="ticker", how="left")
test_aligned["last_price_dollars"] = test_aligned["decision_mid"]
test_aligned["yes_bid_dollars"]    = test_aligned["decision_bid"]
test_aligned["yes_ask_dollars"]    = test_aligned["decision_ask"]

# Also attach decision prices to cal set for the run_backtest call
cal_markets    = markets[markets["ticker"].isin(cal_aligned["ticker"])].reset_index(drop=True)
cal_prices     = get_decision_prices(cal_markets, fetch_if_missing=False)
cal_aligned    = cal_aligned.merge(cal_prices, on="ticker", how="left")
cal_aligned["last_price_dollars"] = cal_aligned["decision_mid"]
cal_aligned["yes_bid_dollars"]    = cal_aligned["decision_bid"]
cal_aligned["yes_ask_dollars"]    = cal_aligned["decision_ask"]
print()

# ── Build rules ───────────────────────────────────────────────────────────────
print("4/4  Building rules …")

# Gaussian base rule (trained on 2021-2022)
gauss_base = make_forecast_rule(
    labels, forecasts,
    train_years=BIAS_TRAIN_YEARS,
    station=nws_station,
    model=FCST_MODELS[0],
    distribution="gaussian",
    rule_name="gfs_gaussian_2021_2022",
)

# Isotonic calibration on 2023
iso_rule = make_isotonic_calibrated_rule(
    gauss_base,
    cal_aligned,
    rule_name="gfs_gaussian_isotonic_cal2023",
)
print()

# ── Backtests ─────────────────────────────────────────────────────────────────
# Primary: test on 2024 only
print("─" * 60)
print("Gaussian base rule on 2024 test set:")
gauss_result = run_backtest(test_aligned, gauss_base,
                             rule_name="gfs_gaussian_2021_2022 [2024 only]",
                             atm_min=ATM_MIN, atm_max=ATM_MAX)

print("─" * 60)
print("Isotonic-calibrated rule on 2024 test set:")
iso_result   = run_backtest(test_aligned, iso_rule,
                             rule_name="gfs_gaussian_isotonic_cal2023 [2024 only]",
                             atm_min=ATM_MIN, atm_max=ATM_MAX)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "═" * 64)
print(f"  2024 test — ATM [{ATM_MIN},{ATM_MAX}] — Brier + edge comparison")
print("═" * 64)
for label, result in [("Gaussian base", gauss_result), ("Isotonic cal", iso_result)]:
    b = result.metrics_pess.get("brier_score", float("nan"))
    n = result.metrics_pess.get("n_estimates", 0)
    print(f"  {label:<20}  Brier={b:.5f}  n={n}")
print()

# Reliability curves side-by-side
for label, result in [("Gaussian base", gauss_result), ("Isotonic cal", iso_result)]:
    rel = reliability_curve(result.estimates)
    if rel.empty:
        continue
    print(f"  Reliability — {label}:")
    print(f"  {'Bin':>6} {'Pred':>8} {'Actual':>8} {'N':>6}")
    for _, r in rel.iterrows():
        flag = " ←" if abs(r["mean_pred"] - r["mean_actual"]) > 0.10 else ""
        print(f"  {r['bin_center']:>6.2f} {r['mean_pred']:>8.3f} {r['mean_actual']:>8.3f} {r['count']:>6}{flag}")
    print()

print("═" * 64)
print("  NOTE: fills are modeled (pessimistic=taker+half-spread,")
print("  optimistic=maker). Real edge depends on achievable fill prices.")
print("═" * 64)
