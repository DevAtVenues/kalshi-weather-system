"""
2025 out-of-sample validation — liquid ATM contracts only.

Test year:          2025 (completely fresh; never used in any training or calibration)
Bias-train:         2021-2022 GFS residuals → Gaussian rule
Isotonic cal:       2023 (same as M2c)
Liquid ATM filter:  two-sided (bid > 0.01) AND spread < 0.15 AND mid ∈ [0.15, 0.85]

This is the cleanest possible out-of-sample test — 2025 data was untouched
until this script was written (2026-06-08) with the model frozen beforehand.

Run: .venv/bin/python scripts/run_2025_validation.py
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
from kalshi_weather.rules import make_forecast_rule, make_isotonic_calibrated_rule

# ── Config ────────────────────────────────────────────────────────────────────
STATION_KEY      = "NYC"
BIAS_TRAIN_YEARS = [2021, 2022]
CAL_YEARS        = [2023]
TEST_YEARS       = [2025]
ALL_YEARS        = sorted(set(BIAS_TRAIN_YEARS + CAL_YEARS + TEST_YEARS))
FCST_MODELS      = ["gfs_seamless"]

# Liquid ATM: two-sided market + tight spread + mid in range
ATM_MIN, ATM_MAX  = 0.15, 0.85
MAX_SPREAD        = 0.15
MIN_BID           = 0.01   # bid > 0 means two-sided market

print("=== 2025 Out-of-Sample Validation — Liquid ATM Contracts ===\n")
print(f"Bias train: {BIAS_TRAIN_YEARS}  |  Cal: {CAL_YEARS}  |  Test: {TEST_YEARS}")
print(f"Liquid ATM: mid ∈ [{ATM_MIN}, {ATM_MAX}], spread < {MAX_SPREAD}, two-sided\n")

# ── Setup ─────────────────────────────────────────────────────────────────────
station     = get_station(STATION_KEY)
nws_station = station["nws_station"]
lat, lon    = station["lat"], station["lon"]

# ── Ingest ────────────────────────────────────────────────────────────────────
print("1/4  Loading data …")
markets   = fetch_all_markets()
labels    = fetch_labels(nws_station, ALL_YEARS)
forecasts = fetch_forecasts(nws_station, lat, lon, ALL_YEARS, models=FCST_MODELS)
print(f"     {len(markets)} markets | {len(labels)} label days | {len(forecasts)} forecast rows\n")

# ── Align ─────────────────────────────────────────────────────────────────────
print("2/4  Aligning data …")

def year_filter(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    mask = df["settlement_date"].apply(lambda d: hasattr(d, "year") and d.year in set(years))
    return df[mask].reset_index(drop=True)

# Need labels for 2023 (cal) and 2025 (test); forecasts for same years
# build_aligned_dataset aligns on settlement_date, so we pass what we need
aligned_full = build_aligned_dataset(
    markets, labels, forecasts, station=nws_station, model=FCST_MODELS[0]
)
aligned_full = threshold_contracts(aligned_full)

cal_aligned  = year_filter(aligned_full, CAL_YEARS)
test_aligned = year_filter(aligned_full, TEST_YEARS)
print(f"     Cal (2023): {len(cal_aligned)} contracts  |  Test (2025): {len(test_aligned)} contracts\n")

# ── Decision prices (fetch candles for test + cal if missing) ─────────────────
print("3/4  Fetching decision prices (will download missing 2025 candles) …")
print("     This may take several minutes for the first run …")

test_markets = markets[markets["ticker"].isin(test_aligned["ticker"])].reset_index(drop=True)
cal_markets  = markets[markets["ticker"].isin(cal_aligned["ticker"])].reset_index(drop=True)

print(f"     Fetching candles for {len(test_markets)} test contracts …")
test_prices = get_decision_prices(test_markets, fetch_if_missing=True)
print(f"     Got prices for {len(test_prices)}/{len(test_aligned)} test contracts.")

print(f"     Fetching candles for {len(cal_markets)} cal contracts …")
cal_prices  = get_decision_prices(cal_markets, fetch_if_missing=True)
print(f"     Got prices for {len(cal_prices)}/{len(cal_aligned)} cal contracts.\n")

# Attach prices to aligned frames
test_aligned = test_aligned.merge(test_prices, on="ticker", how="left")
test_aligned["last_price_dollars"] = test_aligned["decision_mid"]
test_aligned["yes_bid_dollars"]    = test_aligned["decision_bid"]
test_aligned["yes_ask_dollars"]    = test_aligned["decision_ask"]

cal_aligned = cal_aligned.merge(cal_prices, on="ticker", how="left")
cal_aligned["last_price_dollars"] = cal_aligned["decision_mid"]
cal_aligned["yes_bid_dollars"]    = cal_aligned["decision_bid"]
cal_aligned["yes_ask_dollars"]    = cal_aligned["decision_ask"]

# ── Liquid ATM filter ─────────────────────────────────────────────────────────
def liquid_atm_filter(df: pd.DataFrame, label: str) -> pd.DataFrame:
    bid    = pd.to_numeric(df["yes_bid_dollars"], errors="coerce")
    ask    = pd.to_numeric(df["yes_ask_dollars"], errors="coerce")
    mid    = pd.to_numeric(df["last_price_dollars"], errors="coerce")
    spread = ask - bid

    two_sided  = bid > MIN_BID
    tight      = spread < MAX_SPREAD
    in_range   = mid.between(ATM_MIN, ATM_MAX)
    mask       = two_sided & tight & in_range

    filtered = df[mask].reset_index(drop=True)
    print(f"     Liquid ATM ({label}): {len(filtered)}/{len(df)} contracts pass filter")
    if len(filtered) > 0:
        print(f"       spread: median=${spread[mask].median():.3f}, max=${spread[mask].max():.3f}")
    return filtered

print("Applying liquid ATM filter:")
test_liquid = liquid_atm_filter(test_aligned, "2025 test")
cal_liquid  = liquid_atm_filter(cal_aligned,  "2023 cal")
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

# Isotonic calibration on 2023 (all ATM, same as M2c)
iso_rule = make_isotonic_calibrated_rule(
    gauss_base,
    cal_aligned,    # train isotonic on full 2023 ATM set (same as M2c)
    rule_name="gfs_gaussian_isotonic_cal2023",
)
print()

# ── Backtests ─────────────────────────────────────────────────────────────────
print("─" * 60)
print("Gaussian base rule on 2025 liquid ATM:")
gauss_result = run_backtest(
    test_liquid, gauss_base,
    rule_name="gfs_gaussian [2025 liquid ATM]",
    atm_min=ATM_MIN, atm_max=ATM_MAX,
)

print("─" * 60)
print("Isotonic-calibrated rule on 2025 liquid ATM:")
iso_result = run_backtest(
    test_liquid, iso_rule,
    rule_name="gfs_gaussian_isotonic_cal2023 [2025 liquid ATM]",
    atm_min=ATM_MIN, atm_max=ATM_MAX,
)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  2025 OUT-OF-SAMPLE — Liquid ATM — Pessimistic vs. Optimistic fill")
print("═" * 70)

for label, result in [("Gaussian base", gauss_result), ("Isotonic cal", iso_result)]:
    p = result.metrics_pess
    o = result.metrics_opt
    print(f"\n  {label}:")
    print(f"    Estimates: {p.get('n_estimates', 0)}  |  Trades: {p.get('n_trades', 0)}")
    print(f"    Brier:     {p.get('brier_score', float('nan')):.5f}")
    print(f"    Pess:  total={p.get('total_return','—'):.4f}  avg={p.get('avg_return','—'):.5f}  "
          f"CI=({p.get('edge_ci_low','—'):.3f}, {p.get('edge_ci_high','—'):.3f})  win={p.get('win_rate','—'):.3f}")
    print(f"    Opt:   total={o.get('total_return','—'):.4f}  avg={o.get('avg_return','—'):.5f}  "
          f"CI=({o.get('edge_ci_low','—'):.3f}, {o.get('edge_ci_high','—'):.3f})  win={o.get('win_rate','—'):.3f}")

print("\n" + "═" * 70)
print("  Comparison to 2024 result (for reference):")
print("    Gaussian 2024 liquid ATM (pess): CI=(0.019, 0.144), 57 trades")
print("═" * 70)

# Reliability curves
print()
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
