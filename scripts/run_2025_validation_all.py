"""
2025 out-of-sample validation for all cities with 2025 candle data.

Model parameters match the new production model:
  Bias train:   2022-2023
  Isotonic cal: 2024
  Test:         2025

Cities: NYC, CHI, MIA, AUS
Liquid ATM filter: two-sided, spread < 0.15, mid in [0.15, 0.85]

Run: .venv/bin/python scripts/run_2025_validation_all.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd
import numpy as np

from kalshi_weather.align import build_aligned_dataset, threshold_contracts
from kalshi_weather.backtest import reliability_curve, run_backtest
from kalshi_weather.config import get_station
from kalshi_weather.ingest.forecasts import fetch_forecasts
from kalshi_weather.ingest.kalshi import get_decision_prices
from kalshi_weather.ingest.labels import fetch_labels
from kalshi_weather.rules import make_forecast_rule, make_isotonic_calibrated_rule

BIAS_TRAIN_YEARS = [2022, 2023]
ISO_CAL_YEARS    = [2024]
TEST_YEARS       = [2025]
FCST_MODELS      = ["gfs_seamless"]
ATM_MIN, ATM_MAX = 0.15, 0.85
MAX_SPREAD       = 0.15
MIN_BID          = 0.01

CACHE_ROOT = Path(__file__).parents[1] / "data" / "raw" / "kalshi"

CITIES = {
    "NYC": "markets.parquet",
    "CHI": "markets_KXHIGHCHI.parquet",
    "MIA": "markets_KXHIGHMIA.parquet",
    "AUS": "markets_KXHIGHAUS.parquet",
}

def year_filter(df, years):
    mask = df["settlement_date"].apply(lambda d: hasattr(d, "year") and d.year in set(years))
    return df[mask].reset_index(drop=True)

def liquid_atm_filter(df):
    bid    = pd.to_numeric(df.get("yes_bid_dollars", pd.Series(dtype=float)), errors="coerce")
    ask    = pd.to_numeric(df.get("yes_ask_dollars", pd.Series(dtype=float)), errors="coerce")
    mid    = pd.to_numeric(df.get("last_price_dollars", pd.Series(dtype=float)), errors="coerce")
    spread = ask - bid
    mask   = (bid > MIN_BID) & (spread < MAX_SPREAD) & mid.between(ATM_MIN, ATM_MAX)
    return df[mask].reset_index(drop=True)

all_results = {}

for city_key, markets_file in CITIES.items():
    print(f"\n{'='*65}")
    print(f"  {city_key}")
    print(f"{'='*65}")

    station     = get_station(city_key)
    nws_station = station["nws_station"]
    lat, lon    = station["lat"], station["lon"]

    all_years = sorted(set(BIAS_TRAIN_YEARS + ISO_CAL_YEARS + TEST_YEARS))
    markets   = pd.read_parquet(CACHE_ROOT / markets_file)
    labels    = fetch_labels(nws_station, all_years)
    forecasts = fetch_forecasts(nws_station, lat, lon, all_years, models=FCST_MODELS)
    print(f"  Labels: {len(labels)} days | Forecasts: {len(forecasts)} rows")

    aligned_full = build_aligned_dataset(
        markets, labels, forecasts, station=nws_station, model=FCST_MODELS[0]
    )
    aligned_full = threshold_contracts(aligned_full)

    cal_aligned  = year_filter(aligned_full, ISO_CAL_YEARS)
    test_aligned = year_filter(aligned_full, TEST_YEARS)

    # Attach prices — fetch any missing candles
    for label, df_subset in [("cal (2024)", cal_aligned), ("test (2025)", test_aligned)]:
        subset_markets = markets[markets["ticker"].isin(df_subset["ticker"])].reset_index(drop=True)
        prices = get_decision_prices(subset_markets, fetch_if_missing=True)
        if prices.empty or "ticker" not in prices.columns:
            prices = pd.DataFrame(columns=["ticker","decision_mid","decision_bid","decision_ask"])
        df_subset = df_subset.merge(prices, on="ticker", how="left")
        df_subset["last_price_dollars"] = df_subset["decision_mid"]
        df_subset["yes_bid_dollars"]    = df_subset["decision_bid"]
        df_subset["yes_ask_dollars"]    = df_subset["decision_ask"]
        if label.startswith("cal"):
            cal_aligned = df_subset
        else:
            test_aligned = df_subset

    test_liquid = liquid_atm_filter(test_aligned)
    cal_priced  = cal_aligned["decision_mid"].notna().sum() if "decision_mid" in cal_aligned.columns else 0
    print(f"  Cal 2024: {len(cal_aligned)} contracts, {cal_priced} priced")
    print(f"  Test 2025 liquid ATM: {len(test_liquid)}/{len(test_aligned)} contracts pass filter")

    # Build rules
    gauss_rule = make_forecast_rule(
        labels, forecasts,
        train_years=BIAS_TRAIN_YEARS,
        station=nws_station,
        model=FCST_MODELS[0],
        distribution="gaussian",
        rule_name=f"gfs_gaussian_val_{city_key}",
    )

    iso_rule = None
    if cal_priced >= 50:
        iso_rule = make_isotonic_calibrated_rule(
            gauss_rule, cal_aligned,
            rule_name=f"gfs_isotonic_val_{city_key}",
        )

    if len(test_liquid) < 10:
        print(f"  SKIP: only {len(test_liquid)} liquid ATM contracts in 2025")
        continue

    gauss_result = run_backtest(
        test_liquid, gauss_rule,
        rule_name=f"gaussian [{city_key} 2025]",
        atm_min=ATM_MIN, atm_max=ATM_MAX,
    )

    iso_result = None
    if iso_rule:
        iso_result = run_backtest(
            test_liquid, iso_rule,
            rule_name=f"isotonic [{city_key} 2025]",
            atm_min=ATM_MIN, atm_max=ATM_MAX,
        )

    all_results[city_key] = {"gaussian": gauss_result, "isotonic": iso_result}

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'═'*75}")
print("  2025 OOS SUMMARY — Liquid ATM — Pessimistic fill")
print(f"{'═'*75}")
print(f"  {'City':<6} {'Rule':<10} {'Trades':>7} {'Win%':>6} {'AvgRet':>8} {'CI low':>8} {'CI high':>8} {'Brier':>8}")
print(f"  {'-'*70}")

for city_key, res in all_results.items():
    for rule_label, result in [("gaussian", res["gaussian"]), ("isotonic", res["isotonic"])]:
        if result is None:
            continue
        p = result.metrics_pess
        print(
            f"  {city_key:<6} {rule_label:<10} "
            f"{p.get('n_trades',0):>7} "
            f"{p.get('win_rate',float('nan')):>6.1%} "
            f"{p.get('avg_return',float('nan')):>8.4f} "
            f"{p.get('edge_ci_low',float('nan')):>8.3f} "
            f"{p.get('edge_ci_high',float('nan')):>8.3f} "
            f"{p.get('brier_score',float('nan')):>8.5f}"
        )
    print(f"  {'-'*70}")

print(f"\n  Note: Pessimistic fill model. Edge CI is 95% block-bootstrap.")
print(f"  Preferred rule = rule whose CI low is higher (more robust).")
