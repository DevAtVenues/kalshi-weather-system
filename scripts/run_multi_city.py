"""
Multi-city validation — same isotonic GFS pipeline on all cities with 2021+ history.

Currently: NYC and Chicago both have data back to August 2021.
Pipeline per city:
  Bias-train:     2021-2022 GFS residuals → Gaussian rule
  Isotonic cal:   2023
  Test:           2025 liquid ATM (two-sided, spread < 0.15, mid ∈ [0.15, 0.85])

Run: .venv/bin/python scripts/run_multi_city.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import yaml
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
BIAS_TRAIN_YEARS = [2021, 2022]
CAL_YEARS        = [2023]
TEST_YEARS       = [2025]
ALL_YEARS        = sorted(set(BIAS_TRAIN_YEARS + CAL_YEARS + TEST_YEARS))
FCST_MODELS      = ["gfs_seamless"]

ATM_MIN    = 0.15
ATM_MAX    = 0.85
MAX_SPREAD = 0.15
MIN_BID    = 0.01

# Cities with 2021+ Kalshi history: NYC, CHI.
# Miami and Austin started mid-2023, but GFS + IEM labels go back to 2021,
# so we can use the same bias-train window (2021-2022) with 2023 isotonic cal.
CITY_KEYS = ["NYC", "CHI", "MIA", "AUS"]

CACHE_ROOT = Path(__file__).parents[1] / "data" / "raw" / "kalshi"

print("=== Multi-City Validation — Isotonic GFS Rule ===")
print(f"Bias train: {BIAS_TRAIN_YEARS}  |  Cal: {CAL_YEARS}  |  Test: {TEST_YEARS}")
print(f"Liquid ATM: mid ∈ [{ATM_MIN},{ATM_MAX}], spread < {MAX_SPREAD}, two-sided\n")


def year_filter(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    mask = df["settlement_date"].apply(lambda d: hasattr(d, "year") and d.year in set(years))
    return df[mask].reset_index(drop=True)


def liquid_atm_filter(df: pd.DataFrame) -> pd.DataFrame:
    bid    = pd.to_numeric(df["yes_bid_dollars"], errors="coerce")
    ask    = pd.to_numeric(df["yes_ask_dollars"], errors="coerce")
    mid    = pd.to_numeric(df["last_price_dollars"], errors="coerce")
    mask   = (bid > MIN_BID) & ((ask - bid) < MAX_SPREAD) & mid.between(ATM_MIN, ATM_MAX)
    return df[mask].reset_index(drop=True)


summary_rows: list[dict] = []

for city_key in CITY_KEYS:
    print("=" * 65)
    print(f"  CITY: {city_key}")
    print("=" * 65)

    station     = get_station(city_key)
    nws_station = station["nws_station"]
    lat, lon    = station["lat"], station["lon"]
    series      = station["kalshi_series"]

    # Load markets for this city
    cache_name = "markets.parquet" if series == "KXHIGHNY" else f"markets_{series}.parquet"
    markets = pd.read_parquet(CACHE_ROOT / cache_name)
    print(f"  Markets: {len(markets)}")

    # Fetch labels + forecasts (cached after first run)
    print(f"  Loading labels and forecasts for {nws_station} …")
    labels    = fetch_labels(nws_station, ALL_YEARS)
    forecasts = fetch_forecasts(nws_station, lat, lon, ALL_YEARS, models=FCST_MODELS)
    print(f"  Labels: {len(labels)} days | Forecasts: {len(forecasts)} rows")

    # Align
    aligned_full = build_aligned_dataset(
        markets, labels, forecasts, station=nws_station, model=FCST_MODELS[0]
    )
    aligned_full = threshold_contracts(aligned_full)

    cal_aligned  = year_filter(aligned_full, CAL_YEARS)
    test_aligned = year_filter(aligned_full, TEST_YEARS)
    print(f"  Aligned — Cal (2023): {len(cal_aligned)}  |  Test (2025): {len(test_aligned)}")

    # Decision prices — fetch candles if missing
    print(f"  Fetching decision prices for {len(test_aligned)} test + {len(cal_aligned)} cal contracts …")
    test_markets = markets[markets["ticker"].isin(test_aligned["ticker"])].reset_index(drop=True)
    cal_markets  = markets[markets["ticker"].isin(cal_aligned["ticker"])].reset_index(drop=True)

    test_prices = get_decision_prices(test_markets, station=nws_station, fetch_if_missing=True)
    cal_prices  = get_decision_prices(cal_markets,  station=nws_station, fetch_if_missing=True)

    test_aligned = test_aligned.merge(test_prices, on="ticker", how="left")
    test_aligned["last_price_dollars"] = test_aligned["decision_mid"]
    test_aligned["yes_bid_dollars"]    = test_aligned["decision_bid"]
    test_aligned["yes_ask_dollars"]    = test_aligned["decision_ask"]

    cal_aligned = cal_aligned.merge(cal_prices, on="ticker", how="left")
    cal_aligned["last_price_dollars"] = cal_aligned["decision_mid"]
    cal_aligned["yes_bid_dollars"]    = cal_aligned["decision_bid"]
    cal_aligned["yes_ask_dollars"]    = cal_aligned["decision_ask"]

    # Liquid ATM filter
    test_liquid = liquid_atm_filter(test_aligned)
    cal_liquid  = liquid_atm_filter(cal_aligned)
    print(f"  Liquid ATM — Test: {len(test_liquid)}/{len(test_aligned)}  |  Cal: {len(cal_liquid)}/{len(cal_aligned)}")

    if len(test_liquid) < 10:
        print(f"  Skipping {city_key} — too few liquid ATM test contracts.\n")
        continue

    # Build rules
    print("  Building rules …")
    gauss_base = make_forecast_rule(
        labels, forecasts,
        train_years=BIAS_TRAIN_YEARS,
        station=nws_station,
        model=FCST_MODELS[0],
        distribution="gaussian",
        rule_name=f"gfs_gaussian_2021_2022_{city_key}",
    )

    iso_rule = make_isotonic_calibrated_rule(
        gauss_base,
        cal_aligned,
        rule_name=f"gfs_gaussian_isotonic_cal2023_{city_key}",
    )

    # Backtests
    print()
    gauss_result = run_backtest(
        test_liquid, gauss_base,
        rule_name=f"{city_key} Gaussian [2025 liquid ATM]",
        atm_min=ATM_MIN, atm_max=ATM_MAX,
    )

    iso_result = run_backtest(
        test_liquid, iso_rule,
        rule_name=f"{city_key} Isotonic [2025 liquid ATM]",
        atm_min=ATM_MIN, atm_max=ATM_MAX,
    )

    for rule_label, result in [("Gaussian", gauss_result), ("Isotonic", iso_result)]:
        p = result.metrics_pess
        o = result.metrics_opt
        summary_rows.append({
            "city":       city_key,
            "rule":       rule_label,
            "n_test":     len(test_liquid),
            "n_trades":   p.get("n_trades", 0),
            "brier":      p.get("brier_score", float("nan")),
            "win_rate":   p.get("win_rate", float("nan")),
            "total_pess": p.get("total_return", float("nan")),
            "avg_pess":   p.get("avg_return", float("nan")),
            "ci_lo_pess": p.get("edge_ci_low", float("nan")),
            "ci_hi_pess": p.get("edge_ci_high", float("nan")),
            "total_opt":  o.get("total_return", float("nan")),
            "avg_opt":    o.get("avg_return", float("nan")),
        })

    print()

# ── Combined summary ──────────────────────────────────────────────────────────
print("\n" + "═" * 82)
print("  COMBINED SUMMARY — 2025 Liquid ATM Out-of-Sample")
print("═" * 82)
print(f"  {'City':<6} {'Rule':<10} {'Tests':>6} {'Trades':>7} {'Brier':>7} "
      f"{'Win':>5} {'Pess CI':>18} {'Avg$':>8}")
print(f"  {'-'*6} {'-'*10} {'-'*6} {'-'*7} {'-'*7} {'-'*5} {'-'*18} {'-'*8}")

for row in summary_rows:
    ci = f"({row['ci_lo_pess']:.3f}, {row['ci_hi_pess']:.3f})"
    flag = " ✓" if row['ci_lo_pess'] > 0 else ("  " if row['ci_lo_pess'] > -0.01 else "  ")
    print(f"  {row['city']:<6} {row['rule']:<10} {row['n_test']:>6} {row['n_trades']:>7} "
          f"{row['brier']:>7.4f} {row['win_rate']:>5.3f} {ci:>18}{flag} {row['avg_pess']:>8.4f}")

print("═" * 82)
print("  ✓ = pessimistic CI lower bound > 0 (both positive)")
print()

# Aggregate: if both cities profitable, show combined implied annual P&L at $50/contract
print("  Implied annual P&L at $50/contract position size:")
iso_rows = [r for r in summary_rows if r["rule"] == "Isotonic"]
total_trades = sum(r["n_trades"] for r in iso_rows)
if total_trades > 0:
    weighted_avg = sum(r["avg_pess"] * r["n_trades"] for r in iso_rows) / total_trades
    print(f"    Total isotonic trades/year: {total_trades}")
    print(f"    Weighted avg pess return:   ${weighted_avg:.4f}/trade")
    print(f"    Annual P&L at $50:          ${weighted_avg * total_trades * 50:.0f}")
