"""
2025 out-of-sample validation for B-type (bracket) contracts.

Compares gaussian vs isotonic-calibrated B-type rules on 2025 test data
for cities that have 2024 calibration history (NYC, CHI, MIA, AUS).

Model parameters:
  Bias train:   2022-2023
  Isotonic cal: 2024 B-type outcomes
  Test:         2025 B-type outcomes

Liquid filter: two-sided market, spread < 0.20, mid in [0.05, 0.80]
(B-type brackets typically price 0.05–0.35; using a looser floor than
 T-type [0.15] to include near-center brackets.)

Run: .venv/bin/python scripts/run_2025_b_validation.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd
import numpy as np

from kalshi_weather.align import build_aligned_dataset
from kalshi_weather.backtest import run_backtest
from kalshi_weather.config import get_station
from kalshi_weather.ingest.forecasts import fetch_forecasts
from kalshi_weather.ingest.kalshi import get_decision_prices
from kalshi_weather.ingest.labels import fetch_labels
from kalshi_weather.rules import make_b_gaussian_rule, make_b_isotonic_rule

BIAS_TRAIN_YEARS = [2022, 2023]
ISO_CAL_YEARS    = [2024]
TEST_YEARS       = [2025]
FCST_MODELS      = ["gfs_seamless"]
ATM_MIN, ATM_MAX = 0.05, 0.80   # B-type brackets price lower than T-type
MAX_SPREAD       = 0.20
MIN_BID          = 0.01
MIN_YES_CAL      = 100           # minimum YES outcomes needed for isotonic

CACHE_ROOT = Path(__file__).parents[1] / "data" / "raw" / "kalshi"

CITIES = {
    "NYC": "markets.parquet",
    "CHI": "markets_KXHIGHCHI.parquet",
    "MIA": "markets_KXHIGHMIA.parquet",
    "AUS": "markets_KXHIGHAUS.parquet",
}


def year_filter(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    mask = df["settlement_date"].apply(lambda d: hasattr(d, "year") and d.year in set(years))
    return df[mask].reset_index(drop=True)


def liquid_atm_filter(df: pd.DataFrame) -> pd.DataFrame:
    bid    = pd.to_numeric(df.get("yes_bid_dollars",    pd.Series(dtype=float)), errors="coerce")
    ask    = pd.to_numeric(df.get("yes_ask_dollars",    pd.Series(dtype=float)), errors="coerce")
    mid    = pd.to_numeric(df.get("last_price_dollars", pd.Series(dtype=float)), errors="coerce")
    spread = ask - bid
    mask   = (bid > MIN_BID) & (spread < MAX_SPREAD) & mid.between(ATM_MIN, ATM_MAX)
    return df[mask].reset_index(drop=True)


all_results: dict = {}

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

    # Build full aligned dataset (all types) — no threshold_contracts() filter
    aligned_all = build_aligned_dataset(
        markets, labels, forecasts, station=nws_station, model=FCST_MODELS[0]
    )

    # Filter to B-type only
    b_all = aligned_all[aligned_all["contract_type"] == "B"].copy()
    print(f"  Total B-type aligned: {len(b_all)}")

    cal_b  = year_filter(b_all, ISO_CAL_YEARS)
    test_b = year_filter(b_all, TEST_YEARS)

    # Fetch pre-settlement decision prices for test set.
    # Parquet last_price_dollars for settled contracts = final settlement price ($0.01/$0.99),
    # not tradeable price. Candle-based decision prices are required for a meaningful backtest.
    print(f"  Fetching candles for {len(test_b)} 2025 B-type contracts (≈{len(test_b)*0.3/60:.0f} min, cached)...")
    test_markets = markets[markets["ticker"].isin(test_b["ticker"])].reset_index(drop=True)
    prices = get_decision_prices(test_markets, station=nws_station, fetch_if_missing=True)
    if prices.empty or "ticker" not in prices.columns:
        prices = pd.DataFrame(columns=["ticker", "decision_mid", "decision_bid", "decision_ask"])
    test_b = test_b.merge(prices, on="ticker", how="left")
    if "decision_mid" in test_b.columns:
        test_b["last_price_dollars"] = test_b["decision_mid"]
        test_b["yes_bid_dollars"]    = test_b["decision_bid"]
        test_b["yes_ask_dollars"]    = test_b["decision_ask"]
    n_with_prices = test_b["last_price_dollars"].notna().sum()
    print(f"  Decision prices retrieved: {n_with_prices}/{len(test_b)}")

    test_liquid = liquid_atm_filter(test_b)
    n_yes_cal   = int((cal_b["result"] == "yes").sum())

    print(f"  Cal 2024: {len(cal_b)} B-type contracts, {n_yes_cal} YES")
    print(f"  Test 2025: {len(test_b)} total, {len(test_liquid)} pass liquid ATM filter")
    print(f"  Test 2025 YES: {(test_liquid['result'] == 'yes').sum()} / {len(test_liquid)}")

    if len(test_liquid) < 20:
        print(f"  SKIP: only {len(test_liquid)} liquid ATM B-type contracts in 2025")
        continue

    # Build rules
    gauss_b_rule = make_b_gaussian_rule(
        labels, forecasts,
        station=nws_station, model=FCST_MODELS[0],
        train_years=BIAS_TRAIN_YEARS,
        rule_name=f"b_gaussian_{city_key}",
    )

    iso_b_rule = None
    if n_yes_cal >= MIN_YES_CAL:
        try:
            iso_b_rule = make_b_isotonic_rule(
                labels, forecasts, cal_b,
                station=nws_station, model=FCST_MODELS[0],
                train_years=BIAS_TRAIN_YEARS,
                rule_name=f"b_isotonic_{city_key}",
            )
        except ValueError as e:
            print(f"  WARNING: isotonic skipped: {e}")

    # Run backtests
    gauss_result = run_backtest(
        test_liquid, gauss_b_rule,
        rule_name=f"b_gaussian [{city_key} 2025]",
        atm_min=ATM_MIN, atm_max=ATM_MAX,
    )

    iso_result = None
    if iso_b_rule:
        iso_result = run_backtest(
            test_liquid, iso_b_rule,
            rule_name=f"b_isotonic [{city_key} 2025]",
            atm_min=ATM_MIN, atm_max=ATM_MAX,
        )

    all_results[city_key] = {"gaussian": gauss_result, "isotonic": iso_result}


# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'═'*80}")
print("  2025 OOS SUMMARY — B-type Brackets — Liquid ATM filter [0.05,0.80]")
print(f"{'═'*80}")
print(f"  {'City':<6} {'Rule':<12} {'Trades':>7} {'Win%':>6} {'AvgRet':>8} "
      f"{'CI low':>8} {'CI high':>8} {'Brier':>8}")
print(f"  {'-'*75}")

for city_key, res in all_results.items():
    for rule_label, result in [("b_gaussian", res["gaussian"]), ("b_isotonic", res["isotonic"])]:
        if result is None:
            continue
        p = result.metrics_pess
        print(
            f"  {city_key:<6} {rule_label:<12} "
            f"{p.get('n_trades',0):>7} "
            f"{p.get('win_rate',float('nan')):>6.1%} "
            f"{p.get('avg_return',float('nan')):>8.4f} "
            f"{p.get('edge_ci_low',float('nan')):>8.3f} "
            f"{p.get('edge_ci_high',float('nan')):>8.3f} "
            f"{p.get('brier_score',float('nan')):>8.5f}"
        )
    print(f"  {'-'*75}")

print(f"\n  Note: Pessimistic fill model. Edge CI is 95% block-bootstrap.")
print(f"  B-type isotonic cal=2024, test=2025 (same cal/test split as T-type).")
print(f"  Preferred = rule whose CI low is higher.")
