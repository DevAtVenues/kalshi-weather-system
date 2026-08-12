"""
2025 out-of-sample validation for ALL cities — T-type, using the EXACT deployed
production rule (not a reconstruction), so the resulting CI describes the model
that actually trades.

Leakage safety (HARD RULE 4): a city is validated on 2025 only if its DEPLOYED
rule never saw 2025. Verified from production_rules.pkl metadata:
  • isotonic-preferred cities (NYC/CHI/MIA/AUS): iso_cal = 2024  → 2025 clean
  • gaussian-preferred cities (everyone else):   bias_train = 2022-23, no 2025 → clean
A city whose preferred rule is isotonic AND iso_cal includes 2025 is SKIPPED.

Output: a summary table + a ready-to-paste _OOS_CI block (T-type) for store.py.
Edge CI = 95% block-bootstrap, pessimistic fill.

Run: .venv/bin/python scripts/run_2025_validation_full.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import cloudpickle
import pandas as pd

from kalshi_weather.align import build_aligned_dataset, threshold_contracts
from kalshi_weather.backtest import run_backtest
from kalshi_weather.config import get_station
from kalshi_weather.ingest.forecasts import fetch_forecasts
from kalshi_weather.ingest.kalshi import get_decision_prices
from kalshi_weather.ingest.labels import fetch_labels

ALL_YEARS   = [2022, 2023, 2024, 2025]
TEST_YEARS  = [2025]
FCST_MODELS = ["gfs_seamless"]
ATM_MIN, ATM_MAX = 0.15, 0.85
MAX_SPREAD  = 0.15
MIN_BID     = 0.01
MIN_TRADES  = 10

ROOT       = Path(__file__).parents[1]
CACHE_ROOT = ROOT / "data" / "raw" / "kalshi"
PROD       = cloudpickle.load(open(ROOT / "data" / "models" / "production_rules.pkl", "rb"))


def markets_file(city: str, series: str) -> Path:
    return CACHE_ROOT / ("markets.parquet" if city == "NYC" else f"markets_{series}.parquet")


def year_filter(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    mask = df["settlement_date"].apply(lambda d: hasattr(d, "year") and d.year in set(years))
    return df[mask].reset_index(drop=True)


def liquid_atm_filter(df: pd.DataFrame) -> pd.DataFrame:
    bid    = pd.to_numeric(df.get("yes_bid_dollars", pd.Series(dtype=float)), errors="coerce")
    ask    = pd.to_numeric(df.get("yes_ask_dollars", pd.Series(dtype=float)), errors="coerce")
    mid    = pd.to_numeric(df.get("last_price_dollars", pd.Series(dtype=float)), errors="coerce")
    spread = ask - bid
    mask   = (bid > MIN_BID) & (spread < MAX_SPREAD) & mid.between(ATM_MIN, ATM_MAX)
    return df[mask].reset_index(drop=True)


results: dict[str, dict] = {}

for city, entry in PROD.items():
    series   = entry["series"]
    pref     = entry.get("preferred_rule")
    iso_cal  = entry.get("iso_cal_years") or []
    rule     = entry["active_rule"]      # the EXACT deployed T rule

    # Leakage gate: skip if the deployed rule is isotonic and was calibrated on 2025.
    if pref == "isotonic" and 2025 in iso_cal:
        print(f"\n{city}: SKIP — deployed isotonic rule calibrated on 2025 (would leak)")
        continue

    mfile = markets_file(city, series)
    if not mfile.exists():
        print(f"\n{city}: SKIP — no cached market data ({mfile.name})")
        continue

    print(f"\n{'='*60}\n  {city}  (deployed rule: {pref})\n{'='*60}")
    st  = get_station(city)
    nws, lat, lon = st["nws_station"], st["lat"], st["lon"]

    markets   = pd.read_parquet(mfile)
    labels    = fetch_labels(nws, ALL_YEARS)
    forecasts = fetch_forecasts(nws, lat, lon, ALL_YEARS, models=FCST_MODELS)

    aligned = build_aligned_dataset(markets, labels, forecasts, station=nws, model=FCST_MODELS[0])
    aligned = threshold_contracts(aligned)
    test    = year_filter(aligned, TEST_YEARS)
    if test.empty:
        print("  SKIP — no 2025 contracts")
        continue

    sub = markets[markets["ticker"].isin(test["ticker"])].reset_index(drop=True)
    prices = get_decision_prices(sub, fetch_if_missing=True)
    if prices.empty or "ticker" not in prices.columns:
        print("  SKIP — no decision prices")
        continue
    test = test.merge(prices, on="ticker", how="left")
    test["last_price_dollars"] = test["decision_mid"]
    test["yes_bid_dollars"]    = test["decision_bid"]
    test["yes_ask_dollars"]    = test["decision_ask"]

    test_liquid = liquid_atm_filter(test)
    print(f"  2025 liquid ATM T-contracts: {len(test_liquid)}")
    if len(test_liquid) < MIN_TRADES:
        print(f"  SKIP — only {len(test_liquid)} (< {MIN_TRADES})")
        continue

    res = run_backtest(test_liquid, rule, rule_name=f"{pref} [{city} 2025]",
                       atm_min=ATM_MIN, atm_max=ATM_MAX)
    results[city] = res.metrics_pess


# ── Summary + paste-ready _OOS_CI block ────────────────────────────────────────
print(f"\n\n{'═'*72}")
print("  2025 OOS — T-type — deployed rule — pessimistic fill — 95% block-bootstrap")
print(f"{'═'*72}")
print(f"  {'City':<6}{'Trades':>7}{'Win%':>7}{'AvgRet':>9}{'CI low':>9}{'CI high':>9}")
for city, p in results.items():
    print(f"  {city:<6}{p.get('n_trades',0):>7}{p.get('win_rate',float('nan')):>7.1%}"
          f"{p.get('avg_return',float('nan')):>9.4f}{p.get('edge_ci_low',float('nan')):>9.3f}"
          f"{p.get('edge_ci_high',float('nan')):>9.3f}")

print("\n  --- paste-ready T-type CI (positive lower bound = validated +EV) ---")
for city, p in results.items():
    lo, hi = p.get("edge_ci_low"), p.get("edge_ci_high")
    if lo is None or hi is None:
        continue
    flag = "  # validated +EV" if lo > 0 else "  # CI crosses 0"
    print(f'    "{city}": {{"T": ({lo:.3f}, {hi:.3f}), "B": None}},{flag}')
