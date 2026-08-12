"""
Build and cache the production model for 2026 trading — all 20 active cities.

Training data:
  Bias correction: 2022-2023 GFS residuals  (Gaussian per-month mu/sigma)
  Isotonic cal:    2024 decision prices      (for cities with 2024 candles)

City tiers (T-type rules):
  Tier 1 — Isotonic available (Kalshi candles from 2023 or earlier):
    NYC (isotonic preferred), CHI, MIA, AUS
  Tier 1b — Isotonic on 2025 cal (started late 2024 / early 2025):
    DEN, PHL (started Nov 2024), LAX (started Jan 2025)
  Tier 2 — Gaussian only (Kalshi started 2026 — no candle history):
    ATL, BOS, DAL, DCA, HOU, LAS, MSP, MSY, OKC, PHX, SAT, SEA, SFO

B-type rules (bracket contracts):
  B-isotonic built wherever cal year has ≥ 100 YES outcomes.
  NYC/CHI/MIA/AUS: 2024 cal → isotonic
  DEN/PHL/LAX:     2025 cal → isotonic
  Tier 2 cities:   gaussian fallback (no cal data)

Outputs:
  data/models/production_rules.pkl  — dict of {city_key: entry}

Run: .venv/bin/python scripts/build_production_model.py
"""
from __future__ import annotations

import cloudpickle as pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from kalshi_weather.align import build_aligned_dataset, threshold_contracts
from kalshi_weather.config import get_station
from kalshi_weather.ingest.forecasts import fetch_forecasts
from kalshi_weather.ingest.kalshi import fetch_all_markets, get_decision_prices
from kalshi_weather.ingest.labels import fetch_labels
from kalshi_weather.rules import make_forecast_rule, make_isotonic_calibrated_rule, make_b_isotonic_rule

CACHE_ROOT    = Path(__file__).parents[1] / "data" / "raw" / "kalshi"
MODEL_DIR     = Path(__file__).parents[1] / "data" / "models"
FCST_MODELS   = ["gfs_seamless"]

# Bias training years: 2022+2023 GFS data exists for ALL 20 cities.
# 2021 data only exists for NYC/CHI/MIA/AUS in the calibration cache; using
# 2022-2023 gives a consistent, reproducible baseline across all cities.
BIAS_TRAIN_YEARS = [2022, 2023]

# Default isotonic cal year. Cities with a different candle window can override
# this via iso_cal_years in their per-city config below.
ISO_CAL_YEARS_DEFAULT = [2024]

# ─── Per-city configuration ────────────────────────────────────────────────
#
# build_isotonic:    True  → build T-type isotonic rule
# preferred_rule:    T-type rule the live system uses ("isotonic" | "gaussian")
# b_preferred_rule:  B-type rule override (default "isotonic" if built, else "gaussian")
# iso_cal_years:     override ISO_CAL_YEARS_DEFAULT when candles cover a different year
#
# B-type OOS results (2025, pessimistic fill, CI = 95% block-bootstrap):
#   NYC gaussian CI [0.004, 0.074]  isotonic CI [0.049, 0.118]  → isotonic
#   CHI gaussian CI [-0.056, 0.010] isotonic CI [-0.047, 0.040] → neither profitable; isotonic slightly less bad
#   MIA gaussian CI [0.015, 0.088]  isotonic CI [-0.005, 0.092] → gaussian (isotonic crosses zero)
#   AUS gaussian CI [-0.021, 0.047] isotonic CI [-0.012, 0.062] → isotonic (better Brier, CI shifts right)
#
CITY_CONFIGS = {
    # ── Tier 1a: isotonic on 2024 (Kalshi launched 2021-2023) ─────────────
    # NYC: T-type isotonic + B-type isotonic both outperformed gaussian on 2025 OOS.
    "NYC": {"preferred_rule": "isotonic", "build_isotonic": True},
    # CHI: T-type isotonic preferred. B-type: neither model profitable OOS; isotonic slightly less bad.
    "CHI": {"preferred_rule": "isotonic", "build_isotonic": True},
    # MIA: T-type isotonic preferred. B-type: gaussian CI fully positive [0.015,0.088]; isotonic crosses zero.
    "MIA": {"preferred_rule": "isotonic", "build_isotonic": True, "b_preferred_rule": "gaussian"},
    # AUS: T-type isotonic preferred. B-type: isotonic has best Brier; CI crosses zero but shifts right vs gaussian.
    "AUS": {"preferred_rule": "isotonic", "build_isotonic": True},

    # ── Tier 1b: isotonic on 2025 (Kalshi launched late 2024 / early 2025) ─
    # DEN/PHL launched Nov 2024 → ~730 T-contracts in 2025.
    # LAX launched Jan 2025   → ~730 T-contracts in 2025.
    # 2025 is clean OOS for these cities (no prior model existed).
    "DEN": {"preferred_rule": "gaussian", "build_isotonic": True,  "iso_cal_years": [2025]},
    "PHL": {"preferred_rule": "gaussian", "build_isotonic": True,  "iso_cal_years": [2025]},
    "LAX": {"preferred_rule": "gaussian", "build_isotonic": True,  "iso_cal_years": [2025]},

    # ── Tier 2: gaussian only (Kalshi started 2026 — no candle history) ───
    "ATL": {"preferred_rule": "gaussian", "build_isotonic": False},
    "BOS": {"preferred_rule": "gaussian", "build_isotonic": False},
    "DAL": {"preferred_rule": "gaussian", "build_isotonic": False},
    "DCA": {"preferred_rule": "gaussian", "build_isotonic": False},
    "HOU": {"preferred_rule": "gaussian", "build_isotonic": False},
    "LAS": {"preferred_rule": "gaussian", "build_isotonic": False},
    "MSP": {"preferred_rule": "gaussian", "build_isotonic": False},
    "MSY": {"preferred_rule": "gaussian", "build_isotonic": False},
    "OKC": {"preferred_rule": "gaussian", "build_isotonic": False},
    "PHX": {"preferred_rule": "gaussian", "build_isotonic": False},
    "SAT": {"preferred_rule": "gaussian", "build_isotonic": False},
    "SEA": {"preferred_rule": "gaussian", "build_isotonic": False},
    "SFO": {"preferred_rule": "gaussian", "build_isotonic": False},
}

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=== Building Production Models for 2026 Trading ===")
print(f"Bias train:   {BIAS_TRAIN_YEARS}  (all 20 cities)")
print(f"Isotonic cal: {ISO_CAL_YEARS_DEFAULT} default / [2025] for DEN, PHL, LAX")
print(f"Cities:       {len(CITY_CONFIGS)}\n")


def year_filter(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    mask = df["settlement_date"].apply(lambda d: hasattr(d, "year") and d.year in set(years))
    return df[mask].reset_index(drop=True)


production_rules: dict = {}

for city_key, config in CITY_CONFIGS.items():
    print(f"── {city_key} {'─' * (46 - len(city_key))}")
    station     = get_station(city_key)
    nws_station = station["nws_station"]
    lat, lon    = station["lat"], station["lon"]
    series      = station["kalshi_series"]

    build_iso    = config["build_isotonic"]
    iso_cal_yrs  = config.get("iso_cal_years", ISO_CAL_YEARS_DEFAULT)
    all_years    = sorted(set(BIAS_TRAIN_YEARS + (iso_cal_yrs if build_iso else [])))

    # ── Load data ──────────────────────────────────────────────────────────
    cache_name = "markets.parquet" if series == "KXHIGHNY" else f"markets_{series}.parquet"
    markets_path = CACHE_ROOT / cache_name
    if not markets_path.exists():
        print(f"  SKIP: {cache_name} not found")
        continue

    markets   = pd.read_parquet(markets_path)
    labels    = fetch_labels(nws_station, all_years)
    forecasts = fetch_forecasts(nws_station, lat, lon, all_years, models=FCST_MODELS)
    print(f"  Labels: {len(labels)} days | Forecasts: {len(forecasts)} rows")

    # ── Gaussian base rule ─────────────────────────────────────────────────
    gauss_rule = make_forecast_rule(
        labels, forecasts,
        train_years=BIAS_TRAIN_YEARS,
        station=nws_station,
        model=FCST_MODELS[0],
        distribution="gaussian",
        rule_name=f"gfs_gaussian_2022_2023_{city_key}_prod",
    )

    # ── Isotonic calibrated rule (Tier 1 only) ─────────────────────────────
    iso_rule   = None
    b_iso_rule = None
    n_priced   = 0
    if build_iso:
        aligned_all  = build_aligned_dataset(
            markets, labels, forecasts, station=nws_station, model=FCST_MODELS[0]
        )
        aligned_full = threshold_contracts(aligned_all)   # T-type only
        cal_aligned  = year_filter(aligned_full, iso_cal_yrs)

        cal_markets = markets[markets["ticker"].isin(cal_aligned["ticker"])].reset_index(drop=True)
        cal_prices  = get_decision_prices(cal_markets, station=nws_station, fetch_if_missing=False)
        if cal_prices.empty or "ticker" not in cal_prices.columns:
            cal_prices = pd.DataFrame(columns=["ticker", "decision_mid", "decision_bid", "decision_ask"])
        cal_aligned = cal_aligned.merge(cal_prices, on="ticker", how="left")
        cal_aligned["last_price_dollars"] = cal_aligned["decision_mid"]
        cal_aligned["yes_bid_dollars"]    = cal_aligned["decision_bid"]
        cal_aligned["yes_ask_dollars"]    = cal_aligned["decision_ask"]

        n_priced = cal_aligned["last_price_dollars"].notna().sum()
        print(f"  Cal contracts ({iso_cal_yrs}): {len(cal_aligned)}  with prices: {n_priced}")

        if n_priced >= 50:
            cal_label = "_".join(str(y) for y in iso_cal_yrs)
            iso_rule = make_isotonic_calibrated_rule(
                gauss_rule,
                cal_aligned,
                rule_name=f"gfs_gaussian_isotonic_{cal_label}_{city_key}_prod",
            )
        else:
            print(f"  WARNING: only {n_priced} priced cal contracts — skipping isotonic")
            build_iso = False

        # ── B-type isotonic calibrated rule ────────────────────────────────
        # Uses same calibration years as T-type isotonic.
        # B-type contracts use a narrow 2°F bracket so the gaussian→actual
        # mapping is structurally different from T-type; they need their own
        # isotonic fit.
        b_aligned_all = aligned_all[aligned_all["contract_type"] == "B"].copy()
        cal_b = year_filter(b_aligned_all, iso_cal_yrs)
        n_yes_b = int((cal_b["result"] == "yes").sum())
        print(f"  B-type cal ({iso_cal_yrs}): {len(cal_b)} contracts, {n_yes_b} YES")
        if n_yes_b >= 100:
            cal_label = "_".join(str(y) for y in iso_cal_yrs)
            try:
                b_iso_rule = make_b_isotonic_rule(
                    labels, forecasts, cal_b,
                    station=nws_station, model=FCST_MODELS[0],
                    train_years=BIAS_TRAIN_YEARS,
                    rule_name=f"b_isotonic_{cal_label}_{city_key}_prod",
                )
            except ValueError as e:
                print(f"  WARNING: B-type isotonic skipped: {e}")
        else:
            print(f"  B-type isotonic skipped: only {n_yes_b} YES outcomes (need ≥ 100)")

        # Apply b_preferred_rule override (default: isotonic when available)
        b_preferred = config.get("b_preferred_rule", "isotonic")
        if b_preferred == "gaussian" or b_iso_rule is None:
            b_active_rule = None   # None → live runner uses gaussian fallback via bias.py
            b_active_label = "gaussian"
        else:
            b_active_rule = b_iso_rule
            b_active_label = "isotonic"
        if b_preferred == "gaussian" and b_iso_rule is not None:
            print(f"  B-type: using gaussian (isotonic built but OOS prefers gaussian)")
    else:
        b_active_rule  = None
        b_active_label = "gaussian"

    preferred = config["preferred_rule"]
    if preferred == "isotonic" and iso_rule is None:
        preferred = "gaussian"
        print(f"  NOTE: isotonic unavailable, falling back to gaussian")
    active_rule = iso_rule if preferred == "isotonic" else gauss_rule
    print(f"  Preferred rule: {preferred}  (tier={'1' if iso_rule else '2'})"
          + f"  B-type: {b_active_label}")

    production_rules[city_key] = {
        "gaussian_rule":     gauss_rule,
        "isotonic_rule":     iso_rule,
        "active_rule":       active_rule,
        "preferred_rule":    preferred,
        "b_active_rule":     b_active_rule,   # None → live runner uses gaussian via bias.py
        "nws_station":       nws_station,
        "series":            series,
        "bias_train_years":  BIAS_TRAIN_YEARS,
        "iso_cal_years":     iso_cal_yrs if iso_rule else [],
        "tier":              1 if iso_rule else 2,
    }
    print()

# ── Persist ────────────────────────────────────────────────────────────────
model_path = MODEL_DIR / "production_rules.pkl"
with open(model_path, "wb") as fh:
    pickle.dump(production_rules, fh)

print(f"Production rules saved → {model_path}")
print()
print("Summary:")
tier1 = [(k, v) for k, v in production_rules.items() if v["tier"] == 1]
tier2 = [(k, v) for k, v in production_rules.items() if v["tier"] == 2]
print(f"  Tier 1 (isotonic available, {len(tier1)} cities): "
      + ", ".join(k for k, _ in tier1))
print(f"  Tier 2 (gaussian only,      {len(tier2)} cities): "
      + ", ".join(k for k, _ in tier2))
print()
for city_key, entry in production_rules.items():
    iso_str = f", iso cal {entry['iso_cal_years']}" if entry["iso_cal_years"] else ""
    print(f"  {city_key:4s}: {entry['preferred_rule']:8s}  "
          f"(bias train {entry['bias_train_years']}{iso_str})")
print()
print("Next out-of-sample test: 2026 forward. Do not re-evaluate on 2025.")
