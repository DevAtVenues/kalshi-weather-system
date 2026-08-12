"""
Live rule runner — applies production rules to currently open Kalshi markets.

Flow per city:
  1. Fetch open T-type markets from Kalshi live API
  2. Fetch GFS tmax forecast for each settlement date (Open-Meteo)
  3. Apply production rule → P(YES) per contract
  4. Compute edge = |P(YES) - market_mid|
  5. Return signals where edge > threshold

No look-ahead: forecasts are fetched live from the current model run, matching
the decision-time information available to a trader right now.
"""
from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from kalshi_weather.calibration.bias import apply_correction, contract_probability, load_bias_table
from kalshi_weather.calibration.center_bias import center_shift as _ens_center_shift
from kalshi_weather.calibration.ensemble_dist import (
    fetch_ensemble_maxes, corrected_members, recent_grid_bias,
    fetch_nws_high, fetch_hrrr_high, mixture_prob, calibrate_dispersion, summary as ens_summary,
)
from kalshi_weather.calibration.recalibrate import (
    get_calibration, apply_prob_calibration, lead_bucket)
from kalshi_weather.calibration.shrinkage import shrink_prob
from kalshi_weather.provenance import provenance
from kalshi_weather.ingest.metar import fetch_running_high_f
from kalshi_weather.live.intraday import intraday_prob
from kalshi_weather.live.risk import RiskManager
from kalshi_weather.tz import lst_date_for_utc, lst_offset as _lst_offset_td, utc_now

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
_KALSHI_BASE     = "https://api.elections.kalshi.com/trade-api/v2"
_OM_FCST_BASE    = "https://api.open-meteo.com/v1/forecast"          # live/future
_OM_HIST_BASE    = "https://historical-forecast-api.open-meteo.com/v1/forecast"  # recent past

EDGE_THRESHOLD   = 0.05    # must match backtest pre-registered value
ATM_MIN, ATM_MAX = 0.15, 0.85
MAX_SPREAD       = 0.15
MIN_BID          = 0.01    # bid > 0 = two-sided market

# Probability recalibration (backlog P1.1) — corrects the rule/gaussian path's
# systematic YES over-prediction. Fit offline (scripts/build_calibration_map.py),
# applied per prob_source. The ensemble path is left IDENTITY on purpose: calibrating
# it was measured to HURT out-of-sample (Brier 0.206→0.223). Read via get_calibration()
# at call time — mtime-cached, so a refit applies without an engine restart.

# ── Ensemble pricing (Move 1: ensemble engine drives live P(YES)) ─────────────
# Validated cities/types (OOS-positive) get priced from the multi-model ensemble
# distribution (calibration/ensemble_dist.py) instead of the single-GFS-run rule.
# Mirrors scripts/run_ensemble_picks.py — the by-hand cross-check, automated.
# Cities NOT listed here, or contract types not validated, fall back to the rule.
W_NWS                              = 0.4              # weight on the NWS-Normal in the mixture
SIGMA_FLOOR, SIGMA_CAP, SIGMA_DEF  = 1.2, 4.0, 2.0   # NWS-Normal spread bounds (°F)
ENSEMBLE_CITIES: dict[str, dict] = {
    # tz: IANA zone for the ensemble daily-max; types: validated contract types;
    # iem_net/iem_stn: ASOS station for the ensemble-basis recent grid->sensor bias.
    #
    # OOS-validated (fire-eligible) cities — 2025 holdout CI wholly positive:
    "NYC": {"tz": "America/New_York", "types": {"T", "B"}, "iem_net": "NY_ASOS", "iem_stn": "NYC"},
    "MIA": {"tz": "America/New_York", "types": {"T", "B"}, "iem_net": "FL_ASOS", "iem_stn": "MIA"},
    "CHI": {"tz": "America/Chicago",  "types": {"T"},      "iem_net": "IL_ASOS", "iem_stn": "MDW"},
    "AUS": {"tz": "America/Chicago",  "types": {"T"},      "iem_net": "TX_ASOS", "iem_stn": "AUS"},
    "PHL": {"tz": "America/New_York", "types": {"T"},      "iem_net": "PA_ASOS", "iem_stn": "PHL"},
    # Calibrated but no 2025 price-history holdout (Tier 1b — can fire if edge ∈ [0.18,0.25]):
    "LAX": {"tz": "America/Los_Angeles", "types": {"T"}, "iem_net": "CA_ASOS", "iem_stn": "LAX"},
    "DEN": {"tz": "America/Denver",      "types": {"T"}, "iem_net": "CO_ASOS", "iem_stn": "DEN"},
    # Tier 2 — no Kalshi history cached (min_edge=0.25, ceiling=0.25 = effectively never fires).
    # Listed for: ensemble pricing, forward-logging (builds validation corpus), gate-log entries.
    "ATL": {"tz": "America/New_York",    "types": {"T"}, "iem_net": "GA_ASOS", "iem_stn": "ATL"},
    "BOS": {"tz": "America/New_York",    "types": {"T"}, "iem_net": "MA_ASOS", "iem_stn": "BOS"},
    "DAL": {"tz": "America/Chicago",     "types": {"T"}, "iem_net": "TX_ASOS", "iem_stn": "DFW"},
    "DCA": {"tz": "America/New_York",    "types": {"T"}, "iem_net": "VA_ASOS", "iem_stn": "DCA"},
    "HOU": {"tz": "America/Chicago",     "types": {"T"}, "iem_net": "TX_ASOS", "iem_stn": "HOU"},
    "LAS": {"tz": "America/Los_Angeles", "types": {"T"}, "iem_net": "NV_ASOS", "iem_stn": "LAS"},
    "MSP": {"tz": "America/Chicago",     "types": {"T"}, "iem_net": "MN_ASOS", "iem_stn": "MSP"},
    "MSY": {"tz": "America/Chicago",     "types": {"T"}, "iem_net": "LA_ASOS", "iem_stn": "MSY"},
    "OKC": {"tz": "America/Chicago",     "types": {"T"}, "iem_net": "OK_ASOS", "iem_stn": "OKC"},
    "PHX": {"tz": "America/Phoenix",     "types": {"T"}, "iem_net": "AZ_ASOS", "iem_stn": "PHX"},
    "SAT": {"tz": "America/Chicago",     "types": {"T"}, "iem_net": "TX_ASOS", "iem_stn": "SAT"},
    "SEA": {"tz": "America/Los_Angeles", "types": {"T"}, "iem_net": "WA_ASOS", "iem_stn": "SEA"},
    "SFO": {"tz": "America/Los_Angeles", "types": {"T"}, "iem_net": "CA_ASOS", "iem_stn": "SFO"},
}

CITY_CONFIGS: dict[str, dict] = {
    "NYC": {
        "series":      "KXHIGHNY",
        "nws_station": "KNYC",
        "lat":         40.7789,
        "lon":         -73.9692,
        "lst_offset":  -5,
    },
    "CHI": {
        "series":      "KXHIGHCHI",
        "nws_station": "KMDW",
        "lat":         41.7860,
        "lon":         -87.7522,
        "lst_offset":  -6,
    },
    "MIA": {
        "series":      "KXHIGHMIA",
        "nws_station": "KMIA",
        "lat":         25.7959,
        "lon":         -80.2870,
        "lst_offset":  -5,
    },
    "PHX": {
        "series":      "KXHIGHTPHX",
        "nws_station": "KPHX",
        "lat":         33.4373,
        "lon":         -112.0078,
        "lst_offset":  -7,
    },
    "BOS": {
        "series":      "KXHIGHTBOS",
        "nws_station": "KBOS",
        "lat":         42.3643,
        "lon":         -71.0052,
        "lst_offset":  -5,
    },
    "SFO": {
        "series":      "KXHIGHTSFO",
        "nws_station": "KSFO",
        "lat":         37.6188,
        "lon":         -122.3750,
        "lst_offset":  -8,
    },
    "LAX": {
        "series":      "KXHIGHLAX",
        "nws_station": "KLAX",
        "lat":         33.9425,
        "lon":         -118.4081,
        "lst_offset":  -8,
    },
    "DEN": {
        "series":      "KXHIGHDEN",
        "nws_station": "KDEN",
        "lat":         39.8561,
        "lon":         -104.6737,
        "lst_offset":  -7,
    },
    "ATL": {
        "series":      "KXHIGHTATL",
        "nws_station": "KATL",
        "lat":         33.6367,
        "lon":         -84.4281,
        "lst_offset":  -5,
    },
    "HOU": {
        "series":      "KXHIGHTHOU",
        "nws_station": "KHOU",
        "lat":         29.6454,
        "lon":         -95.2789,
        "lst_offset":  -6,
    },
    "DCA": {
        "series":      "KXHIGHTDC",
        "nws_station": "KDCA",
        "lat":         38.8521,
        "lon":         -77.0377,
        "lst_offset":  -5,
    },
    "MSY": {
        "series":      "KXHIGHTNOLA",
        "nws_station": "KMSY",
        "lat":         29.9934,
        "lon":         -90.2580,
        "lst_offset":  -6,
    },
    "PHL": {
        "series":      "KXHIGHPHIL",
        "nws_station": "KPHL",
        "lat":         39.8719,
        "lon":         -75.2411,
        "lst_offset":  -5,
    },
    "SEA": {
        "series":      "KXHIGHTSEA",
        "nws_station": "KSEA",
        "lat":         47.4502,
        "lon":         -122.3088,
        "lst_offset":  -8,
    },
    "LAS": {
        "series":      "KXHIGHTLV",
        "nws_station": "KLAS",
        "lat":         36.0840,
        "lon":         -115.1537,
        "lst_offset":  -8,
    },
    "MSP": {
        "series":      "KXHIGHTMIN",
        "nws_station": "KMSP",
        "lat":         44.8848,
        "lon":         -93.2223,
        "lst_offset":  -6,
    },
    "SAT": {
        "series":      "KXHIGHTSATX",
        "nws_station": "KSAT",
        "lat":         29.5337,
        "lon":         -98.4698,
        "lst_offset":  -6,
    },
    "DAL": {
        "series":      "KXHIGHTDAL",
        "nws_station": "KDFW",
        "lat":         32.8998,
        "lon":         -97.0403,
        "lst_offset":  -6,
    },
    "AUS": {
        "series":      "KXHIGHAUS",
        "nws_station": "KAUS",
        "lat":         30.1945,
        "lon":         -97.6699,
        "lst_offset":  -6,
    },
    "OKC": {
        "series":      "KXHIGHTOKC",
        "nws_station": "KOKC",
        "lat":         35.3931,
        "lon":         -97.6007,
        "lst_offset":  -6,
    },
}

_TICKER_RE = re.compile(r"^[A-Z]+-(\d{2})([A-Z]{3})(\d{2})-([TB])([\d.]+)$")
_MONTH_MAP  = {m: i for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"], 1
)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _headers() -> dict:
    key = os.getenv("KALSHI_API_KEY")
    if not key:
        raise EnvironmentError("KALSHI_API_KEY not set. Check your .env file.")
    return {"Authorization": f"Bearer {key}"}


def _kalshi_get(path: str, params: dict | None = None, max_retries: int = 3) -> dict:
    delay = 2.0
    for attempt in range(max_retries):
        resp = requests.get(f"{_KALSHI_BASE}{path}", headers=_headers(),
                            params=params, timeout=15)
        if resp.status_code == 429 and attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def _parse_ticker(ticker: str) -> tuple[date | None, str, float | None]:
    """
    Parse a Kalshi ticker into (settlement_date, contract_type, threshold_f).
    Returns (None, '?', None) on parse failure.
    """
    m = _TICKER_RE.match(ticker)
    if not m:
        return None, "?", None
    yy, mon, dd, ctype, val_str = m.groups()
    month_num = _MONTH_MAP.get(mon)
    if month_num is None:
        return None, "?", None
    try:
        sdate = date(2000 + int(yy), month_num, int(dd))
    except ValueError:
        return None, "?", None
    try:
        threshold = float(val_str)
    except ValueError:
        threshold = None
    return sdate, ctype, threshold


# ── Market fetch ──────────────────────────────────────────────────────────────

def fetch_open_markets(series: str, station: str) -> pd.DataFrame:
    """
    Fetch currently open T-type and B-type contracts for a series.
    Returns a DataFrame ready for scoring.
    """
    markets: list[dict] = []
    cursor: str | None = None

    while True:
        params: dict = {"series_ticker": series, "status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data    = _kalshi_get("/markets", params)
        batch   = data.get("markets", [])
        markets.extend(batch)
        cursor  = data.get("cursor")
        if not cursor or len(batch) < 200:
            break

    if not markets:
        return pd.DataFrame()

    rows: list[dict] = []
    for m in markets:
        ticker    = m.get("ticker", "")
        sdate, ctype, threshold = _parse_ticker(ticker)
        if sdate is None or threshold is None:
            continue

        direction = m.get("strike_type", "")
        fs_raw    = m.get("floor_strike")
        cap_raw   = m.get("cap_strike")

        if ctype == "T":
            if direction not in ("greater", "less"):
                continue
            # Prefer floor_strike for "greater", cap_strike for "less"
            if direction == "greater" and fs_raw is not None:
                try:
                    threshold = float(fs_raw)
                except (TypeError, ValueError):
                    pass
            elif direction == "less" and cap_raw is not None:
                try:
                    threshold = float(cap_raw)
                except (TypeError, ValueError):
                    pass
            cap_f = float("nan")

        elif ctype == "B":
            if direction != "between":
                continue
            # threshold is the .5 midpoint (e.g. 83.5). Kalshi settles B83.5 as
            # floor=83, cap=84 INCLUSIVE both ends (bracket = {83, 84}; canonical
            # rules in kalshi_weather.settlement). Ensemble pricing uses
            # floor_strike / cap_strike from the API directly; cap_f is logging.
            cap_f     = threshold + 0.5
            direction = "between"

        else:
            continue

        bid   = _to_float(m.get("yes_bid_dollars"))
        ask   = _to_float(m.get("yes_ask_dollars"))
        last  = _to_float(m.get("last_price_dollars"))
        mid   = ((bid + ask) / 2) if (np.isfinite(bid) and np.isfinite(ask)) else last

        rows.append({
            "ticker":             ticker,
            "settlement_date":    sdate,
            "contract_type":      ctype,
            "threshold_f":        threshold,
            "cap_f":              cap_f,
            "t_direction":        direction,
            # Raw Kalshi strike fields, kept verbatim so the ensemble path prices
            # exactly as scripts/run_ensemble_picks.py (the validated cross-check).
            "strike_type":        m.get("strike_type"),
            "floor_strike":       _to_float(fs_raw)  if fs_raw  is not None else float("nan"),
            "cap_strike":         _to_float(cap_raw) if cap_raw is not None else float("nan"),
            "yes_bid_dollars":    bid,
            "yes_ask_dollars":    ask,
            "last_price_dollars": mid,
            "volume_fp":          _to_float(m.get("volume_fp")),
        })

    return pd.DataFrame(rows)


# ── Forecast fetch ────────────────────────────────────────────────────────────

def fetch_gfs_tmax(
    lat: float,
    lon: float,
    target_dates: list[date],
    model: str = "gfs_seamless",
    lst_offset: int = 0,
) -> dict[date, float]:
    """
    Fetch GFS tmax (°F) for each target_date from the Open-Meteo API.

    For dates up to ~today + 7 days: uses the live forecast API.
    For dates more than 7 days in the future: returns {} (no reliable forecast).
    For dates in the recent past: falls back to the Historical Forecast API.

    lst_offset: UTC offset for Local Standard Time (e.g. -5 for EST, -7 for MST).
    Uses Etc/GMT fixed-offset timezones so daily boundaries match NWS LST day,
    not the DST-shifted local clock. Kalshi settles on LST, not DST clock time.
    """
    if not target_dates:
        return {}

    # Etc/GMT+N = UTC-N (POSIX sign reversal). For lst_offset=-7 → "Etc/GMT+7".
    if lst_offset < 0:
        om_timezone = f"Etc/GMT+{abs(lst_offset)}"
    elif lst_offset > 0:
        om_timezone = f"Etc/GMT-{lst_offset}"
    else:
        om_timezone = "UTC"

    today = date.today()
    results: dict[date, float] = {}

    # Split into past (≤ today) and future (> today)
    past_dates   = sorted(d for d in target_dates if d <= today)
    future_dates = sorted(d for d in target_dates if today < d <= today + timedelta(days=7))

    for dates, base_url in [(past_dates, _OM_HIST_BASE), (future_dates, _OM_FCST_BASE)]:
        if not dates:
            continue
        start, end = dates[0].isoformat(), dates[-1].isoformat()
        params = {
            "latitude":          lat,
            "longitude":         lon,
            "daily":             "temperature_2m_max",
            "temperature_unit":  "fahrenheit",
            "timezone":          om_timezone,
            "start_date":        start,
            "end_date":          end,
            "models":            model,
        }
        try:
            resp = requests.get(base_url, params=params, timeout=20)
            resp.raise_for_status()
            data  = resp.json()
            daily = data.get("daily", {})
            for d_str, tmax in zip(daily.get("time", []), daily.get("temperature_2m_max", [])):
                d = date.fromisoformat(d_str)
                if d in target_dates and tmax is not None:
                    results[d] = float(tmax)
        except Exception as e:
            print(f"  Warning: forecast fetch failed for {start}–{end}: {e}")

    return results


# ── Scoring ───────────────────────────────────────────────────────────────────

def build_ensemble_context(
    city_key: str,
    cfg: dict,
    target_dates: list[date],
    bias_table: dict | None = None,
    running_highs: dict[date, float] | None = None,
) -> dict[date, dict]:
    """For a validated city, build the per-settlement-date ensemble distribution
    used to price contracts: corrected + dispersion-shrunk + center-bias-corrected
    members, NWS anchor (HRRR fallback), and the NWS-Normal sigma. Returns {} for
    non-validated cities or on data failure.

    Mirrors scripts/run_ensemble_picks.py exactly (the by-hand cross-check):
    the recent grid->sensor bias is computed once (point-in-time, asof today) and
    reused across target dates; each date gets its own ensemble member fetch.

    Center-bias correction (P1.0 partial — wired 2026-07-09):
    After dispersion calibration we apply an additional shrunk center shift from
    center_bias.center_shift(), which corrects for the remaining systematic error
    in the ensemble center AFTER the grid->sensor (gbias) correction.

    Intraday floor (P3.1): if running_highs contains an entry for a settlement date,
    all ensemble members below that observed running high are raised to it. The final
    daily high must be >= H_now (it's already been observed), so members below H_now
    are impossible — replacing them narrows the distribution without look-ahead.
    """
    ec = ENSEMBLE_CITIES.get(city_key)
    if ec is None:
        return {}
    gbias, gsigma = recent_grid_bias(cfg["lat"], cfg["lon"], ec["iem_stn"], ec["iem_net"], date.today())
    sigma = float(np.clip(gsigma if np.isfinite(gsigma) else SIGMA_DEF, SIGMA_FLOOR, SIGMA_CAP))
    station = cfg.get("nws_station", "")

    # OM timezone string for the HRRR fallback fetch (same LST zone as settlement).
    lst_off = cfg.get("lst_offset", 0)
    if lst_off < 0:
        _om_tz = f"Etc/GMT+{abs(lst_off)}"
    elif lst_off > 0:
        _om_tz = f"Etc/GMT-{lst_off}"
    else:
        _om_tz = "UTC"

    out: dict[date, dict] = {}
    for sdate in target_dates:
        members = fetch_ensemble_maxes(cfg["lat"], cfg["lon"], str(sdate), ec["tz"])
        if members.size == 0:
            continue
        corr = corrected_members(members, gbias, 0.0)
        corr, kshrink = (calibrate_dispersion(corr, gsigma) if np.isfinite(gsigma) else (corr, 1.0))

        # Center-bias correction (P1.0): shrunk shift = (n·measured + PRIOR·prior)/(n+PRIOR).
        # Applied to the fully-corrected members so ens_p50 in the log reflects the
        # corrected center and ens_center_err tracks the RESIDUAL, not the raw error.
        cs = _ens_center_shift(station, sdate.month, bias_table)
        if abs(cs) > 0.05:
            corr = corr + cs   # shift all members by cs °F

        # Intraday floor (P3.1): truncate distribution from below at observed running high.
        # final_high >= H_now is guaranteed by physics; members below H_now are impossible.
        # Applied AFTER all model corrections so p50/sd stats reflect the constrained dist.
        h_now: float | None = running_highs.get(sdate) if running_highs else None
        if h_now is not None:
            corr = np.maximum(corr, h_now)

        # NWS daytime-high anchor.  Fall back to HRRR (gfs_seamless) when the NWS API
        # fails — HRRR MAE 1.22°F at 1d lead is a reliable center even without the
        # human-adjusted layer; prevents the degraded pure-ensemble mode.
        nws = fetch_nws_high(cfg["lat"], cfg["lon"], str(sdate))
        if nws is None:
            nws = fetch_hrrr_high(cfg["lat"], cfg["lon"], str(sdate), om_timezone=_om_tz)

        # Log-only HRRR cross-check (2026-07-20): HRRR reruns hourly vs. NWS's few-
        # times-a-day cadence, so an unconditional read here (not just the NWS-failure
        # fallback above) gives a much faster-refreshing signal to compare against the
        # ensemble center over time. Advisory only — does NOT feed prob_estimate/the
        # anchor; wire in only after forward-validating it against settled outcomes
        # (same discipline the same-day taper needed and initially skipped).
        hrrr_check = fetch_hrrr_high(cfg["lat"], cfg["lon"], str(sdate), om_timezone=_om_tz)

        s = ens_summary(corr)
        out[sdate] = {
            "members": corr, "nws": nws, "sigma": sigma, "types": ec["types"],
            "gbias": gbias, "k": kshrink, "center_shift": round(cs, 2),
            "running_high": h_now,
            "n": s.get("n"), "p50": s.get("p50"), "sd": s.get("sd"),
            "disagree": (nws - s["p50"]) if (nws is not None and s.get("p50") is not None) else None,
            "hrrr_check": hrrr_check,
            "hrrr_vs_ens": (hrrr_check - s["p50"]) if (hrrr_check is not None and s.get("p50") is not None) else None,
        }
    return out


def score_contracts(
    contracts: pd.DataFrame,
    rule: Callable[[pd.Series], float],
    forecasts: dict[date, float],
    nws_station: str = "KNYC",
    gaussian_rule: Callable[[pd.Series], float] | None = None,
    bias_table: dict | None = None,
    b_rule: Callable[[pd.Series], float] | None = None,
    ensemble_ctx: dict[date, dict] | None = None,
) -> pd.DataFrame:
    """
    Apply rule to each row, attach forecast, compute edge.

    T-type contracts use `rule` (isotonic when available).
    B-type contracts use `b_rule` when available (isotonic calibrated),
    otherwise fall back to contract_probability() from bias.py.

    Returns one row per contract with columns:
        ticker, contract_type, settlement_date, threshold_f, cap_f, t_direction,
        tmax_f_fcst, prob_estimate, market_mid,
        yes_bid_dollars, yes_ask_dollars, spread,
        edge_raw, direction, liquid_atm
    """
    if contracts.empty:
        return pd.DataFrame()

    # P3.1 intraday path: precompute today's LST date + current local hour for this city.
    # Used below to route same-day picks through intraday_prob (time-of-day-tapered) instead
    # of mixture_prob (which treats the distribution as fully unresolved).
    _intraday_today: date | None = None
    _intraday_hour: float | None = None
    try:
        _td = _lst_offset_td(nws_station)
        _lst_now = utc_now() + _td
        _intraday_today = _lst_now.date()
        _intraday_hour = _lst_now.hour + _lst_now.minute / 60.0
    except Exception:
        pass  # unknown station → intraday path unavailable, falls back to mixture_prob

    rows: list[dict] = []
    tz_failures = 0   # settlement-window computations that failed (contract skipped from trading)
    for _, c in contracts.iterrows():
        sdate = c["settlement_date"]
        tmax  = forecasts.get(sdate)
        if tmax is None:
            continue

        ctype = c.get("contract_type", "T")
        prob: float | None = None
        prob_source = "rule"

        # ── Ensemble-first pricing for validated cities/types (Move 1) ─────────
        # The multi-model ensemble distribution is the source of truth for P(YES)
        # where validated; the single-GFS-run rule below is only the fallback.
        # Price EVERY contract with the ensemble when we have it (Fix A): the single-run
        # rule is measured to be badly miscalibrated, so it must not price anything we
        # can price better. The OOS-validated set (`types`) governs what may TRADE, in
        # the push gate — not what gets priced. `ctype in types` = validated-for-trading.
        ens = (ensemble_ctx or {}).get(sdate)
        if ens is not None:
            fs = c.get("floor_strike", float("nan"))
            cs = c.get("cap_strike", float("nan"))
            h_now = ens.get("running_high")    # observed intraday running high (P3.1)

            # P3.1 intraday path: if we have a running high AND this is today's settlement
            # date, use intraday_prob instead of mixture_prob. intraday_prob applies:
            #   (a) a hard floor at h_now (final high >= h_now)
            #   (b) a fitted β(h) multiplier on the remaining forecast upside
            #       (data/calibration/diurnal_warming_curve.json; the old linear taper
            #       was falsified live 2026-07-12 — see scripts/fit_diurnal_warming.py).
            # Fitted β: ~0.98 at 9 AM, 0.83 at noon, 0.63 at 1 PM, 0.10 at 3 PM. The
            # simple floor in build_ensemble_context (P3.1) handles the β=1 special
            # case; intraday_prob gives the full model. Same-day picks remain
            # WATCH-ONLY (run_live SAME_DAY_WATCH_ONLY) until forward-validated.
            same_day_intraday = (
                h_now is not None
                and _intraday_today is not None
                and sdate == _intraday_today
                and _intraday_hour is not None
            )
            if same_day_intraday:
                p_ens = intraday_prob(
                    ens["members"], float(h_now), float(_intraday_hour),
                    c.get("strike_type"),
                    float(fs) if np.isfinite(fs) else None,
                    float(cs) if np.isfinite(cs) else None,
                    ens.get("nws"), ens["sigma"], W_NWS,
                )
            else:
                p_ens = mixture_prob(
                    ens["members"], c.get("strike_type"),
                    float(fs) if np.isfinite(fs) else None,
                    float(cs) if np.isfinite(cs) else None,
                    ens.get("nws"), ens["sigma"], W_NWS,
                )
            if p_ens is not None and np.isfinite(p_ens):
                prob = float(p_ens)
                prob_source = "ensemble"

        # ── Fallback: production rule (single-GFS-run path) ────────────────────
        if prob is None:
            if ctype == "B":
                row = c.copy()
                row["tmax_f_fcst"] = tmax
                if b_rule is not None:
                    prob = b_rule(row)
                    if not np.isfinite(prob):
                        continue
                else:
                    month      = sdate.month if hasattr(sdate, "month") else sdate.month
                    bt         = bias_table or {}
                    entry      = bt.get(nws_station, {}).get(month)
                    if entry is None:
                        continue
                    calibrated = apply_correction(tmax, nws_station, month, bt)
                    if calibrated is None:
                        continue
                    # floor/cap = the API integer strikes (B83.5 -> 83/84); the
                    # inclusive-bracket boundaries are applied inside
                    # contract_probability via kalshi_weather.settlement.
                    floor_f = float(c["threshold_f"]) - 0.5
                    cap_f_b = float(c["threshold_f"]) + 0.5
                    prob    = contract_probability(calibrated, entry["std_f"],
                                                  floor=floor_f, cap=cap_f_b,
                                                  strike_type="between")
                    if prob is None or not np.isfinite(prob):
                        continue
            else:
                # T-type: production rule (isotonic when available)
                row = c.copy()
                row["tmax_f_fcst"] = tmax
                prob = rule(row)
                if not np.isfinite(prob):
                    continue

        prob  = float(np.clip(prob, 1e-6, 1 - 1e-6))

        # Pre-settlement check: tradeable before the LST day begins (day-ahead)
        # OR during the same-day morning window (until 2 PM LST = 14h after midnight).
        # The push gate applies a higher edge bar for same-day picks; see SAME_DAY_EDGE_BUMP.
        # Computed HERE (before recalibration) because the calibration map is keyed by
        # lead bucket — day-ahead rows are measured to need a correction same-day rows
        # don't (skill_monitor bias +0.053 vs −0.006).
        from kalshi_weather.tz import settlement_window_utc
        from datetime import timedelta
        win_start = None
        is_same_day = False
        SAME_DAY_CUTOFF_H = 14   # allow same-day picks until 2 PM LST (typical daily-high window)
        try:
            win_start, _ = settlement_window_utc(sdate, nws_station)
            same_day_cutoff = win_start + timedelta(hours=SAME_DAY_CUTOFF_H)
            pre_settlement = utc_now() < same_day_cutoff
            is_same_day = (utc_now() >= win_start) and pre_settlement
        except Exception:
            # Unknown settlement window = NOT tradeable. Defaulting to tradeable
            # would let a tz failure route an in-progress/settled day into the
            # forecast-pick path; going quiet is the correct failure mode — but it
            # must be LOUD quiet (counted + warned below), or a systematic tz bug
            # would silently zero out every actionable signal and starve grading.
            pre_settlement = False
            tz_failures += 1
        try:
            hours_to_settle = (win_start - utc_now()).total_seconds() / 3600.0 if win_start else float("nan")
        except Exception:
            hours_to_settle = float("nan")

        # Recalibrate the raw model P(YES) → realized frequency (fixes over-prediction).
        # Keep prob_raw for logging so the map can always be refit from the model's own
        # output; log calib_lead so replay_check can re-apply the exact same map key.
        #
        # calib_lead MUST use the same hours_to_settle bucketing as skill_monitor's
        # LEAD_BUCKETS (lead_bucket() below) — NOT same_day_intraday. skill_monitor
        # drops every row with hours_to_settle <= 0 (window already open), so BOTH its
        # "same-day" (0-14h) and "day-ahead" (14-40h) buckets are exclusively pre-window
        # mixture_prob rows for the ensemble source; intraday_prob rows are never in
        # either bucket. "same-day" here really means "near-term day-ahead" (forecast
        # has sharpened, measured bias -0.006) vs "far day-ahead" (measured bias +0.053)
        # — a real, evidence-backed distinction. Keying off same_day_intraday instead
        # would break the fit/apply population match and over-correct near-term picks
        # that the evidence shows are already fine. (Caught in review 2026-07-18 before
        # this variant ever shipped — see git history if this comment looks stale.)
        prob_raw = prob
        calib_lead = lead_bucket(hours_to_settle)
        prob = apply_prob_calibration(prob_raw, prob_source, get_calibration(), lead=calib_lead)
        prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
        mid   = float(c.get("last_price_dollars", float("nan")))
        bid   = float(c.get("yes_bid_dollars",    float("nan")))
        ask   = float(c.get("yes_ask_dollars",    float("nan")))
        spread = (ask - bid) if (np.isfinite(bid) and np.isfinite(ask)) else float("nan")

        if not np.isfinite(mid):
            continue

        edge_signed = prob - mid
        if edge_signed >= 0:
            direction = "BUY_YES"
            edge_raw  = edge_signed
        else:
            direction = "BUY_NO"
            edge_raw  = -edge_signed

        # Market-shrinkage posterior (LOG-ONLY, 2026-07-21). w fit per
        # source@lead on graded outcomes (scripts/build_shrinkage.py): LODO
        # shows the w≈0.4 blend beats BOTH the model alone and the market alone
        # — the model loses head-to-head but carries independent information.
        # Logged so the posterior accrues a forward record; nothing gates or
        # sizes on these fields until that record supports promotion (intended
        # endgame: edge_shrunk replaces the heuristic ceiling gates).
        prob_shrunk = shrink_prob(prob, mid, prob_source, calib_lead)
        edge_shrunk = None if prob_shrunk is None else round(
            (prob_shrunk - mid) if direction == "BUY_YES" else (mid - prob_shrunk), 4)

        # Liquid ATM check
        two_sided  = np.isfinite(bid) and bid > MIN_BID
        tight      = np.isfinite(spread) and spread < MAX_SPREAD
        in_range   = ATM_MIN <= mid <= ATM_MAX
        liquid_atm = two_sided and tight and in_range

        # ── Holistic action score (Move 4) ─────────────────────────────────────
        # 0-100 fusion so EVERY contract is scored and rankable — nothing is silently
        # ruled out; the score just orders them and marks a tier. The hard safety
        # vetoes (model disagreement, calibration-error ceiling) still decide what
        # actually FIRES downstream in push_forecast_picks. Components: edge magnitude,
        # settlement horizon (nearer=better), forecast stability (ensemble/NWS
        # agreement + tightness), liquidity. (hours_to_settle computed above, at the
        # calibration seam, so the logged value and the applied calib_lead agree.)
        # Edge component rewards CREDIBLE edge: rises to ~0.12, plateaus to 0.20, then
        # TAPERS for implausibly large edges. A 0.4 "edge" on a liquid market is far more
        # likely our calibration error than a real mispricing (same intuition as the
        # push-side ceiling), so it must NOT outrank a clean 0.10 edge.
        if edge_raw <= 0.12:
            e_c = 0.85 * (edge_raw / 0.12)
        elif edge_raw <= 0.20:
            e_c = 0.85 + 0.15 * ((edge_raw - 0.12) / 0.08)
        else:
            e_c = max(0.30, 1.0 - (edge_raw - 0.20) / 0.30)   # 0.20->1.0 ... 0.50->0.30 floor
        if not np.isfinite(hours_to_settle):
            h_c = 0.5
        elif hours_to_settle <= 12:
            h_c = 1.0
        else:
            h_c = max(0.0, 1.0 - (hours_to_settle - 12.0) / 36.0)   # -> 0 by 48h out
        e_sd  = ens.get("sd")       if ens else None
        e_dis = ens.get("disagree") if ens else None
        if e_sd is None and e_dis is None:
            s_c = 0.7   # rule-priced row: stability unknown -> neutral-ish
        else:
            s_c = 1.0
            if e_dis is not None and np.isfinite(e_dis):
                s_c *= max(0.2, 1.0 - abs(float(e_dis)) / 5.0)      # 2.5°F->0.5, 5°F->0.2
            if e_sd is not None and np.isfinite(e_sd):
                s_c *= float(np.clip(1.0 - max(0.0, float(e_sd) - 1.5) / 2.0, 0.3, 1.0))
        l_c = 1.0 if liquid_atm else (0.5 if (two_sided and tight) else 0.2)
        action_score = round(100.0 * (0.45 * e_c + 0.25 * h_c + 0.20 * s_c + 0.10 * l_c), 1)
        if not pre_settlement:
            action_score = 0.0   # settlement window open — not a forecast trade
        # Tier is a RANKING/triage label, not a trade trigger. The actual fire decision
        # stays with push_forecast_picks (its live model-agreement + ceiling vetoes);
        # ★ in the table marks what actually fires. A "strong" contract the push gate
        # holds means high apparent edge that failed a safety veto — worth a human look.
        score_tier = "strong" if action_score >= 55 else ("watch" if action_score >= 35 else "low")

        cap_f = c.get("cap_f", float("nan"))
        rows.append({
            "ticker":           c["ticker"],
            "contract_type":    ctype,
            "settlement_date":  sdate,
            "threshold_f":      c["threshold_f"],
            "cap_f":            cap_f if np.isfinite(cap_f) else float("nan"),
            "t_direction":      c["t_direction"],
            "strike_type":      c.get("strike_type"),
            "floor_strike":     (c.get("floor_strike") if c.get("floor_strike") is not None and np.isfinite(c.get("floor_strike")) else None),
            "cap_strike":       (c.get("cap_strike")   if c.get("cap_strike")   is not None and np.isfinite(c.get("cap_strike"))   else None),
            "tmax_f_fcst":      round(tmax, 1),
            "prob_estimate":    round(prob, 4),
            "prob_raw":         round(prob_raw, 4),
            "prob_source":      prob_source,
            "calib_lead":       calib_lead,
            "ens_p50":          (ens.get("p50")            if ens else None),
            "ens_sd":           (ens.get("sd")             if ens else None),
            "ens_k":            (round(ens["k"], 2)           if ens and ens.get("k")             is not None else None),
            "ens_center_shift": (round(ens["center_shift"], 2) if ens and ens.get("center_shift") is not None else None),
            "grid_bias":        (round(ens["gbias"], 1)        if ens and ens.get("gbias")         is not None else None),
            "running_high":     (round(ens["running_high"], 1) if ens and ens.get("running_high")  is not None else None),
            "local_hour":       (round(_intraday_hour, 2) if _intraday_hour is not None else None),
            "nws_disagree":     (round(ens["disagree"], 1) if ens and ens.get("disagree") is not None else None),
            "hrrr_check_f":     (round(ens["hrrr_check"], 1) if ens and ens.get("hrrr_check") is not None else None),
            "hrrr_vs_ens_diff": (round(ens["hrrr_vs_ens"], 1) if ens and ens.get("hrrr_vs_ens") is not None else None),
            "prob_shrunk":      (round(prob_shrunk, 4) if prob_shrunk is not None else None),
            "edge_shrunk":      edge_shrunk,
            "market_mid":       round(mid, 4),
            "yes_bid_dollars":  round(bid, 4) if np.isfinite(bid) else float("nan"),
            "yes_ask_dollars":  round(ask, 4) if np.isfinite(ask) else float("nan"),
            "spread":           round(spread, 4) if np.isfinite(spread) else float("nan"),
            "edge_raw":         round(edge_raw, 4),
            "direction":        direction,
            "liquid_atm":       liquid_atm,
            "pre_settlement":   pre_settlement,
            "is_same_day":      is_same_day,
            "hours_to_settle":  round(hours_to_settle, 1) if np.isfinite(hours_to_settle) else None,
            "action_score":     action_score,
            "score_tier":       score_tier,
            # Provenance: which code + calibration artifacts produced this row
            # (reproducibility; replay_check compares only same-artifact rows).
            **provenance(),
        })

    if tz_failures:
        print(f"  ⚠️  WARNING: settlement-window computation failed for {tz_failures} "
              f"contract(s) — they are treated as NOT tradeable. If this repeats every "
              f"run, tz/settlement_window_utc is broken and no forecast pick can fire.")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("action_score", ascending=False).reset_index(drop=True)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_signals(
    model_path: Path,
    city_keys: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load production rules, fetch open markets + forecasts, return signal DataFrame.

    city_keys defaults to all cities in CITY_CONFIGS.
    """
    import cloudpickle

    if not model_path.exists():
        raise FileNotFoundError(
            f"Production model not found: {model_path}\n"
            "Run scripts/build_production_model.py first."
        )

    with open(model_path, "rb") as fh:
        production_rules: dict = cloudpickle.load(fh)

    city_keys  = city_keys or list(CITY_CONFIGS.keys())
    _bias_table = load_bias_table()
    all_signals: list[pd.DataFrame] = []

    for city_key in city_keys:
        cfg = CITY_CONFIGS.get(city_key)
        rule_entry = production_rules.get(city_key)
        if cfg is None or rule_entry is None:
            print(f"  {city_key}: no config or model — skipping")
            continue

        rule          = rule_entry["active_rule"]
        gaussian_rule = rule_entry.get("gaussian_rule")
        b_rule        = rule_entry.get("b_active_rule")   # None for Tier 2 / old models
        rule_name     = rule_entry["preferred_rule"]
        series     = cfg["series"]
        station    = cfg["nws_station"]

        b_label = "isotonic" if b_rule is not None else "gaussian"
        print(f"\n── {city_key} ({series}, T={rule_name}, B={b_label}) ─────────────")

        # 1. Fetch open markets
        contracts = fetch_open_markets(series, station)
        n_t = (contracts["contract_type"] == "T").sum() if not contracts.empty else 0
        n_b = (contracts["contract_type"] == "B").sum() if not contracts.empty else 0
        print(f"   Open contracts: {len(contracts)} ({n_t}T + {n_b}B)")
        if contracts.empty:
            print(f"   No open T-type markets.")
            continue

        # 2. Fetch forecasts for each settlement date
        target_dates = sorted(contracts["settlement_date"].unique())
        forecasts_map = fetch_gfs_tmax(cfg["lat"], cfg["lon"], target_dates,
                                       lst_offset=cfg.get("lst_offset", 0))
        n_fcst = sum(1 for d in target_dates if d in forecasts_map)
        print(f"   Settlement dates: {[str(d) for d in target_dates]}")
        print(f"   GFS tmax fetched: {n_fcst}/{len(target_dates)}")
        for d in target_dates:
            if d in forecasts_map:
                print(f"     {d}: {forecasts_map[d]:.1f}°F")

        # 2b. Intraday floor: fetch observed running high for same-day dates (P3.1).
        # The final daily high must be >= H_now, so ensemble members below H_now are
        # physically impossible — truncating from below sharpens same-day pricing.
        lst_off = cfg.get("lst_offset", 0)
        try:
            lst_today = lst_date_for_utc(utc_now(), station)
        except Exception:
            lst_today = None
        running_highs: dict[date, float] = {}
        if lst_today is not None and lst_today in set(target_dates):
            rh = fetch_running_high_f(station, lst_off, on_date=lst_today)
            if rh is not None:
                running_highs[lst_today] = rh
                print(f"   Running high (intraday floor): {rh:.1f}°F — truncating ensemble below")
            else:
                print(f"   Running high: IEM fetch failed — unconstrained ensemble")

        # 2c. Build ensemble distribution for validated cities (Move 1: ensemble
        #     drives P(YES); single-GFS rule is fallback only).
        ensemble_ctx = build_ensemble_context(city_key, cfg, list(target_dates),
                                              bias_table=_bias_table,
                                              running_highs=running_highs or None)
        if ensemble_ctx:
            ddiag = next(iter(ensemble_ctx.values()))
            cs = ddiag.get("center_shift", 0.0)
            cs_str = f", center-shift {cs:+.2f}°F" if abs(cs) > 0.05 else ""
            print(f"   Ensemble: {len(ensemble_ctx)}/{len(target_dates)} dates priced "
                  f"(n={ddiag.get('n')}, grid-bias {ddiag.get('gbias'):+.1f}°F, "
                  f"k={ddiag.get('k'):.2f}, σ={ddiag['sigma']:.1f}{cs_str}) — overrides single-run rule")
        elif city_key in ENSEMBLE_CITIES:
            print(f"   Ensemble: no member data — falling back to single-run rule")

        # 3. Score
        scored = score_contracts(contracts, rule, forecasts_map,
                                 nws_station=station, gaussian_rule=gaussian_rule,
                                 bias_table=_bias_table, b_rule=b_rule,
                                 ensemble_ctx=ensemble_ctx)
        if scored.empty:
            print("   No scoreable contracts.")
            continue

        scored["city"] = city_key
        all_signals.append(scored)

    if not all_signals:
        return pd.DataFrame()

    return pd.concat(all_signals, ignore_index=True)


# ── Formatting ────────────────────────────────────────────────────────────────

def _to_float(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def print_signals(signals: pd.DataFrame, risk: RiskManager | None = None) -> None:
    """Print a human-readable signal report with optional risk sizing."""
    now = utc_now().strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'═'*70}")
    print(f"  LIVE TRADE SIGNALS  —  {now}")
    print(f"{'═'*70}")

    if risk is not None:
        print(f"  {risk.status_line()}")

    if signals.empty:
        print("  No open contracts scored.")
        print(f"{'═'*70}")
        return

    # Actionable: liquid ATM + edge ≥ threshold + settlement window not yet open
    pre_col = signals.get("pre_settlement", pd.Series(True, index=signals.index))
    actionable = signals[
        signals["liquid_atm"] &
        (signals["edge_raw"] >= EDGE_THRESHOLD) &
        pre_col
    ]

    if not actionable.empty:
        print(f"\n  *** ACTIONABLE ({len(actionable)} contracts — liquid ATM, edge ≥ {EDGE_THRESHOLD}) ***\n")
        for _, r in actionable.iterrows():
            sizing = risk.size_trade(r) if risk is not None else None
            _print_signal_row(r, highlight=True, sizing=sizing)
    else:
        print(f"\n  No liquid ATM contracts with edge ≥ {EDGE_THRESHOLD} right now.")

    # All scored contracts (informational) — ranked by holistic action score (Move 4)
    # so nothing is silently dropped; the score orders everything and marks a tier.
    print(f"\n  All scored contracts ({len(signals)} total, ranked by score):\n")
    print(f"  {'Ticker':<32} {'T':>2} {'Settle':>10} {'P(YES)':>7} "
          f"{'Mid':>6} {'Edge':>6} {'Score':>6} {'Tier':>6} {'Dir':>8} {'Flags':>7}")
    print(f"  {'-'*32} {'-'*2} {'-'*10} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*7}")
    _sort_col = "action_score" if "action_score" in signals.columns else "edge_raw"
    for _, r in signals.sort_values(_sort_col, ascending=False).iterrows():
        pre   = r.get("pre_settlement", True)
        flags = ""
        if r["liquid_atm"]:
            flags += "✓"
        if r["liquid_atm"] and r["edge_raw"] >= EDGE_THRESHOLD and pre:
            flags += "★"
        if r.get("is_same_day"):
            flags += "📅"  # same-day morning pick (higher edge bar applies)
        if not pre:
            flags += "⏰"  # settling now (past 2 PM LST cutoff)
        ctype = r.get("contract_type", "T")
        tier  = r.get("score_tier", "")
        tmark = {"strong": "◆", "watch": "◇", "low": "·"}.get(tier, "")
        print(f"  {r['ticker']:<32} {ctype:>2} {str(r['settlement_date']):>10} "
              f"{r['prob_estimate']:>7.4f} {r['market_mid']:>6.4f} {r['edge_raw']:>6.4f} "
              f"{r.get('action_score', 0):>6.1f} {tmark + tier:>6} {r['direction']:>8} {flags:>7}")

    print(f"\n{'═'*70}")
    print(f"  Edge threshold: {EDGE_THRESHOLD} | ATM: mid ∈ [{ATM_MIN},{ATM_MAX}], "
          f"spread < {MAX_SPREAD}, two-sided")
    print(f"  Fills are MODELED — pessimistic=taker+spread, optimistic=maker.")
    print(f"  ★ = actionable  ✓ = liquid ATM  📅 = same-day morning (+{int(0.10*100)}% edge bar)  ⏰ = past 2PM cutoff")
    print(f"{'═'*70}\n")


def _print_signal_row(
    r: pd.Series,
    highlight: bool = False,
    sizing: dict | None = None,
) -> None:
    star = "★ " if highlight else "  "
    bid  = r.get("yes_bid_dollars", float("nan"))
    ask  = r.get("yes_ask_dollars", float("nan"))
    bid_ask = f"${bid:.2f} / ${ask:.2f}" \
              if (np.isfinite(bid) and np.isfinite(ask)) else "— / —"

    ctype = r.get("contract_type", "T")
    if ctype == "B":
        floor_d = int(r["threshold_f"] - 0.5)
        cap_d   = floor_d + 1
        contract_desc = f"Range: {floor_d}–{cap_d}°F (between)"
    else:
        contract_desc = f"Threshold: {r['threshold_f']:.0f}°F ({r['t_direction']})"

    rh = r.get("running_high")
    rh_tag = f"  |  Running high: {rh:.1f}°F (obs floor)" if rh is not None else ""
    print(f"  {star}{r['ticker']}  [{r.get('city', '?')}]  [{ctype}]")
    print(f"      Settlement: {r['settlement_date']}  |  "
          f"GFS tmax: {r['tmax_f_fcst']:.1f}°F  |  "
          f"{contract_desc}{rh_tag}")
    print(f"      Rule P(YES): {r['prob_estimate']:.4f}  |  "
          f"Market mid: {r['market_mid']:.4f}  |  "
          f"Edge: {r['edge_raw']:.4f}  |  {r['direction']}")
    print(f"      Bid/Ask: {bid_ask}  |  Spread: ${r['spread']:.3f}")

    if r["direction"] == "BUY_YES":
        place_line = (
            f"      → Place YES limit at mid ~${r['market_mid']:.2f}  "
            f"(taker: ask ~${ask:.2f})" if np.isfinite(ask) else ""
        )
    else:
        if np.isfinite(bid):
            no_ask = round(1.0 - float(bid), 4)
            place_line = (
                f"      → Place NO limit at mid ~${1-r['market_mid']:.2f}  "
                f"(taker: NO ask ~${no_ask:.2f})"
            )
        else:
            place_line = ""

    if sizing is not None:
        if sizing["contracts"] > 0:
            cap_tag = f"  [{sizing['cap']} cap]" if sizing["cap"] else ""
            size_line = (
                f"  |  Size: {sizing['contracts']} contracts"
                f" (${sizing['dollars']:.2f})"
                f"  Kelly {sizing['kelly_f']:.3f}→{sizing['kelly_f'] * 100:.1f}%"
                f"{cap_tag}"
            )
        else:
            size_line = f"  |  Size: BLOCKED ({sizing['blocked']})"
        if place_line:
            print(place_line + size_line)
        else:
            print(f"      {size_line.strip()}")
    elif place_line:
        print(place_line)
    print()
