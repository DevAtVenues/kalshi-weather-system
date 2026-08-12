"""
Ensemble bucket-probability engine.

Replaces the production model's failure mode — a SINGLE deterministic GFS run fed
to a coarse rule — with the real, day-specific distribution from the 71-member
multi-model ensemble (GEFS + ECMWF-ENS + ICON). This is exactly the by-hand
cross-check that has been overriding the live engine all week, made into code.

Two things it fixes at once:
  1. SINGLE-RUN OUTLIERS — the deterministic forecast is often a tail member
     (Chicago: 95.8°F vs ensemble mean 91.8). The ensemble's spread is real and
     day-specific (calm ridge = tight, frontal/marine day = wide), unlike the
     fixed historical std the gaussian path uses.
  2. THE "less than X" BOUNDARY BUG + incoherent buckets — probabilities are
     computed straight from each contract's strike_type/floor/cap with explicit
     integer rounding, so a coherent distribution and correct tail boundaries
     fall out by construction.

Bias discipline (no double-counting — same rule as sensor_drift.py): the ensemble
is on the model grid, so members are corrected by the climatological monthly bias
AND the recent sensor drift before bucketing. corrected = member - bias_f - drift_f.

VALIDATION CAVEAT: Open-Meteo retains ensemble MEMBERS only ~3 days, so a long
multi-month out-of-sample backtest of this engine is not possible from the API —
that is the whole reason the live ensemble logger exists. Until enough logged
history accrues, this runs as a PARALLEL engine whose job is to reproduce (and
replace) the manual cross-check, not yet to silently gate live capital alone.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_ENS_URL  = "https://ensemble-api.open-meteo.com/v1/ensemble"
_FCST_URL = "https://api.open-meteo.com/v1/forecast"
_IEM_URL  = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"
_MODELS   = "gfs_seamless,ecmwf_ifs025,icon_seamless"
# Minimum member series each model FAMILY must contribute. The API resolves
# requested names to internal ones (gfs_seamless -> ncep_gefs_seamless,
# icon_seamless -> icon_seamless_eps), so families are matched by substring.
# A family below its floor means the feed silently degraded — 2026-07-30
# incident: ecmwf_ifs04 was deprecated upstream and returned ZERO members for
# weeks while GFS+ICON kept the total count plausible (71), so every consumer
# priced off a distribution missing its best model. Fail the WHOLE fetch
# loudly instead of returning a partial ensemble.
_MIN_MEMBERS = {"gefs": 25, "ecmwf": 40, "icon": 30}
_H        = {"User-Agent": "kalshi-weather research"}
# Daily cache for the recent grid->sensor bias: computed once per station per day and
# reused all day, so a transient IEM flake on a later cycle can't silently drop a city
# into the sigma=2.0/no-bias/no-shrink degraded mode (and it cuts IEM load).
_GRID_BIAS_CACHE = Path(__file__).parents[3] / "data" / "cache" / "grid_bias"


def recent_grid_bias(lat: float, lon: float, iem_station: str, iem_network: str,
                     asof, window: int = 14, min_days: int = 7, lag_days: int = 1,
                     max_bias: float = 3.0) -> tuple[float, float]:
    """Recent grid->sensor stats on the ENSEMBLE/ANALYSIS basis (NOT the deterministic
    forecast basis that bias.py/sensor_drift use — those run hotter and over-correct
    the ensemble). Compares the Open-Meteo gridded analysis daily-max (`past_days`,
    the same family the ensemble lives in) to the actual station high, point-in-time.

    Returns (bias, sigma):
      bias  = median(grid_analysis - sensor_actual), robust, soft-thresholded by 1 SEM,
              capped, min-sample gated -> 0. °F to SUBTRACT from each ensemble member.
      sigma = robust std (MAD) of the residuals = the REALIZED recent forecast
              uncertainty of the daily high. Used as the NWS-Normal spread, so the
              human forecast is treated as sharp where the station is predictable
              and broad where it is not. NaN -> caller falls back to a default.
    """
    # Daily cache: reuse today's successfully-computed value so a transient IEM flake
    # on a later cycle can't silently degrade the city to sigma=2.0/no-bias/no-shrink.
    asof_d   = pd.Timestamp(asof).date()
    cache_f  = _GRID_BIAS_CACHE / f"{iem_station}_{asof_d}.json"
    if cache_f.exists():
        try:
            d = json.loads(cache_f.read_text())
            return float(d["bias"]), (float(d["sigma"]) if d.get("sigma") is not None else float("nan"))
        except Exception:
            pass

    hi = pd.Timestamp(asof) - pd.Timedelta(days=lag_days)
    lo = hi - pd.Timedelta(days=window + 5)
    result: tuple[float, float] | None = None
    for attempt in range(2):
        try:
            u = (f"{_IEM_URL}?network={iem_network}&stations={iem_station}"
                 f"&year1={lo.year}&month1={lo.month}&day1={lo.day}"
                 f"&year2={hi.year}&month2={hi.month}&day2={hi.day}"
                 "&vars=max_temp_f&format=comma&na=blank")
            a = pd.read_csv(io.StringIO(requests.get(u, headers=_H, timeout=60).text), comment="#")
            a["day"] = pd.to_datetime(a["day"]).dt.date
            a = a.dropna(subset=["max_temp_f"])
            g = requests.get(_FCST_URL, params={"latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max", "temperature_unit": "fahrenheit",
                "timezone": "auto", "past_days": window + 6, "forecast_days": 1}, timeout=25).json()["daily"]
            gd = pd.DataFrame({"day": pd.to_datetime(g["time"]).date, "grid": g["temperature_2m_max"]})
            m = a.merge(gd, on="day")
            m = m[(m["day"] >= lo.date()) & (m["day"] <= hi.date())]
            diff = (m["grid"] - m["max_temp_f"].astype(float)).dropna().values
            n = len(diff)
            if n < min_days:
                result = (0.0, float("nan"))     # genuinely insufficient data (a real answer)
            else:
                raw = float(np.median(diff))
                mad = float(np.median(np.abs(diff - raw))) * 1.4826
                sem = mad / np.sqrt(n)
                bias = float(np.clip(np.sign(raw) * max(0.0, abs(raw) - sem), -max_bias, max_bias))
                result = (bias, float(mad))
            break
        except Exception:
            if attempt == 0:
                time.sleep(1.5)                  # transient IEM/Open-Meteo flake — retry once
                continue
            result = None                        # give up this cycle; do NOT cache the failure

    if result is None:
        return 0.0, float("nan")                 # degraded fallback; next cycle retries fresh
    try:
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        sigma_out = None if not np.isfinite(result[1]) else result[1]
        cache_f.write_text(json.dumps({"bias": result[0], "sigma": sigma_out}))
    except Exception:
        pass
    return result


def calibrate_dispersion(members: np.ndarray, realized_sigma: float,
                         w: float = 0.6, k_min: float = 0.5) -> tuple[np.ndarray, float]:
    """Shrink an OVER-dispersed ensemble toward the realized recent forecast error,
    keeping its center (median) and day-specific shape (skew). The multi-model
    ensemble runs ~1.5-2x wider than the high actually varies from forecast, which
    manufactures phantom tail edges; this rescales the spread to match reality.

        target_sigma = w·realized + (1-w)·ensemble_sd
        k = clip(target_sigma / ensemble_sd, k_min, 1.0)     # only SHRINK, never inflate
        member' = median + (member - median)·k

    Only shrinks: over-dispersion is the demonstrated failure; inflating an already-
    tight ensemble would invent confidence we have not validated. Returns (members, k)."""
    if members.size < 3 or not (realized_sigma and realized_sigma > 0):
        return members, 1.0
    es = float(members.std())
    if es <= 0:
        return members, 1.0
    target = w * float(realized_sigma) + (1.0 - w) * es
    k = float(np.clip(target / es, k_min, 1.0))
    med = float(np.median(members))
    return med + (members - med) * k, k


def nws_normal_prob(strike_type: str, floor: float | None, cap: float | None,
                    nws: float, sigma: float) -> float | None:
    """P(YES) for a contract under Normal(nws, sigma) — the human forecast treated as
    a sharp-ish distribution, with the SAME integer-rounding boundaries as the ensemble."""
    from scipy.stats import norm
    if nws is None or not (sigma and sigma > 0):
        return None
    from kalshi_weather.settlement import yes_bounds
    b = yes_bounds(strike_type, floor, cap)
    if b is None:
        return None
    lo, hi = b
    return float(norm.cdf(hi, nws, sigma) - norm.cdf(lo, nws, sigma))


def mixture_prob(members: np.ndarray, strike_type: str, floor: float | None, cap: float | None,
                 nws: float | None, sigma: float, w_nws: float = 0.4) -> float | None:
    """Blend the empirical ensemble bucket prob with the NWS-Normal bucket prob:
        P = (1 - w)·P_ensemble + w·P_nws.
    Falls back to the ensemble alone when NWS is unavailable. This is the NWS-in-bucket
    weighting: where sigma is small (predictable station), the NWS term concentrates on
    its bucket; where sigma is large, it stays broad and barely shifts the ensemble."""
    pe = contract_prob(members, strike_type, floor, cap)
    if pe is None:
        return None
    pn = nws_normal_prob(strike_type, floor, cap, nws, sigma) if nws is not None else None
    if pn is None:
        return pe
    return (1.0 - w_nws) * pe + w_nws * pn


# Every successful fetch is cached per (lat, lon, date) with a PER-FAMILY
# member breakdown, so verification/analysis reads the members the engine
# already paid for instead of re-hitting the API (2026-08-04: duplicate
# verifier fetches + first full-uptime day exhausted the 10k/day free tier
# by 2 PM). Latest fetch wins; files are a few KB.
_MEMBER_CACHE = Path(__file__).parents[3] / "data" / "cache" / "ensemble_members"


def _cache_path(lat: float, lon: float, date_str: str) -> Path:
    return _MEMBER_CACHE / f"{lat:.3f}_{lon:.3f}_{date_str}.json"


def cached_ensemble_maxes(lat: float, lon: float, date_str: str,
                          max_age_min: float | None = None) -> dict | None:
    """Latest cached member maxes for (lat, lon, date): {"fetch_ts", "age_min",
    "families": {family: np.ndarray}}. None if absent or older than max_age_min.
    This is the read path for pick verification — prefer it over a fresh
    fetch_ensemble_maxes whenever the engine has run recently."""
    try:
        j = json.loads(_cache_path(lat, lon, date_str).read_text())
        fetched = pd.Timestamp(j["fetch_ts"])
        age_min = (pd.Timestamp.utcnow() - fetched).total_seconds() / 60.0
        if max_age_min is not None and age_min > max_age_min:
            return None
        return {"fetch_ts": j["fetch_ts"], "age_min": round(age_min, 1),
                "families": {f: np.array(v, dtype=float)
                             for f, v in j["families"].items()}}
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


# Upstream model cycles only refresh ~every 6 h; re-fetching every 30-min engine
# run burned the 10k/day Open-Meteo quota by late morning (2026-08-04 and -05).
# Same-day TTL stays short so intraday picks see new cycles quickly.
_CACHE_TTL_MIN_SAME_DAY = 45.0
_CACHE_TTL_MIN_AHEAD = 150.0


def fetch_ensemble_maxes(lat: float, lon: float, date_str: str, tz: str) -> np.ndarray:
    """Daily-max (°F) per ensemble member for one station-date. Empty array on failure.

    Serves from the member cache within a freshness TTL before touching the API."""
    try:
        today_local = pd.Timestamp.now(tz=tz).strftime("%Y-%m-%d")
        ttl = _CACHE_TTL_MIN_SAME_DAY if date_str == today_local else _CACHE_TTL_MIN_AHEAD
    except Exception:
        ttl = _CACHE_TTL_MIN_SAME_DAY
    c = cached_ensemble_maxes(lat, lon, date_str, max_age_min=ttl)
    if c is not None and all(len(c["families"].get(f, ())) >= n
                             for f, n in _MIN_MEMBERS.items()):
        return np.concatenate([np.asarray(v, dtype=float)
                               for v in c["families"].values()])
    try:
        j = requests.get(_ENS_URL, params={
            "latitude": lat, "longitude": lon, "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit", "timezone": tz,
            "start_date": date_str, "end_date": date_str, "models": _MODELS,
        }, timeout=30).json()
        if j.get("error"):
            return np.array([])
        maxes = []
        fam_counts = {fam: 0 for fam in _MIN_MEMBERS}
        fam_series: dict[str, list[float]] = {fam: [] for fam in _MIN_MEMBERS}
        for k, v in j.get("hourly", {}).items():
            if k == "time":
                continue
            vals = [x for x in v if x is not None]
            if vals:
                mx = max(vals)
                maxes.append(mx)
                for fam in fam_counts:
                    if fam in k:
                        fam_counts[fam] += 1
                        fam_series[fam].append(mx)
        missing = {fam: n for fam, n in fam_counts.items()
                   if n < _MIN_MEMBERS[fam]}
        if missing:
            print(f"CRITICAL: ensemble feed degraded — member counts below floor "
                  f"{missing} (floors {_MIN_MEMBERS}, got {fam_counts}); "
                  f"refusing partial ensemble for {date_str}", file=sys.stderr)
            return np.array([])
        try:
            _MEMBER_CACHE.mkdir(parents=True, exist_ok=True)
            _cache_path(lat, lon, date_str).write_text(json.dumps({
                "fetch_ts": pd.Timestamp.utcnow().isoformat(),
                "lat": lat, "lon": lon, "date": date_str,
                "families": fam_series,
            }))
        except OSError:
            pass                                  # cache failure must not break pricing
        return np.array(maxes, dtype=float)
    except Exception:
        return np.array([])


def corrected_members(members: np.ndarray, bias_f: float = 0.0, drift_f: float = 0.0) -> np.ndarray:
    """Apply the climatological monthly bias and the recent sensor drift to every
    member (both are grid->sensor corrections; subtracting matches bias.py sign:
    positive bias/drift = model runs hot = lower the forecast)."""
    return members - float(bias_f) - float(drift_f)


def contract_prob(members: np.ndarray, strike_type: str,
                  floor: float | None, cap: float | None) -> float | None:
    """Empirical P(YES) for one contract from the (already corrected) member set.

    Boundaries come from kalshi_weather.settlement.yes_bounds — the conventions
    derived from 8,040 settled markets (see that module; enforced by
    tests/test_settlement_convention.py). In integer-settle terms:

        between  YES iff floor <= round(m) <= cap   (INCLUSIVE; brackets are 2 wide)
        greater  YES iff round(m) >  floor
        less     YES iff round(m) <  cap
    """
    if members.size == 0:
        return None
    from kalshi_weather.settlement import yes_bounds
    b = yes_bounds(strike_type, floor, cap)
    if b is None:
        return None
    lo, hi = b
    return float(np.mean((members >= lo) & (members < hi)))


def fetch_nws_high(lat: float, lon: float, date_str: str) -> float | None:
    """NWS daytime high (°F) for the exact target date, matched by the forecast
    period's start date (not the day-name, which rolls). The settlement station's
    human-adjusted forecast — independent of, and a center-anchor for, the ensemble."""
    try:
        m = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=_H, timeout=20).json()
        periods = requests.get(m["properties"]["forecast"], headers=_H, timeout=20).json()["properties"]["periods"]
        for p in periods:
            if p.get("isDaytime") and p.get("startTime", "")[:10] == date_str:
                return float(p["temperature"])
    except Exception:
        return None
    return None


def fetch_hrrr_high(lat: float, lon: float, date_str: str,
                    om_timezone: str = "auto") -> float | None:
    """HRRR-based deterministic daily-max (°F) from Open-Meteo gfs_seamless.

    At leads ≤ ~48h, gfs_seamless IS the HRRR model (highest-resolution operational
    forecast in the CONUS, MAE 1.22°F at 1d lead across 20 cities). Used as a
    fallback center-anchor when fetch_nws_high() fails, preventing the degraded
    pure-ensemble mode that loses the stabilising anchor.

    om_timezone should match the station's LST zone so 'daily' boundaries align
    with the settlement day. Defaults to 'auto' (Open-Meteo selects from lat/lon).
    """
    try:
        resp = requests.get(_FCST_URL, params={
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": om_timezone,
            "start_date": date_str, "end_date": date_str,
            "models": "gfs_seamless",
        }, headers=_H, timeout=20)
        resp.raise_for_status()
        vals = resp.json().get("daily", {}).get("temperature_2m_max", [])
        if vals and vals[0] is not None:
            return float(vals[0])
    except Exception:
        pass
    return None


def blend_to_nws(members: np.ndarray, nws_high: float | None,
                 weight: float = 0.5) -> tuple[np.ndarray, float]:
    """Recenter the (already grid-bias-corrected) ensemble toward the NWS high,
    keeping the ensemble's day-specific spread. Returns (shifted members, disagreement
    = NWS - ensemble median). weight=0 trusts the ensemble fully, 1.0 trusts NWS fully;
    0.5 = equal. Conservative default; tune forward (no historical NWS to backtest)."""
    if nws_high is None or members.size == 0:
        return members, 0.0
    med = float(np.median(members))
    disagreement = float(nws_high) - med
    shift = weight * disagreement
    return members + shift, disagreement


def summary(members: np.ndarray) -> dict:
    """Diagnostics for transparency on a pick card."""
    if members.size == 0:
        return {"n": 0}
    return {"n": int(members.size), "mean": round(float(members.mean()), 1),
            "sd": round(float(members.std()), 1),
            "p10": round(float(np.percentile(members, 10)), 1),
            "p50": round(float(np.percentile(members, 50)), 1),
            "p90": round(float(np.percentile(members, 90)), 1)}
