"""
Per-station GFS forecast bias estimation.

Computes historical GFS error (forecast − actual NWS high) per station
per calendar month from the Historical Forecast API vs NWS CLI labels.

  bias_f > 0  →  GFS runs HOT (over-predicts); calibrated = GFS − bias
  bias_f < 0  →  GFS runs COLD (under-predicts)

Cache layout:
  data/calibration/forecasts/{station}/{year}.parquet  — per-station GFS archive
  data/calibration/bias.parquet                         — all stations, all months

The calibration script (scripts/run_calibration.py) populates this cache.
The dashboard loads it read-only at startup — no API calls at runtime.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from scipy import stats as scipy_stats
import numpy as np

from kalshi_weather.ingest.labels import fetch_labels

_CALIB_ROOT = Path(__file__).parents[3] / "data" / "calibration"
_FCST_URL   = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Years used for bias estimation.
# Using 2022-2024: avoids model changes pre-2022, keeps training recent.
CALIB_YEARS = [2022, 2023, 2024]

# Stations that use the LINEAR (slope) calibration  actual = a + b·forecast
# instead of the flat monthly offset  forecast − bias.  Chosen by an out-of-sample
# test — train 2022-23, test 2024 held out (scripts/calibration_upgrade_analysis.py):
# only the marine-layer coast has a forecast→actual slope meaningfully below 1
# (SFO b≈0.83, SEA b≈0.93), where the linear form beats the offset out-of-sample
# (SFO −0.18°F MAE, −8% CRPS). For the other 18 cities slope ≈ 1 and the offset
# ties or wins, so they keep the simpler, lower-variance offset — forcing slope
# everywhere fit noise and hurt. The gate is deliberately narrow.
SLOPE_STATIONS: set[str] = {"KSFO", "KSEA"}


def _fit_monthly_slope(
    merged: pd.DataFrame, min_n: int = 25,
) -> dict[int, tuple[float, float, float]]:
    """Per-month OLS  actual = a + b·gfs  →  {month: (a, b, residual_std)}.

    `merged` needs columns gfs_h (forecast), high (actual), month. Thin months
    (< min_n rows) fall back to the station's pooled global fit. This mirrors the
    fit validated out-of-sample in scripts/calibration_upgrade_analysis.py exactly.
    """
    x_all = merged["gfs_h"].to_numpy(dtype=float)
    y_all = merged["high"].to_numpy(dtype=float)
    gb, ga = np.polyfit(x_all, y_all, 1)                 # slope, intercept
    g_resid = y_all - (ga + gb * x_all)
    g_fit = (float(ga), float(gb), float(g_resid.std(ddof=1)))

    out: dict[int, tuple[float, float, float]] = {}
    for mo, g in merged.groupby("month"):
        if len(g) < min_n:
            out[int(mo)] = g_fit
            continue
        x = g["gfs_h"].to_numpy(dtype=float)
        y = g["high"].to_numpy(dtype=float)
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        out[int(mo)] = (float(a), float(b), float(resid.std(ddof=1)))
    return out


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fcst_cache_path(station: str, year: int, cache_dir: Path) -> Path:
    return cache_dir / "forecasts" / station / f"{year}.parquet"


def _fetch_gfs_year(lat: float, lon: float, year: int) -> pd.DataFrame:
    """Fetch GFS daily high forecast for one station-year in 90-day chunks."""
    chunks: list[pd.DataFrame] = []
    cursor   = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")

    while cursor <= year_end:
        chunk_end = min(cursor + pd.Timedelta(days=89), year_end)
        resp = requests.get(_FCST_URL, params={
            "latitude":          lat,
            "longitude":         lon,
            "start_date":        cursor.strftime("%Y-%m-%d"),
            "end_date":          chunk_end.strftime("%Y-%m-%d"),
            "daily":             "temperature_2m_max",
            "temperature_unit":  "fahrenheit",
            "timezone":          "UTC",
            "models":            "gfs_seamless",
        }, timeout=30)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        chunks.append(pd.DataFrame({
            "date":  pd.to_datetime(daily["time"]).date,
            "gfs_h": pd.to_numeric(daily.get("temperature_2m_max"), errors="coerce"),
        }))
        cursor = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.15)

    return pd.concat(chunks, ignore_index=True)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_station_forecasts(
    station_cfg: dict,
    years: list[int],
    cache_dir: Path,
) -> pd.DataFrame:
    """
    Fetch and cache historical GFS daily high forecasts for one station.
    Cache is per-station per-year to avoid the global-path collision in
    ingest/forecasts.py. Pull once; never re-fetch if already cached.
    """
    station = station_cfg["metar_station"]
    lat, lon = station_cfg["lat"], station_cfg["lon"]
    frames: list[pd.DataFrame] = []

    for year in sorted(years):
        path = _fcst_cache_path(station, year, cache_dir)
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df = _fetch_gfs_year(lat, lon, year)
            df["station"] = station
            df.to_parquet(path, index=False)
            frames.append(df)
        except Exception as exc:
            print(f"    [warn] GFS fetch failed for {station} {year}: {exc}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_bias_table(
    stations_cfg: dict,
    years: list[int] | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    For all stations, compute per-month GFS bias vs NWS CLI labels.

    Returns and saves a DataFrame with columns:
      station, month, bias_f, std_f, mae_f, n
    where bias_f = mean(GFS − actual); positive means GFS runs hot.
    """
    if years is None:
        years = CALIB_YEARS
    if cache_dir is None:
        cache_dir = _CALIB_ROOT

    rows: list[dict] = []

    for city_key, cfg in stations_cfg.items():
        station = cfg["metar_station"]
        print(f"  {city_key} ({station}) ...", end=" ", flush=True)

        try:
            forecasts = fetch_station_forecasts(cfg, years, cache_dir)
            labels    = fetch_labels(station, years)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        if forecasts.empty or labels.empty:
            print("no data")
            continue

        forecasts["date"] = pd.to_datetime(forecasts["date"]).dt.date
        labels["date"]    = pd.to_datetime(labels["date"]).dt.date

        merged = (
            forecasts[["date", "gfs_h"]]
            .merge(labels[["date", "high"]], on="date", how="inner")
            .dropna(subset=["gfs_h", "high"])
        )
        if merged.empty:
            print("no overlap")
            continue

        merged["error"] = merged["gfs_h"] - merged["high"].astype(float)
        merged["month"] = merged["date"].apply(lambda d: d.month)

        global_bias = float(merged["error"].mean())
        global_n    = len(merged)
        print(f"n={global_n}  bias={global_bias:+.2f}°F")

        # Linear (slope) calibration params per month — stored for every station
        # for transparency; only SLOPE_STATIONS apply them at runtime.
        merged["high"] = merged["high"].astype(float)
        slope = _fit_monthly_slope(merged)

        for month_val, grp in merged.groupby("month"):
            errors = grp["error"].dropna()
            if len(errors) < 3:
                continue
            a_m, b_m, sstd_m = slope[int(month_val)]
            rows.append({
                "station": station,
                "month":   int(month_val),
                "bias_f":  round(float(errors.mean()), 3),
                "std_f":   round(float(errors.std(ddof=1)), 3),
                "mae_f":   round(float(errors.abs().mean()), 3),
                "n":       int(len(errors)),
                "slope_a":     round(a_m, 4),
                "slope_b":     round(b_m, 4),
                "slope_std_f": round(sstd_m, 3),
            })

    if not rows:
        return pd.DataFrame()

    bias_df = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "bias.parquet"
    bias_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(bias_df)} station-month pairs)")
    return bias_df


def load_bias_table(cache_dir: Path | None = None) -> dict[str, dict[int, dict]]:
    """
    Load the saved bias table into a fast lookup dict:
      result[station][month] = {"bias_f": ..., "std_f": ..., "n": ...}

    Returns empty dict if the table hasn't been computed yet.
    """
    if cache_dir is None:
        cache_dir = _CALIB_ROOT
    path = cache_dir / "bias.parquet"
    if not path.exists():
        return {}

    df = pd.read_parquet(path)
    has_slope = {"slope_a", "slope_b", "slope_std_f"}.issubset(df.columns)
    result: dict[str, dict[int, dict]] = {}
    for _, row in df.iterrows():
        st = str(row["station"])
        mo = int(row["month"])
        entry = {
            "bias_f": float(row["bias_f"]),
            "std_f":  float(row["std_f"]),
            "n":      int(row["n"]),
        }
        if has_slope and pd.notna(row.get("slope_b")):
            entry["slope_a"]     = float(row["slope_a"])
            entry["slope_b"]     = float(row["slope_b"])
            entry["slope_std_f"] = float(row["slope_std_f"])
            # For a slope station the ACTIVE center model is linear, so the sigma
            # fed to contract_probability must be the linear model's residual std
            # — not the offset model's error std. Swap it in here so every consumer
            # that reads std_f stays consistent with apply_correction()'s center.
            if st in SLOPE_STATIONS:
                entry["std_f"] = float(row["slope_std_f"])
        result.setdefault(st, {})[mo] = entry
    return result


def apply_correction(
    gfs_h: float,
    station: str,
    month: int,
    bias_table: dict,
) -> float | None:
    """
    Correct a raw GFS high forecast toward the settlement sensor.

    Most stations: subtract the historical monthly mean bias (forecast − bias).
    SLOPE_STATIONS (SFO, SEA): apply the OOS-validated linear map a + b·forecast,
    which also gently shrinks extreme forecasts toward the seasonal mean (b < 1).
    Returns None if no calibration data is available for this station/month.
    """
    entry = bias_table.get(station, {}).get(month)
    if entry is None:
        return None
    # Slope stations (marine-layer coast) use the OOS-validated linear form
    # actual = a + b·forecast; every other station uses the flat monthly offset.
    if station in SLOPE_STATIONS and "slope_b" in entry:
        return round(entry["slope_a"] + entry["slope_b"] * float(gfs_h), 1)
    return round(float(gfs_h) - entry["bias_f"], 1)


def contract_probability(
    calibrated_h: float,
    std_f: float,
    floor: float | None,
    cap: float | None,
    strike_type: str,
) -> float | None:
    """
    Estimate P(YES) for a contract given a calibrated GFS forecast and
    historical residual spread (std_f).

    Models actual high as Normal(calibrated_h, std_f), mapped through the CANONICAL
    settlement boundaries (kalshi_weather.settlement.yes_bounds — empirically derived
    from settled markets; between is INCLUSIVE both ends, greater strict >, less
    strict <). Same boundaries as the ensemble path and grading.
    """
    if std_f < 0.1:
        return None

    from kalshi_weather.settlement import yes_bounds
    import math as _math
    b = yes_bounds(strike_type, floor, cap)
    if b is None:
        return None
    lo, hi = b
    lo_p = 0.0 if lo == -_math.inf else scipy_stats.norm.cdf(lo, loc=calibrated_h, scale=std_f)
    hi_p = 1.0 if hi == _math.inf else scipy_stats.norm.cdf(hi, loc=calibrated_h, scale=std_f)
    return float(hi_p - lo_p)
