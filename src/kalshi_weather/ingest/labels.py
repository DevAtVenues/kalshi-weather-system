"""
Fetch and cache NWS Daily Climate Report (CLI) data from Iowa Environmental Mesonet.
IEM is the authoritative source — it parses the exact CLI reports Kalshi settles on.

Cache layout: data/raw/labels/{station}/{year}.parquet
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from kalshi_weather.tz import utc_now

_IEM_URL = "https://mesonet.agron.iastate.edu/json/cli.py"
_CACHE_ROOT = Path(__file__).parents[3] / "data" / "raw" / "labels"


def _cache_path(station: str, year: int) -> Path:
    return _CACHE_ROOT / station / f"{year}.parquet"


def _fetch_year_raw(station: str, year: int) -> list[dict]:
    resp = requests.get(_IEM_URL, params={"station": station, "year": year}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    records = data.get("results", [])
    if not records:
        raise ValueError(f"IEM returned no records for {station} {year}")
    return records


def _parse(records: list[dict], station: str) -> pd.DataFrame:
    df = pd.DataFrame(records)

    df = df.rename(columns={"valid": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # IEM relays the morning/afternoon CLI for the CURRENT day too — a "high so
    # far", not the settlement value (seen 2026-07-12: DEN cached 74 on a 95°F
    # day). Never cache an unfinished day; the daily current-year force-refresh
    # picks the date up once it's final. (Grading additionally guards date<today
    # itself, which also covers the pre-3am-ET window for West Coast LST days.)
    df = df[df["date"] < date.today()]

    # Coerce numeric fields; "M" (missing) and "T" (trace) become NaN
    for col in ("high", "low", "precip", "snow", "snowdepth", "high_normal"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = ["station", "date", "high", "low", "precip", "snow", "snowdepth",
            "high_normal", "retrieved_utc"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def fetch_labels(
    station: str,
    years: list[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily climate records for the given station and years.
    Pulls once per year and caches to parquet; never re-fetches unless force_refresh=True.

    Columns: station, date (Python date), high (°F), low (°F), precip (in),
             snow (in), snowdepth (in), high_normal (°F), retrieved_utc
    """
    (_CACHE_ROOT / station).mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for year in sorted(years):
        path = _cache_path(station, year)
        if path.exists() and not force_refresh:
            frames.append(pd.read_parquet(path))
            continue

        records = _fetch_year_raw(station, year)
        raw_df = pd.DataFrame(records)
        raw_df["retrieved_utc"] = utc_now().isoformat()
        parsed = _parse(raw_df.to_dict("records"), station)
        parsed.to_parquet(path, index=False)
        frames.append(parsed)
        time.sleep(0.25)  # polite rate limiting

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)
    return result


def climatological_prob(
    labels: pd.DataFrame,
    threshold_f: float,
    target_dates: list[date],
    train_years: range,
) -> pd.Series:
    """
    Compute the historical fraction of days where tmax >= threshold_f,
    for each target date's (month, day) combination, using only train_years.

    This is the Milestone 1 baseline — no forecast data required.
    Returns a Series indexed by target_date.
    """
    train = labels[labels["date"].apply(lambda d: d.year in set(train_years))].copy()
    train["month_day"] = train["date"].apply(lambda d: (d.month, d.day))
    train["exceeded"] = train["high"] >= threshold_f

    climo = train.groupby("month_day")["exceeded"].mean()

    result = {}
    for d in target_dates:
        md = (d.month, d.day)
        result[d] = climo.get(md, float("nan"))

    return pd.Series(result, name="climo_prob")
