"""
Timezone isolation module. This is the ONLY module that may perform timezone
conversions. All other modules call functions from here.

Kalshi settles temperature contracts on Local Standard Time (LST) day boundaries,
not DST-adjusted local time. NYC LST = UTC-5 year-round.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

UTC = timezone.utc


def _load_lst_offsets() -> dict[str, timedelta]:
    """Load per-station LST offsets from config/stations.yaml at import time."""
    from pathlib import Path
    import yaml  # type: ignore[import-untyped]
    stations_file = Path(__file__).parent.parent.parent / "config" / "stations.yaml"
    try:
        with open(stations_file) as fh:
            stations = yaml.safe_load(fh)
        return {
            cfg["nws_station"]: timedelta(hours=cfg["lst_offset_hours"])
            for cfg in stations.values()
            if "nws_station" in cfg and "lst_offset_hours" in cfg
        }
    except (FileNotFoundError, KeyError):
        return {"KNYC": timedelta(hours=-5)}


# LST offsets per NWS station ID (UTC offset, fixed year-round for settlement purposes)
_LST_OFFSETS: dict[str, timedelta] = _load_lst_offsets()


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def to_utc(dt: datetime) -> pd.Timestamp:
    """Convert a tz-aware datetime to a UTC pandas Timestamp. Raises on naive input."""
    if dt.tzinfo is None:
        raise ValueError(
            f"Naive datetime passed to to_utc: {dt!r}. "
            "All datetimes must carry timezone info."
        )
    return pd.Timestamp(dt).tz_convert("UTC")


def lst_offset(station: str) -> timedelta:
    if station not in _LST_OFFSETS:
        raise ValueError(f"No LST offset configured for station {station!r}. "
                         f"Available: {list(_LST_OFFSETS)}")
    return _LST_OFFSETS[station]


def lst_date_for_utc(ts: pd.Timestamp, station: str) -> date:
    """
    Return the LST calendar date a UTC timestamp belongs to.
    Used to assign a forecast or observation to the correct settlement day.
    """
    offset = lst_offset(station)
    lst_ts = ts + offset
    return lst_ts.date()


def settlement_window_utc(day: date, station: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return [start, end) UTC bounds for an LST settlement day.
    E.g. NYC 2024-07-04 LST = [2024-07-04 05:00 UTC, 2024-07-05 05:00 UTC)
    """
    offset = lst_offset(station)
    # LST midnight in UTC = UTC midnight shifted by -offset
    start = pd.Timestamp(datetime(day.year, day.month, day.day), tz="UTC") - offset
    return start, start + timedelta(days=1)
