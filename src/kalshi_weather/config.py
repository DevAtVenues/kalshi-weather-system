from __future__ import annotations

from pathlib import Path

import yaml

_STATIONS_FILE = Path(__file__).parent.parent.parent / "config" / "stations.yaml"


def load_stations() -> dict:
    with open(_STATIONS_FILE) as f:
        return yaml.safe_load(f)


def get_station(key: str) -> dict:
    stations = load_stations()
    if key not in stations:
        raise KeyError(f"Station {key!r} not in config. Available: {list(stations)}")
    return stations[key]
