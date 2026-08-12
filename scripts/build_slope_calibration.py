"""
Augment data/calibration/bias.parquet with the linear (slope) calibration columns
  slope_a, slope_b, slope_std_f
for every station — network-free, from already-cached data.

Reads the cached historical forecasts (data/calibration/forecasts/{st}/{year}.parquet)
and NWS labels (data/raw/labels/{st}/{year}.parquet), fits the SAME per-month OLS
used in the out-of-sample validation (kalshi_weather.calibration.bias._fit_monthly_slope),
and writes the three columns back WITHOUT modifying any existing column.

Only KSFO/KSEA actually USE the slope at runtime (bias.SLOPE_STATIONS); the columns
are stored for all stations for transparency and future inspection. Re-runnable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_weather.calibration.bias import (
    CALIB_YEARS, SLOPE_STATIONS, _fit_monthly_slope,
)

ROOT = Path(__file__).parents[1]
FCST = ROOT / "data" / "calibration" / "forecasts"
LABEL = ROOT / "data" / "raw" / "labels"
BIAS = ROOT / "data" / "calibration" / "bias.parquet"


def load_merged(station: str) -> pd.DataFrame | None:
    ff = [FCST / station / f"{y}.parquet" for y in CALIB_YEARS]
    lf = [LABEL / station / f"{y}.parquet" for y in CALIB_YEARS]
    ff = [p for p in ff if p.exists()]
    lf = [p for p in lf if p.exists()]
    if not ff or not lf:
        return None
    fc = pd.concat([pd.read_parquet(p) for p in ff], ignore_index=True)
    lb = pd.concat([pd.read_parquet(p) for p in lf], ignore_index=True)
    fc["date"] = pd.to_datetime(fc["date"])
    lb["date"] = pd.to_datetime(lb["date"])
    m = (fc[["date", "gfs_h"]]
         .merge(lb[["date", "high"]], on="date", how="inner")
         .dropna(subset=["gfs_h", "high"]))
    if m.empty:
        return None
    m["high"] = m["high"].astype(float)
    m["gfs_h"] = m["gfs_h"].astype(float)
    m["month"] = m["date"].dt.month
    return m


def main() -> None:
    bias = pd.read_parquet(BIAS)
    orig_cols = list(bias.columns)
    print(f"Loaded bias.parquet: {len(bias)} rows, columns {orig_cols}")

    sa: dict[tuple[str, int], float] = {}
    sb: dict[tuple[str, int], float] = {}
    ss: dict[tuple[str, int], float] = {}
    for st in sorted(bias["station"].unique()):
        m = load_merged(st)
        if m is None:
            print(f"  {st}: no cached data — leaving slope blank")
            continue
        for mo, (a, b, s) in _fit_monthly_slope(m).items():
            sa[(st, mo)] = round(a, 4)
            sb[(st, mo)] = round(b, 4)
            ss[(st, mo)] = round(s, 3)

    keys = list(zip(bias["station"].astype(str), bias["month"].astype(int)))
    bias["slope_a"] = [sa.get(k, np.nan) for k in keys]
    bias["slope_b"] = [sb.get(k, np.nan) for k in keys]
    bias["slope_std_f"] = [ss.get(k, np.nan) for k in keys]

    # Guarantee we did not perturb any pre-existing column.
    assert list(bias.columns)[: len(orig_cols)] == orig_cols, "existing columns reordered!"

    bias.to_parquet(BIAS, index=False)
    print(f"Wrote slope columns → {BIAS}")
    print(f"Active slope stations (bias.SLOPE_STATIONS): {sorted(SLOPE_STATIONS)}\n")
    for st in sorted(SLOPE_STATIONS):
        sub = bias[bias.station == st][
            ["month", "bias_f", "std_f", "slope_a", "slope_b", "slope_std_f"]
        ].sort_values("month")
        print(f"{st}  (slope < 1 = shrink warm forecasts toward the seasonal mean)")
        print(sub.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
