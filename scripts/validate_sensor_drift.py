"""
Walk-forward, point-in-time validation of the recent-drift correction.

For each station and each recent test day d, estimate the drift using ONLY days
strictly before d, then compare the residual of the current system
(climatology-corrected forecast) against the proposed system
(climatology + recent drift). No look-ahead: the drift for day d never sees d.

Decision rule (per station): enable the drift only where it reduces both
out-of-sample MAE and |mean residual| by a non-trivial margin. "No improvement"
is a valid outcome and the station simply keeps climatology-only.

    .venv/bin/python scripts/validate_sensor_drift.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd

from kalshi_weather.calibration.bias import load_bias_table
from kalshi_weather.calibration.sensor_drift import (
    estimate_drift, _fetch_recent_forecasts, WINDOW_DAYS, LAG_DAYS,
)
from kalshi_weather.ingest.labels import fetch_labels

STATIONS = {
    "KNYC": (40.7789, -73.9692), "KMIA": (25.7959, -80.2870),
    "KPHL": (39.8719, -75.2411), "KAUS": (30.1945, -97.6699),
    "KMDW": (41.7860, -87.7522),
}
TEST_DAYS = 60   # how many recent days to score


def main() -> None:
    bt = load_bias_table()
    today = date.today()
    test_end   = today - timedelta(days=LAG_DAYS)
    test_start = test_end - timedelta(days=TEST_DAYS - 1)
    # widest range we need: earliest test day needs WINDOW_DAYS before it
    fetch_start = test_start - timedelta(days=WINDOW_DAYS + LAG_DAYS + 3)

    print(f"Walk-forward drift validation | test {test_start}..{test_end} "
          f"({TEST_DAYS}d), window {WINDOW_DAYS}d, lag {LAG_DAYS}d\n")
    print(f"{'stn':5} {'n':>3} {'%appl':>6} {'mean|drift|':>11} "
          f"{'MAE_clim':>9} {'MAE_+drift':>10} {'bias_clim':>9} {'bias_+drift':>11}  verdict")

    for stn, (lat, lon) in STATIONS.items():
        clim_entry_exists = any(m in bt.get(stn, {}) for m in range(1, 13))
        fc = _fetch_recent_forecasts(lat, lon, fetch_start, test_end)
        lb = fetch_labels(stn, sorted({fetch_start.year, test_end.year}))
        lb = lb.copy(); lb["date"] = pd.to_datetime(lb["date"]).dt.date
        act = dict(zip(lb["date"], pd.to_numeric(lb["high"], errors="coerce")))
        fcd = fc.copy(); fcd["date"] = pd.to_datetime(fcd["date"]).dt.date
        fch = dict(zip(fcd["date"], pd.to_numeric(fcd["gfs_h"], errors="coerce")))

        rows = []
        d = test_start
        while d <= test_end:
            f, a = fch.get(d), act.get(d)
            clim = bt.get(stn, {}).get(d.month)
            if f is not None and a is not None and clim is not None and np.isfinite(f) and np.isfinite(a):
                est = estimate_drift(stn, lat, lon, d, bt, forecasts=fc, labels=lb)
                clim_fc = f - clim["bias_f"]
                rows.append({
                    "before": clim_fc - a,                 # current system residual
                    "after":  clim_fc - est.drift_f - a,   # proposed residual
                    "drift":  est.drift_f, "applied": est.applied,
                })
            d += timedelta(days=1)

        if not rows:
            print(f"{stn:5}   --  (no climatological anchor / no data)")
            continue
        df = pd.DataFrame(rows)
        pct = 100.0 * df["applied"].mean()
        mad = df.loc[df["applied"], "drift"].abs().mean() if df["applied"].any() else 0.0
        mae_b, mae_a = df["before"].abs().mean(), df["after"].abs().mean()
        bias_b, bias_a = df["before"].mean(), df["after"].mean()
        better = (mae_a < mae_b - 0.05) and (abs(bias_a) < abs(bias_b) - 0.05)
        verdict = "ENABLE" if better else ("worse" if mae_a > mae_b + 0.05 else "no-gain")
        print(f"{stn:5} {len(df):>3} {pct:>5.0f}% {mad:>11.2f} "
              f"{mae_b:>9.2f} {mae_a:>10.2f} {bias_b:>+9.2f} {bias_a:>+11.2f}  {verdict}")


if __name__ == "__main__":
    main()
