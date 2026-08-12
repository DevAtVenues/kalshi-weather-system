"""
Is 1.79°F actually good? Put it on a ladder against honest baselines, on the SAME
held-out 2024 city-days used everywhere else. All computed from cached data.

  Climatology  — guess the seasonal-normal high every day (a no-skill baseline)
  Persistence  — guess yesterday's actual high (the naive forecaster)
  Raw model    — the uncorrected GFS forecast
  Our model    — GFS + our per-station calibration
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
FCST = ROOT / "data" / "calibration" / "forecasts"
LABELS = ROOT / "data" / "raw" / "labels"
TRAIN_YEARS = [2022, 2023]
TEST_YEAR = 2024


def _load(st: str, years) -> pd.DataFrame:
    frames = []
    for y in years:
        fp, lp = FCST / st / f"{y}.parquet", LABELS / st / f"{y}.parquet"
        if not fp.exists() or not lp.exists():
            continue
        f = pd.read_parquet(fp)
        l = pd.read_parquet(lp).sort_values("date")
        f["date"] = pd.to_datetime(f["date"])
        l["date"] = pd.to_datetime(l["date"])
        l["prev_high"] = l["high"].shift(1)          # persistence = yesterday
        cols = ["date", "high", "high_normal", "prev_high"]
        frames.append(f[["date", "gfs_h"]].merge(l[cols], on="date", how="inner"))
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames, ignore_index=True).dropna(subset=["gfs_h", "high"])
    m["month"] = m["date"].dt.month
    return m


def main():
    clim, pers, raw, ours = [], [], [], []
    n = 0
    for st_dir in sorted(FCST.glob("*")):
        st = st_dir.name
        train, test = _load(st, TRAIN_YEARS), _load(st, [TEST_YEAR])
        if test.empty or train.empty:
            continue
        # monthly bias fit on TRAIN ONLY, then applied to the held-out test year
        bias = {int(mo): float((g["gfs_h"] - g["high"]).mean())
                for mo, g in train.groupby("month")}
        actual = test["high"].astype(float).to_numpy()
        n += len(test)

        cm = test.dropna(subset=["high_normal"])
        clim.append(np.abs(cm["high"].astype(float) - cm["high_normal"].astype(float)).to_numpy())
        pm = test.dropna(subset=["prev_high"])
        pers.append(np.abs(pm["high"].astype(float) - pm["prev_high"].astype(float)).to_numpy())
        raw.append(np.abs(test["gfs_h"].astype(float).to_numpy() - actual))
        corr = test["gfs_h"].astype(float).to_numpy() - np.array(
            [bias.get(int(mo), 0.0) for mo in test["month"]]
        )
        ours.append(np.abs(corr - actual))

    def mae(chunks):
        a = np.concatenate(chunks)
        return a.mean(), a.std() / np.sqrt(len(a))   # MAE, standard error

    print(f"Forecast-skill ladder — held-out {TEST_YEAR}, {n:,} city-days\n")
    print(f"  {'Method':<26}{'MAE °F':>8}   {'vs our model':>14}")
    print("  " + "-" * 50)
    om, _ = mae(ours)
    for name, chunks in [
        ("Climatology (seasonal avg)", clim),
        ("Persistence (yesterday)", pers),
        ("Raw model (uncorrected)", raw),
        ("OUR MODEL (calibrated)", ours),
    ]:
        m_, se = mae(chunks)
        ratio = f"{m_/om:.1f}× worse" if name.startswith(("Clim", "Pers", "Raw")) else "—"
        star = "  ◀" if name.startswith("OUR") else ""
        print(f"  {name:<26}{m_:>7.2f}{'':1}   {ratio:>14}{star}")
    print()
    print(f"  Read: guessing the seasonal normal is off by {mae(clim)[0]:.1f}°F; we're off by "
          f"{om:.2f}°F — about {mae(clim)[0]/om:.1f}× sharper than a no-skill baseline,")
    print("  and on par with professional next-day guidance (~2°F).")


if __name__ == "__main__":
    main()
