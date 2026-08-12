"""
Spread recalibration test — the real lever.

The center analysis showed the forecast→actual slope is ~1.0 for 18/20 cities
(a flat monthly offset is enough), but PIT coverage showed the predictive SPREAD
is miscalibrated: too wide for most interior cities (underconfident), too narrow
for Miami (overconfident).

This tests the fix: scale each station's residual sigma by a factor k fit ONLY on
the training years, where k makes the standardized residuals have unit variance:
    k = sqrt( mean( ((actual - mu) / sigma)^2 ) )   on 2022-2023
Then apply N(mu, k*sigma) unchanged to held-out 2024 and re-score.

k < 1  → the model was too uncertain; tightening reveals more real edge.
k > 1  → the model was overconfident; widening protects against oversized bets.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parents[1]
FCST_DIR = ROOT / "data" / "calibration" / "forecasts"
LABEL_DIR = ROOT / "data" / "raw" / "labels"
TRAIN, TEST = [2022, 2023], 2024

CITY = {
    "KNYC": "NYC", "KORD": "Chicago", "KPHL": "Philadelphia", "KSEA": "Seattle",
    "KSFO": "SFO", "KLAX": "LAX", "KDFW": "Dallas", "KHOU": "Houston",
    "KATL": "Atlanta", "KMIA": "Miami", "KBOS": "Boston", "KDCA": "DC",
    "KDEN": "Denver", "KPHX": "Phoenix", "KLAS": "Las Vegas", "KMSP": "Minneapolis",
    "KOKC": "OKC", "KSAT": "San Antonio", "KMSY": "New Orleans", "KAUS": "Austin",
}


def load_station(st):
    ff = sorted(glob.glob(str(FCST_DIR / st / "*.parquet")))
    lf = sorted(glob.glob(str(LABEL_DIR / st / "*.parquet")))
    if not ff or not lf:
        return None
    f = pd.concat([pd.read_parquet(x) for x in ff], ignore_index=True)
    l = pd.concat([pd.read_parquet(x) for x in lf], ignore_index=True)
    f["date"] = pd.to_datetime(f["date"]); l["date"] = pd.to_datetime(l["date"])
    m = f[["date", "gfs_h"]].merge(l[["date", "high"]], on="date", how="inner").dropna()
    m["actual"] = m["high"].astype(float); m["gfs"] = m["gfs_h"].astype(float)
    m["month"] = m["date"].dt.month; m["year"] = m["date"].dt.year
    return m


def gaussian_crps(mu, sigma, y):
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return sigma * (z * (2*stats.norm.cdf(z)-1) + 2*stats.norm.pdf(z) - 1/np.sqrt(np.pi))


def cov(mu, sigma, y, level):
    zc = stats.norm.ppf(0.5 + level/2)
    return float(np.mean((y >= mu-zc*sigma) & (y <= mu+zc*sigma)))


def main():
    stations = sorted(d for d in CITY if (FCST_DIR / d).exists())
    rows = []
    P = {"y": [], "mu": [], "sd": [], "sdk": []}

    for st in stations:
        df = load_station(st)
        if df is None:
            continue
        tr, te = df[df.year.isin(TRAIN)], df[df.year == TEST]
        if len(tr) < 200 or len(te) < 100:
            continue

        # M2 center + monthly sigma, fit on train
        mb = {int(mo): ((g.gfs-g.actual).mean(), (g.gfs-g.actual).std(ddof=1))
              for mo, g in tr.groupby("month")}

        def apply(dset):
            mu = np.array([r.gfs - mb[int(r.month)][0] for r in dset.itertuples()])
            sd = np.array([mb[int(r.month)][1] for r in dset.itertuples()])
            return mu, sd

        mu_tr, sd_tr = apply(tr)
        z_tr = (tr.actual.to_numpy() - mu_tr) / sd_tr
        k = float(np.sqrt(np.mean(z_tr**2)))          # variance-matching factor (train only)

        mu, sd = apply(te)
        y = te.actual.to_numpy()
        sdk = k * sd

        rows.append({
            "city": CITY[st], "k": k,
            "crps_base": gaussian_crps(mu, sd, y).mean(),
            "crps_fix":  gaussian_crps(mu, sdk, y).mean(),
            "cov50_base": cov(mu, sd, y, .50), "cov50_fix": cov(mu, sdk, y, .50),
        })
        P["y"].append(y); P["mu"].append(mu); P["sd"].append(sd); P["sdk"].append(sdk)

    R = pd.DataFrame(rows).sort_values("k")
    Y = np.concatenate(P["y"]); MU = np.concatenate(P["mu"])
    SD = np.concatenate(P["sd"]); SDK = np.concatenate(P["sdk"])

    print("=" * 74)
    print("SPREAD RECALIBRATION — held-out 2024 (variance factor k fit on 2022-23)")
    print("=" * 74)
    print(f"{'City':<13}{'k':>6}{'  CRPS base→fix':>18}{'   50% cov base→fix':>20}")
    print("-" * 74)
    for _, r in R.iterrows():
        d = r.crps_base - r.crps_fix
        mark = "✓" if d > 0.005 else ("·" if d > -0.005 else "✗")
        print(f"{r.city:<13}{r.k:>6.2f}   {r.crps_base:>5.2f} → {r.crps_fix:<5.2f}"
              f"     {r.cov50_base:>4.2f} → {r.cov50_fix:<4.2f}   {mark}")
    print("-" * 74)
    cb = gaussian_crps(MU, SD, Y).mean(); cf = gaussian_crps(MU, SDK, Y).mean()
    print(f"{'POOLED':<13}{'':>6}   {cb:>5.2f} → {cf:<5.2f}"
          f"     {cov(MU,SD,Y,.50):>4.2f} → {cov(MU,SDK,Y,.50):<4.2f}")
    print(f"\nPooled CRPS improvement: {100*(cb-cf)/cb:+.1f}%   (ideal 50% coverage = 0.50)")
    print(f"Stations improved: {(R.crps_base > R.crps_fix + 0.005).sum()}/{len(R)}   "
          f"| k<1 (was too wide): {(R.k < 0.97).sum()}   k>1 (too narrow): {(R.k > 1.03).sum()}")


if __name__ == "__main__":
    main()
