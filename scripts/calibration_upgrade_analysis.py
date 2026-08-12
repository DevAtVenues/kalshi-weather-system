"""
Calibration upgrade analysis — is the forecast error in the CENTER or the SPREAD?

Compares three calibration models per station, with STRICT out-of-sample discipline
(HARD RULE 4): fit on 2022-2023, evaluate ONLY on held-out 2024.

  M0  Raw GFS                       — no correction (baseline)
  M2  Monthly mean subtraction      — CURRENT PRODUCTION MODEL
  M3  OLS linear  a + b*GFS         — corrects center AND slope (per-month)

For the pricing distribution we output N(mu, sigma). We therefore score the full
predictive distribution with CRPS (proper scoring rule), not just point MAE, and
check calibration of the SPREAD via PIT interval coverage.

Answers three questions with real numbers:
  Q1  Is there conditional bias? (does slope b differ from 1?)  -> center problem
  Q2  Is the ensemble/residual spread right? (PIT coverage)     -> spread problem
  Q3  Does the upgrade actually help OUT OF SAMPLE? (MAE, CRPS on 2024)

Deterministic GFS archive only (data/calibration/forecasts). No look-ahead:
every parameter is fit on train years and applied unchanged to the test year.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parents[1]
FCST_DIR = ROOT / "data" / "calibration" / "forecasts"
LABEL_DIR = ROOT / "data" / "raw" / "labels"

TRAIN_YEARS = [2022, 2023]
TEST_YEAR = 2024

CITY = {
    "KNYC": "NYC", "KORD": "Chicago", "KPHL": "Philadelphia", "KSEA": "Seattle",
    "KSFO": "SFO", "KLAX": "LAX", "KDFW": "Dallas", "KHOU": "Houston",
    "KATL": "Atlanta", "KMIA": "Miami", "KBOS": "Boston", "KDCA": "DC",
    "KDEN": "Denver", "KPHX": "Phoenix", "KLAS": "Las Vegas", "KMSP": "Minneapolis",
    "KOKC": "OKC", "KSAT": "San Antonio", "KMSY": "New Orleans", "KAUS": "Austin",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_station(station: str) -> pd.DataFrame | None:
    """Return matched (date, month, year, gfs, actual) for one station, all years."""
    fcst_files = sorted(glob.glob(str(FCST_DIR / station / "*.parquet")))
    label_files = sorted(glob.glob(str(LABEL_DIR / station / "*.parquet")))
    if not fcst_files or not label_files:
        return None

    fcst = pd.concat([pd.read_parquet(f) for f in fcst_files], ignore_index=True)
    labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)

    fcst["date"] = pd.to_datetime(fcst["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    m = (fcst[["date", "gfs_h"]]
         .merge(labels[["date", "high"]], on="date", how="inner")
         .dropna(subset=["gfs_h", "high"]))
    m["actual"] = m["high"].astype(float)
    m["gfs"] = m["gfs_h"].astype(float)
    m["month"] = m["date"].dt.month
    m["year"] = m["date"].dt.year
    return m[["date", "month", "year", "gfs", "actual"]].reset_index(drop=True)


# ── Scoring ───────────────────────────────────────────────────────────────────

def gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive (Gneiting & Raftery 2007). Lower=better."""
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1)
                    + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


# ── Model fitting (train only) ────────────────────────────────────────────────

def fit_monthly_mean(train: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """M2 (current production): per-month (bias, residual_std)."""
    out = {}
    for mo, g in train.groupby("month"):
        err = g["gfs"] - g["actual"]
        out[int(mo)] = (float(err.mean()), float(err.std(ddof=1)))
    return out


def fit_monthly_ols(train: pd.DataFrame, min_n: int = 25) -> dict[int, tuple[float, float, float]]:
    """M3: per-month OLS actual = a + b*gfs; returns (a, b, residual_std).
    Falls back to a global fit for thin months."""
    # global fallback
    gx, gy = train["gfs"].to_numpy(), train["actual"].to_numpy()
    gb, ga = np.polyfit(gx, gy, 1)  # slope, intercept
    g_resid = gy - (ga + gb * gx)
    g_fit = (float(ga), float(gb), float(g_resid.std(ddof=1)))

    out = {}
    for mo, g in train.groupby("month"):
        if len(g) < min_n:
            out[int(mo)] = g_fit
            continue
        x, y = g["gfs"].to_numpy(), g["actual"].to_numpy()
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        out[int(mo)] = (float(a), float(b), float(resid.std(ddof=1)))
    return out


# ── Prediction (test) ─────────────────────────────────────────────────────────

def predict(test: pd.DataFrame, m2, m3):
    n = len(test)
    mu0 = test["gfs"].to_numpy()                       # M0 raw
    mu2 = np.empty(n); sd2 = np.empty(n)               # M2 monthly-mean
    mu3 = np.empty(n); sd3 = np.empty(n)               # M3 monthly-OLS
    for i, (_, r) in enumerate(test.iterrows()):
        mo = int(r["month"])
        bias, std = m2[mo]
        mu2[i] = r["gfs"] - bias
        sd2[i] = std
        a, b, rstd = m3[mo]
        mu3[i] = a + b * r["gfs"]
        sd3[i] = rstd
    return mu0, (mu2, sd2), (mu3, sd3)


def coverage(mu, sigma, y, level):
    """Fraction of obs inside the central `level` predictive interval."""
    zc = stats.norm.ppf(0.5 + level / 2)
    lo, hi = mu - zc * sigma, mu + zc * sigma
    return float(np.mean((y >= lo) & (y <= hi)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    stations = sorted(d for d in CITY if (FCST_DIR / d).exists())
    rows = []
    slope_notes = []
    pooled = {"y": [], "m0": [], "m2mu": [], "m2sd": [], "m3mu": [], "m3sd": []}

    for st in stations:
        df = load_station(st)
        if df is None or df.empty:
            continue
        train = df[df["year"].isin(TRAIN_YEARS)]
        test = df[df["year"] == TEST_YEAR]
        if len(train) < 200 or len(test) < 100:
            continue

        m2 = fit_monthly_mean(train)
        m3 = fit_monthly_ols(train)
        mu0, (mu2, sd2), (mu3, sd3) = predict(test, m2, m3)
        y = test["actual"].to_numpy()

        mae0 = np.mean(np.abs(mu0 - y))
        mae2 = np.mean(np.abs(mu2 - y))
        mae3 = np.mean(np.abs(mu3 - y))
        crps2 = gaussian_crps(mu2, sd2, y).mean()
        crps3 = gaussian_crps(mu3, sd3, y).mean()

        # pooled global slope (for the Q1 conditional-bias read)
        xtr, ytr = train["gfs"].to_numpy(), train["actual"].to_numpy()
        b_glob, a_glob = np.polyfit(xtr, ytr, 1)
        # slope std error
        n = len(xtr)
        resid = ytr - (a_glob + b_glob * xtr)
        se_b = np.sqrt((resid.var(ddof=2)) / np.sum((xtr - xtr.mean()) ** 2))

        cov80_2 = coverage(mu2, sd2, y, 0.80)
        cov80_3 = coverage(mu3, sd3, y, 0.80)

        rows.append({
            "city": CITY[st], "station": st, "n_test": len(test),
            "mae0": mae0, "mae2": mae2, "mae3": mae3,
            "crps2": crps2, "crps3": crps3,
            "slope": b_glob, "se_b": se_b,
            "cov80_m2": cov80_2, "cov80_m3": cov80_3,
        })
        slope_notes.append((CITY[st], b_glob, se_b))
        for k, v in [("y", y), ("m0", mu0), ("m2mu", mu2), ("m2sd", sd2),
                     ("m3mu", mu3), ("m3sd", sd3)]:
            pooled[k].append(v)

    R = pd.DataFrame(rows).sort_values("mae2", ascending=False)

    print("=" * 78)
    print("CALIBRATION UPGRADE — OUT-OF-SAMPLE TEST (train 2022-23, test 2024 held out)")
    print("=" * 78)
    print(f"{len(R)} stations · {R.n_test.sum():,} held-out test days\n")

    # Q3 — does the upgrade help? Point accuracy + full-distribution score
    print("Q3. OUT-OF-SAMPLE ACCURACY  (MAE °F; lower better)")
    print(f"  {'City':<13}{'raw':>7}{'monthly':>9}{'OLS':>7}   {'Δ MAE':>7}   {'CRPS↓ mo→OLS':>16}")
    print("  " + "-" * 66)
    for _, r in R.iterrows():
        d = r.mae2 - r.mae3
        mark = "  ✓" if d > 0.02 else ("  ·" if abs(d) <= 0.02 else "  ✗")
        print(f"  {r.city:<13}{r.mae0:>7.2f}{r.mae2:>9.2f}{r.mae3:>7.2f}   "
              f"{d:>+7.2f}   {r.crps2:>6.2f} → {r.crps3:<5.2f}{mark}")

    # aggregate pooled
    Y = np.concatenate(pooled["y"])
    M0 = np.concatenate(pooled["m0"])
    M2mu, M2sd = np.concatenate(pooled["m2mu"]), np.concatenate(pooled["m2sd"])
    M3mu, M3sd = np.concatenate(pooled["m3mu"]), np.concatenate(pooled["m3sd"])
    print("  " + "-" * 66)
    print(f"  {'POOLED':<13}{np.mean(np.abs(M0-Y)):>7.2f}"
          f"{np.mean(np.abs(M2mu-Y)):>9.2f}{np.mean(np.abs(M3mu-Y)):>7.2f}   "
          f"{np.mean(np.abs(M2mu-Y))-np.mean(np.abs(M3mu-Y)):>+7.2f}   "
          f"{gaussian_crps(M2mu,M2sd,Y).mean():>6.2f} → {gaussian_crps(M3mu,M3sd,Y).mean():<5.2f}")

    # Q1 — conditional bias: slope != 1?
    print("\nQ1. CONDITIONAL BIAS  (forecast→actual slope; =1.0 means flat offset is enough)")
    print(f"  {'City':<13}{'slope b':>9}{'±2se':>8}   verdict")
    print("  " + "-" * 52)
    for _, r in R.sort_values("slope").iterrows():
        far = abs(r.slope - 1.0) > 2 * r.se_b and abs(r.slope - 1.0) > 0.05
        v = "slope ≠ 1 → shrink extremes" if far else "≈ flat offset ok"
        print(f"  {r.city:<13}{r.slope:>9.3f}{2*r.se_b:>8.3f}   {v}")

    # Q2 — spread calibration via 80% coverage (ideal 0.80)
    print("\nQ2. SPREAD CALIBRATION  (80% interval coverage on test; ideal ≈ 0.80)")
    print(f"  {'City':<13}{'monthly':>9}{'OLS':>8}   read")
    print("  " + "-" * 50)
    for _, r in R.sort_values("cov80_m2").iterrows():
        under = r.cov80_m2 < 0.72
        over = r.cov80_m2 > 0.88
        read = "too NARROW (underconfident σ)" if under else ("too WIDE" if over else "ok")
        print(f"  {r.city:<13}{r.cov80_m2:>9.2f}{r.cov80_m3:>8.2f}   {read}")
    print(f"\n  POOLED 80% coverage:  monthly {coverage(M2mu,M2sd,Y,0.80):.3f}   "
          f"OLS {coverage(M3mu,M3sd,Y,0.80):.3f}   (ideal 0.80)")
    print(f"  POOLED 50% coverage:  monthly {coverage(M2mu,M2sd,Y,0.50):.3f}   "
          f"OLS {coverage(M3mu,M3sd,Y,0.50):.3f}   (ideal 0.50)")

    R.to_csv(ROOT / "data" / "calibration" / "upgrade_analysis_2024.csv", index=False)
    print(f"\nSaved per-station results → data/calibration/upgrade_analysis_2024.csv")


if __name__ == "__main__":
    main()
