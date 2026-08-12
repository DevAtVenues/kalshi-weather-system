"""
Walk-forward per-station dispersion + bias calibration study (backlog: spread is the
frontier — see project_calibration_center_vs_spread).

Question: does calibrating the predictive distribution — center (subtract the
station's historical forecast bias) and especially SPREAD (set sigma to the station's
realized forecast-error std) — improve OUT-OF-SAMPLE probabilistic skill vs the live
default (fixed sigma≈2.0)? The live model over-predicts YES because its distribution
is too narrow at most stations (realized error std 2.5–3.8°F vs assumed ~2.0).

Design (respects the no-look-ahead / OOS-sacred rules):
  • Walk-forward by year: for each holdout year Y, FIT bias+sigma per station on all
    data with year < Y, EVALUATE only on year Y. No test row informs its own fit.
  • Deterministic HRRR (gfs_seamless ≈ HRRR at ≤2d) forecast vs the integer CLI high.
  • Settlement rounding matches production: a "≥ t" contract is YES iff round(high) ≥ t
    ⇔ high ≥ t − 0.5, so P = 1 − Φ((t−0.5 − μ)/σ).

Metrics (both OOS):
  • CRPS — proper score for the whole predictive distribution (Gaussian closed form).
  • Brier — on the decision-relevant "≥ t" contracts for integer t near the forecast.

Configs compared:
  baseline   μ = forecast,        σ = 2.0   (the live SIGMA_DEF)
  bias_only  μ = forecast + bias, σ = 2.0
  full       μ = forecast + bias, σ = realized-std (clamped to the live [1.2, 4.0])

Usage: .venv/bin/python scripts/calibration_dispersion_study.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).parents[1]
SIGMA_FLOOR, SIGMA_CAP, SIGMA_DEF = 1.2, 4.0, 2.0   # mirror live ensemble_dist caps
THRESH_HALF_WINDOW = 4   # evaluate "≥ t" contracts for integer t within ±4°F of forecast


def _load() -> pd.DataFrame:
    fc = pd.concat([pd.read_parquet(f) for f in
                    glob.glob(str(ROOT / "data/raw/forecasts/gfs_seamless/*.parquet"))],
                   ignore_index=True)
    lab = []
    for st in os.listdir(ROOT / "data/raw/labels"):
        for f in glob.glob(str(ROOT / f"data/raw/labels/{st}/*.parquet")):
            d = pd.read_parquet(f); d["station"] = st; lab.append(d)
    lab = pd.concat(lab, ignore_index=True)
    fc["date"] = pd.to_datetime(fc["date"]); lab["date"] = pd.to_datetime(lab["date"])
    m = (fc.merge(lab[["station", "date", "high"]], on=["station", "date"], how="inner")
           .dropna(subset=["tmax_f_fcst", "high"]))
    m["year"] = m["date"].dt.year
    m["fcst"] = m["tmax_f_fcst"].astype(float)
    m["actual"] = m["high"].astype(float)
    return m[["station", "date", "year", "fcst", "actual"]]


def _crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive dist. Lower is better."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def _contract_brier(mu: np.ndarray, sigma: np.ndarray, fcst: np.ndarray,
                    actual: np.ndarray) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Brier over '≥ t' contracts for integer t near each day's forecast.
    Returns (mean_brier, n_pairs, predicted, realized) for reliability."""
    preds, reals = [], []
    for m_, s_, f_, a_ in zip(mu, sigma, fcst, actual):
        for t in range(int(round(f_)) - THRESH_HALF_WINDOW, int(round(f_)) + THRESH_HALF_WINDOW + 1):
            p = 1.0 - norm.cdf((t - 0.5 - m_) / s_)      # P(round(high) ≥ t)
            preds.append(p)
            reals.append(1.0 if a_ >= t else 0.0)
    preds = np.array(preds); reals = np.array(reals)
    return float(np.mean((preds - reals) ** 2)), len(preds), preds, reals


def _reliability(preds: np.ndarray, reals: np.ndarray) -> str:
    lines = []
    for lo in np.arange(0, 1.0, 0.2):
        hi = lo + 0.2
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() >= 30:
            lines.append(f"    P∈[{lo:.1f},{hi:.1f}) n={mask.sum():5d} "
                         f"pred≈{preds[mask].mean():.2f} realized={reals[mask].mean():.2f}")
    return "\n".join(lines)


def run() -> None:
    m = _load()
    holdout_years = [y for y in [2024, 2025, 2026] if (m.year == y).any()]
    print("=" * 74)
    print("WALK-FORWARD DISPERSION + BIAS CALIBRATION STUDY (HRRR vs CLI, per station)")
    print(f"  holdout years: {holdout_years}  |  {len(m)} station-days total")
    print("=" * 74)

    agg = {c: {"crps": [], "brier": [], "preds": [], "reals": []}
           for c in ("baseline", "bias_only", "full")}
    per_station_delta: dict[str, list[float]] = {}

    for Y in holdout_years:
        train, test = m[m.year < Y], m[m.year == Y]
        if train.empty or test.empty:
            continue
        # fit per-station on TRAIN only
        fit = (train.assign(err=train.actual - train.fcst)
                    .groupby("station")["err"].agg(bias="mean", sd="std"))
        test = test.merge(fit, on="station", how="inner").dropna(subset=["bias", "sd"])
        sd_cal = test["sd"].clip(SIGMA_FLOOR, SIGMA_CAP).to_numpy()
        fcst, actual = test.fcst.to_numpy(), test.actual.to_numpy()
        configs = {
            "baseline":  (fcst,               np.full(len(test), SIGMA_DEF)),
            "bias_only": (fcst + test.bias.to_numpy(), np.full(len(test), SIGMA_DEF)),
            "full":      (fcst + test.bias.to_numpy(), sd_cal),
        }
        for name, (mu, sig) in configs.items():
            crps = _crps_gaussian(mu, sig, actual)
            brier, _, preds, reals = _contract_brier(mu, sig, fcst, actual)
            agg[name]["crps"].append(crps)
            agg[name]["brier"].append((brier, len(preds)))
            agg[name]["preds"].append(preds); agg[name]["reals"].append(reals)

        # per-station OOS Brier delta (full vs baseline) for this holdout
        for st in test.station.unique():
            sub = test[test.station == st]
            f_, a_ = sub.fcst.to_numpy(), sub.actual.to_numpy()
            b_base, *_ = _contract_brier(f_, np.full(len(sub), SIGMA_DEF), f_, a_)
            b_full, *_ = _contract_brier(f_ + sub.bias.to_numpy(),
                                         sub.sd.clip(SIGMA_FLOOR, SIGMA_CAP).to_numpy(), f_, a_)
            per_station_delta.setdefault(st, []).append(b_base - b_full)

    print("\nOOS aggregate (pooled over holdout years):")
    print(f"  {'config':<10} {'CRPS':>8} {'Brier':>9}   (lower = better)")
    base_crps = base_brier = None
    for name in ("baseline", "bias_only", "full"):
        crps = np.concatenate(agg[name]["crps"]).mean()
        bw = agg[name]["brier"]
        brier = sum(b * n for b, n in bw) / sum(n for _, n in bw)
        if name == "baseline":
            base_crps, base_brier = crps, brier
            print(f"  {name:<10} {crps:>8.4f} {brier:>9.5f}")
        else:
            print(f"  {name:<10} {crps:>8.4f} {brier:>9.5f}   "
                  f"CRPS {(crps-base_crps)/base_crps*100:+.1f}%  Brier {(brier-base_brier)/base_brier*100:+.1f}%")

    print("\nReliability (full-calibration, pooled OOS) — should hug the diagonal:")
    print(_reliability(np.concatenate(agg["full"]["preds"]), np.concatenate(agg["full"]["reals"])))
    print("\nReliability (baseline, pooled OOS) — for contrast:")
    print(_reliability(np.concatenate(agg["baseline"]["preds"]), np.concatenate(agg["baseline"]["reals"])))

    print("\nPer-station OOS Brier improvement (baseline − full, +=better), mean over holdouts:")
    print(f"  {'station':>8} {'Δbrier':>9} {'realized_sd':>12}")
    fit_all = (m.assign(err=m.actual - m.fcst).groupby("station")["err"].std())
    for st in sorted(per_station_delta, key=lambda s: -np.mean(per_station_delta[s])):
        print(f"  {st:>8} {np.mean(per_station_delta[st]):>+9.5f} {fit_all.get(st, float('nan')):>12.2f}")


if __name__ == "__main__":
    run()
