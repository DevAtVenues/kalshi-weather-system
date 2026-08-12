"""
EMOS spread-calibration readiness — is the LIVE ensemble's spread right yet?

The out-of-sample deterministic analysis said the CENTER of our forecast is fine
but the predictive SPREAD is miscalibrated (Miami overconfident). The proper fix is
EMOS: recalibrate  σ_corrected = c + d·σ_ensemble  on the REAL ensemble spread.
That needs the live-logged ensemble members (the API only retains them ~3 days), so
this harness measures spread calibration on what we've logged so far.

It is a MEASUREMENT, not a shippable change. With only a few weeks of autocorrelated
days the block-bootstrap CI on the spread factor is wide — this tells us WHERE the
ensemble is over/under-confident and starts the clock; it does not yet justify
changing live pricing (same discipline as the forward-validation finding).

Method, per (station, settlement-date D), strict no-look-ahead:
  • take the latest model run whose init_time < the start of D's LST day,
  • daily-max per member (per model, then pool GFS+ECMWF+ICON → one distribution),
  • compare to the actual NWS settlement high.
Center bias is removed per city (the live engine centers via recent_grid_bias), so the
spread factor  k = sqrt(mean(z²)),  z = (actual − ens_mean − city_bias) / ens_sd
isolates WIDTH:  k > 1 ⇒ ensemble too NARROW (overconfident);  k < 1 ⇒ too wide.
k is exactly the multiplicative EMOS coefficient d (with c = 0).
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parents[1]
ENS = ROOT / "data" / "logger" / "ensemble"
LABELS = ROOT / "data" / "raw" / "labels"
STATIONS = ROOT / "config" / "stations.yaml"
MODELS = ["gfs025", "ecmwf_ifs025", "icon_seamless"]
UTC = timezone.utc
_MIN_BLOCKS = 10          # distinct settlement days below which a CI is an artifact
_MIN_MEMBERS = 40         # need a real distribution to measure spread
_MIN_CITY_DAYS = 6        # per-city table floor

CITY = {
    "KNYC": "NYC", "KORD": "Chicago", "KPHL": "Philadelphia", "KSEA": "Seattle",
    "KSFO": "SFO", "KLAX": "LAX", "KDFW": "Dallas", "KHOU": "Houston",
    "KATL": "Atlanta", "KMIA": "Miami", "KBOS": "Boston", "KDCA": "DC",
    "KDEN": "Denver", "KPHX": "Phoenix", "KLAS": "Las Vegas", "KMSP": "Minneapolis",
    "KOKC": "OKC", "KSAT": "San Antonio", "KMSY": "New Orleans", "KAUS": "Austin",
}


def load_offsets() -> dict[str, int]:
    cfg = yaml.safe_load(STATIONS.read_text())
    off: dict[str, int] = {}
    for c in cfg.values():
        for k in ("nws_station", "metar_station"):
            if c.get(k):
                off[c[k]] = int(c["lst_offset_hours"])
    off.setdefault("KORD", -6)   # O'Hare logged for Chicago instead of KMDW
    return off


def _init_dt(name: str):
    m = re.match(r"(\d{8})_(\d{2})Z", name)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=UTC) if m else None


def list_inits(model: str) -> list[tuple[datetime, Path]]:
    return sorted(
        (dt, p) for p in (ENS / model).glob("*Z") if (dt := _init_dt(p.name))
    )


def load_labels() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for st in CITY:
        d: dict = {}
        for f in (LABELS / st).glob("*.parquet"):
            df = pd.read_parquet(f)
            for _, r in df.iterrows():
                d[str(pd.to_datetime(r["date"]).date())] = float(r["high"])
        out[st] = d
    return out


def member_daily_max(path: Path, win_start, win_end) -> np.ndarray | None:
    """Per-member daily max within the LST-day UTC window, for one model/init/station."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    vt = pd.to_datetime(df["valid_time"], utc=True)
    m = df[(vt >= win_start) & (vt < win_end)]
    if m.empty:
        return None
    return m.groupby("member")["temp_f"].max().to_numpy(dtype=float)


def build_rows(offsets, labels, inits_by_model) -> pd.DataFrame:
    rows = []
    # candidate settlement dates: union of logged init dates, minus the last day
    all_inits = [dt for m in MODELS for dt, _ in inits_by_model[m]]
    d0 = min(all_inits).date()
    d1 = max(all_inits).date()
    dates = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]

    for st in CITY:
        off = offsets.get(st)
        if off is None:
            continue
        for D in dates:
            # LST day D  →  UTC window [D 00:00 − off, +24h);  off is negative (e.g. −5)
            win_start = datetime(D.year, D.month, D.day, tzinfo=UTC) - timedelta(hours=off)
            win_end = win_start + timedelta(hours=24)
            actual = labels.get(st, {}).get(str(D))
            if actual is None:
                continue

            pooled, leads = [], []
            for model in MODELS:
                cand = [(dt, p) for dt, p in inits_by_model[model] if dt < win_start]
                if not cand:
                    continue
                idt, ipath = max(cand, key=lambda x: x[0])
                assert idt < win_start, "look-ahead!"            # HARD RULE 2
                arr = member_daily_max(ipath / f"{st}.parquet", win_start, win_end)
                if arr is not None and arr.size:
                    pooled.append(arr)
                    leads.append((win_start - idt).total_seconds() / 3600)
            if not pooled:
                continue
            members = np.concatenate(pooled)
            if members.size < _MIN_MEMBERS:
                continue

            rows.append({
                "station": st, "city": CITY[st], "date": str(D),
                "n_mem": int(members.size),
                "ens_mean": float(members.mean()),
                "ens_sd": float(members.std(ddof=1)),
                "p10": float(np.percentile(members, 10)),
                "p90": float(np.percentile(members, 90)),
                "actual": actual,
                "err": actual - float(members.mean()),   # +: actual warmer than ensemble
                "lead_h": float(np.median(leads)),
            })
    return pd.DataFrame(rows)


def block_bootstrap_k(df: pd.DataFrame, n_boot=3000, seed=0) -> tuple[float, float, float]:
    """CI on the pooled spread factor k, resampling whole settlement DATES (blocks)."""
    rng = np.random.default_rng(seed)
    by_date = defaultdict(list)
    for _, r in df.iterrows():
        by_date[r["date"]].append(r["z"])
    dates = list(by_date)
    ks = []
    for _ in range(n_boot):
        pick = rng.choice(len(dates), size=len(dates), replace=True)
        z = np.concatenate([by_date[dates[i]] for i in pick])
        ks.append(np.sqrt(np.mean(z ** 2)))
    return float(np.percentile(ks, 2.5)), float(np.median(ks)), float(np.percentile(ks, 97.5))


def main():
    offsets = load_offsets()
    labels = load_labels()
    inits_by_model = {m: list_inits(m) for m in MODELS}
    df = build_rows(offsets, labels, inits_by_model)
    if df.empty:
        print("No paired ensemble/settlement data found.")
        return

    # per-city center bias (proxy for the engine's recenter), then standardized residual z
    df["city_bias"] = df.groupby("station")["err"].transform("mean")
    df["z"] = (df["err"] - df["city_bias"]) / df["ens_sd"]
    df["cover80"] = ((df["actual"] >= df["p10"] + df["city_bias"]) &
                     (df["actual"] <= df["p90"] + df["city_bias"]))

    n_days = df["date"].nunique()
    print("=" * 76)
    print("EMOS SPREAD-CALIBRATION READINESS — live ensemble vs settlement")
    print("=" * 76)
    print(f"{len(df)} station-days · {n_days} distinct settlement dates "
          f"({df.date.min()} → {df.date.max()}) · median lead {df.lead_h.median():.0f} h")
    print(f"pooled ensemble ≈ {int(df.n_mem.median())} members (GFS+ECMWF+ICON)")
    if n_days < _MIN_BLOCKS:
        print(f"⚠  only {n_days} distinct days (< {_MIN_BLOCKS}) — MEASUREMENT ONLY, not shippable.")
    print()

    # per-city
    print("PER-CITY  (k>1 = ensemble too NARROW/overconfident;  ~1 = calibrated)")
    print(f"  {'City':<13}{'n':>4}{'bias':>7}{'ens σ':>7}{'real σ':>7}{'k':>6}{'80% cov':>9}")
    print("  " + "-" * 56)
    g = df.groupby("station")
    city_rows = []
    for st, sub in g:
        if len(sub) < _MIN_CITY_DAYS:
            continue
        k = float(np.sqrt(np.mean(sub["z"] ** 2)))
        real_sd = float((sub["err"] - sub["city_bias"]).std(ddof=1))
        city_rows.append((CITY[st], len(sub), sub["city_bias"].iloc[0],
                          sub["ens_sd"].mean(), real_sd, k, sub["cover80"].mean()))
    for city, n, bias, esd, rsd, k, cov in sorted(city_rows, key=lambda x: -x[5]):
        flag = "  NARROW" if k > 1.12 else ("  wide" if k < 0.88 else "")
        print(f"  {city:<13}{n:>4}{bias:>+7.1f}{esd:>7.1f}{rsd:>7.1f}{k:>6.2f}{cov:>8.0%}{flag}")

    # pooled with block-bootstrap CI
    k_all = float(np.sqrt(np.mean(df["z"] ** 2)))
    lo, med, hi = block_bootstrap_k(df)
    print("  " + "-" * 56)
    print(f"  {'POOLED':<13}{len(df):>4}{'':>7}{df.ens_sd.mean():>7.1f}"
          f"{(df.err-df.city_bias).std(ddof=1):>7.1f}{k_all:>6.2f}{df.cover80.mean():>8.0%}")
    print(f"\n  spread factor k (block-bootstrap over {n_days} days): "
          f"{med:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"  ideal k = 1.00 · ideal 80% coverage = 0.80")

    verdict = ("ensemble runs NARROW → live pricing overconfident → widen σ (EMOS d>1)"
               if med > 1.05 else
               "ensemble runs WIDE → underconfident" if med < 0.95 else
               "ensemble spread ≈ calibrated")
    print(f"\n  READ: {verdict}")
    if lo < 1.0 < hi:
        print("  BUT the CI spans 1.0 — cannot yet distinguish from calibrated. Keep logging.")

    out = ROOT / "data" / "calibration" / "emos_readiness.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved per-station-day rows → {out.relative_to(ROOT)}")
    print("  Re-run as data accrues; ship EMOS only when the CI clears 1.0 at ~40–60 days.")


if __name__ == "__main__":
    main()
