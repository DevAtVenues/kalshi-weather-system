"""
Per-model-family ensemble probabilities + minimax-family EV for one contract.

The production blend mixes all model families (GFS/ECMWF/ICON members) into one
distribution, which HIDES epistemic disagreement: on 2026-07-20 the blended MSP
T80 probability looked confidently one-sided while ICON's members sat 87% on the
OPPOSITE side of the strike from GFS's (13%). The same hand check passed SAT
B98.5 because all three families priced YES below 50% — unanimous direction,
+EV even under the most adverse family.

This module formalizes that check from the raw members the live logger already
records (data/logger/ensemble/): P(YES) computed separately per family via the
exact Kalshi settlement convention, then the trade's EV per $1-payout contract
under EACH family against the actual market mid. The decision quantity is
`worst_ev` — the EV if the most adverse family turns out to be the right one.
A blend can only hide a family; the minimum over families cannot.

ADVISORY ONLY (log-only discipline, same as the HRRR cross-check): consumed by
candidate_pipeline as an advisory section that does NOT move verdicts, until
the check itself has a graded forward record. Member maxes are RAW grid basis —
no grid→sensor bias applied — so compare families to each other and to the
market, not to the bias-corrected production center.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Family → logger directory. Mirrors the ensemble member logger's layout:
# data/logger/ensemble/<dir>/<YYYYMMDD_HHZ>/<STATION>.parquet
# with columns init_time / valid_time / member / temp_f.
FAMILY_DIRS = {"gfs": "gfs025", "icon": "icon_seamless", "ecmwf": "ecmwf_ifs025"}
ENSEMBLE_ROOT = Path(__file__).parents[3] / "data" / "logger" / "ensemble"

# A file must cover the settlement window at least this far toward its end to
# count — the daily max forms in the local afternoon, and a horizon-truncated
# file would systematically miss it (biasing the family cool). 4h of slack
# tolerates 3-hourly ECMWF steps at the window edge.
_MIN_COVER_TO_END_H = 4.0


def settles_yes(strike_type: str, floor: float | None, cap: float | None,
                high_f: float) -> bool:
    """Kalshi settlement on the integer CLI high — delegates to the canonical
    empirically-derived rules in kalshi_weather.settlement."""
    from kalshi_weather.settlement import settles_yes as _canon
    r = _canon(strike_type, floor, cap, high_f)
    return bool(r) if r is not None else False


def family_maxes(station: str, sdate: date | str,
                 root: Path | None = None,
                 window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
                 ) -> dict[str, dict]:
    """Per-family member daily maxes for one station's LST settlement day.

    For each family, the NEWEST logged init whose file covers the settlement
    window is used (an older init never covers a target day the newest can't).
    Families with no covering file are simply absent — the caller must treat
    fewer than 2 families as inconclusive, not as agreement.

    `window` overrides the derived LST settlement window (tests); production
    callers leave it None and the tz module (the single tz seam) derives it.
    """
    if window is None:
        from kalshi_weather.tz import settlement_window_utc
        d = pd.Timestamp(sdate).date() if not isinstance(sdate, date) else sdate
        window = settlement_window_utc(d, station)
    win_start, win_end = window
    root = ENSEMBLE_ROOT if root is None else Path(root)

    out: dict[str, dict] = {}
    for fam, sub in FAMILY_DIRS.items():
        base = root / sub
        if not base.is_dir():
            continue
        for init_dir in sorted((p for p in base.iterdir() if p.is_dir()),
                               reverse=True):
            f = init_dir / f"{station}.parquet"
            if not f.exists():
                continue
            try:
                df = pd.read_parquet(f, columns=["valid_time", "member", "temp_f"])
            except Exception:
                continue
            w = df[(df["valid_time"] >= win_start) & (df["valid_time"] < win_end)]
            if w.empty or w["valid_time"].max() < win_end - pd.Timedelta(hours=_MIN_COVER_TO_END_H):
                continue    # doesn't reach the afternoon peak — try an older init
            maxes = w.groupby("member")["temp_f"].max().to_numpy(dtype=float)
            if len(maxes) < 3:
                continue
            out[fam] = {"init": init_dir.name, "n": int(len(maxes)), "maxes": maxes}
            break
    return out


def family_probs(station: str, sdate: date | str, strike_type: str,
                 floor: float | None, cap: float | None,
                 root: Path | None = None,
                 window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
                 fams: dict[str, dict] | None = None,
                 ) -> dict[str, dict]:
    """P(YES) per model family for one contract, from raw member daily maxes.

    `fams` lets a caller vetting many contracts on one (station, day) reuse a
    single family_maxes() read instead of re-reading the parquets per contract.
    """
    if fams is None:
        fams = family_maxes(station, sdate, root=root, window=window)
    out: dict[str, dict] = {}
    for fam, d in fams.items():
        p = float(np.mean([settles_yes(strike_type, floor, cap, h)
                           for h in d["maxes"]]))
        out[fam] = {"p_yes": round(p, 3), "n": d["n"], "init": d["init"],
                    "mean_f": round(float(np.mean(d["maxes"])), 1)}
    return out


def minimax_family_ev(fam_probs: dict[str, dict], direction: str,
                      market_mid: float) -> dict | None:
    """EV per $1-payout contract under each family, and the worst case.

    BUY_YES: cost=mid, EV = p − mid.  BUY_NO: cost=1−mid, EV = mid − p.
    `worst_ev` is the trade's EV if the most adverse family is the right one —
    the blend can average a hostile family away; the minimum cannot.
    `direction_unanimous`: every family independently prices our side as the
    favorite (all p<0.5 for BUY_NO, all p≥0.5 for BUY_YES) — the 2026-07-20
    hand criterion that failed MSP and passed SAT.
    None (inconclusive, NOT agreement) with fewer than 2 families.
    """
    if not fam_probs or len(fam_probs) < 2:
        return None
    sign = 1.0 if direction == "BUY_YES" else -1.0
    evs = {fam: round(sign * (d["p_yes"] - float(market_mid)), 3)
           for fam, d in fam_probs.items()}
    worst = min(evs, key=evs.get)
    ps = [d["p_yes"] for d in fam_probs.values()]
    unanimous = (min(ps) >= 0.5) if direction == "BUY_YES" else (max(ps) < 0.5)
    return {"per_family_ev": evs, "worst_family": worst, "worst_ev": evs[worst],
            "n_families": len(evs), "direction_unanimous": bool(unanimous),
            "family_p_yes": {fam: d["p_yes"] for fam, d in fam_probs.items()}}
