"""
Probability recalibration — corrects the model's systematic YES over-prediction.

The single-run rule / gaussian path over-predicts YES at every level (forward
reliability: predicted 0.50 → realized 0.18). This module applies a monotone
isotonic map, fit on graded outcomes (raw P(YES) → realized YES frequency), so the
probability we act on matches reality.

Maps are keyed by `prob_source`, optionally refined by LEAD bucket
(`"{source}@{lead}"`, e.g. `"ensemble@day-ahead"`). Lookup is most-specific-first:
the lead-keyed map if present, else the plain source map, else IDENTITY. The
same-day ensemble path and the intraday (window-open) path deliberately have no
map: same-day is measured well-calibrated (bias −0.006) and intraday locks must
keep their observed near-certainties.

The `ensemble@day-ahead` map exists because the day-ahead board is measured to
systematically over-predict (bias +0.053, CI excluding 0, LODO-by-day Brier
0.0936 → 0.0795 vs market 0.0886 — see build_calibration_map.py, which only
emits this key when that out-of-sample proof holds).

Fit offline by `scripts/build_calibration_map.py` → `data/models/prob_calibration.json`.
The map stores the isotonic interpolation knots so inference is a dependency-free
`np.interp` (no sklearn needed at runtime) and the artifact is human-inspectable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DEFAULT_PATH = Path(__file__).parents[3] / "data" / "models" / "prob_calibration.json"


def load_calibration(path: str | Path | None = None) -> dict:
    """Load the fitted calibration artifact. Returns {} if missing (→ identity)."""
    p = Path(path) if path else _DEFAULT_PATH
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


_cache: dict = {"mtime": None, "maps": {}}


def get_calibration(path: str | Path | None = None) -> dict:
    """
    Like load_calibration but mtime-cached: reloads only when the artifact file changes,
    so a fresh `build_calibration_map.py` refit takes effect in a long-running engine
    WITHOUT a restart (and without re-reading the file every call). Prevents a refit from
    silently doing nothing.
    """
    p = Path(path) if path else _DEFAULT_PATH
    try:
        mt = p.stat().st_mtime
    except OSError:
        return {}
    if _cache["mtime"] != mt:
        _cache["maps"] = load_calibration(p)
        _cache["mtime"] = mt
    return _cache["maps"]


def lead_bucket(hours_to_settle: float | None) -> str | None:
    """
    Bucket a row's hours-until-settlement-window into the lead classes the maps
    (and scripts/skill_monitor.py) are keyed by. MUST stay in lockstep with the
    monitor's LEAD_BUCKETS — fit population and apply population must agree.

    None outside (0, 40]: window already open (intraday path — never mapped) or
    too far out to be a tradeable forecast row.
    """
    if hours_to_settle is None:
        return None
    try:
        h = float(hours_to_settle)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(h):
        return None
    if 0.0 < h <= 14.0:
        return "same-day"
    if 14.0 < h <= 40.0:
        return "day-ahead"
    return None


def apply_prob_calibration(prob: float | None, source: str, maps: dict,
                           lead: str | None = None) -> float | None:
    """
    Map a raw model P(YES) to a calibrated P(YES) using `source`'s isotonic knots.

    Lookup order: `"{source}@{lead}"` (when lead given) → `source` → identity.
    Identity when: no artifact loaded, prob is None, or no key matches
    (deliberately, e.g. the same-day/intraday ensemble paths). Clipped to (0,1).
    """
    if not maps or prob is None:
        return prob
    sources = maps.get("sources") or {}
    m = sources.get(f"{source}@{lead}") if lead else None
    if not m:
        m = sources.get(source)
    if not m or not m.get("x") or not m.get("y"):
        return prob
    cal = float(np.interp(float(prob), m["x"], m["y"]))
    # Safety cap: a recalibrated probability must never assert near certainty —
    # mapped paths are measured unreliable at high confidence. Genuine locks
    # come from the intraday observation path, which is never mapped.
    return float(np.clip(cal, 0.01, 0.97))
