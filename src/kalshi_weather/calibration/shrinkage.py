"""
Market-shrinkage posterior: blend our probability with the market's in log-odds.

The pricing stack treats model-P as truth and market-P as noise, then patches
the contradiction with heuristic ceilings ("edge > 0.25 = implausible"). But
skill_monitor MEASURES the market beating our day-ahead board — so the honest
probability is a posterior between the two, weighted by measured relative
skill:

    p_post = sigmoid( w * logit(p_model) + (1 - w) * logit(p_market) )

w is fit per "{source}@{lead}" by scripts/build_shrinkage.py on graded
outcomes (log-loss, leave-one-day-out evidence recorded) and stored in
data/models/prob_shrinkage.json. w=1 → trust the model fully; w=0 → the
market already knows everything we do.

LOG-ONLY (2026-07-21): prob_shrunk / edge_shrunk ride signal rows and boards
so the posterior accrues a forward record; nothing gates or sizes on them.
The intended endgame is edge_shrunk replacing the ad-hoc ceiling gates —
after (not before) the forward record supports it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DEFAULT_PATH = Path(__file__).parents[3] / "data" / "models" / "prob_shrinkage.json"
_cache: dict = {"path": None, "mtime": None, "maps": None}

_P_CLIP = 1e-4   # logit domain guard; also the posterior's output clip


def logit(p: float) -> float:
    p = float(np.clip(p, _P_CLIP, 1 - _P_CLIP))
    return float(np.log(p / (1 - p)))


def blend(p_model: float, p_market: float, w: float) -> float:
    z = w * logit(p_model) + (1.0 - w) * logit(p_market)
    return float(np.clip(1.0 / (1.0 + np.exp(-z)), _P_CLIP, 1 - _P_CLIP))


def load_shrinkage(path: str | Path | None = None) -> dict:
    """Cached artifact load; {} when absent (all shrink_prob calls → None)."""
    p = Path(path) if path else _DEFAULT_PATH
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    if _cache["path"] == str(p) and _cache["mtime"] == mtime:
        return _cache["maps"]
    try:
        maps = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    _cache.update(path=str(p), mtime=mtime, maps=maps)
    return maps


def shrink_prob(p_model: float | None, p_market: float | None, source: str,
                lead: str | None, maps: dict | None = None) -> float | None:
    """Posterior P(YES), or None when it cannot be computed (no artifact, no
    fitted weight for this source@lead, missing inputs). None means UNKNOWN —
    callers must not treat it as agreement with either side."""
    if p_model is None or p_market is None or lead is None:
        return None
    if maps is None:
        maps = load_shrinkage()
    if not maps:
        return None
    w = (maps.get("weights") or {}).get(f"{source}@{lead}", {}).get("w")
    if w is None:
        return None
    return blend(float(p_model), float(p_market), float(w))
