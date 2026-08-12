"""
Intraday high-distribution — the observation-conditioned twin of the day-ahead ensemble.

The daily high H = max over the LST day. At time t we've already observed a running high M,
which is a hard FLOOR on H (the day can't settle below what already happened). As the day
runs from sunrise toward the afternoon peak, less warming remains, so the forecast upside
above M shrinks; by the peak, H ≈ M (an observation lock).

This module conditions the day-ahead ensemble members on M and time-of-day, so ONE
distribution serves both regimes:
  • early morning  → warming_fraction ≈ 1 → members ≈ the raw forecast (obs don't bind yet)
  • afternoon peak → warming_fraction ≈ 0 → members collapse onto M (obs-lock falls out)

P(YES) is then the same NWS-blended bucket probability the day-ahead path uses
(mixture_prob), just on the conditioned members. Advisory only until forward-validated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kalshi_weather.calibration.ensemble_dist import mixture_prob

# Fitted diurnal curve (scripts/fit_diurnal_warming.py): β(h) regression of realized
# remaining warming on forecast upside, pooled over 20 stations × ~57 days of hourly
# obs. The original linear taper was FALSIFIED live 2026-07-12 — it assumed ~62% of
# warming done by noon; the fit says ~17% of forecast upside is gone by then (β=0.83).
_CURVE_PATH = Path(__file__).parents[3] / "data" / "calibration" / "diurnal_warming_curve.json"
_CURVE_CACHE: dict | None | bool = False   # False = not loaded yet; None = missing


def _fitted_curve() -> dict | None:
    global _CURVE_CACHE
    if _CURVE_CACHE is False:
        try:
            _CURVE_CACHE = json.loads(_CURVE_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            _CURVE_CACHE = None
    return _CURVE_CACHE if isinstance(_CURVE_CACHE, dict) else None


def warming_fraction(local_hour: float, sunrise: float = 7.0, peak: float = 15.0) -> float:
    """
    LEGACY linear taper (falsified 2026-07-12; kept as the no-curve fallback and for
    the historical record): fraction of remaining warming, ≈1 before sunrise,
    linear to 0 at `peak`. Prefer `warming_beta`, which uses the fitted curve.
    """
    if local_hour <= sunrise:
        return 1.0
    if local_hour >= peak:
        return 0.0
    return float((peak - local_hour) / (peak - sunrise))


def _interp_curve(field: str, local_hour: float) -> float | None:
    """Interpolate a fitted-curve field (beta / resid_sd) at local_hour, or None."""
    curve = _fitted_curve()
    vals = (curve or {}).get(field) or {}
    if not vals:
        return None
    hours = sorted(float(h) for h in vals)
    ys = [float(vals[str(int(h))]) for h in hours]
    if local_hour <= hours[0]:
        return ys[0]
    if local_hour >= hours[-1]:
        # after the last fitted hour the day is resolved: beta → 0, width → last value
        return 0.0 if field == "beta" else ys[-1]
    return float(np.interp(local_hour, hours, ys))


def warming_beta(local_hour: float, sunrise: float = 7.0, peak: float = 15.0) -> float:
    """
    Multiplier on forecast upside above the running high at `local_hour` (LST):
    the FITTED β(h) when data/calibration/diurnal_warming_curve.json exists,
    else the legacy linear taper. β is clipped to [0, 1].
    """
    b = _interp_curve("beta", local_hour)
    if b is None:
        return warming_fraction(local_hour, sunrise, peak)
    return float(np.clip(b, 0.0, 1.0))


def intraday_adjust(highs: np.ndarray, running_high: float, frac: float) -> np.ndarray:
    """
    Condition forecast member highs on the observed running high M and remaining-warming
    fraction: each member keeps only `frac` of its forecast upside above M, and can never
    fall below M (the floor). frac=1 → ≈ the raw forecast (floored at M); frac=0 → all = M.
    """
    highs = np.asarray(highs, dtype=float)
    upside = np.clip(highs - running_high, 0.0, None)   # forecast gap above what we've seen
    return running_high + upside * frac


def intraday_prob(
    members: np.ndarray,
    running_high: float,
    local_hour: float,
    strike_type: str,
    floor: float | None,
    cap: float | None,
    nws: float | None = None,
    sigma: float = 3.0,
    w_nws: float = 0.4,
    sunrise: float = 7.0,
    peak: float = 15.0,
) -> float | None:
    """
    Observation-conditioned P(YES) for a high-temp bucket, using the SAME NWS-blended
    bucket model as the day-ahead path (mixture_prob) on the intraday-adjusted members.

    The NWS anchor is floored/shrunk the same way, and sigma narrows with the day (floored
    so the blend never becomes a false spike). At the peak this collapses to an obs-lock.
    Returns None if the bucket can't be evaluated; else clipped to [0.005, 0.995].
    """
    if members is None or np.asarray(members).size == 0 or not np.isfinite(running_high):
        return None
    frac = warming_beta(local_hour, sunrise, peak)
    adj_members = intraday_adjust(members, running_high, frac)
    adj_nws = (
        running_high + max(0.0, nws - running_high) * frac
        if (nws is not None and np.isfinite(nws)) else None
    )
    # Blend width: the fitted curve's residual sd IS the measured remaining-outcome
    # uncertainty at this hour (floored so a near-peak blend stays a probability, not
    # a 0-width spike). Without a fitted curve, fall back to the legacy frac-scaling.
    fitted_sd = _interp_curve("resid_sd", local_hour)
    if fitted_sd is not None:
        adj_sigma = float(max(0.5, fitted_sd))
    else:
        adj_sigma = float(max(0.75, sigma * (0.35 + 0.65 * frac)))
    p = mixture_prob(adj_members, strike_type, floor, cap, adj_nws, adj_sigma, w_nws)
    if p is None:
        return None
    return float(np.clip(p, 0.005, 0.995))
