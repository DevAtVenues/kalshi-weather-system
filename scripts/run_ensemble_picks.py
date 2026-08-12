"""
Ensemble pick engine (parallel to the production runner).

Scores each VALIDATED city's live contracts using the multi-model ensemble
distribution (calibration/ensemble_dist.py), bias- and drift-corrected — i.e.
the by-hand cross-check, automated. Prints ensemble P(YES) vs market for every
contract and flags edges. Goal: reproduce the manual analysis so it can later
replace the single-run path in the production runner.

    .venv/bin/python scripts/run_ensemble_picks.py 2026-06-29
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import requests
from dotenv import load_dotenv

from kalshi_weather.calibration.ensemble_dist import (
    fetch_ensemble_maxes, corrected_members, summary, recent_grid_bias,
    fetch_nws_high, mixture_prob, calibrate_dispersion,
)

DISAGREE_FLAG = 2.5   # |NWS - ensemble median| above this = forecasts diverge, low trust
SIGMA_FLOOR, SIGMA_CAP, SIGMA_DEFAULT = 1.2, 4.0, 2.0   # NWS-Normal spread bounds (°F)
W_NWS = 0.4           # weight on the NWS-Normal in the mixture

load_dotenv()
_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_HDR  = {"Authorization": f"Bearer {os.getenv('KALSHI_API_KEY')}"}
EDGE_MIN = 0.08

# Validated cities only (OOS-positive types).
# station, series, lat, lon, tz, validated types, (iem_network, iem_station)
CITIES = {
    "NYC": ("KNYC", "KXHIGHNY",  40.7789, -73.9692, "America/New_York", {"T", "B"}, ("NY_ASOS", "NYC")),
    "MIA": ("KMIA", "KXHIGHMIA", 25.7959, -80.2870, "America/New_York", {"T", "B"}, ("FL_ASOS", "MIA")),
    "CHI": ("KMDW", "KXHIGHCHI", 41.7860, -87.7522, "America/Chicago",   {"T"},      ("IL_ASOS", "MDW")),
    "AUS": ("KAUS", "KXHIGHAUS", 30.1945, -97.6699, "America/Chicago",   {"T"},      ("TX_ASOS", "AUS")),
    "PHL": ("KPHL", "KXHIGHPHIL",39.8719, -75.2411, "America/New_York",  {"T"},      ("PA_ASOS", "PHL")),
}


def _mid(m: dict) -> float | None:
    b, a = m.get("yes_bid_dollars"), m.get("yes_ask_dollars")
    if b is None or a is None:
        return None
    return (float(b) + float(a)) / 2.0


def _open_markets(series: str, date_tag: str) -> list[dict]:
    r = requests.get(f"{_BASE}/markets", headers=_HDR,
                     params={"series_ticker": series, "status": "open", "limit": 100}, timeout=30)
    return [m for m in r.json().get("markets", []) if date_tag in m["ticker"]]


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    d = datetime.strptime(target, "%Y-%m-%d").date()
    date_tag = d.strftime("%y%b%d").upper()   # 2026-06-29 -> 26JUN29

    print(f"=== Ensemble picks for {target} (validated cities; edge ≥ {EDGE_MIN}) ===")
    for city, (stn, series, lat, lon, tz, vtypes, (net, iem)) in CITIES.items():
        mkts = _open_markets(series, date_tag)
        if not mkts:
            continue
        members = fetch_ensemble_maxes(lat, lon, target, tz)
        if members.size == 0:
            print(f"\n── {city}: no ensemble data"); continue
        # ensemble-basis grid->sensor correction (NOT the deterministic bias.py/drift)
        gbias, gsigma = recent_grid_bias(lat, lon, iem, net, date.today())
        corr = corrected_members(members, gbias, 0.0)
        raw_sd = float(np.std(corr))
        # calibrate the over-dispersed ensemble spread toward the realized forecast error
        if np.isfinite(gsigma):
            corr, kshrink = calibrate_dispersion(corr, gsigma)
        else:
            kshrink = 1.0
        # NWS-Normal spread = realized recent forecast error (sharp where predictable)
        sigma = float(np.clip(gsigma if np.isfinite(gsigma) else SIGMA_DEFAULT, SIGMA_FLOOR, SIGMA_CAP))
        nws = fetch_nws_high(lat, lon, target)
        s = summary(corr)
        disagree = (nws - s["p50"]) if nws is not None else 0.0
        flag = "  ⚠ FORECASTS DIVERGE" if abs(disagree) > DISAGREE_FLAG else ""
        nws_str = f"{nws:.0f}" if nws is not None else "n/a"
        print(f"\n── {city} ({stn}) | ens n={s['n']} mean {s['mean']} sd {raw_sd:.1f}→{s['sd']} (k={kshrink:.2f}) "
              f"[p10 {s['p10']} p50 {s['p50']} p90 {s['p90']}] | grid-bias {gbias:+.1f} | "
              f"NWS {nws_str} (Δ{disagree:+.1f}) σ{sigma:.1f}{flag}")
        rows = []
        for m in mkts:
            st = m.get("strike_type"); ctype = "B" if st == "between" else "T"
            if ctype not in vtypes:
                continue
            p = mixture_prob(corr, st, m.get("floor_strike"), m.get("cap_strike"), nws, sigma, W_NWS)
            mid = _mid(m)
            if p is None or mid is None:
                continue
            edge = p - mid
            side = "BUY_YES" if edge > 0 else "BUY_NO"
            rows.append((abs(edge), m.get("yes_sub_title") or m.get("subtitle"),
                         ctype, p, mid, edge, side))
        for ae, sub, ctype, p, mid, edge, side in sorted(rows, reverse=True):
            flag = "  ★" if ae >= EDGE_MIN else ""
            print(f"   [{ctype}] {sub:16}  ens {p:.2f}  mkt {mid:.2f}  edge {edge:+.2f}  {side}{flag}")


if __name__ == "__main__":
    main()
