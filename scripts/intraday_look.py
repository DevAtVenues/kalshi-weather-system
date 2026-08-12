"""
Intraday advisory look — today's settlement, conditioned on observations so far.

The obs-informed twin of the day-ahead engine. For each ensemble city it takes today's
ensemble members, conditions them on the running high so far + time-of-day (live/intraday.py),
and prints P(YES) vs the market for each open TODAY contract. ADVISORY — surfaces edges but
does not trade; the day-ahead engine (run_live_engine.py) is the auto-fire path.

Needs the dashboard running (reads the live running-high tracker from its API).

Usage:
    .venv/bin/python scripts/intraday_look.py [--min-edge 0.08]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalshi_weather.live.runner import (  # noqa: E402
    CITY_CONFIGS, ENSEMBLE_CITIES, build_ensemble_context, fetch_open_markets, W_NWS,
)
from kalshi_weather.live.intraday import intraday_prob, warming_fraction  # noqa: E402
from kalshi_weather.calibration.center_bias import center_shift  # noqa: E402
from kalshi_weather.calibration.bias import load_bias_table  # noqa: E402

DASH = "http://localhost:5555/api/data"
_BIAS_TBL = load_bias_table()


def _running_highs() -> dict[str, float]:
    """station → running high so far today, from the live dashboard tracker."""
    try:
        d = requests.get(DASH, timeout=10).json()
    except Exception:
        print("⚠ dashboard not reachable at :5555 — start scripts/run_web_dashboard.py"); return {}
    return {r.get("station"): r.get("day_h") for r in d.get("rows", []) if r.get("day_h") is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-edge", type=float, default=0.08)
    args = ap.parse_args()

    today = date.today()
    now = datetime.now(timezone.utc)
    highs = _running_highs()
    if not highs:
        return

    print("=" * 74)
    print(f"  INTRADAY LOOK (advisory) — {today}  {now:%H:%M} UTC")
    print("  P(YES) conditions today's ensemble on the running high + time of day.")
    print("=" * 74)

    any_pick = False
    for city, cfg in ((c, CITY_CONFIGS[c]) for c in ENSEMBLE_CITIES):
        station = cfg["nws_station"]
        M = highs.get(station)
        if M is None:
            continue
        local_hour = (now.hour + now.minute / 60 + cfg.get("lst_offset", 0)) % 24
        frac = warming_fraction(local_hour)

        ec = build_ensemble_context(city, cfg, [today]).get(today)
        if not ec:
            continue
        # Per-city center correction (P1.0): shift the ensemble members/center by the
        # historical-prior-anchored bias so the board stops fighting our own known bias.
        # Advisory — the NWS anchor is left uncorrected (independent second opinion).
        shift = center_shift(station, today.month, _BIAS_TBL)
        members = ec["members"] + shift
        try:
            contracts = fetch_open_markets(cfg["series"], station)
        except Exception:
            continue
        today_c = contracts[contracts["settlement_date"] == today] if not contracts.empty else contracts
        if today_c.empty:
            continue

        regime = "≈forecast" if frac > 0.66 else ("blending" if frac > 0.1 else "≈locked (post-peak)")

        # Market's implied CENTER = midpoint of its most-likely between bucket. If it's
        # far from our ensemble p50, the disagreement is about the forecast center, not a
        # single mispriced bucket — that's much more likely OUR error than real edge (same
        # discipline as the day-ahead credibility gate). We flag it and hold those picks.
        betweens = [c for _, c in today_c.iterrows()
                    if c.get("strike_type") == "between" and c.get("floor_strike") is not None
                    and c.get("last_price_dollars") == c.get("last_price_dollars")]
        mkt_center = None
        if betweens:
            top = max(betweens, key=lambda c: c.get("last_price_dollars") or 0)
            mkt_center = (float(top["floor_strike"]) + float(top.get("cap_strike", top["floor_strike"]) )) / 2
        ens_c = (ec.get("p50") + shift) if ec.get("p50") is not None else None
        center_gap = (mkt_center - ens_c) if (mkt_center is not None and ens_c is not None) else None
        center_dispute = center_gap is not None and abs(center_gap) >= 2.0

        rows = []
        for _, c in today_c.iterrows():
            mid = c.get("last_price_dollars")
            if mid is None or not (mid == mid):   # NaN guard
                continue
            p = intraday_prob(
                members, float(M), local_hour,
                c.get("strike_type"), c.get("floor_strike"), c.get("cap_strike"),
                ec.get("nws"), ec["sigma"], W_NWS,
            )
            if p is None:
                continue
            edge = p - float(mid)
            # Implausible-edge guard: a >25-pt intraday edge on a liquid market is almost
            # always our error, not opportunity — hold it, don't surface as actionable.
            if abs(edge) >= args.min_edge and abs(edge) <= 0.25 and not center_dispute:
                side = "BUY YES" if edge > 0 else "BUY NO"
                rows.append((abs(edge), c.get("ticker", ""), side, float(mid), p))

        hdr = f"\n  {city}  (high so far {M:.0f}°F · local {local_hour:04.1f}h · {regime}"
        if ens_c is not None:
            hdr += f" · our center {ens_c:.0f}°F (bias-corr {shift:+.1f})"
        hdr += ")"
        if center_dispute:
            print(hdr)
            print(f"    ⚠ HELD — market center ~{mkt_center:.0f}°F vs our {ens_c:.0f}°F "
                  f"({center_gap:+.1f}°F). Likely our forecast, not edge — do not trade on this.")
            continue
        if rows:
            any_pick = True
            print(hdr)
            for _, tk, side, mid, p in sorted(rows, reverse=True):
                print(f"    {side:<7} {tk:<26} market {mid*100:4.0f}% → intraday {p*100:4.0f}%  "
                      f"(edge {abs(p-mid)*100:+.0f})")

    if not any_pick:
        print("\n  No intraday edges ≥ threshold right now "
              "(early-day obs may not bind yet — recheck as the afternoon peak develops).")
    print("\n  Advisory only. Size small; the market watches the same thermometer.")


if __name__ == "__main__":
    main()
