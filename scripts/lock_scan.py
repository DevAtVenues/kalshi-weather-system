"""
Live high-side lock scanner — playbook trade #1.

For every same-day contract in the latest run: fetch the station's LST-day
running METAR high, mark outcomes that are already physically decided (the high
only rises), pull the live book, and print anything still priced below fair.

High-side certainties only (all monotone-safe as temp rises):
  greater → YES locked when the high has cleared the floor
  between → NO locked when the high has cleared the cap (bracket overshot)
  less    → NO locked when the high has cleared the cap
Low-side is NEVER a lock (falsified: CLI min can undercut hourly METARs).

BOUNDARY GUARD (learned live 2026-07-12): first scan flagged two "free" trades
where the running high sat EXACTLY on the cap (85.0 vs cap 85) while the market
priced 99% the other way — METAR-vs-CLI integer rounding makes exact-boundary
locks unsafe. A lock requires SLACK_F (default 0.5°F) of clearance past the
strike, so a one-rounding disagreement can't flip the outcome.

Usage: .venv/bin/python scripts/lock_scan.py [--min-value 0.02] [--slack 0.5]
Best run 20-60 min after each city's diurnal peak — by evening the books are
fully priced (bots own the late window; the value is in being early).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kalshi_weather.live.runner import CITY_CONFIGS          # noqa: E402
from kalshi_weather.ingest.metar import fetch_running_high_f  # noqa: E402
from candidate_pipeline import kalshi_verify                  # noqa: E402

SIGNALS = ROOT / "data" / "signals" / "signals_log.jsonl"


def same_day_rows() -> list[dict]:
    rows = [json.loads(l) for l in open(SIGNALS) if l.strip()]
    latest = max(r["run_ts"] for r in rows)
    return [r for r in rows if r["run_ts"] == latest and r.get("is_same_day")]


def locked_outcome(r: dict, rh: float, slack: float) -> str | None:
    # Canonical boundaries (kalshi_weather.settlement): greater YES iff max >=
    # floor+0.5; between NO iff max >= cap+0.5 (INCLUSIVE cap); less NO iff
    # max >= cap-0.5. Lock = true boundary + slack.
    stt, fl, cp = r.get("strike_type"), r.get("floor_strike"), r.get("cap_strike")
    if stt == "greater" and fl is not None and rh >= fl + 0.5 + slack:
        return "YES"
    if stt == "between" and cp is not None and rh >= cp + 0.5 + slack:
        return "NO"
    if stt == "less" and cp is not None and rh >= cp - 0.5 + slack:
        return "NO"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-value", type=float, default=0.02,
                    help="min gross ¢/contract below fair to flag (default 0.02)")
    ap.add_argument("--slack", type=float, default=0.5,
                    help="required °F clearance past the strike (boundary guard)")
    ap.add_argument("--scheduled", action="store_true",
                    help="launchd mode: only run 1-9 PM ET (peak windows), push hits "
                         "to the phone, dedup pushes per (day, ticker, side)")
    args = ap.parse_args()

    if args.scheduled:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if not (13 <= now_et.hour <= 20):
            return                            # outside peak windows — quiet no-op

    sd = same_day_rows()
    highs: dict[str, float | None] = {}
    for city in sorted({r["city"] for r in sd}):
        cfg = CITY_CONFIGS.get(city) or {}
        if not cfg.get("nws_station") or cfg.get("lst_offset") is None:
            continue
        if args.scheduled:
            # Locks form in each city's post-peak window; skip cities whose LST
            # afternoon hasn't arrived or is long over — 60% fewer IEM calls and
            # the scan finishes faster where it matters.
            from datetime import datetime as _dt, timedelta as _td, timezone as _tz
            lst_hour = (_dt.now(_tz.utc) + _td(hours=cfg["lst_offset"])).hour
            if not (13 <= lst_hour <= 19):
                continue
        highs[city] = fetch_running_high_f(cfg["nws_station"], cfg["lst_offset"])
        time.sleep(0.3)                          # IEM rate limit
    print(f"scanning {len(sd)} same-day contracts, {len(highs)} cities in-window "
          f"(slack {args.slack}°F)")

    hit_lines = []
    for r in sd:
        rh = highs.get(r["city"])
        if rh is None:
            continue
        lock = locked_outcome(r, rh, args.slack)
        if lock is None:
            continue
        m = kalshi_verify(str(r["ticker"]))
        if not m or m.get("yes_ask") is None or m.get("yes_bid") is None:
            continue
        cost = m["yes_ask"] if lock == "YES" else (1 - m["yes_bid"])
        value = 1.0 - cost
        if value >= args.min_value and m.get("status") == "active":
            line = (f"🎯 {r['ticker']} BUY {lock} @ {cost:.2f} → +{value:.2f} gross "
                    f"(rh {rh:.1f} vs {r.get('strike_type')} f={r.get('floor_strike')} "
                    f"c={r.get('cap_strike')}; book {m['yes_bid']:.2f}/{m['yes_ask']:.2f}, "
                    f"vol {m.get('volume'):.0f})")
            print(f"  {line}")
            hit_lines.append((f"{r['ticker']}:{lock}", line))
    if not hit_lines:
        print("  no mispriced locks right now — books fully priced (scan earlier "
              "in each city's post-peak window)")

    if args.scheduled and hit_lines:
        # Push new hits only — a lock that stays mispriced must not ping every 10 min.
        from datetime import date as _date
        state_p = ROOT / "logs" / "health" / "lock_scan_pushed.json"
        try:
            state = json.loads(state_p.read_text())
        except (OSError, json.JSONDecodeError):
            state = {}
        today = str(_date.today())
        pushed = set(state.get(today, []))
        fresh = [(k, l) for k, l in hit_lines if k not in pushed]
        if fresh:
            from kalshi_weather.dashboard.notifications import notify_health
            notify_health(f"Lock scan HIT ({len(fresh)})",
                          "\n".join(l for _, l in fresh), priority="high", tags="dart")
            state_p.parent.mkdir(parents=True, exist_ok=True)
            state_p.write_text(json.dumps(
                {today: sorted(pushed | {k for k, _ in fresh})}))


if __name__ == "__main__":
    main()
