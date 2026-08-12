"""
Fast-obs boundary watcher — the thermometer-side sibling of position_watch.

position_watch tells you when the PRICE moves; this tells you when the
THERMOMETER moves, which is upstream of the price (2026-07-21: the market
crushed a SAT position 22c->7c on hourly tenths our stack read an hour late).

Reads the SAME watchlist (data/analysis/watchlist.json — see position_watch.py
for the entry shape). For each entry settling TODAY (station-local), pulls the
merged fast-obs running band (5-min feed + hourly METAR tenths) and pushes on
STAGE ESCALATION only:

  between/BUY_NO :  warn   high_max >= floor-0.5  (coarse feed says the bracket
                                                   may already be in reach)
                    threat high_min >= floor-0.5  (tenths confirm — in the
                                                   bracket's settle zone; the
                                                   bracket is {floor..cap} INCLUSIVE)
                    win    high_min >= cap+1.0    (overshot past the inclusive cap)
  greater/BUY_YES:  near   high_max >= floor-0.5  (good news — nearly there)
                    win    high_min >= floor+1.0  (greater is strict >)
  less/BUY_NO    :  win    high_min >= cap+0.0    (less is strict <)
  any entry      :  dead   the high locked the outcome AGAINST the position

Boundaries are the canonical settlement rules (kalshi_weather.settlement,
empirically derived) + lock_scan's 0.5 F slack: an exact-boundary reading
never counts as decided (METAR-vs-CLI rounding).
Stages only escalate, one push per (entry, day, stage); state in
logs/health/fast_obs_state.json (auto-resets on date change).

Usage: .venv/bin/python scripts/fast_obs_watch.py [--scheduled] [--force]
Scheduled: launchd com.plec.fast-obs-watch every 5 min; per-station self-gate
to LOCAL 10:00-20:00 (afternoon boundary hours). No health-check row on
purpose: a missed poll loses awareness, not data (lock-scanner doctrine).
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                       # noqa: E402

from kalshi_weather.ingest.fastobs import day_running_band  # noqa: E402
from kalshi_weather.tz import utc_now                       # noqa: E402

WATCHLIST = ROOT / "data" / "analysis" / "watchlist.json"
STATE = ROOT / "logs" / "health" / "fast_obs_state.json"
SLACK_F = 0.5                     # lock boundary guard — keep equal to lock_scan
GATE_LOCAL_HOURS = range(10, 20)  # poll only during station-local afternoon

_STAGE_RANK = {None: 0, "warn": 1, "near": 1, "threat": 2, "win": 3, "dead": 3}


def stage_for(entry: dict, band: dict) -> str | None:
    """Highest stage the running band supports for this entry (None = quiet)."""
    stt = entry.get("strike_type")
    direction = entry.get("direction")
    fl = entry.get("floor")
    cp = entry.get("cap")
    hi_min, hi_max = band["high_min_f"], band["high_max_f"]

    # Canonical boundaries (kalshi_weather.settlement): between is INCLUSIVE of
    # cap (bracket = {floor..cap}); greater YES needs max >= floor+0.5; less NO
    # needs max >= cap-0.5. Lock = true boundary + slack.
    locked = None                                   # side the high has decided
    if stt == "greater" and fl is not None and hi_min >= float(fl) + 0.5 + SLACK_F:
        locked = "YES"
    elif stt == "between" and cp is not None \
            and hi_min >= float(cp) + 0.5 + SLACK_F:
        locked = "NO"
    elif stt == "less" and cp is not None \
            and hi_min >= float(cp) - 0.5 + SLACK_F:
        locked = "NO"
    if locked:
        won = (direction == "BUY_YES") == (locked == "YES")
        return "win" if won else "dead"

    if stt == "between" and direction == "BUY_NO" and fl is not None:
        if hi_min >= float(fl) - 0.5:
            return "threat"
        if hi_max >= float(fl) - 0.5:
            return "warn"
    if stt == "greater" and direction == "BUY_YES" and fl is not None:
        if hi_max >= float(fl) - 0.5:
            return "near"
    return None


def _fmt(entry: dict, stage: str, band: dict) -> str:
    flat = band.get("flat_minutes")
    parts = [
        f"{entry['ticker']} {entry.get('direction', '')}: {stage.upper()}",
        f"high confirmed >= {band['high_min_f']}F (possible {band['high_max_f']}F)",
    ]
    if band.get("latest_precise_f") is not None:
        p = f"latest tenths {band['latest_precise_f']}F"
        if flat is not None and flat >= 20:
            p += f", flat {flat:.0f} min"
        parts.append(p)
    return " | ".join(parts)


def active_entries(now) -> list[dict]:
    try:
        entries = json.loads(WATCHLIST.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    from kalshi_weather.tz import lst_offset
    out = []
    for e in entries:
        st = e.get("station")
        if not st or not e.get("settlement_date"):
            continue
        try:
            off = lst_offset(st)
        except Exception:
            continue
        local = pd.Timestamp(now) + off
        e["_lst_offset_h"] = int(off.total_seconds() // 3600)
        e["_local_hour"] = local.hour
        if str(e["settlement_date"]) == str(local.date()):
            out.append(e)
    return out


def main() -> None:
    scheduled = "--scheduled" in sys.argv
    now = utc_now()
    entries = active_entries(now)
    if scheduled:
        entries = [e for e in entries if e["_local_hour"] in GATE_LOCAL_HOURS]
    if not entries:
        print("(no watchlist entries in an active window)")
        return

    try:
        state = json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}

    alerts: list[tuple[str, str]] = []              # (stage, line)
    for e in entries:
        band = day_running_band(e["station"], e["_lst_offset_h"])
        if band is None:
            print(f"{e['ticker']}: no obs yet")
            continue
        stage = stage_for(e, band)
        key = f"{e['ticker']}:{e.get('direction', '')}"
        st = state.setdefault(key, {})
        if st.get("date") != str(e["settlement_date"]):
            state[key] = st = {"date": str(e["settlement_date"]), "stage": None}
        prev = st.get("stage")
        line = _fmt(e, stage or "quiet", band)
        print(line)
        if stage and _STAGE_RANK[stage] > _STAGE_RANK.get(prev, 0):
            st["stage"] = stage
            alerts.append((stage, line))

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state))

    if alerts and scheduled:
        from kalshi_weather.dashboard.notifications import notify_health
        worst = max(alerts, key=lambda a: _STAGE_RANK[a[0]])
        notify_health(f"Fast-obs {worst[0].upper()}",
                      "\n".join(l for _, l in alerts),
                      priority="urgent" if _STAGE_RANK[worst[0]] >= 2 else "high",
                      tags="thermometer")


if __name__ == "__main__":
    if "--scheduled" in sys.argv:
        from kalshi_weather.scheduling import acquire_slot, mark_and_release
        _state_dir = ROOT / "logs" / "scheduler"
        _slot = acquire_slot(_state_dir, "fast_obs_watch", min_gap_s=240,
                             force="--force" in sys.argv)
        if _slot is None:
            print("fast_obs_watch: ran recently or already running — skipping.")
            sys.exit(0)
        try:
            main()
        finally:
            mark_and_release(_state_dir, "fast_obs_watch", _slot)
    else:
        main()
