"""
Position watcher: push when something CHANGES on a held/watched contract.

Watchlist entries live in data/analysis/watchlist.json (hand-edited or appended
when a trade goes on). For each entry, every scheduled run checks:

  1. PRICE: live YES mid moved >= price_alert (default 0.07) from the last
     alerted mid → push (both directions — edge evaporating and edge blowing
     out both matter). Baseline then resets to the alerted mid.
  2. THESIS (between/BUY_NO entries): the newest multi-model snapshot for the
     station+date has any model within thesis_margin_f (default 0.5°F) of the
     bracket cap → the "every model is comfortably above" story is eroding →
     push once per day.
  3. OBS (settlement day only): the station's merged running band (5-min feed +
     METAR tenths, same source as fast_obs_watch) is climbing toward the
     position's loss zone. Pushes once per stage as high_min crosses
     loss_floor-2 (WARN), loss_floor-1 (CRITICAL), loss_floor (BREACHED).
     Added 2026-08-03 after KPHL ran 79→87 against a held NO->87 with zero
     phone alerts (watchlist was empty AND no obs check existed).

Watchlist entry shape:
  {"ticker": "KXHIGHNY-26JUL14-B92.5", "direction": "BUY_NO",
   "station": "KNYC", "settlement_date": "2026-07-14",
   "strike_type": "between", "floor": 92, "cap": 93,
   "note": "playbook trade #2, entered ~0.71-0.74",
   "price_alert": 0.07, "thesis_margin_f": 0.5}

Entries whose settlement_date is past are ignored (prune whenever).
State (last alerted mid, thesis-alerted flag): logs/health/position_watch.json

Usage: .venv/bin/python scripts/position_watch.py [--scheduled]
Scheduled: launchd com.plec.position-watch every 30 min.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from candidate_pipeline import kalshi_verify, FORECASTS   # noqa: E402
from kalshi_weather.ingest.fastobs import day_running_band  # noqa: E402

WATCHLIST = ROOT / "data" / "analysis" / "watchlist.json"
STATE     = ROOT / "logs" / "health" / "position_watch.json"

_OBS_STAGES = ("WARN", "CRITICAL", "BREACHED")   # rank order


def loss_floor_f(entry: dict) -> float | None:
    """Lowest integer settle at which this position LOSES, or None if rising
    obs cannot hurt it (loss zone below/absent). threshold conventions are the
    canonical ones in kalshi_weather.settlement: between inclusive both ends,
    greater strict >, less strict <."""
    st, d = entry.get("strike_type"), entry.get("direction")
    thr = entry.get("threshold", entry.get("cap"))
    if st == "between":
        if d == "BUY_NO" and entry.get("floor") is not None:
            return float(entry["floor"])            # high enters the bracket
        if d == "BUY_YES" and entry.get("cap") is not None:
            return float(entry["cap"]) + 1          # high overshoots the bracket
    elif st == "greater" and d == "BUY_NO" and thr is not None:
        return float(thr) + 1                       # >T loses at T+1
    elif st == "less" and d == "BUY_YES" and thr is not None:
        return float(thr)                           # <T loses at T
    return None


def obs_check(entry: dict, st: dict) -> str | None:
    """Running-band alert line if the day's high is climbing into the loss zone."""
    lf = loss_floor_f(entry)
    if lf is None:
        return None
    off = int(entry.get("lst_offset_h", -5))
    from datetime import datetime, timedelta, timezone
    today_lst = (datetime.now(timezone.utc) + timedelta(hours=off)).date()
    if str(entry.get("settlement_date")) != str(today_lst):
        return None
    band = day_running_band(entry.get("station", ""), off)
    if not band or band.get("high_min_f") is None:
        return None
    hm = float(band["high_min_f"])
    stage = None
    if hm >= lf:
        stage = "BREACHED"
    elif hm >= lf - 1:
        stage = "CRITICAL"
    elif hm >= lf - 2:
        stage = "WARN"
    if stage is None:
        return None
    prev = st.get("obs_stage")
    if prev in _OBS_STAGES and _OBS_STAGES.index(prev) >= _OBS_STAGES.index(stage):
        return None
    st["obs_stage"] = stage
    return (f"{entry['ticker']}: OBS {stage} — running high_min {hm:.1f}F vs "
            f"loss floor {lf:.0f}F (band max {band.get('high_max_f')}, latest "
            f"tenths {band.get('latest_precise_f')})  [{entry.get('note','')}]")


def newest_snapshot(station: str, sdate: str) -> dict | None:
    best = None
    try:
        with open(FORECASTS) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("station") == station and r.get("settlement_date") == sdate:
                    if best is None or r["snapshot_ts"] > best["snapshot_ts"]:
                        best = r
    except FileNotFoundError:
        return None
    return best


def check(entry: dict, state: dict) -> list[str]:
    """Return alert lines for this entry; mutates state."""
    alerts: list[str] = []
    key = f"{entry['ticker']}:{entry['direction']}"
    st = state.setdefault(key, {})

    m = kalshi_verify(entry["ticker"])
    if m and m.get("yes_bid") is not None and m.get("yes_ask") is not None:
        mid = (m["yes_bid"] + m["yes_ask"]) / 2
        base = st.get("last_alerted_mid")
        if base is None:
            st["last_alerted_mid"] = mid          # first sighting = baseline, no alert
        elif abs(mid - base) >= entry.get("price_alert", 0.07):
            alerts.append(f"{entry['ticker']}: mid moved {base:.2f} → {mid:.2f} "
                          f"(book {m['yes_bid']:.2f}/{m['yes_ask']:.2f})  [{entry.get('note','')}]")
            st["last_alerted_mid"] = mid

    if entry.get("strike_type") == "between" and entry.get("cap") is not None \
            and entry.get("direction") == "BUY_NO":
        snap = newest_snapshot(entry.get("station", ""), str(entry["settlement_date"]))
        if snap:
            vals = {k: snap[k] for k in ("hrrr", "ecmwf", "nbm", "gfs", "nws")
                    if snap.get(k) is not None}
            margin = entry.get("thesis_margin_f", 0.5)
            close = {k: v for k, v in vals.items()
                     if v <= float(entry["cap"]) + margin}
            today = str(date.today())
            if close and st.get("thesis_alerted") != today:
                alerts.append(f"{entry['ticker']}: THESIS EROSION — model(s) within "
                              f"{margin}°F of cap {entry['cap']}: {close} (all: {vals})")
                st["thesis_alerted"] = today

    obs = obs_check(entry, st)
    if obs:
        alerts.append(obs)
    return alerts


def main() -> None:
    try:
        entries = json.loads(WATCHLIST.read_text())
    except (OSError, json.JSONDecodeError):
        print("(no watchlist)")
        return
    entries = [e for e in entries if str(e.get("settlement_date", "")) >= str(date.today())]
    if not entries:
        print("(watchlist empty or all settled)")
        return
    try:
        state = json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}

    alerts: list[str] = []
    for e in entries:
        alerts += check(e, state)

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state))
    if alerts:
        print("\n".join(alerts))
        if "--scheduled" in sys.argv:
            from kalshi_weather.dashboard.notifications import notify_health
            notify_health(f"Position watch: {len(alerts)} change(s)",
                          "\n".join(alerts), priority="high", tags="eyes")
    else:
        print(f"({len(entries)} watched, no changes past thresholds)")


if __name__ == "__main__":
    main()
