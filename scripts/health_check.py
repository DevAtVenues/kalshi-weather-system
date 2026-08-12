"""
System health watchdog: make silent failure loud.

Every incident so far was a SILENT DEGRADE, not a crash — the job exited 0 and
produced something hollow (book checks returning None for weeks, a 2-day orderbook
hole after the directory rename, grading stalled 14 days on a stale label cache).
This check therefore verifies OUTPUTS, not processes, in two classes:

  LIVENESS — did every scheduled job produce its artifact, recently? (artifact
  freshness, scheduler last_run markers, supervisor status [systemd on the VPS,
  launchd on macOS], logger process)

  CONTENT  — is the artifact non-degenerate? (city/contract breadth of the latest
  run, null fractions, a live canary-market probe that catches Kalshi API contract
  drift, candidate-board meta availability)

Statuses: OK / WARN / RED. Any RED pushes via ntfy (notify_health — bypasses the
risk halt on purpose: a broken logger must always reach the phone), with a
cooldown: re-push only when the RED set changes or 6h pass; one "recovered" push
when a red episode clears. Always exits 0 — the push is the signal, and a nonzero
exit would make this agent's own launchctl row look sick.

What this can NOT catch: model wrongness (that's forward grading + falsification
discipline) and anything with no artifact to inspect.

Usage:
  .venv/bin/python scripts/health_check.py            # table + push on red
  .venv/bin/python scripts/health_check.py --no-push  # table only
Scheduled: systemd plec-health-check.timer on the VPS (launchd com.plec.health-check
01:40/07:40/13:40/19:40 local on macOS).
Also surfaced in daily_review (report()).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

SIGNALS    = ROOT / "data" / "signals" / "signals_log.jsonl"
FORECASTS  = ROOT / "data" / "logger" / "forecasts" / "forecasts.jsonl"
ORDERBOOK  = ROOT / "data" / "logger" / "orderbook"
ENSEMBLE   = ROOT / "data" / "logger" / "ensemble"
CANDIDATES = ROOT / "data" / "analysis" / "candidates"
LABELS     = ROOT / "data" / "raw" / "labels"
SCHED      = ROOT / "logs" / "scheduler"
STATE      = ROOT / "logs" / "health" / "state.json"
WARM_CURVE = ROOT / "data" / "calibration" / "diurnal_warming_curve.json"
PROB_MAP   = ROOT / "data" / "models" / "prob_calibration.json"

AGENTS = ["com.plec.run-logger", "com.plec.run-live", "com.plec.archive-forecasts",
          "com.plec.daily-review", "com.plec.claims-logger",
          "com.plec.candidate-pipeline", "com.plec.health-check",
          "com.plec.lock-scanner", "com.plec.stay-awake",
          "com.plec.position-watch"]

CANARY_SERIES = "KXHIGHNY"      # any liquid daily series works
ALERT_COOLDOWN_S = 6 * 3600


def _age_h(ts: float) -> float:
    return (time.time() - ts) / 3600.0


def _tail_jsonl_ts(path: Path, field: str, scan_last: int = 400) -> float | None:
    """Newest ISO timestamp in `field` near the end of a jsonl file, as epoch."""
    try:
        lines = path.read_bytes().splitlines()[-scan_last:]
    except OSError:
        return None
    best = None
    for line in lines:
        try:
            v = json.loads(line).get(field)
        except json.JSONDecodeError:
            continue
        if v and (best is None or v > best):
            best = v
    if best is None:
        return None
    try:
        return datetime.fromisoformat(str(best).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# ── LIVENESS ─────────────────────────────────────────────────────────────────

def check_signals_fresh() -> tuple[str, str]:
    ts = _tail_jsonl_ts(SIGNALS, "run_ts")
    if ts is None:
        return "RED", "signals_log unreadable or no run_ts"
    age = _age_h(ts)
    s = "OK" if age < 6.7 else "RED"    # cycles every 6h; catch-up on wake counts
    return s, f"latest run {age:.1f}h ago"


def check_forecast_archive() -> tuple[str, str]:
    ts = _tail_jsonl_ts(FORECASTS, "snapshot_ts")
    if ts is None:
        return "RED", "forecasts.jsonl unreadable or no snapshot_ts"
    age = _age_h(ts)
    s = "OK" if age < 8 else "RED"      # archive windows are ≤7h apart
    return s, f"latest snapshot {age:.1f}h ago"


def check_orderbook_logger() -> tuple[str, str]:
    """The one dataset that is UNRECOVERABLE if missed. Today's file must exist and
    be minutes-fresh; any missing day in the last 3 is a hole worth flagging.
    The logger names files by UTC date (logger/orderbook.py), so 'today' must be
    UTC — local date goes falsely stale every night after the 00:00 UTC rollover."""
    today = datetime.now(timezone.utc).date()
    f = ORDERBOOK / f"{today}.parquet"
    if not f.exists():
        # Right after 00:00 UTC the new day's file doesn't exist until the first
        # write lands. If yesterday's file is minutes-fresh the logger is alive.
        prev = ORDERBOOK / f"{today - timedelta(days=1)}.parquet"
        if prev.exists() and (time.time() - prev.stat().st_mtime) / 60 <= 20:
            return "OK", f"awaiting first write of {today} (rollover; yesterday fresh)"
        return "RED", f"no orderbook file for {today}"
    age_min = (time.time() - f.stat().st_mtime) / 60
    missing = [str(today - timedelta(days=d)) for d in (1, 2, 3)
               if not (ORDERBOOK / f"{today - timedelta(days=d)}.parquet").exists()]
    if age_min > 20:
        return "RED", f"today's file stale ({age_min:.0f} min; writes every 5)"
    if missing:
        return "WARN", f"today fresh but HOLE in last 3 days: {missing}"
    return "OK", f"today updated {age_min:.0f} min ago, last 3 days present"


def check_ensemble_logger() -> tuple[str, str]:
    """Layout: ensemble/<model>/<YYYYMMDD_HHZ>/<station>.parquet — the run-dir name
    IS the model init time, so read that instead of statting thousands of files."""
    newest: datetime | None = None
    newest_name = ""
    for d in ENSEMBLE.glob("*/*_*Z"):
        try:
            dt = datetime.strptime(d.name, "%Y%m%d_%HZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if newest is None or dt > newest:
            newest, newest_name = dt, f"{d.parent.name}/{d.name}"
    if newest is None:
        return "RED", "no ensemble run directories at all"
    age = (datetime.now(timezone.utc) - newest).total_seconds() / 3600
    return ("OK" if age < 24 else "RED"), f"newest run {newest_name} ({age:.1f}h ago)"


def check_candidate_boards() -> tuple[str, str]:
    # boards are YYYY-MM-DDTHHMMSS.json; the dir also holds forward_stats.json
    # etc., which sort lexicographically AFTER them ("f" > "2") — filtering by
    # leading digits keeps files[-1] the newest actual board (2026-08-03 bug:
    # RED "newest board 11h ago" all evening while fresh boards existed).
    files = sorted(f for f in CANDIDATES.glob("*.json") if f.name[:4].isdigit())
    if not files:
        return "WARN", "no boards yet (agent first fires 19:20 local)"
    age = _age_h(files[-1].stat().st_mtime)
    return ("OK" if age < 7.5 else "RED"), f"newest board {age:.1f}h ago"


def check_labels_fresh() -> tuple[str, str]:
    """After the 10:00 review, every station should have yesterday. Any time of
    day, a station >2 days behind means its label refresh is broken."""
    cutoff = date.today() - timedelta(days=2)
    year = date.today().year
    behind, total = [], 0
    for f in LABELS.glob(f"*/{year}.parquet"):
        total += 1
        try:
            mx = pd.to_datetime(pd.read_parquet(f, columns=["date"])["date"]).max().date()
        except Exception:
            behind.append(f"{f.parent.name}:unreadable")
            continue
        if mx < cutoff:
            behind.append(f"{f.parent.name}:{mx}")
    if total == 0:
        return "RED", "no current-year label parquets"
    if len(behind) > total * 0.1:
        return "RED", f"{len(behind)}/{total} stations behind: {behind[:5]}"
    if behind:
        return "WARN", f"{len(behind)}/{total} stations behind: {behind[:5]}"
    return "OK", f"all {total} stations within 2 days"


def check_scheduler_markers() -> tuple[str, str]:
    expect = {"run_live": 6.7, "archive_forecasts": 8.0, "candidate_pipeline": 6.7}
    bad, seen = [], []
    for name, max_h in expect.items():
        m = SCHED / f"{name}.last_run"
        if not m.exists():
            bad.append(f"{name}:never")
            continue
        try:
            age = _age_h(float(m.read_text().strip()))
        except ValueError:
            bad.append(f"{name}:unreadable")
            continue
        (bad if age > max_h else seen).append(f"{name}:{age:.1f}h")
    if any(b.startswith(("run_live", "archive_forecasts")) for b in bad):
        return "RED", f"stale/missing: {bad}"
    if bad:
        return "WARN", f"stale/missing: {bad} (ok: {seen})"
    return "OK", ", ".join(seen)


def check_supervisor() -> tuple[str, str]:
    """Process-supervisor health, cross-platform. Ported launchd->systemd for the
    VPS (2026-08-05 migration): systemctl on Linux, launchctl on macOS. Previously
    this hard-coded launchctl and perma-WARNed on the Linux server, masking a real
    monitoring gap (a dead systemd unit went unseen)."""
    if shutil.which("launchctl"):
        return _check_launchd_agents()
    if shutil.which("systemctl"):
        return _check_systemd_units()
    return "WARN", "no supervisor (launchctl/systemctl) found"


def _check_launchd_agents() -> tuple[str, str]:
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception as exc:
        return "WARN", f"launchctl unavailable: {exc}"
    rows = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].startswith("com.plec"):
            rows[parts[2]] = (parts[0], parts[1])       # (pid, last exit)
    missing = [a for a in AGENTS if a not in rows]
    failed = [f"{a}(exit {rows[a][1]})" for a in AGENTS
              if a in rows and rows[a][1] not in ("0", "-")
              and a != "com.plec.health-check"
              # -15 (SIGTERM) with a live pid = a deliberate restart
              # (launchctl kickstart -k), not a crash
              and not (rows[a][1] == "-15" and rows[a][0] != "-")]
    if missing or failed:
        return "RED", f"missing: {missing or '—'}  failing: {failed or '—'}"
    return "OK", f"all {len(AGENTS)} agents loaded, last exits clean"


def _check_systemd_units() -> tuple[str, str]:
    """VPS supervisor check: any failed plec-* unit is RED; the run-logger daemon
    (KeepAlive equivalent) must be active; report the scheduled-timer tally."""
    def _run(args: list[str]) -> str:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout
    try:
        failed = _run(["systemctl", "--failed", "--no-legend", "--plain"])
    except Exception as exc:
        return "WARN", f"systemctl unavailable: {exc}"
    plec_failed = [ln.split()[0] for ln in failed.splitlines()
                   if ln.strip().startswith("plec-")]
    if plec_failed:
        return "RED", f"failed units: {plec_failed}"
    try:
        active = _run(["systemctl", "is-active", "plec-run-logger.service"]).strip()
    except Exception:
        active = "unknown"
    if active != "active":
        return "RED", f"plec-run-logger.service {active} (KeepAlive daemon down?)"
    try:
        timers = _run(["systemctl", "list-timers", "--all", "--no-legend", "--plain"])
        n_timers = sum(1 for ln in timers.splitlines() if "plec-" in ln)
    except Exception:
        n_timers = 0
    return "OK", f"no failed plec units; run-logger active; {n_timers} plec timers scheduled"


def check_run_logger_process() -> tuple[str, str]:
    try:
        out = subprocess.run(["pgrep", "-f", "run_logger.py"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:
        return "WARN", f"pgrep unavailable: {exc}"
    return ("OK", f"alive (pid {out.split()[0]})") if out else \
           ("RED", "run_logger.py not running (KeepAlive agent dead?)")


def check_power_source() -> tuple[str, str]:
    """Lid-close on battery sleeps the machine regardless of caffeinate — the
    whole stack silently stops (learned 2026-07-15: 5h orderbook hole). AC is
    the only safe unattended state; low battery threatens imminent shutdown.
    A server / any host without pmset has no battery to worry about — skip
    cleanly rather than perma-WARN (VPS, 2026-08-07)."""
    if not shutil.which("pmset"):
        return "OK", "n/a (no battery — server/AC host)"
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception as exc:
        return "WARN", f"pmset unavailable: {exc}"
    m = re.search(r"(\d+)%", out)
    pct = f"{m.group(1)}%" if m else "?%"
    if "Battery Power" not in out:
        return "OK", f"on AC power ({pct})"
    if m and int(m.group(1)) < 25:
        return "RED", f"on battery at {pct} — stack dies at 0; plug in NOW"
    return "WARN", f"on battery ({pct}) — lid-close would sleep the stack; plug in"


def check_calibration_age() -> tuple[str, str]:
    msgs, worst = [], "OK"
    for name, path, warn_d, red_d in (("warming_curve", WARM_CURVE, 14, 30),
                                      ("prob_map", PROB_MAP, 9, 16)):
        if not path.exists():
            msgs.append(f"{name}: MISSING")
            worst = "RED"
            continue
        age_d = _age_h(path.stat().st_mtime) / 24
        msgs.append(f"{name}: {age_d:.1f}d")
        if age_d > red_d:
            worst = "RED"
        elif age_d > warn_d and worst == "OK":
            worst = "WARN"
    return worst, ", ".join(msgs)


# ── CONTENT ──────────────────────────────────────────────────────────────────

def _latest_run_rows() -> list[dict]:
    try:
        rows = [json.loads(l) for l in open(SIGNALS) if l.strip()]
    except OSError:
        return []
    if not rows:
        return []
    latest = max(r.get("run_ts", "") for r in rows)
    return [r for r in rows if r.get("run_ts") == latest]


def check_run_breadth() -> tuple[str, str]:
    """Listing-aware: mornings legitimately have ONE settlement day (~120 rows)
    until Kalshi lists tomorrow's markets — judge breadth per listed day, not by
    a fixed total. Thin = cities missing or the fullest day under ~100 rows."""
    rows = _latest_run_rows()
    cities = {r.get("city") for r in rows if r.get("city")}
    by_day: dict = {}
    for r in rows:
        by_day[str(r.get("settlement_date"))] = by_day.get(str(r.get("settlement_date")), 0) + 1
    fullest = max(by_day.values()) if by_day else 0
    detail = (f"{len(rows)} rows / {len(cities)} cities / "
              f"{len(by_day)} settlement day(s) {by_day}")
    if len(cities) < 15 or fullest < 100:
        return "RED", f"latest run thin: {detail} (fullest day should be ~120)"
    return "OK", detail


def check_run_nulls() -> tuple[str, str]:
    rows = _latest_run_rows()
    if not rows:
        return "RED", "no rows in latest run"
    p_ok = sum(r.get("prob_estimate") is not None for r in rows) / len(rows)
    m_ok = sum(r.get("market_mid") is not None for r in rows) / len(rows)
    if p_ok < 0.95:
        return "RED", f"prob_estimate null in {100 * (1 - p_ok):.0f}% of rows"
    if m_ok < 0.50:
        return "WARN", f"market_mid null in {100 * (1 - m_ok):.0f}% of rows (thin books or feed issue)"
    return "OK", f"prob {100 * p_ok:.0f}% / mid {100 * m_ok:.0f}% populated"


def check_kalshi_canary() -> tuple[str, str]:
    """Catch API contract drift (the yes_bid → yes_bid_dollars rename made every
    book check silently dead). Probe one live market and require a parseable book."""
    url = (f"https://api.elections.kalshi.com/trade-api/v2/markets"
           f"?series_ticker={CANARY_SERIES}&status=open&limit=5")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "plec-health-check"})
        markets = json.loads(urllib.request.urlopen(req, timeout=15).read()).get("markets", [])
    except Exception as exc:
        return "WARN", f"canary request failed (transient?): {exc}"
    if not markets:
        return "WARN", f"no open {CANARY_SERIES} markets to probe"
    for m in markets:
        try:
            bid = float(m.get("yes_bid_dollars") or 0)
            ask = float(m.get("yes_ask_dollars") or 0)
        except (TypeError, ValueError):
            break
        if 0 < bid <= ask <= 1:
            return "OK", f"{m.get('ticker')} book {bid:.2f}/{ask:.2f} parses"
    return "RED", ("no probed market yields a parseable yes_bid_dollars/yes_ask_dollars "
                   "book — API contract drift or degenerate books")


def check_push_channel() -> tuple[str, str]:
    """Dead-man for the alerting channel itself: if NTFY_TOPIC doesn't resolve,
    every push in the system (trade AND health) is silently disabled — and the
    senders won't say so (notify_transition returns True with no topic)."""
    from kalshi_weather.dashboard.notifications import _topic
    t = _topic()
    if not t:
        return "RED", "NTFY_TOPIC unresolved — ALL pushes silently disabled (.env not loaded?)"
    return "OK", f"topic '{t[:6]}…' resolves"


def check_board_meta_alive() -> tuple[str, str]:
    """If every vetted candidate says 'Kalshi verify unavailable', the vet's live
    re-verify integration is dead even though boards keep being written."""
    # boards are timestamped YYYY-MM-DDTHHMMSS.json; the dir also holds
    # forward_stats.json etc., which sort after them lexicographically
    files = sorted(f for f in CANDIDATES.glob("*.json") if f.name[:4].isdigit())
    if not files:
        return "WARN", "no boards yet"
    try:
        verdicts = json.loads(files[-1].read_text()).get("verdicts", [])
    except (OSError, json.JSONDecodeError):
        return "RED", f"newest board unreadable: {files[-1].name}"
    metas = [c for v in verdicts for c in v.get("checks", []) if c.get("name") == "meta"]
    if not metas:
        return "WARN", "no meta checks in newest board (ran --no-api?)"
    ok = sum(c.get("sev") == "OK" for c in metas)
    if ok == 0:
        return "RED", f"0/{len(metas)} live meta verifies succeeded — integration dead"
    return "OK", f"{ok}/{len(metas)} live meta verifies succeeded"


CHECKS = [
    ("signals fresh",       check_signals_fresh),
    ("forecast archive",    check_forecast_archive),
    ("orderbook logger",    check_orderbook_logger),
    ("ensemble logger",     check_ensemble_logger),
    ("candidate boards",    check_candidate_boards),
    ("labels fresh",        check_labels_fresh),
    ("scheduler markers",   check_scheduler_markers),
    ("supervisor",          check_supervisor),
    ("run_logger process",  check_run_logger_process),
    ("power source",        check_power_source),
    ("calibration age",     check_calibration_age),
    ("run breadth",         check_run_breadth),
    ("run null fields",     check_run_nulls),
    ("kalshi canary",       check_kalshi_canary),
    ("push channel",        check_push_channel),
    ("board meta alive",    check_board_meta_alive),
]


def run_checks() -> list[tuple[str, str, str]]:
    results = []
    for name, fn in CHECKS:
        try:
            status, detail = fn()
        except Exception as exc:          # a crashing check is itself a red
            status, detail = "RED", f"check crashed: {exc}"
        results.append((status, name, detail))
    return results


def should_alert(reds: list[str], now: float | None = None) -> tuple[bool, str | None]:
    """(push?, kind) with cooldown: push when the RED set changes or 6h pass;
    push 'recovered' once when a red episode fully clears."""
    now = now or time.time()
    try:
        prev = json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        prev = {"reds": [], "ts": 0}
    if reds:
        if sorted(reds) != sorted(prev.get("reds", [])) or \
                now - prev.get("ts", 0) > ALERT_COOLDOWN_S:
            return True, "red"
        return False, None
    if prev.get("reds"):
        return True, "recovered"
    return False, None


def _save_state(reds: list[str], now: float | None = None) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"reds": sorted(reds), "ts": now or time.time()}))


def report(push: bool = True) -> str:
    results = run_checks()
    reds = [n for s, n, _ in results if s == "RED"]
    warns = [n for s, n, _ in results if s == "WARN"]
    icon = {"OK": "✅", "WARN": "⚠️ ", "RED": "🔴"}
    lines = [f"{icon[s]} {s:<4} {n:<20} {d}" for s, n, d in results]
    head = ("ALL GREEN" if not reds and not warns else
            f"{len(reds)} RED / {len(warns)} WARN")
    out = f"SYSTEM HEALTH — {head}\n" + "\n".join(lines)

    if push:
        from kalshi_weather.dashboard.notifications import notify_health
        do, kind = should_alert(reds)
        if do and kind == "red":
            detail = "\n".join(f"{n}: {d}" for s, n, d in results if s == "RED")
            notify_health(f"System health: {len(reds)} RED", detail, priority="high")
        elif do and kind == "recovered":
            notify_health("System health recovered", "all checks green again",
                          priority="default", tags="white_check_mark")
        _save_state(reds)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    print(report(push=not args.no_push))
