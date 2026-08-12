"""
Daily review driver — the heartbeat of the recalibration flywheel.

Runs once a day (cron/launchd, after ~12Z when the prior day's NWS CLI is published):
  1. Grades every newly-settled pick against actuals (unblocks learning).
  2. Prints the forward calibration report (reliability + realized edge by engine).
  3. WATCHDOG: if grading has fallen > STALE_DAYS behind the logged picks, warn loudly
     (and ntfy if configured) — this is what would have caught the 2-week stall.
  4. Appends the whole run to logs/daily_review/YYYY-MM-DD.log for the record.

Usage:
    .venv/bin/python scripts/daily_review.py            # runs at most once per local day
    .venv/bin/python scripts/daily_review.py --force     # re-run (manual / debugging)

Scheduling: a launchd agent (~/Library/LaunchAgents/com.plec.daily-review.plist,
10:00 local) is primary — unlike cron it runs on the next WAKE if the laptop was
asleep at the scheduled time. The legacy 10:00 cron line may still fire as a backup;
the once-per-day lock + marker below make a double trigger safe (the second exits).
"""
from __future__ import annotations

import fcntl
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from kalshi_weather.outcome_tracker import grade_signals, forward_report  # noqa: E402

SIGNALS  = ROOT / "data" / "signals" / "signals_log.jsonl"
OUTCOMES = ROOT / "data" / "outcomes" / "signal_outcomes.jsonl"
STALE_DAYS = 2   # grading is "behind" if graded max is > this many days before the newest settled pick

# Once-per-day + concurrency guard state (so cron + launchd can't double-run and race
# on the outcomes file). LOCK is held for the whole run; DONE_MARKER records the last
# completed local day.
_STATE_DIR  = ROOT / "logs" / "daily_review"
LOCK_PATH   = _STATE_DIR / ".daily_review.lock"
DONE_MARKER = _STATE_DIR / ".last_completed_day"


def _run_captured(script: Path, *args: str) -> str:
    """Run a child script and return its output as a string. Direct subprocess.run
    would write to the raw stdout fd, bypassing the _Tee — the child's output then
    lands out of order in cron logs and never reaches the daily-review log file."""
    import subprocess
    r = subprocess.run([sys.executable, str(script), *args],
                       check=False, capture_output=True, text=True)
    out = r.stdout
    if r.returncode != 0 and r.stderr.strip():
        out += f"\n(child exited {r.returncode}: {r.stderr.strip().splitlines()[-1]})"
    return out


def _max_settled(path: Path, today: date) -> date | None:
    """Newest settlement_date in a jsonl log that is strictly before today (i.e. settled)."""
    mx: date | None = None
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        try:
            d = date.fromisoformat(json.loads(line).get("settlement_date", ""))
        except (json.JSONDecodeError, ValueError):
            continue
        if d < today and (mx is None or d > mx):
            mx = d
    return mx


def _staleness_check(today: date) -> str:
    logged = _max_settled(SIGNALS, today)
    graded = _max_settled(OUTCOMES, today)
    if logged is None:
        return "no settled picks logged yet"
    if graded is None:
        gap = (today - logged).days
        return f"⚠ WATCHDOG: nothing graded yet (newest settled pick {logged})"
    gap = (logged - graded).days
    msg = f"grading up to {graded}, newest settled pick {logged} (gap {gap}d)"
    if gap > STALE_DAYS:
        warn = f"⚠ WATCHDOG: grading is {gap} days behind — the flywheel is stalling. {msg}"
        try:
            from kalshi_weather.dashboard.notifications import notify_feed_stale  # best-effort ntfy
            notify_feed_stale(gap * 24 * 60.0)
        except Exception:
            pass
        return warn
    return "✓ " + msg


MIN_DAYS_TO_CALIBRATE = 10   # charter: don't fit a bias correction below this many independent days


def _calibration_readiness() -> str:
    """
    Per-ensemble-city count of DISTINCT graded settlement days with REAL ens_center_err
    (actual − ens_p50). Only real ens_p50 data counts toward readiness, because:
      • forecast_error_f (GFS single-run vs actual) is a DIFFERENT measurement
      • the proxy diverges from ens_center_err during extreme events (heat waves etc.)
      • center_bias._measured_shift uses only real ens_center_err for the same reason
    Shows the current applied center_shift from center_bias.py so we can see what the
    live engine is using right now and track whether the correction is making sense.
    """
    import statistics as st
    # Real ens_center_err only (no proxy) — must match center_bias._measured_shift.
    # Deduplicate to latest run_ts per ticker first; the outcomes file accumulates many
    # intraday snapshots of each contract as the ensemble is re-priced, and earlier
    # snapshots have different ens_p50 values (stale model run). Using all snapshots
    # unweighted would dilute the correct final reading toward stale early estimates.
    latest_per_ticker: dict[str, dict] = {}
    for line in OUTCOMES.read_text().splitlines() if OUTCOMES.exists() else []:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("prob_source") != "ensemble" or r.get("ens_center_err") is None:
            continue
        ticker = r.get("ticker", "")
        run_ts = str(r.get("run_ts", ""))
        if ticker not in latest_per_ticker or run_ts > str(latest_per_ticker[ticker].get("run_ts", "")):
            latest_per_ticker[ticker] = r
    by_city_day: dict[str, dict[str, list[float]]] = {}
    for r in latest_per_ticker.values():
        by_city_day.setdefault(r.get("city", "?"), {}).setdefault(r.get("settlement_date", ""), []).append(r["ens_center_err"])

    try:
        from kalshi_weather.calibration.center_bias import center_shift as _cs
        from kalshi_weather.calibration.bias import load_bias_table as _lbt
        _bt = _lbt()
        _month = date.today().month
        _cs_fn = lambda station: _cs(station, _month, _bt)
    except Exception:
        _cs_fn = lambda station: float("nan")

    # city → NWS station (for center_shift lookup)
    from kalshi_weather.live.runner import CITY_CONFIGS
    _station_for = {k: v["nws_station"] for k, v in CITY_CONFIGS.items()}

    lines = ["\nENSEMBLE CENTER-BIAS READINESS (P1.0 — real ens_p50 days only; "
             f"need {MIN_DAYS_TO_CALIBRATE}+ to trust measured component):"]
    for city in sorted(by_city_day):
        days = by_city_day[city]
        day_means = [st.mean(v) for v in days.values()]
        n = len(days)
        bias = st.mean(day_means)
        station = _station_for.get(city, "?")
        try:
            applied = _cs_fn(station)
            cs_str = f"  applied now: {applied:+.2f}°F"
        except Exception:
            cs_str = ""
        ready = "✅ READY TO FIT" if n >= MIN_DAYS_TO_CALIBRATE else f"⏳ {MIN_DAYS_TO_CALIBRATE - n} more real days"
        lines.append(f"  {city:<4} {n:>2} real days  mean center err {bias:+.2f}°F{cs_str}   {ready}")
    return "\n".join(lines)


def main() -> None:
    today = date.today()
    print("=" * 64)
    print(f"  DAILY REVIEW — {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 64)

    # System health first: if a feed/logger/agent is silently dead, every number
    # below is suspect. Pushes on RED (cooldown shared with the 4x/day launchd runs).
    try:
        import health_check
        print(health_check.report(push=True), "\n")
    except Exception as exc:
        print(f"(health check skipped: {exc})\n")

    # Replay: does the current code still reproduce the week's production output?
    # Informational here (the push gate enforces it) — drift showing up in the
    # review means something changed OUTSIDE a gated push (artifact edit, hotfix).
    try:
        from replay_check import report as replay_report
        text, drift = replay_report(days=7)
        print(text)
        if drift:
            print("⚠ replay drift outside a gated push — investigate before trusting today's picks")
        print()
    except Exception as exc:
        print(f"(replay check skipped: {exc})\n")

    n = grade_signals()
    print(f"(graded {n} newly-settled signal rows)")
    print("\nFLYWHEEL WATCHDOG:", _staleness_check(today), "\n")

    # Weekly (Mondays): refit the probability recalibration map on the freshly-graded
    # outcomes so it tracks the model as it drifts. get_calibration() is mtime-cached, so
    # running engines pick up the new map without a restart.
    if today.weekday() == 0:
        print("Weekly refit of the probability calibration map…")
        print(_run_captured(ROOT / "scripts" / "build_calibration_map.py"))

    forward_report(min_n=5)
    print(_calibration_readiness())

    # Skill-vs-market control chart: is the model still valid THIS WEEK? Pushes
    # on status transitions (market significantly beats us / systematic bias).
    try:
        import skill_monitor
        print("\n" + skill_monitor.report(push=True))
    except Exception as exc:
        print(f"\n(skill monitor skipped: {exc})")

    # Morning-edge: are our early disagreements actually front-running the market?
    try:
        from morning_edge import analyse, report  # scripts/ is on sys.path via __file__ dir
        print("\nMORNING-EDGE TRACKER (do our early forecasts front-run the market?):")
        print(report(analyse(min_dis=0.10)))
    except Exception as exc:
        print(f"\n(morning-edge tracker skipped: {exc})")

    # Gate analysis: is each push gate saving us (filtering losers) or costing us?
    try:
        import gate_analysis
        print("\nGATE ANALYSIS (is each push gate saving us or costing us?):")
        print(gate_analysis.report())
    except Exception as exc:
        print(f"\n(gate analysis skipped: {exc})")

    # Credibility gate analysis: are REJECT/WARN picks actually losing more?
    try:
        import credibility_analysis
        print("\nCREDIBILITY GATE ANALYSIS (are dashboard REJECT picks losing more?):")
        print(credibility_analysis.report())
    except Exception as exc:
        print(f"\n(credibility analysis skipped: {exc})")

    # Candidate-verdict grading: does the Stage-2 vet add value over the raw board,
    # and what did gate-suppressed near-misses actually return?
    try:
        import grade_candidates
        print("\nCANDIDATE VERDICT GRADING (does the Stage-2 vet add value?):")
        print(grade_candidates.report())
    except Exception as exc:
        print(f"\n(candidate grading skipped: {exc})")

    # Model skill: which forecast model wins per city (from the multi-model archive)?
    try:
        print("\nMODEL SKILL (which model wins per city, lead 1d):")
        print(_run_captured(ROOT / "scripts" / "model_skill.py", "--lead", "1"))
    except Exception as exc:
        print(f"\n(model skill skipped: {exc})")

    # Price-path shadow book (dormant pilot): advance the paper book one day and
    # report the forward per-policy record. Log-only, no orders, no network —
    # accrues the evidence an activation decision would need (STATE Part V).
    try:
        import price_path_shadow
        print("\nPRICE-PATH SHADOW BOOK (dormant pilot):")
        print(price_path_shadow.run_and_report())
    except Exception as exc:
        print(f"\n(price-path shadow skipped: {exc})")

    print("\nNext: read the reliability table + by-engine edge above; log anything off")
    print("into private/docs-archive/recalibration-backlog.md.")


class _Tee:
    """Write stdout to both the console and today's review log."""
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()


def _run_guarded(force: bool) -> None:
    """Run main() at most once per local day, and never concurrently. Returns
    quietly if another instance holds the lock or today already completed."""
    today_str = date.today().isoformat()
    _STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Non-blocking exclusive lock: if a sibling trigger (cron vs launchd) is already
    # running, this instance backs off instead of racing on the outcomes file.
    lock_fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("daily_review: another instance is already running — skipping.")
        return

    try:
        if not force and DONE_MARKER.exists() and DONE_MARKER.read_text().strip() == today_str:
            print(f"daily_review: already completed for {today_str} — skipping (use --force to re-run).")
            return
        with open(_STATE_DIR / f"{today_str}.log", "a") as fh:
            sys.stdout = _Tee(sys.__stdout__, fh)
            try:
                main()
            finally:
                sys.stdout = sys.__stdout__   # restore before fh closes (avoids shutdown flush error)
        DONE_MARKER.write_text(today_str)   # only marked done after a clean run
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


if __name__ == "__main__":
    from kalshi_weather.preflight import preflight
    preflight("daily_review")     # refuse to run hollow
    _run_guarded(force="--force" in sys.argv[1:])
