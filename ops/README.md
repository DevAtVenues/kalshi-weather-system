# Ops / scheduling

> **Production runs 24/7 on a cloud VPS** (DigitalOcean droplet, full stack
> under systemd `plec-*` units, monitored by the hourly health check). This
> runbook documents the **repo-tracked launchd definitions**, which are the
> canonical job specs and also run the identical stack on the development
> laptop as a redundant mirror (the slot guard makes the parallel schedulers
> double-fire-safe). The laptop-specific notes below apply to that mirror.

## The agent fleet (all launchd — this is a laptop that sleeps)

macOS `cron` does **not** run a job missed during sleep, and a plain background
process dies on sleep/reboot and never restarts. Everything here therefore runs as a
**launchd agent** so it either catches up on wake or is kept alive. Tracked plists in
`ops/launchd/`; live copies in `~/Library/LaunchAgents/`.

| Agent | Script | Schedule | Kind |
|-------|--------|----------|------|
| `com.plec.run-logger` | `run_logger.py` | **24/7** (KeepAlive) | orderbook every 5min + ensemble members — data is UNRECOVERABLE if missed |
| `com.plec.run-live` | `run_live.py` | 01/07/13/19 local | signal generation |
| `com.plec.archive-forecasts` | `archive_forecasts.py` | 06/12/18/23 local | multi-model forecast archive |
| `com.plec.daily-review` | `daily_review.py` | 10:00 local | grade + forward report + watchdog |
| `com.plec.candidate-pipeline` | `candidate_pipeline.py --scheduled` | 01:20/07:20/13:20/19:20 local | two-stage candidate board + auto-vet, 20 min after each run_live cycle |
| `com.plec.health-check` | `health_check.py` | hourly + on load/wake | system watchdog: 15 liveness/content checks, ntfy push on RED (transition-gated, 6h cooldown) |
| `com.plec.stay-awake` | `caffeinate -s` | **24/7** (KeepAlive) | prevents system sleep ON AC ONLY (battery sleeps normally); replaces hand-run caffeinate — the 7/12 overnight data gap was a closed terminal |
| `com.plec.position-watch` | `position_watch.py --scheduled` | every 30 min | pushes when a watchlist contract's mid moves ≥7¢ or fresh models erode the thesis; positions in `data/analysis/watchlist.json` |
| `com.plec.lock-scanner` | `lock_scan.py --scheduled` | every 10 min, self-gated to 1–9 PM ET | playbook trade #1: mispriced settled-outcome scanner, push-on-hit (deduped). No health-check row on purpose: a missed scan loses an opportunity, not data. |
| `com.plec.fast-obs-watch` | `fast_obs_watch.py --scheduled` | every 5 min, self-gated to watched stations' local 10:00–20:00 | thermometer-side sibling of position-watch (same watchlist file): merges the 5-min obs feed + hourly METAR tenths into a running band, pushes on stage escalation (warn→threat→win/dead), once per stage per day. No health-check row (lock-scanner doctrine). |

**Env for scheduled jobs:** launchd and cron do NOT inherit shell env. All secrets
(`NTFY_TOPIC`, `KALSHI_API_KEY`, …) live in the project `.env`, loaded centrally by
`src/kalshi_weather/__init__.py` on first package import — never rely on exported
shell variables for anything scheduled. The health check's `push channel` line
verifies this resolves.

## Change safety (three layers — all mandatory)

1. **Commit gate**: `.githooks/pre-commit` + `pre-push` run the unit suite AND
   `scripts/smoke_test.py` (imports every scheduled entry point, exercises the real
   pipelines read-only, pins the contracts that have broken before: ntfy header
   encodability, .env resolution, warming-curve sanity, signals breadth/nulls,
   verdict payload fields, no partial today-labels). Activate once per clone:
   `git config core.hooksPath .githooks`. Never `--no-verify`.
   **pre-push additionally runs `scripts/replay_check.py --days 3`**: re-derives
   the last 3 days of production probabilities/edges/directions through the
   current code and blocks on unacknowledged drift. Intentional model change →
   `REPLAY_ACK=1 git push` (the diff still prints — reviewed, not rubber-stamped).
   Every signals row carries provenance (`code_sha`, `calib_hash`, `curve_hash` —
   `kalshi_weather/provenance.py`), so replay compares only rows produced under
   the current calibration artifacts and any pick is attributable to an exact
   model version.
2. **Entry preflights**: run_live / candidate_pipeline / daily_review call
   `kalshi_weather.preflight.preflight()` and exit 78 (+ push) rather than run
   hollow when env/artifacts are missing.
3. **Runtime watchdog**: `com.plec.health-check` verifies outputs 4x/day, pushes
   on RED (see fleet table).

**New scheduled job checklist:** plist in `ops/launchd/` → `acquire_slot` guard →
`preflight` requirements entry → `health_check` freshness check → smoke contract
if it writes a new artifact → row in the fleet table above.

Install/verify all:
```sh
for p in ops/launchd/com.plec.*.plist; do cp "$p" ~/Library/LaunchAgents/; \
  launchctl load ~/Library/LaunchAgents/$(basename "$p"); done
launchctl list | grep com.plec        # 2nd column 0 = healthy; run-logger shows a live PID
```

`candidate-pipeline` trails each `run-live` window by 20 min so Stage 1 reads the
signals that cycle just wrote; boards land in `data/analysis/candidates/` (launchd
only — no cron backup; a missed board is regenerated next window). The scheduled
agents (run-live/archive/daily-review) still have matching **cron**
entries as a belt-and-suspenders backup; a per-job guard
(`src/kalshi_weather/scheduling.py`: flock + 30-min `last_run` marker, fail-open)
stops cron + launchd double-firing. `run-logger` is KeepAlive (RunAtLoad + restart
on exit) — check it's alive with `pgrep -f run_logger.py`.

### Wake the machine for the jobs (pmset)
launchd catches up *late* (on next wake). For on-time runs and time-critical captures,
schedule the Mac to WAKE. Needs admin:
```sh
sudo pmset repeat wake MTWRFSU 09:55:00    # wake daily before the 10:00 review
```
`pmset repeat` holds one time; for more windows use a `pmset schedule` helper or keep
the lid open during trading hours. Check with `pmset -g sched`.

---

## daily_review — launchd agent (macOS)

`scripts/daily_review.py` is the flywheel heartbeat: grade newly-settled picks,
forward report, watchdog, weekly calibration refit. It must run once a day.

### Why launchd, not cron
This runs on a **laptop**. macOS `cron` does **not** run a missed job after the
machine wakes, so any day the lid is closed at the scheduled time the review is
silently skipped. `launchd` with `StartCalendarInterval` runs the job on the next
**wake** if it was asleep at the scheduled time — which is what we want.

### Install
```sh
cp ops/launchd/com.plec.daily-review.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.plec.daily-review.plist
launchctl list | grep com.plec.daily-review        # 2nd col 0 = healthy
launchctl kickstart -k gui/$(id -u)/com.plec.daily-review   # optional: run now to test
```
Scheduled for **10:00 local**. Output appends to `logs/daily_review/cron.out`;
the run also tees to `logs/daily_review/YYYY-MM-DD.log`.

### Retiring the old cron line
There was a `0 10 * * *` daily_review line in `crontab`. It can stay as a harmless
backup — `daily_review.py` now runs **at most once per local day** (flock mutex +
`logs/daily_review/.last_completed_day` marker), so cron + launchd firing together
is safe (the second exits). To remove it anyway (needs a terminal with Full Disk
Access; a sandboxed agent cannot edit cron):
```sh
crontab -l | grep -v daily_review.py | crontab -
```

### Manual run
```sh
.venv/bin/python scripts/daily_review.py            # no-ops if today already ran
.venv/bin/python scripts/daily_review.py --force     # force a re-run
```

## Directory rename (2026-07-10) — path compatibility symlink

The project moved from `/Users/josiah/PLEC projects/…` (space) to
`/Users/josiah/PLEC-projects/…` (hyphen). That broke the absolute paths in both the
launchd agent and every `crontab` entry (`run_live` 4×/day, `archive_forecasts`
4×/day, `daily_review`).

Fixes applied:
- **launchd agent** repointed to the new canonical path (this plist).
- **crontab** can't be edited from the sandboxed agent (TCC), so a compatibility
  **symlink** keeps every cron entry working with zero edits:
  ```sh
  ln -s /Users/josiah/PLEC-projects "/Users/josiah/PLEC projects"
  ```
  The old space-path now resolves to the new location.

**⚠️ The symlink is LOAD-BEARING — do not just delete it.** The venv's editable
install (`pip install -e`) recorded the old space-path, so `import kalshi_weather` resolves through the symlink. Removing it breaks imports for every
agent and cron job, not just path strings.

To fully migrate off the old path (in a Full-Disk-Access terminal): repoint crontab,
reinstall the editable package at the new path, THEN drop the symlink —
```sh
crontab -l | sed 's#/Users/josiah/PLEC projects/#/Users/josiah/PLEC-projects/#g' | crontab -
cd /Users/josiah/PLEC-projects/Prediction-Markets-Project && .venv/bin/pip install -e .
rm "/Users/josiah/PLEC projects"   # only after the reinstall; removes the symlink, not the dir
```
