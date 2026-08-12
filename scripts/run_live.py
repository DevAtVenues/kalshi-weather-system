"""
Live trade signal runner.

Loads production rules, fetches currently open NYC and Chicago markets,
applies the rule to each T-type contract using today's GFS forecast,
and prints actionable trade signals.

Run:   .venv/bin/python scripts/run_live.py
Quiet: .venv/bin/python scripts/run_live.py --no-log
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd

from kalshi_weather.live.risk import RiskManager
from kalshi_weather.live.executor import maybe_executor
from kalshi_weather.live.runner import (
    CITY_CONFIGS, ENSEMBLE_CITIES, print_signals, run_signals, MIN_BID, MAX_SPREAD,
)
from kalshi_weather.dashboard.notifications import notify_transition
from kalshi_weather.ingest.metar import fetch_running_high_f
from kalshi_weather.monitor.lock import METAR_MARGIN_F
from kalshi_weather.dashboard.store import (
    _lock_actionable_edge_c,
    LOCK_NOTIFY_MIN_EDGE_C,
    LOCK_NOTIFY_MAX_EDGE_C,
    _pick_min_edge,
    _OOS_CI,
    MAX_SPREAD_F,
)
import requests

MODEL_PATH = Path(__file__).parents[1] / "data" / "models" / "production_rules.pkl"
LOG_PATH   = Path(__file__).parents[1] / "data" / "signals" / "signals_log.jsonl"
# Counterfactual gate log: for every ensemble+OOS candidate, which gate held it (or FIRED)
# + the gate inputs, so we can later grade each gate — did it save us (filter losers) or
# cost us (filter winners)? Lets us tune the eyeball-set thresholds from data.
GATE_LOG   = Path(__file__).parents[1] / "data" / "analysis" / "gate_log.jsonl"
# Tracks what was already notified today so repeat runs don't re-spam. Shared with
# scripts/check_locks.py (the intraday lock checker). Resets each calendar day.
NOTIFY_STATE = LOG_PATH.parent / "notified_today.json"

# Notification gating (forecast cron path).
# A "lock" means the OUTCOME is settled by observation — that lives in the obs path
# (push_obs_locks / the dashboard), NEVER here. This cron sees the day-ahead forecast
# and surfaces +EV PICKS, not certainties: a pick is a positive-expected-value bet
# (we expect to win most, lose some). Two gates, both from the dashboard's validated
# framework (replacing the old flat edge ceiling, which suppressed real edge):
#   1. edge >= _pick_min_edge(city, t_dir): the bar scales to out-of-sample evidence
#      (0.10 where the 2025 holdout CI is wholly positive, higher where weaker).
#   2. multi-model agreement: GFS/ECMWF/NBM must agree within MAX_SPREAD_F, which
#      rejects single-model (e.g. GFS-cold) outliers that masquerade as huge edge.


def _model_spread_f(cfg: dict, date_str: str) -> float | None:
    """
    Max−min of three deterministic daily-high forecasts (°F) for a city-date. A wide
    spread means the models can't agree which side of a contract boundary the high
    lands on — the "edge" is single-model noise, not signal.
    Returns None if fewer than two models answer (can't judge agreement).

    NOTE: gfs_seamless IS HRRR within the ~2-day lead where picks can fire
    (HORIZON_NEVER_FIRE_H=36), so the trio is effectively HRRR/ECMWF/NBM — per the
    corrected model-skill table (P1.4a) HRRR is the top model, so keep it. Do NOT
    "fix" this to gfs_global without re-baselining the gate log.
    """
    vals: list[float] = []
    for m in ("gfs_seamless", "ecmwf_ifs025", "ncep_nbm_conus"):
        try:
            p = {
                "latitude": cfg["lat"], "longitude": cfg["lon"],
                "daily": "temperature_2m_max", "temperature_unit": "fahrenheit",
                "timezone": "UTC", "start_date": date_str, "end_date": date_str,
                "models": m,
            }
            j = requests.get("https://api.open-meteo.com/v1/forecast", params=p, timeout=15).json()
            t = j.get("daily", {}).get("temperature_2m_max", [None])[0]
            if t is not None:
                vals.append(float(t))
        except Exception:
            pass
    return (max(vals) - min(vals)) if len(vals) >= 2 else None


# ── Settlement-horizon gating (Move 3) ────────────────────────────────────────
# Forecast reliability decays with lead time: a day-ahead high swings run-to-run,
# so a small "edge" far out is mostly noise, not signal. Scale the required edge up
# with hours-to-settlement and hard-stop beyond the never-fire horizon. Same-day
# picks (after midnight LST, before 2 PM LST) now also fire, with a higher edge
# bar (SAME_DAY_EDGE_BUMP). The obs-lock path handles confirmed peaks after 2 PM.
HORIZON_NEVER_FIRE_H = 36.0   # never fire a forecast pick whose target day starts >36h out
HORIZON_BASE_H       = 12.0   # within this lead, keep the base evidence bar
HORIZON_EDGE_PER_6H  = 0.01   # extra required edge per 6h of lead beyond base
SAME_DAY_EDGE_BUMP   = 0.10   # P3.0: extra required edge for same-day picks (before 2 PM LST)
                               # makes effective Tier-1a bar 0.20 vs 0.10 for day-ahead

# ── P3.2: same-day intraday pricing is WATCH-ONLY (falsified live 2026-07-12) ─
# intraday_prob's linear sunrise→3pm warming taper assumes ~62% of the day's
# warming is done by noon. Checked against live obs 2026-07-12: MSP was already
# AT the model's projected final high at 1 PM with 3-4h of July climb left
# (model P(≤90)=0.96 vs market 0.04; DAL P(≤94)=0.995 at 92°F by 1 PM). The
# taper collapses upside far too early, so every same-day probability is biased
# toward "high stays near running high". Only the edge ceiling stopped these
# from pushing. Same-day picks stay logged + graded (accrue forward evidence)
# but cannot PUSH until the warming curve is refit from hourly obs and
# forward-validated. Flip to False only with that evidence in hand.
SAME_DAY_WATCH_ONLY  = True

# ── Forecast-stability gating (Move 5) ────────────────────────────────────────
# Betting into a forecast that is still moving is how a "real" edge reverts after
# the next model run. Hold the pick when the day is unresolved: the ensemble and the
# NWS human forecast disagree, or the (already dispersion-shrunk) ensemble is still
# wide. Both signals ride on each ensemble-priced signal row (from Move 1).
DISAGREE_MAX_F   = 2.5   # |NWS - ensemble median| above this = forecasts diverge -> hold
ENS_SD_WIDE_F    = 2.0   # post-shrink ensemble sd above this = unresolved day
STABILITY_BUMP   = 0.03  # extra required edge on a wide-dispersion day

# ── P1.7: B-type BUY_YES watch-only gate ──────────────────────────────────────
# The exclusive-cap fix (P1.6) removed FABRICATED YES edge on bracket contracts
# (center-in-bracket P was ~0.50, corrected to ~0.20-0.30). But the corrected
# BUY_YES direction is UNPROVEN forward: across 50,730 graded outcomes, actionable
# ensemble B-type BUY_YES won only 5% of the time and center-in-bracket had
# NEGATIVE true edge. So a B-type BUY_YES that now clears every gate must still be
# WATCH-ONLY — logged and graded (so the watch can accumulate) but NOT pushed —
# until we have direct forward evidence the fix corrected the calibration enough.
# Lifts only when >=10 graded post-fix actionable B-type BUY_YES outcomes realize
# >=15% YES (beating the -1.4% historic rate for the P∈[0.2,0.3) band these fall in).
# Without this gate, only dedup COINCIDENCE stopped the pre-fix NYC B89.5 false
# positive from re-pushing post-fix — a real B-type BUY_YES on a fresh day would fire.
OUTCOMES_PATH             = Path(__file__).parents[1] / "data" / "outcomes" / "signal_outcomes.jsonl"
BTYPE_YES_FIX_DATE        = "2026-07-10"  # exclusive-cap fix live; only grade outcomes on/after
BTYPE_YES_WATCH_MIN_N     = 10            # graded post-fix outcomes required to consider lifting
BTYPE_YES_WATCH_MIN_RATE  = 0.15          # realized YES rate required to lift (P1.7)


def _btype_yes_watch() -> tuple[bool, int, float]:
    """P1.7 watch state for post-fix ensemble B-type BUY_YES picks.

    Returns (watch_active, n_graded, realized_yes_rate). The watch is ACTIVE
    (suppress pushes) until n>=BTYPE_YES_WATCH_MIN_N graded post-fix actionable
    outcomes realize >=BTYPE_YES_WATCH_MIN_RATE YES. Counts only genuine forward
    evidence: prob_source=ensemble, strike_type=between, direction=BUY_YES,
    actionable, settlement_date>=fix date, and graded (yes_settled present).
    Deduplicates to the latest run_ts per ticker so re-priced snapshots of the
    same contract count once.
    """
    if not OUTCOMES_PATH.exists():
        return True, 0, 0.0
    latest: dict[str, dict] = {}
    for line in OUTCOMES_PATH.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("prob_source") != "ensemble" or r.get("strike_type") != "between"
                or r.get("direction") != "BUY_YES" or not r.get("actionable")):
            continue
        if str(r.get("settlement_date", "")) < BTYPE_YES_FIX_DATE:
            continue
        if r.get("yes_settled") is None:      # ungraded (future / not yet settled)
            continue
        tk  = r.get("ticker", "")
        rts = str(r.get("run_ts", ""))
        if tk not in latest or rts > str(latest[tk].get("run_ts", "")):
            latest[tk] = r
    n = len(latest)
    if n == 0:
        return True, 0, 0.0
    yes_rate = sum(1 for r in latest.values() if r.get("yes_settled")) / n
    active = not (n >= BTYPE_YES_WATCH_MIN_N and yes_rate >= BTYPE_YES_WATCH_MIN_RATE)
    return active, n, round(yes_rate, 3)


def _hours_to_settlement(sdate, nws_station: str) -> float | None:
    """Hours from now until the target LST day's settlement window opens (the lead
    time of the forecast bet). None if it can't be computed."""
    from kalshi_weather.tz import settlement_window_utc
    try:
        win_start, _ = settlement_window_utc(sdate, nws_station)
        return (win_start - datetime.now(tz=timezone.utc)).total_seconds() / 3600.0
    except Exception:
        return None


def _horizon_bump(hours: float | None) -> float:
    """Extra required edge added on top of the evidence bar, growing with lead time."""
    if hours is None or hours <= HORIZON_BASE_H:
        return 0.0
    return HORIZON_EDGE_PER_6H * ((hours - HORIZON_BASE_H) / 6.0)


def push_forecast_picks(signals: pd.DataFrame, state: dict,
                        executor=None) -> tuple[int, int]:
    """
    Surface +EV day-ahead forecast PICKS (not certainties). A row is pushed only if
    its edge clears the evidence-scaled bar AND the deterministic models agree.
    Returns (pushed, filtered_for_model_disagreement).
    """
    if signals.empty:
        return 0, 0
    pre_col    = signals.get("pre_settlement", pd.Series(True, index=signals.index))
    # Liquidity: require a two-sided, tight book — NOT the ATM 0.15-0.85 range,
    # which would drop a perfectly tradeable pick that's priced near a tail
    # (e.g. a 14c contract with a 2c spread). Same gate used for obs locks.
    liq = (
        np.isfinite(signals["yes_bid_dollars"]) & (signals["yes_bid_dollars"] > MIN_BID) &
        np.isfinite(signals["yes_ask_dollars"]) &
        np.isfinite(signals["spread"]) & (signals["spread"] < MAX_SPREAD)
    )
    actionable = signals[liq & pre_col]
    spread_cache: dict[tuple, float | None] = {}
    pushed = filtered = 0
    ts = datetime.now(tz=timezone.utc).isoformat()
    decisions: list[dict] = []
    # P1.7: is the B-type BUY_YES watch still active? Computed once (property of the
    # accumulated outcomes, not per-candidate). Suppresses would-fire B-type BUY_YES.
    btype_yes_watch_active, _byw_n, _byw_rate = _btype_yes_watch()

    def _dec(r, gate: str, sp: float | None = None) -> None:
        """Record which gate decided this candidate (or 'FIRED') + the gate inputs."""
        def _v(x):
            return None if (x is None or (isinstance(x, float) and pd.isna(x))) else x
        decisions.append({
            "run_ts": ts, "ticker": r["ticker"], "direction": r["direction"],
            "city": r.get("city"), "settlement_date": str(r["settlement_date"]),
            "t_direction": r.get("t_direction"), "gate": gate,
            "edge_raw": round(float(r["edge_raw"]), 4),
            "prob_estimate": _v(r.get("prob_estimate")), "market_mid": _v(r.get("market_mid")),
            "ens_p50": _v(r.get("ens_p50")), "nws_disagree": _v(r.get("nws_disagree")),
            "ens_sd": _v(r.get("ens_sd")), "hours_to_settle": _v(r.get("hours_to_settle")),
            "running_high": _v(r.get("running_high")),
            "is_same_day": bool(r.get("is_same_day")), "model_spread_f": _v(sp),
        })

    for _, r in actionable.iterrows():
        city = r.get("city")
        tdir = r.get("t_direction")
        edge = float(r["edge_raw"])
        cfg  = CITY_CONFIGS.get(city)

        # Gate -1 (calibration + validation): fire only an ENSEMBLE-priced pick on an
        # OOS-validated (city, type). Forward validation measured the single-run rule
        # badly over-predicts YES (predicted 0.5 -> realized 0.2), so never fire what it
        # priced; and never auto-fire an unvalidated market even when the ensemble now
        # prices it well. Both still appear in the ranked board — they just don't fire.
        # NOT gate-logged: this is the validation boundary, not a tunable threshold, and
        # it would flood the log with every rule/unvalidated contract.
        ens_cfg = ENSEMBLE_CITIES.get(city)
        ctype_v = "B" if tdir == "between" else "T"
        if (r.get("prob_source") != "ensemble"
                or ens_cfg is None or ctype_v not in ens_cfg.get("types", set())):
            continue

        # Gate 0: settlement horizon (Move 3). A forecast edge that far out swings too
        # much to trust — never fire beyond the cutoff, and require a bigger edge the
        # longer the lead, so only near-term, well-resolved days fire on a small edge.
        hrs = _hours_to_settlement(r["settlement_date"], cfg["nws_station"]) if cfg else None
        if hrs is not None and hrs > HORIZON_NEVER_FIRE_H:
            filtered += 1
            _dec(r, "horizon")
            continue

        # Gate 1: evidence-scaled minimum edge, raised by the settlement horizon.
        # P3.0: same-day picks (after midnight LST, before 2 PM LST) require
        # an additional SAME_DAY_EDGE_BUMP on top — reduces adverse-selection risk
        # when the settlement day has already started and variance is less resolved.
        same_day = bool(r.get("is_same_day"))
        min_e = _pick_min_edge(city, tdir) + _horizon_bump(hrs) + (SAME_DAY_EDGE_BUMP if same_day else 0.0)
        if edge < min_e:
            _dec(r, "edge_below_bar")
            continue

        # Gate 2: multi-model agreement (lazy — only for edge-cleared candidates).
        sd  = str(r["settlement_date"])
        ck  = (city, sd)
        if ck not in spread_cache:
            spread_cache[ck] = _model_spread_f(cfg, sd) if cfg else None
        sp = spread_cache[ck]
        if sp is None or sp > MAX_SPREAD_F:
            filtered += 1
            _dec(r, "model_spread", sp)
            continue

        # Gate 2b: forecast stability (Move 5). Hold on an unresolved day — the
        # ensemble and NWS human forecast disagree, or the shrunk ensemble is still
        # wide (require a bigger edge there). Signals ride on the ensemble-priced row.
        disagree = r.get("nws_disagree")
        if disagree is not None and pd.notna(disagree) and abs(float(disagree)) > DISAGREE_MAX_F:
            filtered += 1
            _dec(r, "nws_disagree", sp)
            continue
        ens_sd = r.get("ens_sd")
        if (ens_sd is not None and pd.notna(ens_sd) and float(ens_sd) > ENS_SD_WIDE_F
                and edge < min_e + STABILITY_BUMP):
            filtered += 1
            _dec(r, "ens_wide", sp)
            continue

        ci = (_OOS_CI.get(city) or {}).get("B" if tdir == "between" else "T")
        # Gate 3: evidence-scaled CEILING. Where the 2025 holdout proved a positive
        # edge, trust the magnitude up to the proven range (+buffer). Where we have
        # no holdout (or the CI crosses zero), a large edge is far more likely a
        # calibration error than a real mispricing on a liquid market — cap it.
        ceiling = (ci[1] + 0.10) if (ci and ci[0] > 0) else 0.25
        if edge > ceiling:
            filtered += 1
            _dec(r, "ceiling", sp)
            continue

        key = f"{r['ticker']}:{r['direction']}"
        if key in state["picks"]:
            _dec(r, "dedup", sp)
            continue

        # Gate 4 (P1.7): B-type BUY_YES is WATCH-ONLY until forward-proven. This
        # candidate cleared every edge/agreement/ceiling gate and WOULD fire, but the
        # corrected BUY_YES direction has no forward track record (historical 5% win
        # rate, negative true edge center-in-bracket). Suppress the push — the signal
        # is still logged and graded, so the watch accumulates without risking capital.
        if ctype_v == "B" and r["direction"] == "BUY_YES" and btype_yes_watch_active:
            _dec(r, "btype_yes_watch", sp)
            continue

        # Gate 5 (P3.2): same-day intraday pricing watch. The warming taper was
        # falsified against live obs (see SAME_DAY_WATCH_ONLY) — a same-day pick
        # that cleared every other gate is logged as would-fire but not pushed.
        if same_day and SAME_DAY_WATCH_ONLY:
            _dec(r, "same_day_watch", sp)
            continue

        ci_str = f"CI({ci[0]:+.2f},{ci[1]:+.2f})" if ci else "no-holdout"
        delivered = notify_transition({
            "cat":            "model_edge",
            "city":           city,
            "contract_label": r["ticker"],
            "action":         "BUY YES" if r["direction"] == "BUY_YES" else "BUY NO",
            "reason":         f"edge {edge:.2f}  {ci_str}  models ±{sp:.1f}°F"
                              + (f"  ~{hrs:.0f}h to settle" if hrs is not None else ""),
            "yes_bid":        r.get("yes_bid_dollars"),
            "day_h":          r.get("tmax_f_fcst"),
        }, old_cat=None)
        if not delivered:              # risk-halt suppressed: don't mark notified/counted
            _dec(r, "risk_halted", sp)
            continue
        state["picks"][key] = "model_edge"
        _dec(r, "FIRED", sp)
        pushed += 1
        if executor is not None:
            # Build-order step 5: a FIRED pick becomes a post-only maker limit
            # on the exchange (demo by default), sized by the executor's risk
            # layer. All skip/reject reasons land in execution_log.jsonl.
            try:
                executor.execute_pick(r)
            except Exception as exc:   # execution must never kill the signal run
                print(f"  ⚠️  execute_pick failed for {r['ticker']}: {exc}")

    if decisions:
        GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(GATE_LOG, "a") as fh:
            for d in decisions:
                fh.write(json.dumps(d) + "\n")

    n_watch = sum(1 for d in decisions if d["gate"] == "btype_yes_watch")
    if n_watch or _byw_n:
        state_str = ("ACTIVE — B-type BUY_YES suppressed" if btype_yes_watch_active
                     else "CLEARED — B-type BUY_YES now eligible")
        print(f"  P1.7 watch: {state_str} "
              f"({_byw_n}/{BTYPE_YES_WATCH_MIN_N} graded, YES rate {_byw_rate:.0%}); "
              f"{n_watch} would-fire B-type BUY_YES held this run.")

    return pushed, filtered


def push_obs_locks(signals: pd.DataFrame, state: dict) -> int:
    """
    Intraday OBSERVATION lock detection for the in-progress (today, LST) day.

    The forecast-edge path above only trades clean future days (pre_settlement).
    Today's contracts are gated out of it because a day in progress is no longer a
    forecast bet — but it CAN become a genuine lock once observed temperatures make
    the outcome certain. Running daily high is monotone until the afternoon peak, so:
      • greater (YES if high ≥ floor): high ≥ floor + margin → YES locked
      • less    (YES if high ≤ cap)  : high ≥ cap   + margin → NO  locked
      • between [floor, cap)         : high ≥ cap   + margin → NO  locked
    METAR_MARGIN_F clears METAR's whole-°C rounding so the NWS actual can't flip it.

    This is the ONLY path allowed to emit a real "lock" push — and only when the
    locked side is still buyable inside the dashboard's actionable-edge band
    (5-25¢ net of fee): below it the edge is already priced away, above it the
    market strongly disagrees it's locked (phantom/stale quote).
    """
    if signals.empty:
        return 0

    now = datetime.now(tz=timezone.utc)
    rh_cache: dict[str, float | None] = {}
    pushed = 0

    for _, r in signals.iterrows():
        # Liquidity gate for LOCKS differs from the forecast path: do NOT require
        # an ATM mid (a genuine lock often sits at 0.85-0.95). Require only a
        # two-sided, reasonably tight book so the locked side is actually buyable;
        # the 5-25¢ actionable-edge band below handles "worth it / not phantom".
        bid_ok    = np.isfinite(r.get("yes_bid_dollars")) and r.get("yes_bid_dollars") > MIN_BID
        ask_ok    = np.isfinite(r.get("yes_ask_dollars"))
        spread_ok = np.isfinite(r.get("spread")) and r.get("spread") < MAX_SPREAD
        if not (bid_ok and ask_ok and spread_ok):
            continue
        city = r.get("city")
        cfg  = CITY_CONFIGS.get(city)
        if cfg is None:
            continue
        lst_off   = cfg.get("lst_offset", 0)
        today_lst = (now + timedelta(hours=lst_off)).date()

        sd = r["settlement_date"]
        if hasattr(sd, "date"):
            sd = sd.date()
        if sd != today_lst:          # obs locks only apply to the in-progress day
            continue

        station = cfg["nws_station"]
        if station not in rh_cache:
            rh_cache[station] = fetch_running_high_f(station, lst_off, today_lst)
        rhi = rh_cache[station]
        if rhi is None:
            continue

        tdir = r.get("t_direction")
        thr  = r.get("threshold_f")
        cap  = r.get("cap_f")
        locked_action: str | None = None
        if tdir == "greater" and np.isfinite(thr) and rhi >= float(thr) + METAR_MARGIN_F:
            locked_action = "BUY YES"        # high already cleared floor → YES certain
        elif tdir == "less" and np.isfinite(thr) and rhi >= float(thr) + METAR_MARGIN_F:
            locked_action = "BUY NO"         # high above cap → YES impossible
        elif tdir == "between" and np.isfinite(cap) and rhi >= float(cap) + METAR_MARGIN_F:
            locked_action = "BUY NO"         # high above the band → NO certain
        if locked_action is None:
            continue

        # Actionable-edge gate (same policy as the dashboard's obs-lock pushes).
        yes_bid = r.get("yes_bid_dollars")
        yes_ask = r.get("yes_ask_dollars")
        pick = {
            "action": locked_action,
            "yes_ask": float(yes_ask) * 100 if np.isfinite(yes_ask) else None,
            "no_ask":  (1.0 - float(yes_bid)) * 100 if np.isfinite(yes_bid) else None,
        }
        edge_c = _lock_actionable_edge_c(pick)
        if edge_c is None or edge_c < LOCK_NOTIFY_MIN_EDGE_C or edge_c > LOCK_NOTIFY_MAX_EDGE_C:
            continue

        key = f"obslock:{r['ticker']}:{locked_action}"
        if key in state["picks"]:
            continue

        delivered = notify_transition({
            "cat":            "lock",
            "city":           city,
            "contract_label": r["ticker"],
            "action":         locked_action,
            "reason":         f"OBS LOCK: running high {rhi:.1f}°F — net edge {edge_c:.0f}¢",
            "yes_bid":        yes_bid,
            "day_h":          rhi,
        }, old_cat=None)
        if not delivered:              # risk-halt suppressed (check_locks path has no
            continue                   # run_cycle gate): keep state/counters honest
        state["picks"][key] = "lock"
        pushed += 1

    if pushed:
        print(f"  Obs-locks: {pushed} pushed (intraday LOCK, actionable edge).")
    return pushed


# ── Stale-fight detection (log-only, 2026-07-20) ─────────────────────────────
# The frozen-model-vs-moving-market artifact: between model runs our
# prob_estimate is STATIC, so when the market drifts against our direction the
# logged "edge" GROWS — but that growth is the market disagreeing with a stale
# read, not new signal (MSP T80 spent the evening doing exactly this). These
# fields make the artifact measurable per row; nothing gates on them until the
# forward record says the flag actually separates losers from winners.
STALE_FIGHT_MIN_AGE_MIN = 120.0  # model static at least this long…
STALE_FIGHT_MIN_MOVE    = 0.05   # …while the market moved this far against us
_STALE_PROB_EPS         = 0.005  # prob deltas below this are rounding noise
_STALE_TAIL_BYTES       = 10 * 1024 * 1024   # ~>48h of history at current volume
_STALE_LOOKBACK_H       = 48.0


def _staleness_fields(signals: pd.DataFrame, log_path: Path | None = None,
                      now: datetime | None = None) -> dict[tuple, dict]:
    """(ticker, direction) → prob_age_min / mkt_move_adverse / stale_fight.

    prob_age_min: minutes since prob_estimate last CHANGED (start of the current
    constant streak in the signal log; None with no matching history — unknown,
    not fresh). mkt_move_adverse: mid move against our direction over that
    streak (BUY_YES: mid falling; BUY_NO: mid rising — both grow edge_raw
    without any model update). stale_fight: both thresholds tripped.
    """
    log_path = LOG_PATH if log_path is None else log_path
    now = now or datetime.now(tz=timezone.utc)
    hist: dict[tuple, list] = {}
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as fh:
            if size > _STALE_TAIL_BYTES:
                fh.seek(size - _STALE_TAIL_BYTES)
                fh.readline()                      # drop the partial first line
            cutoff = (now - timedelta(hours=_STALE_LOOKBACK_H)).isoformat()
            for raw in fh.read().decode("utf-8", "replace").splitlines():
                try:
                    r = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if str(r.get("run_ts", "")) < cutoff:
                    continue
                hist.setdefault((r.get("ticker"), r.get("direction")), []).append(
                    (str(r.get("run_ts", "")), r.get("prob_estimate"), r.get("market_mid")))
    except OSError:
        return {}

    out: dict[tuple, dict] = {}
    for _, row in signals.iterrows():
        key = (row["ticker"], row["direction"])
        cur_p = row.get("prob_estimate")
        rows = hist.get(key) or []
        streak = None                   # earliest consecutive entry ≈ current prob
        if cur_p is not None and pd.notna(cur_p):
            for ts, p, mid in reversed(rows):     # file order is append/ascending
                if p is None or abs(float(p) - float(cur_p)) > _STALE_PROB_EPS:
                    break
                streak = (ts, p, mid)
        if streak is None:
            out[key] = {"prob_age_min": (0.0 if rows else None),
                        "mkt_move_adverse": None, "stale_fight": False}
            continue
        try:
            age_min = (now - datetime.fromisoformat(streak[0])).total_seconds() / 60.0
        except ValueError:
            age_min = None
        cur_mid, start_mid = row.get("market_mid"), streak[2]
        adverse = None
        if (cur_mid is not None and pd.notna(cur_mid)
                and start_mid is not None and pd.notna(start_mid)):
            adverse = (float(start_mid) - float(cur_mid)) if row["direction"] == "BUY_YES" \
                else (float(cur_mid) - float(start_mid))
        out[key] = {
            "prob_age_min": (round(age_min, 1) if age_min is not None else None),
            "mkt_move_adverse": (round(adverse, 4) if adverse is not None else None),
            "stale_fight": bool(age_min is not None and adverse is not None
                                and age_min >= STALE_FIGHT_MIN_AGE_MIN
                                and adverse >= STALE_FIGHT_MIN_MOVE),
        }
    return out


def _load_notify_state() -> dict:
    """Load the date-scoped notify dedup state, resetting on a new calendar day."""
    today_str = date.today().isoformat()
    if NOTIFY_STATE.exists():
        try:
            state = json.loads(NOTIFY_STATE.read_text())
            if state.get("date") == today_str:
                return state
        except Exception:
            pass
    return {"date": today_str, "picks": {}}


def _append_signal_log(signals: pd.DataFrame, risk: RiskManager) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).isoformat()
    pre_col_log = signals.get("pre_settlement", pd.Series(True, index=signals.index))
    stale = _staleness_fields(signals)   # computed BEFORE this run is appended
    with open(LOG_PATH, "a") as fh:
        for idx, row in signals.iterrows():
            pre = bool(pre_col_log.iloc[idx] if hasattr(pre_col_log, "iloc") else True)
            is_actionable = bool(row["liquid_atm"] and row["edge_raw"] >= 0.05 and pre)
            sizing = risk.size_trade(row) if is_actionable else None
            record = {
                "run_ts":          ts,
                "city":            row.get("city"),
                "ticker":          row["ticker"],
                "contract_type":   row.get("contract_type", "T"),
                "strike_type":     row.get("strike_type"),      # between/greater/less
                "threshold_f":     row.get("threshold_f"),      # T-type threshold
                "floor_strike":    row.get("floor_strike"),     # B-type floor
                "cap_strike":      row.get("cap_strike"),       # B-type cap
                "settlement_date": str(row["settlement_date"]),
                "tmax_f_fcst":     row["tmax_f_fcst"],
                "prob_estimate":   row["prob_estimate"],
                "prob_raw":        row.get("prob_raw", row["prob_estimate"]),
                "prob_source":     row.get("prob_source"),
                "calib_lead":      row.get("calib_lead"),   # map key used at the calibration seam (replay re-applies it)
                "ens_p50":          row.get("ens_p50"),          # ensemble center — for ens_center_err grading (P1.0)
                "ens_center_shift": row.get("ens_center_shift"),  # shift applied this cycle — tracks correction over time
                "action_score":     row.get("action_score"),
                "score_tier":      row.get("score_tier"),
                "hours_to_settle": row.get("hours_to_settle"),
                "ens_sd":          row.get("ens_sd"),
                "running_high":    row.get("running_high"),  # intraday obs floor (P3.1)
                "local_hour":      row.get("local_hour"),    # LST hour at pricing time (intraday taper)
                "nws_disagree":    row.get("nws_disagree"),
                "hrrr_check_f":     row.get("hrrr_check_f"),      # log-only fast HRRR read (advisory, unwired)
                "hrrr_vs_ens_diff": row.get("hrrr_vs_ens_diff"),
                # Stale-fight trio (log-only): edge growth with a static model is
                # the market disagreeing, not new signal — see _staleness_fields.
                **(stale.get((row["ticker"], row["direction"]))
                   or {"prob_age_min": None, "mkt_move_adverse": None,
                       "stale_fight": False}),
                # Market-shrinkage posterior (log-only; see runner + build_shrinkage.py)
                "prob_shrunk":     row.get("prob_shrunk"),
                "edge_shrunk":     row.get("edge_shrunk"),
                "market_mid":      row["market_mid"],
                "edge_raw":        row["edge_raw"],
                "direction":       row["direction"],
                "liquid_atm":      bool(row["liquid_atm"]),
                "actionable":      is_actionable,
                "is_same_day":     bool(row.get("is_same_day", False)),
                "sized_contracts": sizing["contracts"] if sizing else None,
                "sized_dollars":   sizing["dollars"]   if sizing else None,
                "kelly_f":         sizing["kelly_f"]   if sizing else None,
                "size_blocked":    sizing["blocked"]   if sizing else None,
                # Provenance travels on the runner row; re-project it here or the
                # signals log loses it (this dict is an explicit field selection).
                "code_sha":        row.get("code_sha"),
                "calib_hash":      row.get("calib_hash"),
                "curve_hash":      row.get("curve_hash"),
            }
            fh.write(json.dumps(record) + "\n")
    print(f"  Signals appended → {LOG_PATH}")


def run_cycle(cities: list[str], risk: RiskManager, log: bool = True) -> dict:
    """One full live cycle: ensemble-priced signals (Move 1) -> evidence-gated
    forecast picks + intraday obs-locks (deduped via notified_today.json) -> optional
    signal log. Returns a summary dict. Shared by the one-shot cron (main) and the
    persistent engine (scripts/run_live_engine.py) so both use identical gating."""
    signals = run_signals(MODEL_PATH, city_keys=cities)
    print_signals(signals, risk=risk)

    # Execution layer (opt-in via KALSHI_EXEC=1; demo env unless prod interlock).
    # Reconcile BEFORE gating: fills feed risk exposure + the position watch,
    # settlements feed P&L (which may trip the executor's circuit breaker).
    executor = maybe_executor()
    if executor is not None:
        print(f"  Execution ARMED ({executor.exchange.env}) — "
              f"{len(executor.state['orders'])} resting, "
              f"{len(executor.state['positions'])} open positions. "
              f"{executor.risk.status_line()}")
        executor.reconcile()

    state = _load_notify_state()
    if risk.is_halted():
        # Circuit breaker tripped (fed by log_to_sheets.py from real P/L): stop
        # prompting for trades entirely — a push we can't act on is how a bad day
        # becomes a worse one. Signals are still logged below for grading.
        print("  ⛔ RISK HALTED — daily circuit breaker tripped; suppressing all "
              "pick/lock pushes until tomorrow.")
        n_pushed = n_filtered = n_locks = 0
    else:
        n_pushed, n_filtered = push_forecast_picks(signals, state, executor)
        print(f"  Forecast picks: {n_pushed} pushed (+EV, models agree, near-term, stable), "
              f"{n_filtered} filtered (horizon / model disagreement / unstable forecast / ceiling).")
        n_locks = push_obs_locks(signals, state)

    NOTIFY_STATE.parent.mkdir(parents=True, exist_ok=True)
    NOTIFY_STATE.write_text(json.dumps(state))

    if log and not signals.empty:
        _append_signal_log(signals, risk)

    return {"picks": n_pushed, "filtered": n_filtered, "locks": n_locks,
            "n_signals": 0 if (signals is None or signals.empty) else len(signals)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Kalshi weather trade signals (one-shot)")
    parser.add_argument("--cities", nargs="+", default=list(CITY_CONFIGS.keys()),
                        help="City keys to run (default: all)")
    parser.add_argument("--no-log", action="store_true",
                        help="Skip writing to signals log")
    parser.add_argument("--force", action="store_true",
                        help="Run even if the scheduler guard would skip (see __main__)")
    args = parser.parse_args()

    print(f"=== Live Rule Runner ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Cities: {args.cities}")

    risk = RiskManager()
    risk.reset_if_new_day()
    run_cycle(args.cities, risk, log=not args.no_log)


if __name__ == "__main__":
    # Once-per-window guard so a stray double-launch (cron + launchd, or two agent
    # fires racing) can't double-fire. The agent now runs every 30 min (StartInterval
    # 1800) for low-latency day-ahead push coverage, so the gap must sit BELOW the
    # interval: the run's own duration (~1-2 min) eats into a 1800s gap and would skip
    # every other legitimate run. 1200s (20 min) still blocks a true double-fire
    # (seconds-minutes apart) while letting each 30-min run through. Fails open.
    from kalshi_weather.preflight import preflight
    from kalshi_weather.scheduling import acquire_slot, mark_and_release
    preflight("run_live")     # refuse to run hollow (missing env/calibration/dirs)
    _state = Path(__file__).parents[1] / "logs" / "scheduler"   # <project root>/logs/scheduler
    _slot = acquire_slot(_state, "run_live", min_gap_s=1200, force="--force" in sys.argv)
    if _slot is None:
        print("run_live: ran recently or another instance is active — skipping (--force to override).")
        sys.exit(0)
    try:
        main()
    finally:
        mark_and_release(_state, "run_live", _slot)
