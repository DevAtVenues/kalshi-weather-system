"""
Thread-safe data store with background refresh loops.

Shared between the Flask routes and background polling threads.
All public state is read under self._lock to avoid torn reads.
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta

from kalshi_weather.tz import UTC
from typing import Any

import requests

from pathlib import Path

from kalshi_weather.calibration.bias import apply_correction, contract_probability, load_bias_table
from kalshi_weather.dashboard.notifications import _PRIORITY, notify_transition, notify_feed_stale
from kalshi_weather.ingest.kalshi import _get
from kalshi_weather.ingest.metar import fetch_metar
from kalshi_weather.logger.ensemble import daily_high_distribution
from kalshi_weather.monitor.lock import DailyTracker

APPROACH_F = 1.5        # °F from threshold → flag as "approaching"
# METAR temps are whole °C.  Max rounding error = 0.5°C = 0.9°F.
# Every hard-lock comparison must clear the contract boundary by this margin
# or it is NOT a genuine lock — the NWS actual could still fall the other side.
METAR_MARGIN_F = 0.9
# Max model spread (max - min across GFS/ECMWF/NBM) to allow a gfs_edge pick.
# A 1°F-wide contract with >3.5°F disagreement means models can't agree which
# side of the boundary the temperature lands on — signal is noise.
MAX_SPREAD_F = 3.5
# Model spread above this level triggers a warning badge on calibrated picks.
SPREAD_WARN_F = 4.0
GFS_REFRESH = 1800      # 30 min — catches each new model run quickly

# ── Model-pick credibility gate ──────────────────────────────────────────────
# Dashboard model_edge picks come from run_signals() — which now uses the 71-member
# ensemble for validated cities (Tier 1a/1b) and falls back to the GFS-gaussian rule
# for Tier 2 cities.  Ensemble picks set prob_source="ensemble" and use ens_p50 as
# the model center; rule picks use GFS tmax.  These gates veto picks where the
# probability contradicts our better guidance (stale forecast, model disagreement, etc.).
GFS_DRIFT_MAX_F      = 2.0   # pick's GFS vs the live forecast — beyond this it's stale
ENS_DET_GAP_WARN_F   = 2.5   # ensemble mean vs deterministic mean — soft flag
ENS_DET_GAP_REJECT_F = 4.0   # …hard reject: our own sources can't agree
PROB_IMPLAUSIBLE_LO  = 0.05  # no contract 24h+ out is this certain — false confidence
PROB_IMPLAUSIBLE_HI  = 0.95
BUCKET_HALF_F        = 1.0   # between-bucket half-width around the threshold midpoint

# ── Pick display + notification policy ───────────────────────────────────────
# The dashboard pick section shows only these three conviction tiers. gfs_edge
# and watchlist are raw-heuristic / un-calibrated (no bias-correction, no OOS
# evidence) — surfacing them was the "bullshit" noise. They are excluded from the
# pick section entirely (still computed internally; just not shown as picks).
DISPLAY_CATS = ("lock", "near_lock", "model_edge")

# A LOCK earns a phone push ONLY if the locked side can still be BOUGHT with at
# least this much net edge (¢, after Kalshi's 7¢·C·(1-C) taker fee). A lock the
# market has already priced (e.g. 96¢) is certain but not worth interrupting the
# user for — the edge has already dwindled. This is the "hop on it now" gate.
LOCK_NOTIFY_MIN_EDGE_C = 5.0
# Upper cap: an "edge" bigger than this on a supposedly-locked contract means the
# market strongly DISAGREES that it's locked (or the quote is phantom/stale) — a
# real lock the market just hasn't fully converged on prices ~5-20c, not 30-90c.
# DC '73+ low' showed +64c at a 34c market = the market pricing ~66% it breaks.
# Above this, suppress the push (still visible in the dashboard).
LOCK_NOTIFY_MAX_EDGE_C = 25.0

# Watchdog: if the observation feed (last successful METAR fetch) goes older than
# this, the lock detector is effectively blind and lock pushes can silently stop.
# A dedicated thread fires one ntfy alert so a silent freeze becomes a loud
# "restart me". 10 min = 5 missed 2-min poll cycles — well past transient noise.
METAR_STALE_ALERT_S = 600


def _lock_actionable_edge_c(pick: dict) -> float | None:
    """
    Net edge still on the table (¢) for a locked pick, buying the locked side at
    the ask after the 7¢·C·(1-C) Kalshi taker fee. None when there's no ask to
    act on (can't hop on a quote that isn't there). A lock settles to ~100¢, so
    edge = 100 − ask − fee.
    """
    action = pick.get("action", "")
    cost = pick.get("yes_ask") if action == "BUY YES" else pick.get("no_ask")
    if cost is None:
        return None
    c = float(cost) / 100.0                 # ¢ → dollars
    fee = 0.07 * c * (1.0 - c)
    return (1.0 - c - fee) * 100.0          # back to ¢

# Out-of-sample edge CI (95% block-bootstrap, pessimistic fill) from 2025 holdout.
# T-type: run_2025_validation_all.py (saved → data/models/oos_2025_ttype_results.txt)
# B-type: run_2025_b_validation.py (results recorded in build_production_model.py comments)
# None = no OOS data (city's 2025 data was used for calibration, no holdout yet).
# Update this dict whenever the production model is rebuilt with a new holdout year.
_OOS_CI: dict[str, dict[str, tuple[float, float] | None]] = {
    "NYC": {"T": (0.135, 0.246), "B": (0.049, 0.118)},
    "CHI": {"T": (0.069, 0.182), "B": (-0.047, 0.040)},
    "AUS": {"T": (0.049, 0.174), "B": (-0.012, 0.062)},
    "MIA": {"T": (0.032, 0.171), "B": (0.015, 0.088)},
    # These cities' DEPLOYED rule is gaussian (train 2022-23), so 2025 is a clean
    # holdout. Validated 2026-06-21 via run_2025_validation_full.py:
    #   PHL T = +EV (66 trades, CI +0.074/+0.222).
    #   LAX, DEN: 0 liquid-ATM 2025 contracts — no validatable sample.
    "PHL": {"T": (0.074, 0.222), "B": None},
    "LAX": {"T": None, "B": None},
    "DEN": {"T": None, "B": None},
    # Tier 2 (ATL/BOS/DAL/DCA/HOU/LAS/MSP/MSY/OKC/PHX/SAT/SEA/SFO): no 2025 Kalshi
    # price history cached → cannot validate from candles. As the live logger
    # accumulates orderbook depth, scripts/validate_from_orderbook.py --promote
    # writes their forward-validated CIs to the overlay below.
}


def _load_oos_overlay() -> None:
    """Merge forward-validated CIs (from the orderbook validator) over the baseline,
    so a newly-validated city starts gating its picks without a code edit."""
    overlay_path = Path(__file__).parents[3] / "data" / "models" / "oos_ci_overlay.json"
    try:
        data = json.loads(overlay_path.read_text())
    except Exception:
        return
    for city, ci in data.items():
        t, b = ci.get("T"), ci.get("B")
        _OOS_CI[city] = {
            "T": tuple(t) if isinstance(t, (list, tuple)) else None,
            "B": tuple(b) if isinstance(b, (list, tuple)) else None,
        }


_load_oos_overlay()


def _pick_min_edge(city: str, t_dir: str) -> float:
    """
    Minimum edge_raw to surface a model pick, scaled to OOS evidence quality.
    Tier 2 cities have no Kalshi history so the gaussian model may simply be
    replicating market noise — require a very strong signal before surfacing.
    """
    ci_type  = "B" if t_dir == "between" else "T"
    city_oos = _OOS_CI.get(city)
    if city_oos is None:
        return 0.25   # Tier 2: no Kalshi history — only extreme outliers
    ci = city_oos.get(ci_type)
    if ci is None:
        return 0.18   # Tier 1b: calibrated but no holdout year yet
    if ci[0] >= 0:
        return 0.10   # Tier 1a: both CI bounds positive — validated edge
    return 0.18       # Tier 1a: CI crosses zero — need stronger confirmation


def _model_pick_credibility(mp: dict, fc_day: dict, ens_stats: dict) -> dict:
    """
    Vet a model_edge pick against our OWN better guidance and return
    {"tier": "ok"|"warn"|"reject", "reasons": [...]}.

    REJECT (gated out of the actionable list, still shown dim with the reason) when
    the pick's probability can't be trusted: the forecast it was built on has drifted,
    the models disagree past the noise gate, the probability is implausibly certain, the
    multi-model mean sits inside the very bucket a BUY NO is fading, or the ensemble and
    the deterministic mean split too far. WARN = show it but flag (unvalidated, precip,
    guidance spread, known-overconfident city). OK = calibrated, validated, consistent.
    """
    reject: list[str] = []
    warn:   list[str] = []

    action    = mp.get("action", "")
    prob      = mp.get("prob_estimate")
    thr       = mp.get("threshold")
    t_dir     = mp.get("t_direction", "")
    pick_gfs  = mp.get("gfs_h")
    live_gfs  = fc_day.get("gfs_h")
    mean      = fc_day.get("model_mean_h")
    spread    = mp.get("model_spread_h") if mp.get("model_spread_h") is not None else fc_day.get("model_spread_h")
    precip    = fc_day.get("precip_prob_pct")
    oos       = mp.get("oos_ci")

    # 1. Forecast drift — pick built on a GFS run that no longer matches the live one.
    if pick_gfs is not None and live_gfs is not None and abs(pick_gfs - live_gfs) > GFS_DRIFT_MAX_F:
        reject.append(f"forecast drifted {abs(pick_gfs - live_gfs):.1f}°F since the pick (stale)")

    # 2. Multi-model disagreement past the noise gate.
    if spread is not None and spread > MAX_SPREAD_F:
        reject.append(f"models disagree {spread:.1f}°F (> {MAX_SPREAD_F:.1f})")

    # 3. Implausible certainty for a contract still ~a day out.
    if prob is not None and (prob <= PROB_IMPLAUSIBLE_LO or prob >= PROB_IMPLAUSIBLE_HI):
        reject.append(f"implausible {prob:.0%} certainty a day out")

    # 4. Mean-in-bucket contradiction (between contracts only).
    if t_dir == "between" and thr is not None and mean is not None:
        lo, hi = thr - BUCKET_HALF_F, thr + BUCKET_HALF_F
        in_bucket = lo <= mean <= hi
        if action == "BUY NO" and in_bucket:
            reject.append(f"multi-model mean {mean:.1f}°F sits inside the bucket it fades")
        elif action == "BUY YES" and not in_bucket and abs(mean - thr) > BUCKET_HALF_F + 0.5:
            reject.append(f"multi-model mean {mean:.1f}°F is outside the bucket it backs")

    # 5. Ensemble vs deterministic split — our own sources can't agree.
    # Skipped for ensemble-priced picks (prob_source=="ensemble"): model_mean_h IS the
    # ensemble center (ens_p50), so comparing it to ens_stats would be ensemble vs itself.
    # For rule/gaussian picks, model_mean_h = GFS tmax → the check is meaningful.
    ens_means = [s.get("mean") for s in ens_stats.values() if s.get("mean") is not None]
    if ens_means and mean is not None and mp.get("prob_source") != "ensemble":
        ens_mean = sum(ens_means) / len(ens_means)
        gap = abs(ens_mean - mean)
        if gap > ENS_DET_GAP_REJECT_F:
            reject.append(f"ensemble {ens_mean:.1f}°F vs deterministic {mean:.1f}°F split {gap:.1f}°F")
        elif gap > ENS_DET_GAP_WARN_F:
            warn.append(f"guidance spread {gap:.1f}°F (ens {ens_mean:.1f} vs det {mean:.1f})")

    # WARN-level flags (shown, not gated).
    if oos is None:
        warn.append("unvalidated — no out-of-sample holdout")
    elif oos[0] < 0:
        warn.append("OOS edge unproven (CI crosses zero)")
    if precip is not None and precip >= 35:
        warn.append(f"{precip}% precip may cap the high")
    if mp.get("city") == "MIA":
        warn.append("Miami intervals run narrow — haircut the probability")

    if reject:
        return {"tier": "reject", "reasons": reject}
    if warn:
        return {"tier": "warn", "reasons": warn}
    return {"tier": "ok", "reasons": []}


FORECAST_DAYS = 3

# Deterministic forecast models fetched from Open-Meteo per station.
# Key = short name used in stored data; value = Open-Meteo model param.
# NBM is CONUS-only (covers all 20 target cities). ECMWF IFS 0.25° is global.
_DET_MODELS: dict[str, str] = {
    "gfs":   "gfs_seamless",
    "ecmwf": "ecmwf_ifs025",
    "nbm":   "ncep_nbm_conus",
    # HRRR: hourly updates, 3km resolution, US-only, ~18h horizon (48h at 00Z/12Z).
    # Provides intraday convergence for same-day picks; may be absent for tomorrow.
    "hrrr":  "gfs_hrrr",
}

# Root of the live logger output — written by scripts/run_logger.py
_LOG_DIR = Path(__file__).parents[3] / "data" / "logger"

# Models to summarise (must match keys in logger/ensemble.py MODELS dict)
_ENS_MODELS = {
    "gfs025":        "GEFS",
    "ecmwf_ifs025":  "ECMWF",
    "icon_seamless": "ICON",
}

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _dollars_to_cents(v: str | float | None) -> int | None:
    """Convert Kalshi's dollar-string price (e.g. "0.1400") to integer cents (14).

    0¢ and 100¢ are valid market states (no-bid and ask-at-par) and are preserved.
    """
    if v is None:
        return None
    try:
        c = round(float(v) * 100)
        return c if 0 <= c <= 100 else None
    except (ValueError, TypeError):
        return None


def _ticker_date(ticker: str) -> str | None:
    """Parse the measurement date from a Kalshi ticker like SERIES-26JUN04-STRIKE.

    Returns "YYYY-MM-DD" or None if the ticker doesn't match the expected format.
    The date portion (e.g. '26JUN04') encodes the day the temperature is measured,
    not the settlement/publication date (which is typically the next morning).
    """
    parts = ticker.split("-")
    for part in parts:
        if len(part) == 7 and part[:2].isdigit() and part[5:].isdigit() and part[2:5].upper() in _MONTHS:
            yy   = int(part[:2])
            mon  = _MONTHS[part[2:5].upper()]
            day  = int(part[5:])
            year = 2000 + yy
            return f"{year:04d}-{mon:02d}-{day:02d}"
    return None


def _place_window_est(pick: dict) -> dict:
    """Heuristic 'best time to place (ET)' for a pick.

    Not empirical — grounded in when the forecast that drives the pick is
    freshest and the market has not yet converged to it:
      • lock / near_lock → act now; the outcome is intraday-certain and the edge
        bleeds away as the market catches up.
      • same-day model edge → late morning to early afternoon, once obs have
        pinned enough of the day's heating to trust the running high.
      • day-ahead model edge → early afternoon, after the midday (12z) model
        cycle has fully propagated and the bracket quote has settled.
    Returns {"window": str, "note": str}.  "—" window = watch only.
    """
    cat = pick.get("cat", "")
    if cat in ("lock", "near_lock"):
        return {"window": "now",
                "note": "intraday certainty — place before the market converges"}
    if cat == "model_edge":
        if pick.get("is_same_day"):
            return {"window": "10 AM – 1 PM ET",
                    "note": "same-day — wait for late-morning obs to pin the running high, then place"}
        return {"window": "12 – 3 PM ET",
                "note": "day-ahead — place after the 12z run propagates and the quote settles"}
    return {"window": "—", "note": "watch only — no confirmed edge to place yet"}


class DataStore:
    def __init__(self, stations_cfg: dict, poll_interval: int = 600):
        self.stations_cfg   = stations_cfg
        self.poll_interval  = poll_interval

        self._lock  = threading.Lock()
        self._obs:       dict[str, dict]        = {}
        self._forecasts: dict[str, dict]        = {}
        self._contracts: dict[str, list[dict]]  = {}
        self._trackers:  dict[str, DailyTracker] = {
            cfg["metar_station"]: DailyTracker(
                station=cfg["metar_station"],
                lst_offset=cfg["lst_offset_hours"],
            )
            for cfg in stations_cfg.values()
        }
        self._track_dates: dict[str, date] = {}

        # ensemble_stats: station → {model_key → {mean,std,p10,p90,n,pct_above}}
        self._ensemble_stats: dict[str, dict] = {}

        # bias_table: station → month → {bias_f, std_f, n}
        # Loaded once from data/calibration/bias.parquet (populated by run_calibration.py)
        self._bias_table: dict = load_bias_table()

        # Previous pick categories keyed by ticker, used to detect lock transitions.
        # Format: {ticker: cat}  e.g. {"KXHIGHNY-26JUN05-T91": "gfs_edge"}
        self._prev_pick_cats: dict[str, str] = {}
        # Last computed picks — updated by snapshot(), read by _check_transitions().
        # Avoids calling the heavy snapshot() from the background thread.
        self._last_picks: list[dict] = []

        # Calibrated model picks from the production rules (refreshed every GFS_REFRESH).
        self._model_picks: list[dict] = []

        self.last_metar_utc:    datetime | None = None
        self.last_prices_utc:   datetime | None = None
        self.last_forecast_utc: datetime | None = None
        self.last_ensemble_utc: datetime | None = None
        self.last_model_utc:    datetime | None = None

        self._feed_stale_alerted = False   # watchdog: one alert per stale episode

    # ── Internal fetchers ────────────────────────────────────────────────────

    def _refresh_metar(self) -> None:
        stations = [cfg["metar_station"] for cfg in self.stations_cfg.values()]
        obs      = fetch_metar(stations)
        utc_now  = datetime.now(UTC)

        with self._lock:
            self._obs = obs
            self.last_metar_utc = utc_now
            for station, tracker in self._trackers.items():
                new_date = tracker.settlement_date(utc_now)
                old_date = self._track_dates.get(station)
                if old_date is not None and new_date != old_date:
                    tracker.reset()
                self._track_dates[station] = new_date
                r = obs.get(station)
                if r:
                    tracker.update(r["temp_f"])

    def _refresh_forecasts(self) -> None:
        # Start from existing data so failures preserve last-known values.
        with self._lock:
            result: dict[str, dict] = dict(self._forecasts)

        for cfg in self.stations_cfg.values():
            station  = cfg["metar_station"]
            # Collect each model's highs/lows per date, then merge.
            by_date: dict[str, dict] = {}
            for model_key, model_id in _DET_MODELS.items():
                params = {
                    "latitude":         cfg["lat"],
                    "longitude":        cfg["lon"],
                    "daily":            "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "temperature_unit": "fahrenheit",
                    "timezone":         cfg["timezone"],
                    "forecast_days":    FORECAST_DAYS,
                    "models":           model_id,
                }
                for attempt in range(3):
                    try:
                        resp = requests.get(
                            "https://api.open-meteo.com/v1/forecast",
                            params=params,
                            timeout=20,
                        )
                        if resp.status_code == 429:
                            time.sleep(2 ** attempt * 2)
                            continue
                        resp.raise_for_status()
                        daily = resp.json().get("daily", {})
                        dates = daily.get("time", [])
                        highs = daily.get("temperature_2m_max", [])
                        lows  = daily.get("temperature_2m_min", [])
                        pps   = daily.get("precipitation_probability_max", [])
                        for i, d in enumerate(dates):
                            h = highs[i] if i < len(highs) else None
                            l = lows[i]  if i < len(lows)  else None
                            p = pps[i]   if i < len(pps)   else None
                            if d not in by_date:
                                by_date[d] = {}
                            by_date[d][f"{model_key}_h"]  = round(h, 1) if h is not None else None
                            by_date[d][f"{model_key}_l"]  = round(l, 1) if l is not None else None
                            by_date[d][f"{model_key}_pp"] = int(p)      if p is not None else None
                        break
                    except Exception:
                        time.sleep(1)
                time.sleep(0.4)  # stay within Open-Meteo rate limit across 3 models × 20 stations

            # Add mean and spread across available models for each date.
            # ECMWF IFS 0.25° snaps to coarse grid points and can land 5-10 miles
            # from coastal/elevated stations, producing artificial spread and wildly
            # inflated precip probabilities vs GFS/NBM at the same location.
            # Exclude ECMWF from spread AND precip; keep it only in the mean for
            # directional reference.
            _NO_ECMWF = [m for m in _DET_MODELS if m != "ecmwf"]
            for d, vals in by_date.items():
                highs_avail  = [vals[k] for k in (f"{m}_h"  for m in _DET_MODELS) if vals.get(k) is not None]
                lows_avail   = [vals[k] for k in (f"{m}_l"  for m in _DET_MODELS) if vals.get(k) is not None]
                highs_spread = [vals[k] for k in (f"{m}_h"  for m in _NO_ECMWF)   if vals.get(k) is not None]
                lows_spread  = [vals[k] for k in (f"{m}_l"  for m in _NO_ECMWF)   if vals.get(k) is not None]
                pps_avail    = [vals[k] for k in (f"{m}_pp" for m in _NO_ECMWF)   if vals.get(k) is not None]
                vals["model_mean_h"]   = round(sum(highs_avail) / len(highs_avail), 1) if highs_avail else None
                vals["model_spread_h"] = round(max(highs_spread) - min(highs_spread), 1) if len(highs_spread) > 1 else None
                vals["model_mean_l"]   = round(sum(lows_avail)  / len(lows_avail),  1) if lows_avail  else None
                vals["model_spread_l"] = round(max(lows_spread) - min(lows_spread), 1) if len(lows_spread) > 1 else None
                # Take max across GFS/NBM/HRRR only — ECMWF precip is unreliable at this grid resolution
                vals["precip_prob_pct"] = max(pps_avail) if pps_avail else None
                # Keep gfs_h as primary fallback for legacy code paths
                vals["gfs_h"] = vals.get("gfs_h")
                vals["gfs_l"] = vals.get("gfs_l")

            if by_date:
                result[station] = by_date

        with self._lock:
            self._forecasts        = result
            self.last_forecast_utc = datetime.now(UTC)

    def _refresh_ensemble_stats(self) -> None:
        """
        Read stored ensemble member parquets (written by run_logger.py) and
        compute per-station per-model statistics: mean, std, p10, p90, and the
        fraction of members that forecast a daily high above the ATM strike.

        This is read-only from the logger's output — no API calls are made here.
        If the logger hasn't run yet the stats dict will be empty (graceful).
        """
        utc_now = datetime.now(UTC)
        ens_dir = _LOG_DIR / "ensemble"
        if not ens_dir.exists():
            return

        # Snapshot contracts so we can compute pct_above without holding lock long
        with self._lock:
            contracts_snap = dict(self._contracts)

        result: dict[str, dict] = {}

        for cfg in self.stations_cfg.values():
            station     = cfg["metar_station"]
            lst_offset  = cfg["lst_offset_hours"]
            settle_date = (
                utc_now + timedelta(hours=lst_offset)
            ).date()

            # ATM strike for this station (the "greater" contract closest to current temp)
            contracts  = contracts_snap.get(station, [])
            greaters   = [c for c in contracts
                          if c["strike_type"] == "greater" and c.get("floor_strike")]
            with self._lock:
                obs = self._obs.get(station)
            temp_f    = obs["temp_f"] if obs else None
            atm_strike = None
            if greaters and temp_f is not None:
                atm = min(greaters, key=lambda c: abs(float(c["floor_strike"]) - temp_f))
                atm_strike = float(atm["floor_strike"])

            model_stats: dict[str, dict] = {}
            for model_key, label in _ENS_MODELS.items():
                try:
                    df = daily_high_distribution(
                        ens_dir, station, settle_date,
                        lst_offset_h=lst_offset,
                        model_key=model_key,
                        max_age_h=14,
                    )
                except Exception:
                    continue
                if df is None or df.empty:
                    continue

                highs = df["daily_high_f"].dropna()
                if len(highs) < 5:
                    continue

                stats: dict[str, Any] = {
                    "label": label,
                    "mean":  round(float(highs.mean()), 1),
                    "std":   round(float(highs.std()),  1),
                    "p10":   round(float(highs.quantile(0.10)), 1),
                    "p90":   round(float(highs.quantile(0.90)), 1),
                    "n":     int(len(highs)),
                }
                if atm_strike is not None:
                    pct = float((highs >= atm_strike).mean()) * 100
                    stats["pct_above"] = round(pct, 1)
                    stats["atm_strike"] = atm_strike

                model_stats[model_key] = stats

            if model_stats:
                result[station] = model_stats

        with self._lock:
            self._ensemble_stats    = result
            self.last_ensemble_utc  = utc_now

    @staticmethod
    def _event_date_str(dt: datetime) -> str:
        _M = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        return f"{dt.year % 100:02d}{_M[dt.month - 1]}{dt.day:02d}"

    def _refresh_contracts(self) -> None:
        from datetime import timedelta as _td
        by_station: dict[str, list] = defaultdict(list)
        utc_now   = datetime.now(UTC)
        date_strs = [
            self._event_date_str(utc_now),
            self._event_date_str(utc_now + _td(days=1)),
        ]
        for cfg in self.stations_cfg.values():
            for series_key, market_type in [
                ("kalshi_series",     "high"),
                ("kalshi_low_series", "low"),
            ]:
                series = cfg.get(series_key, "")
                if not series:
                    continue
                try:
                    markets_raw: list[dict] = []
                    for ds in date_strs:
                        data = _get("/markets", {
                            "event_ticker": f"{series}-{ds}",
                            "limit":        200,
                        })
                        markets_raw.extend(data.get("markets", []))
                    for m in markets_raw:
                        st = m.get("strike_type")
                        if st not in ("greater", "less", "between"):
                            continue
                        fs  = m.get("floor_strike")
                        cap = m.get("cap_strike")
                        # Use Kalshi's own subtitle as the label (exact match to UI).
                        # Fallback construction: greater floor=N → "N+1° or above",
                        # less cap=N → "N-1° or below" (Kalshi's exclusive-boundary convention).
                        label = m.get("subtitle") or m.get("yes_sub_title")
                        if not label:
                            if st == "greater" and fs is not None:
                                label = f"{int(float(fs)) + 1}° or above"
                            elif st == "less" and cap is not None:
                                label = f"{int(float(cap)) - 1}° or below"
                            elif st == "between" and fs is not None and cap is not None:
                                label = f"{int(float(fs))}° to {int(float(cap))}°"
                            else:
                                label = None
                        meas_date = _ticker_date(m["ticker"])
                        by_station[cfg["metar_station"]].append({
                            "ticker":         m["ticker"],
                            "city":           cfg["city"],
                            "market_type":    market_type,
                            "strike_type":    st,
                            "floor_strike":   fs,
                            "cap_strike":     cap,
                            "kalshi_label":   label,
                            "yes_bid":        _dollars_to_cents(m.get("yes_bid_dollars")),
                            "yes_ask":        _dollars_to_cents(m.get("yes_ask_dollars")),
                            "no_bid":         _dollars_to_cents(m.get("no_bid_dollars")),
                            "no_ask":         _dollars_to_cents(m.get("no_ask_dollars")),
                            "occ_date":       meas_date,
                        })
                except Exception:
                    pass
                time.sleep(0.15)

        with self._lock:
            self._contracts      = dict(by_station)
            self.last_prices_utc = datetime.now(UTC)

    # ── Lock status (read-only, no _alerted side-effect) ────────────────────

    @staticmethod
    def _lock_status(tracker: DailyTracker, contracts: list[dict],
                     settle_date_str: str | None = None) -> dict:
        if not tracker._readings:
            return {"status": "no_data", "label": "—", "detail": ""}

        rhi = tracker.running_high_f

        # Only evaluate contracts for the current settlement date.
        # Tomorrow's open contracts must never be compared against today's running high.
        if settle_date_str:
            contracts = [c for c in contracts if c.get("occ_date") == settle_date_str]

        for c in sorted(
            contracts,
            key=lambda x: float(x.get("floor_strike") or x.get("cap_strike") or 0),
        ):
            if c["strike_type"] == "greater" and c.get("floor_strike") is not None:
                fs = float(c["floor_strike"])
                if rhi >= fs + METAR_MARGIN_F:
                    return {
                        "status":    "yes_locked",
                        "label":     f"YES LOCK ≥{fs:.0f}°F",
                        "detail":    f"running high {rhi:.1f}°F ≥ threshold {fs:.0f}°F (+{METAR_MARGIN_F}°F margin)",
                        "threshold": fs,
                        "ticker":    c["ticker"],
                    }
                if rhi >= fs - APPROACH_F:
                    return {
                        "status":    "approaching_yes",
                        "label":     f"→ {fs:.0f}°F  ({fs - rhi:.1f}° away)",
                        "detail":    f"{fs - rhi:.1f}°F from YES lock on {c['ticker']}",
                        "threshold": fs,
                        "ticker":    c["ticker"],
                    }

            elif c["strike_type"] == "less" and c.get("cap_strike") is not None:
                cs = float(c["cap_strike"])
                if rhi >= cs + METAR_MARGIN_F:
                    return {
                        "status":    "no_locked",
                        "label":     f"NO LOCK >{cs:.0f}°F",
                        "detail":    f"running high {rhi:.1f}°F ≥ cap {cs:.0f}°F (+{METAR_MARGIN_F}°F margin)",
                        "threshold": cs,
                        "ticker":    c["ticker"],
                    }
                if rhi >= cs - APPROACH_F:
                    return {
                        "status":    "approaching_no",
                        "label":     f"→ {cs:.0f}°F  ({cs - rhi:.1f}° away)",
                        "detail":    f"{cs - rhi:.1f}°F from NO lock on {c['ticker']}",
                        "threshold": cs,
                        "ticker":    c["ticker"],
                    }

        clo = tracker.confirmed_low_f
        if clo is not None:
            return {
                "status":    "low_confirmed",
                "label":     f"LOW {clo:.0f}°F",
                "detail":    f"confirmed daily low {clo:.1f}°F (reversed ≥2°F)",
                "threshold": clo,
            }

        return {"status": "clear", "label": "—", "detail": ""}

    # ── Public snapshot for JSON API ─────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        def fmt_utc(dt: datetime | None) -> str:
            return dt.strftime("%H:%M UTC") if dt else "—"

        with self._lock:
            rows = []
            for cfg in self.stations_cfg.values():
                station   = cfg["metar_station"]
                obs       = self._obs.get(station)
                tracker   = self._trackers.get(station)
                fc        = self._forecasts.get(station, {})
                contracts = self._contracts.get(station, [])

                temp_f = obs["temp_f"] if obs else None

                if tracker and tracker._readings:
                    rhi = tracker.running_high_f
                    rlo = tracker.running_low_f
                    day_h = round(rhi, 1) if rhi != float("-inf") else None
                    day_l = round(rlo, 1) if rlo != float("inf")  else None
                    readings = len(tracker._readings)
                else:
                    day_h = day_l = None
                    readings = 0

                utc_month  = datetime.now(UTC).month
                bias_entry = self._bias_table.get(station, {}).get(utc_month)
                bias_std   = bias_entry["std_f"] if bias_entry else None

                settle_iso = (
                    tracker.settlement_date(datetime.now(UTC)).isoformat()
                    if tracker else None
                )

                high_contracts = [c for c in contracts if c.get("market_type") != "low"]
                lock = self._lock_status(tracker, high_contracts, settle_date_str=settle_iso) if tracker else {
                    "status": "no_data", "label": "—", "detail": ""
                }

                def _betweens_for(d: str | None) -> list[dict]:
                    if not d:
                        return []
                    return [
                        c for c in contracts
                        if c["strike_type"] == "between"
                        and c.get("floor_strike") is not None
                        and c.get("cap_strike") is not None
                        and c.get("occ_date") == d
                    ]

                # If today's between-contracts are all at extreme certainty
                # (>95¢ or <5¢) there's nothing to trade — shift to tomorrow.
                betweens = _betweens_for(settle_iso)
                active_date = settle_iso
                if betweens and all(
                    (b.get("yes_bid") or 0) >= 95 or (b.get("yes_ask") or 100) <= 5
                    for b in betweens
                ):
                    tomorrow = (
                        (date.fromisoformat(settle_iso) + timedelta(days=1)).isoformat()
                        if settle_iso else None
                    )
                    next_bets = _betweens_for(tomorrow)
                    if next_bets:
                        betweens    = next_bets
                        active_date = tomorrow

                # Multi-model forecast for the active date
                fc_day      = fc.get(active_date, {}) if active_date else {}
                gfs_h_raw   = fc_day.get("gfs_h")
                ecmwf_h_raw = fc_day.get("ecmwf_h")
                nbm_h_raw   = fc_day.get("nbm_h")
                hrrr_h_raw  = fc_day.get("hrrr_h")
                mean_h_raw  = fc_day.get("model_mean_h")
                spread_h    = fc_day.get("model_spread_h")

                # Calibrate GFS only — the bias table was derived from GFS vs NWS actuals.
                # Applying a GFS-only bias to the ensemble mean overcorrects ECMWF/NBM/HRRR,
                # which each have their own (unmeasured) biases.
                calib_h = (
                    apply_correction(gfs_h_raw, station, utc_month, self._bias_table)
                    if gfs_h_raw is not None else None
                )
                # Per-model calibrated values for display
                calib_gfs   = apply_correction(gfs_h_raw,   station, utc_month, self._bias_table) if gfs_h_raw   is not None else None
                calib_ecmwf = apply_correction(ecmwf_h_raw, station, utc_month, self._bias_table) if ecmwf_h_raw is not None else None
                calib_nbm   = apply_correction(nbm_h_raw,   station, utc_month, self._bias_table) if nbm_h_raw   is not None else None
                calib_hrrr  = apply_correction(hrrr_h_raw,  station, utc_month, self._bias_table) if hrrr_h_raw  is not None else None

                # ATM = between-contract whose range contains the calibrated mean forecast
                ref_temp = calib_h or mean_h_raw or temp_f
                atm = None
                if betweens and ref_temp is not None:
                    in_range = [
                        c for c in betweens
                        if float(c["floor_strike"]) <= ref_temp < float(c["cap_strike"])
                    ]
                    atm = (
                        in_range[0] if in_range
                        else min(betweens, key=lambda c: abs(
                            (float(c["floor_strike"]) + float(c["cap_strike"])) / 2 - ref_temp
                        ))
                    )

                # Model probability for ATM contract using calibrated mean + historical spread
                atm_model_prob = None
                if atm and calib_h is not None and bias_std is not None:
                    atm_model_prob = contract_probability(
                        calibrated_h=calib_h,
                        std_f=bias_std,
                        floor=atm.get("floor_strike"),
                        cap=atm.get("cap_strike"),
                        strike_type=atm["strike_type"],
                    )
                    if atm_model_prob is not None:
                        atm_model_prob = round(atm_model_prob * 100, 1)

                # Bid/ask for the locked contract (for lock card display)
                lock_bid = lock_ask = None
                if lock.get("ticker"):
                    lock_c = next(
                        (c for c in contracts if c["ticker"] == lock["ticker"]), None
                    )
                    if lock_c:
                        lock_bid = lock_c.get("yes_bid")
                        lock_ask = lock_c.get("yes_ask")

                weather_risk = obs["weather_risk"] if obs and "weather_risk" in obs else {"level": "safe", "tags": []}

                rows.append({
                    "key":            cfg.get("key", station),
                    "city":           cfg["city"],
                    "station":        station,
                    "temp_f":         temp_f,
                    "weather_risk":   weather_risk,
                    "obs_time":       obs["obs_time"].strftime("%H:%MZ") if obs else None,
                    # Per-model raw highs for the active date
                    "gfs_h":          gfs_h_raw,
                    "ecmwf_h":        ecmwf_h_raw,
                    "nbm_h":          nbm_h_raw,
                    "hrrr_h":         hrrr_h_raw,
                    "model_mean_h":   mean_h_raw,
                    "model_spread_h": spread_h,
                    "gfs_l":          fc_day.get("gfs_l"),
                    "model_mean_l":   fc_day.get("model_mean_l"),
                    "model_spread_l": fc_day.get("model_spread_l"),
                    # Calibrated values
                    "calib_h":        calib_h,
                    "calib_gfs":      calib_gfs,
                    "calib_ecmwf":    calib_ecmwf,
                    "calib_nbm":      calib_nbm,
                    "calib_hrrr":     calib_hrrr,
                    "bias_f":         round(bias_entry["bias_f"], 1) if bias_entry else None,
                    "bias_std":       round(bias_std, 1)             if bias_std   else None,
                    "day_h":          day_h,
                    "day_l":          day_l,
                    "readings":       readings,
                    "lock":           lock,
                    "lock_bid":       lock_bid,
                    "lock_ask":       lock_ask,
                    "atm_ticker":     atm["ticker"]               if atm else None,
                    "atm_label":      atm.get("kalshi_label")     if atm else None,
                    "atm_bid":        atm.get("yes_bid")          if atm else None,
                    "atm_ask":        atm.get("yes_ask")          if atm else None,
                    "atm_model_prob": atm_model_prob,
                    "ensemble":       self._ensemble_stats.get(station, {}),
                    "_fc":            fc,
                    "_bias_table":    self._bias_table.get(station, {}),
                })

            picks = self._build_picks(rows)

            # Merge model picks — skip if rule-based system already has a
            # lock or near_lock on the same ticker (those take precedence).
            rule_tickers = {p["ticker"] for p in picks if p["cat"] in ("lock", "near_lock")}
            for mp in self._model_picks:
                if mp["ticker"] not in rule_tickers:
                    mp = dict(mp)   # don't mutate the cached original

                    # Enrich with multi-model spread and precip probability for the
                    # pick's city+date.  run_signals only sees GFS so spread/precip
                    # must be looked up from self._forecasts (which holds all models).
                    station  = mp.get("station", "")
                    occ_date = mp.get("occ_date", "")
                    fc_day: dict = {}
                    ens_stats: dict = {}
                    if station and occ_date:
                        fc_day = self._forecasts.get(station, {}).get(occ_date, {})
                        ens_stats = self._ensemble_stats.get(station, {})
                        if not mp.get("model_spread_h"):
                            mp["model_spread_h"] = fc_day.get("model_spread_h")
                        mp["precip_prob_pct"] = fc_day.get("precip_prob_pct")
                        mp["model_mean_disp"] = fc_day.get("model_mean_h")   # live multi-model mean
                        ens_means = [s.get("mean") for s in ens_stats.values() if s.get("mean") is not None]
                        mp["ens_mean_disp"] = round(sum(ens_means) / len(ens_means), 1) if ens_means else None

                    # Attach OOS CI for the pick's city and contract type.
                    # t_direction "between" → B-type; "less"/"greater" → T-type.
                    city_key  = mp.get("city", "")
                    t_dir     = mp.get("t_direction", "between")
                    ci_type   = "B" if t_dir == "between" else "T"
                    city_oos  = _OOS_CI.get(city_key, {})
                    mp["oos_ci"] = city_oos.get(ci_type)   # (low, high) or None

                    # Credibility verdict — gates self-contradicting / stale picks.
                    mp["credibility"] = _model_pick_credibility(mp, fc_day, ens_stats)

                    picks.append(mp)

            self._last_picks = picks   # cache for _check_transitions

            # Sort hierarchy:
            #   0  lock / near_lock  (intraday certainties)
            #   2  model_edge calibrated, OOS tier 0: CI both positive  (best evidence)
            #   3  model_edge calibrated, OOS tier 1: CI crosses zero   (uncertain)
            #   4  model_edge calibrated, OOS tier 2: no holdout yet    (unverified)
            #   5  model_edge gaussian (Tier 2 cities — no Kalshi history)
            #   6  gfs_edge
            #   7  watchlist
            # Within each tier, sort by edge descending.
            def _pick_sort_key(p: dict) -> tuple:
                cat = p.get("cat", "")
                if cat == "lock":
                    return (0, 0, 0.0, p.get("city", ""))
                if cat == "near_lock":
                    return (1, 0, 0.0, p.get("city", ""))
                if cat == "model_edge":
                    edge = -(p.get("edge_raw") or 0.0)
                    # Credibility first: rejects sink below everything actionable so a big
                    # (but self-contradicting) edge can never top the ranking.
                    cred_rank = {"ok": 0, "warn": 1, "reject": 9}.get(
                        (p.get("credibility") or {}).get("tier", "warn"), 1
                    )
                    # "ensemble" and "calibrated" both have oos_ci set; "ensemble" picks
                    # rank at least as high (Tier 1a ensemble = the primary traded path).
                    if p.get("model_conf") in ("calibrated", "ensemble"):
                        ci = p.get("oos_ci")
                        if ci is None:
                            oos_tier = 2   # Tier 1b: calibrated/ensemble but no holdout
                        elif ci[0] >= 0:
                            oos_tier = 0   # Tier 1a: CI both positive
                        else:
                            oos_tier = 1   # Tier 1a: CI crosses zero
                        # Ensemble picks rank ahead of calibrated-rule at the same OOS tier.
                        src_rank = 0 if p.get("model_conf") == "ensemble" else 1
                        return (2, cred_rank, oos_tier, src_rank, edge, p.get("city", ""))
                    return (2, cred_rank, 5, 2, edge, p.get("city", ""))   # gaussian
                if cat == "gfs_edge":
                    return (6, 0, -(p.get("edge_raw") or 0.0), p.get("city", ""))
                return (7, 0, 0.0, p.get("city", ""))   # watchlist

            picks.sort(key=_pick_sort_key)

            # Strip internal keys before sending to browser
            for r in rows:
                r.pop("_fc", None)
                r.pop("_bias_table", None)

            # ── Bucket + dedup for display ────────────────────────────────
            # Three surfaces the user asked for, built from the priority-sorted
            # `picks` (locks first, then best-evidence model edges):
            #   place    — trades to actually place, one per city+market_type,
            #              each with a heuristic time-to-place (ET).
            #   watch    — near-miss trades worth monitoring: the credible model
            #              edges that lost the per-city dedup, plus genuine
            #              near-threshold trackers (cat=watchlist).
            #   filtered — model edges that failed credibility or lack calibration
            #              (gaussian); shown dim, never actionable.
            # Deliberately dropped from display: raw gfs_edge (uncalibrated single-
            # model heuristic — the low-signal firehose). Dedup keeps only the
            # strongest pick per city+market_type in PLACE ("no abundance").
            def _is_gaussian(p: dict) -> bool:
                return p.get("cat") == "model_edge" and p.get("model_conf") == "gaussian"

            def _is_reject(p: dict) -> bool:
                return (p.get("cat") == "model_edge"
                        and (p.get("credibility") or {}).get("tier") == "reject")

            place: list[dict] = []
            watch: list[dict] = []
            filtered: list[dict] = []
            seen_place: set = set()
            for p in picks:
                cat = p.get("cat")
                key = (p.get("city"), p.get("market_type", "high"))
                if cat == "model_edge" and (_is_reject(p) or _is_gaussian(p)):
                    filtered.append(p)
                elif cat in ("lock", "near_lock", "model_edge"):
                    if key in seen_place:
                        # A credible model-edge sibling that lost the dedup is still
                        # worth watching (the city's next-best bracket). Locks that
                        # lost dedup are redundant — drop them.
                        if cat == "model_edge":
                            p = dict(p)
                            p["bucket"] = "watch"
                            watch.append(p)
                        continue
                    seen_place.add(key)
                    p = dict(p)
                    p["bucket"] = "place"
                    p["place_window"] = _place_window_est(p)
                    place.append(p)
                elif cat == "watchlist":
                    p = dict(p)
                    p["bucket"] = "watch"
                    watch.append(p)

            return {
                "rows":           rows,
                # Bucketed for the redesigned Picks tab. `picks` kept as a flat,
                # back-compat list (place + watch), each tagged with its bucket.
                "picks":          place + watch,
                "place":          place,
                "watch":          watch,
                "filtered":       filtered,
                "last_metar":     fmt_utc(self.last_metar_utc),
                "last_prices":    fmt_utc(self.last_prices_utc),
                "last_forecast":  fmt_utc(self.last_forecast_utc),
                "last_ensemble":  fmt_utc(self.last_ensemble_utc),
                "last_model":     fmt_utc(self.last_model_utc),
                "utc_now":        datetime.now(UTC).strftime("%H:%M UTC"),
            }

    # ── Picks engine ─────────────────────────────────────────────────────────

    def _build_picks(self, rows: list[dict]) -> list[dict]:
        """
        Categorise all open contracts into actionable picks.

        Categories (returned as the 'cat' field):
          lock       — temperature physically observed past threshold (near-certain)
          near_lock  — within 3°F of threshold, temp still at daily peak
          gfs_edge   — GFS forecast ≥5°F above/below threshold (moderate confidence)
          watchlist  — 3-8°F away or GFS within 2-5°F of threshold

        NOTE: GFS edge picks use raw GFS forecast only — no bias correction or
        calibration yet. Confidence is moderate until M2c is wired in.
        """
        picks: list[dict] = []
        seen: set[str] = set()   # deduplicate: one entry per ticker per category

        utc_today = datetime.now(UTC).date().isoformat()  # "YYYY-MM-DD"

        for row in rows:
            station   = row["station"]
            contracts = self._contracts.get(station, [])
            tracker   = self._trackers.get(station)
            if not tracker or not tracker._readings:
                continue

            rhi       = tracker.running_high_f
            temp_f    = row["temp_f"]
            day_h     = row["day_h"]
            day_l     = row.get("day_l")
            fc_all    = row.get("_fc", {})           # full date → {gfs_h, gfs_l}
            bias_tbl  = row.get("_bias_table", {})   # month → {bias_f, std_f, n}
            utc_month = datetime.now(UTC).month

            # Settlement date in LST (what day the tracker is accumulating)
            settle_date_str = tracker.settlement_date(
                datetime.now(UTC)
            ).isoformat()  # "YYYY-MM-DD"
            next_date_str   = (
                date.fromisoformat(settle_date_str) + timedelta(days=1)
            ).isoformat()

            # "Still at peak" = current temp within 2°F of day high
            at_peak = (
                temp_f is not None and day_h is not None
                and (day_h - temp_f) <= 2.0
            )

            high_contracts = [c for c in contracts if c.get("market_type") != "low"]
            low_contracts  = [c for c in contracts if c.get("market_type") == "low"]

            for c in high_contracts:
                st       = c.get("strike_type")
                ticker   = c.get("ticker", "")
                bid      = c.get("yes_bid")   # None = no quote
                ask      = c.get("yes_ask")   # None = no quote
                occ_date = c.get("occ_date")

                # Use calibrated GFS as the pick signal — the bias table was derived
                # from GFS vs NWS actuals, so the correction belongs on GFS alone.
                # Applying the GFS bias to the ensemble mean would overcorrect
                # ECMWF/NBM/HRRR, which each have their own unmeasured biases.
                # Raw ensemble mean is retained for display and confidence context.
                _fc_day   = fc_all.get(occ_date, {}) if occ_date else {}
                _mean_raw = _fc_day.get("model_mean_h")     # raw ensemble mean (display only)
                _gfs_raw  = _fc_day.get("gfs_h")
                _spread   = _fc_day.get("model_spread_h")
                _bias_e   = bias_tbl.get(utc_month)
                gfs_h     = (                               # calibrated GFS — the pick signal
                    round(float(_gfs_raw) - _bias_e["bias_f"], 1)
                    if _gfs_raw is not None and _bias_e
                    else (_mean_raw if _mean_raw is not None else _gfs_raw)
                )

                def _add(cat: str, action: str, reason: str, threshold: float,
                         no_ask: int | None = None,
                         contract_label: str | None = None) -> None:
                    key = f"{cat}:{ticker}"
                    if key in seen:
                        return

                    # Locks with no actionable edge left (≥95¢ to buy) — same threshold as
                    # non-lock picks so nothing fully priced leaks into the display.
                    if cat == "lock":
                        if action == "BUY YES":
                            if bid is not None and bid >= 95:
                                return   # market has already decided YES
                            if ask is not None and ask >= 95:
                                return   # YES costs ≥95¢ — no edge
                        if action == "BUY NO":
                            if ask is not None and ask <= 5:
                                return   # YES only worth 5¢ → NO costs ≥95¢
                            _no_cost = no_ask if no_ask is not None else (
                                (100 - bid) if bid is not None else None
                            )
                            if _no_cost is not None and _no_cost >= 95:
                                return   # NO costs ≥95¢ — no edge

                    if cat != "lock":
                        # Skip when prices confirm the outcome is fully priced in.
                        # Null prices = no quote yet → still potentially tradeable.
                        if bid == 0 and ask == 0:
                            return   # zero market — no orderbook at all
                        # Market has decided YES (bid ≥95¢) → no edge on either side
                        if bid is not None and bid >= 95:
                            return
                        # Market has decided NO (ask ≤5¢ or bid=0 & ask=1) → no edge
                        if ask is not None and ask <= 5:
                            return
                        if action == "BUY NO" and no_ask is not None and no_ask >= 95:
                            return   # NO costs ≥95¢ — Kalshi has already priced it in
                        if action == "BUY YES" and ask is not None and ask >= 95:
                            return   # YES costs ≥95¢ — no edge left

                    seen.add(key)
                    picks.append({
                        "cat":            cat,
                        "market_type":    "high",
                        "city":           row["city"],
                        "station":        station,
                        "ticker":         ticker,
                        "occ_date":       occ_date,
                        "contract_label": contract_label,
                        "action":         action,
                        "reason":         reason,
                        "yes_bid":        bid,
                        "yes_ask":        ask,
                        "no_ask":         no_ask,
                        "threshold":      threshold,
                        "temp_f":         temp_f,
                        "day_h":          day_h,
                        "model_mean_h":   _mean_raw,
                        "model_spread_h": _spread,
                        "gfs_h":          _fc_day.get("gfs_h"),
                        "ecmwf_h":        _fc_day.get("ecmwf_h"),
                        "nbm_h":          _fc_day.get("nbm_h"),
                        "hrrr_h":         _fc_day.get("hrrr_h"),
                        "calib_h":        gfs_h,
                        "bias_f":         row.get("bias_f"),
                    })

                # Intraday lock signals are only valid for today's contract.
                # GFS forecast signals fire for today AND tomorrow's contracts.
                same_day   = (occ_date == settle_date_str) if occ_date else True
                next_day   = (occ_date == next_date_str)   if occ_date else False
                gfs_ok     = same_day or next_day  # forecast-based signals span both days

                label    = c.get("kalshi_label") or ticker.split("-")[-1]
                _spd_tag = f", spread {_spread:.1f}°F" if _spread is not None else ""
                # Spread gate: suppress model-based picks when models disagree more
                # than MAX_SPREAD_F. None spread (single model) is always allowed.
                spread_ok = _spread is None or _spread <= MAX_SPREAD_F

                # ── BETWEEN contracts (YES if floor ≤ tmax < cap) ─────────────
                # These are the main liquid contracts. Kalshi labels: "86-87" etc.
                if st == "between" and c.get("floor_strike") and c.get("cap_strike"):
                    fs     = float(c["floor_strike"])
                    cs     = float(c["cap_strike"])
                    no_ask = c.get("no_ask") or ((100 - bid) if bid is not None else None)

                    if same_day and rhi >= cs + METAR_MARGIN_F:
                        # High already exceeded the range → settles NO
                        _add("lock", "BUY NO",
                             f"High {rhi:.1f}°F has cleared the top of '{label}' — settles NO",
                             cs, no_ask=no_ask, contract_label=label)
                    elif same_day and rhi >= fs + METAR_MARGIN_F and rhi < cs and at_peak:
                        # High is firmly inside the range (not touching the cap) and near peak
                        _add("near_lock", "BUY YES",
                             f"High {rhi:.1f}°F is inside '{label}' and temp is near its peak",
                             fs, contract_label=label)
                    elif spread_ok and gfs_ok and gfs_h is not None and fs <= gfs_h < cs and not (same_day and rhi >= cs):
                        _add("gfs_edge", "BUY YES",
                             f"GFS cal {gfs_h:.1f}°F — inside '{label}'{_spd_tag}",
                             fs, contract_label=label)
                    elif spread_ok and gfs_ok and gfs_h is not None and gfs_h >= cs + 3.0:
                        _add("gfs_edge", "BUY NO",
                             f"GFS cal {gfs_h:.1f}°F — above '{label}' range{_spd_tag}",
                             cs, no_ask=no_ask, contract_label=label)

                # ── GREATER contracts (YES if tmax ≥ floor_strike) ────────────
                # Kalshi labels: "92 or above". These are the top-tail contracts.
                elif st == "greater" and c.get("floor_strike") is not None:
                    fs  = float(c["floor_strike"])

                    if same_day and rhi >= fs + METAR_MARGIN_F:
                        _add("lock", "BUY YES",
                             f"High {rhi:.1f}°F — already locked '{label}'",
                             fs, contract_label=label)
                    elif same_day and at_peak and rhi >= fs - APPROACH_F:
                        _add("near_lock", "BUY YES",
                             f"High {rhi:.1f}°F, only {fs - rhi:.1f}°F from '{label}' settling YES",
                             fs, contract_label=label)
                    elif spread_ok and gfs_ok and gfs_h is not None and gfs_h >= fs + 3.0:
                        _add("gfs_edge", "BUY YES",
                             f"GFS cal {gfs_h:.1f}°F — {gfs_h - fs:.1f}°F above floor '{label}'{_spd_tag}",
                             fs, contract_label=label)
                    elif gfs_ok and (
                        (same_day and at_peak and APPROACH_F < (fs - rhi) <= 8.0) or
                        (spread_ok and gfs_h is not None and 2.0 <= gfs_h - fs < 5.0)
                    ):
                        _add("watchlist", "WATCH YES",
                             f"Tracking '{label}' — high {rhi:.1f}°F, GFS cal {gfs_h:.1f}°F{_spd_tag}",
                             fs, contract_label=label)

                # ── LESS contracts (YES if tmax < cap_strike) ─────────────────
                # Kalshi labels: "below 82". These are the bottom-tail contracts.
                # BUY NO = you expect high will meet or exceed the cap.
                elif st == "less" and c.get("cap_strike") is not None:
                    cs     = float(c["cap_strike"])
                    no_ask = c.get("no_ask") or ((100 - bid) if bid is not None else None)

                    if same_day and rhi >= cs + METAR_MARGIN_F:
                        _add("lock", "BUY NO",
                             f"High {rhi:.1f}°F — '{label}' can no longer settle YES",
                             cs, no_ask=no_ask, contract_label=label)
                    elif same_day and at_peak and rhi >= cs - APPROACH_F:
                        _add("near_lock", "BUY NO",
                             f"High {rhi:.1f}°F, only {cs - rhi:.1f}°F from locking out '{label}'",
                             cs, no_ask=no_ask, contract_label=label)
                    elif spread_ok and gfs_ok and gfs_h is not None and gfs_h <= cs - 3.0:
                        _add("gfs_edge", "BUY YES",
                             f"GFS cal {gfs_h:.1f}°F — {cs - gfs_h:.1f}°F below cap, '{label}' likely YES{_spd_tag}",
                             cs, contract_label=label)
                    elif spread_ok and gfs_ok and gfs_h is not None and gfs_h >= cs + 3.0:
                        _add("gfs_edge", "BUY NO",
                             f"GFS cal {gfs_h:.1f}°F — {gfs_h - cs:.1f}°F above cap, '{label}' YES ruled out{_spd_tag}",
                             cs, no_ask=no_ask, contract_label=label)

            # ── LOW TEMPERATURE CONTRACTS (T-type only: greater / less) ──────
            # IMPORTANT — why low "locks" are NOT real locks (unlike high locks):
            # The contract settles on the NWS CLI daily *minimum*, which is the true
            # sub-hourly trough. Our hourly METAR min is only an UPPER BOUND on it —
            # the CLI low can (and routinely does) come in a degree BELOW what we
            # sampled. So "daily-low ≥ threshold" can never be locked YES from hourly
            # obs: the settlement low can always slip lower. (High locks are the
            # mirror image and ARE sound: CLI captures sub-hourly PEAKS above our obs,
            # so an observed high is a lower bound — favorable.) The market prices
            # this: our old confirmed-low "locks" showed as 100% while the market sat
            # at 35-90¢, because it correctly discounts the CLI-vs-METAR gap.
            #   → Confirmed-low signals are demoted to "low_soft" (NOT in DISPLAY_CATS,
            #     never pushed). The physically-valid direction — a NO lock once we've
            #     already OBSERVED the low past the boundary (min can't rise) — needs
            #     running_lo-based logic + calibration and is deferred; lows are off
            #     the OOS-validated strategy (ensemble prices highs) for now.
            confirmed_lo = tracker.confirmed_low_f
            running_lo   = tracker.running_low_f

            for c in low_contracts:
                st       = c.get("strike_type")
                ticker   = c.get("ticker", "")
                bid      = c.get("yes_bid")
                ask      = c.get("yes_ask")
                occ_date = c.get("occ_date")

                if st not in ("greater", "less"):
                    continue  # low markets are T-type only

                _fc_day_l    = fc_all.get(occ_date, {}) if occ_date else {}
                gfs_l        = _fc_day_l.get("gfs_l")
                model_mean_l = _fc_day_l.get("model_mean_l")
                spread_l     = _fc_day_l.get("model_spread_l")

                label     = c.get("kalshi_label") or ticker.split("-")[-1]
                _spd_tag  = f", spread {spread_l:.1f}°F" if spread_l is not None else ""
                spread_ok = spread_l is None or spread_l <= MAX_SPREAD_F

                same_day = (occ_date == settle_date_str) if occ_date else True
                next_day = (occ_date == next_date_str)   if occ_date else False
                gfs_ok   = same_day or next_day

                no_ask = c.get("no_ask") or ((100 - bid) if bid is not None else None)

                def _add_low(cat: str, action: str, reason: str, threshold: float,
                             no_ask: int | None = no_ask,
                             contract_label: str | None = None) -> None:
                    key = f"{cat}:{ticker}"
                    if key in seen:
                        return
                    if cat == "lock":
                        if action == "BUY YES":
                            if bid is not None and bid >= 95:
                                return
                            if ask is not None and ask >= 95:
                                return
                        if action == "BUY NO":
                            if ask is not None and ask <= 5:
                                return
                            _no_cost = no_ask if no_ask is not None else (
                                (100 - bid) if bid is not None else None
                            )
                            if _no_cost is not None and _no_cost >= 95:
                                return
                    if cat != "lock":
                        if bid == 0 and ask == 0:
                            return
                        if bid is not None and bid >= 95:
                            return
                        if ask is not None and ask <= 5:
                            return
                        if action == "BUY NO" and no_ask is not None and no_ask >= 95:
                            return
                        if action == "BUY YES" and ask is not None and ask >= 95:
                            return
                    seen.add(key)
                    rlo_val = round(running_lo, 1) if running_lo != float("inf") else None
                    picks.append({
                        "cat":             cat,
                        "market_type":     "low",
                        "city":            row["city"],
                        "station":         station,
                        "ticker":          ticker,
                        "occ_date":        occ_date,
                        "contract_label":  contract_label,
                        "action":          action,
                        "reason":          reason,
                        "yes_bid":         bid,
                        "yes_ask":         ask,
                        "no_ask":          no_ask,
                        "threshold":       threshold,
                        "temp_f":          temp_f,
                        "day_l":           day_l,
                        "running_lo":      rlo_val,
                        "confirmed_lo":    confirmed_lo,
                        "model_mean_l":    model_mean_l,
                        "model_spread_l":  spread_l,
                        "gfs_l":           gfs_l,
                    })

                if st == "greater" and c.get("floor_strike") is not None:
                    fs = float(c["floor_strike"])
                    if same_day and confirmed_lo is not None:
                        # Soft (NOT a lock): CLI settlement low can slip below our
                        # hourly METAR low, so "low ≥ floor" is never truly locked YES.
                        if confirmed_lo >= fs + METAR_MARGIN_F:
                            _add_low("low_soft", "BUY YES",
                                     f"Low {confirmed_lo:.1f}°F so far (soft — CLI low may come in lower)",
                                     fs, contract_label=label)
                        elif confirmed_lo < fs - METAR_MARGIN_F:
                            _add_low("low_soft", "BUY NO",
                                     f"Low {confirmed_lo:.1f}°F so far below floor '{label}' (soft)",
                                     fs, no_ask=no_ask, contract_label=label)
                    elif gfs_ok and gfs_l is not None and spread_ok:
                        if gfs_l >= fs + 3.0:
                            _add_low("gfs_edge", "BUY YES",
                                     f"GFS low {gfs_l:.1f}°F — {gfs_l - fs:.1f}°F above floor '{label}'{_spd_tag}",
                                     fs, contract_label=label)
                        elif gfs_l <= fs - 3.0:
                            _add_low("gfs_edge", "BUY NO",
                                     f"GFS low {gfs_l:.1f}°F — {fs - gfs_l:.1f}°F below floor '{label}'{_spd_tag}",
                                     fs, no_ask=no_ask, contract_label=label)

                elif st == "less" and c.get("cap_strike") is not None:
                    cs = float(c["cap_strike"])
                    if same_day and confirmed_lo is not None:
                        # Soft (NOT a lock): CLI settlement low can slip below our
                        # hourly METAR low, so "low < cap" is never truly locked from obs.
                        if confirmed_lo < cs - METAR_MARGIN_F:
                            _add_low("low_soft", "BUY YES",
                                     f"Low {confirmed_lo:.1f}°F so far under cap '{label}' (soft)",
                                     cs, contract_label=label)
                        elif confirmed_lo >= cs + METAR_MARGIN_F:
                            _add_low("low_soft", "BUY NO",
                                     f"Low {confirmed_lo:.1f}°F so far above cap '{label}' (soft — CLI low may come in lower)",
                                     cs, no_ask=no_ask, contract_label=label)
                    elif gfs_ok and gfs_l is not None and spread_ok:
                        if gfs_l <= cs - 3.0:
                            _add_low("gfs_edge", "BUY YES",
                                     f"GFS low {gfs_l:.1f}°F — {cs - gfs_l:.1f}°F below cap '{label}' YES{_spd_tag}",
                                     cs, contract_label=label)
                        elif gfs_l >= cs + 3.0:
                            _add_low("gfs_edge", "BUY NO",
                                     f"GFS low {gfs_l:.1f}°F — {gfs_l - cs:.1f}°F above cap, '{label}' YES ruled out{_spd_tag}",
                                     cs, no_ask=no_ask, contract_label=label)

        # Sort: locks first, then near_lock, gfs_edge, watchlist
        order = {"lock": 0, "near_lock": 1, "gfs_edge": 2, "watchlist": 3}
        picks.sort(key=lambda p: (order.get(p["cat"], 9), p["city"]))
        return picks

    # ── Notification / transition detection ──────────────────────────────────

    def _check_transitions(self) -> None:
        """
        Compare the cached pick list against the previous one and fire a push
        notification ONLY when a contract newly LOCKS *and* still carries
        actionable edge (≥ LOCK_NOTIFY_MIN_EDGE_C ¢ after fee).

        Rationale: a phone push interrupts the user when they can't consult, so it
        must mean "drop what you're doing and place this now." Near-locks, model
        edges and heuristics do NOT qualify — those are reviewed in the dashboard.
        And a lock the market has already priced (no edge left) is not worth a ping.

        Reads self._last_picks (written by snapshot()) rather than calling
        snapshot() itself — avoids lock contention with the Flask request thread.
        """
        with self._lock:
            picks = list(self._last_picks)

        new_cats: dict[str, str] = {}
        for pick in picks:
            ticker = pick.get("ticker")
            cat    = pick.get("cat", "")
            if not ticker:
                continue
            new_cats[ticker] = cat

            old_cat = self._prev_pick_cats.get(ticker)

            # Push only a HIGH-temp lock that newly locked and is genuinely mispriced.
            # Excluded on purpose:
            #   • low-temp locks — physically soft (the daily low can still be undercut
            #     by late-night cooling before midnight; the confirmed-low rule is
            #     overconfident) and not yet validated. Still shown in the dashboard.
            #   • edge outside [MIN, MAX] — too small to act on, or so large the market
            #     disagrees it's locked / the quote is phantom (the DC/Miami false alarms).
            if (cat == "lock"
                    and old_cat != "lock"
                    and pick.get("market_type") != "low"):
                edge_c = _lock_actionable_edge_c(pick)
                if (edge_c is not None
                        and LOCK_NOTIFY_MIN_EDGE_C <= edge_c <= LOCK_NOTIFY_MAX_EDGE_C):
                    notify_transition(pick, old_cat)

        self._prev_pick_cats = new_cats

    # ── Startup backfill ─────────────────────────────────────────────────────

    def _backfill_today(self) -> None:
        """
        Backfill today's METAR readings using the IEM ASOS historical API.
        The NOAA Aviation Weather API only returns one observation per station;
        IEM returns the full day's hourly records as CSV.
        """
        utc_now      = datetime.now(UTC)
        all_stations = [cfg["metar_station"] for cfg in self.stations_cfg.values()]

        # IEM end date is exclusive, so use tomorrow to include today's data.
        # Cover 2 days back so all LST offsets (-8 to -5) get their full window.
        yesterday = (utc_now - timedelta(days=1)).date()
        tomorrow  = (utc_now + timedelta(days=1)).date()

        params: list[tuple[str, str | int]] = [
            ("data",   "tmpf"),
            ("year1",  yesterday.year), ("month1", yesterday.month), ("day1", yesterday.day),
            ("year2",  tomorrow.year),  ("month2", tomorrow.month),  ("day2", tomorrow.day),
            ("tz",     "UTC"),
            ("format", "onlycomma"),
            ("direct", "no"),
        ]
        for s in all_stations:
            params.append(("station", s.lstrip("K")))

        try:
            resp = requests.get(
                "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
        except Exception:
            return

        # Parse CSV: station,valid,tmpf  (IEM uses "M" for missing)
        raw_readings: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        for line in resp.text.splitlines():
            line = line.strip()
            if not line or line.startswith("station") or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            station_id, valid_str, tmpf_str = parts[0], parts[1], parts[2]
            if tmpf_str in ("M", "", "null"):
                continue
            try:
                obs_time = datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
                temp_f   = round(float(tmpf_str), 1)
            except (ValueError, TypeError):
                continue
            raw_readings["K" + station_id].append((obs_time, temp_f))

        # Seed each tracker with only today's LST-bounded readings
        with self._lock:
            for station, tracker in self._trackers.items():
                settle_date = tracker.settlement_date(utc_now)
                # Midnight LST in UTC = settlement midnight + |lst_offset| hours
                lst_midnight_utc = datetime(
                    settle_date.year, settle_date.month, settle_date.day,
                    tzinfo=UTC,
                ) + timedelta(hours=-tracker.lst_offset)

                today = sorted(
                    [(t, f) for t, f in raw_readings.get(station, [])
                     if lst_midnight_utc <= t <= utc_now],
                    key=lambda x: x[0],
                )
                for _, temp_f in today:
                    tracker.update(temp_f)
                self._track_dates[station] = settle_date

    # ── Background refresh ───────────────────────────────────────────────────

    def _refresh_model_picks(self) -> None:
        """
        Run the production Gaussian/isotonic rule against currently open markets
        and store actionable picks (liquid ATM, edge ≥ 5%, pre-settlement).
        Refreshed every GFS_REFRESH alongside the forecast cycle.
        """
        from pathlib import Path as _Path
        from kalshi_weather.live.runner import run_signals, CITY_CONFIGS as _CITY_CFGS
        import pandas as _pd

        model_path = _Path(__file__).parents[3] / "data" / "models" / "production_rules.pkl"
        if not model_path.exists():
            return

        monitored = set(self.stations_cfg.keys())
        city_keys = [k for k in _CITY_CFGS if k in monitored]
        if not city_keys:
            return

        try:
            signals = run_signals(model_path, city_keys=city_keys)
        except Exception:
            return

        if signals.empty:
            with self._lock:
                self._model_picks = []
                self.last_model_utc = datetime.now(UTC)
            return

        # Load tier info before filtering so OOS-tier thresholds can be applied.
        _rules: dict = {}
        try:
            import cloudpickle as _cp
            with open(model_path, "rb") as _fh:
                _rules = _cp.load(_fh)
        except Exception:
            pass

        pre_col = signals.get("pre_settlement", _pd.Series(True, index=signals.index))
        base_mask   = signals["liquid_atm"] & pre_col
        thresh_mask = _pd.Series(
            [
                row["edge_raw"] >= _pick_min_edge(
                    row.get("city", ""), row.get("t_direction", "")
                )
                for _, row in signals.iterrows()
            ],
            index=signals.index,
        )
        actionable = signals[base_mask & thresh_mask]

        picks: list[dict] = []
        for _, row in actionable.iterrows():
            city     = row.get("city", "?")
            cfg      = _CITY_CFGS.get(city, {})
            station  = cfg.get("nws_station", "?")
            action   = "BUY YES" if row["direction"] == "BUY_YES" else "BUY NO"
            yes_bid  = row.get("yes_bid_dollars")
            yes_ask  = row.get("yes_ask_dollars")
            bid_c    = int(round(float(yes_bid) * 100)) if yes_bid is not None and _pd.notna(yes_bid) else None
            ask_c    = int(round(float(yes_ask) * 100)) if yes_ask is not None and _pd.notna(yes_ask) else None
            no_ask_c = (100 - bid_c) if bid_c is not None else None

            prob_source = row.get("prob_source", "rule")
            is_ensemble = (prob_source == "ensemble")
            ens_p50_val = row.get("ens_p50")
            ens_sd_val  = row.get("ens_sd")
            # Use ensemble center as the model mean when available; GFS tmax as fallback.
            # The drift-check in _model_pick_credibility still uses gfs_h (tmax_f_fcst)
            # as the staleness anchor so it measures GFS staleness, not ensemble.
            center_f = (
                float(ens_p50_val) if (is_ensemble and ens_p50_val is not None and _pd.notna(ens_p50_val))
                else float(row["tmax_f_fcst"])
            )
            spread_f = (
                float(ens_sd_val) if (is_ensemble and ens_sd_val is not None and _pd.notna(ens_sd_val))
                else None
            )
            rule_tier  = _rules.get(city, {}).get("tier", 2)
            if is_ensemble:
                model_conf = "ensemble"
            elif rule_tier == 1:
                model_conf = "calibrated"
            else:
                model_conf = "gaussian"

            rh = row.get("running_high")
            rh_tag = f"  ·  obs floor {rh:.1f}°F" if rh is not None else ""
            src_label = "Ensemble" if is_ensemble else "Model"
            center_label = "ens p50" if is_ensemble else "GFS"
            picks.append({
                "cat":            "model_edge",
                "market_type":    "high",
                "model_conf":     model_conf,
                "prob_source":    prob_source,
                "is_same_day":    bool(row.get("is_same_day", False)),
                "city":           city,
                "station":        station,
                "ticker":         row["ticker"],
                "occ_date":       str(row["settlement_date"]),
                "contract_label": f"T{row['threshold_f']:.0f} ({row['t_direction']})",
                "action":         action,
                # Raw "edge" deliberately omitted — the P-vs-P compare line + OOS band
                # carry the honest picture; a big edge here is usually model error.
                "reason":         (
                    f"{src_label} P(YES) {row['prob_estimate']:.1%}  ·  "
                    f"market {row['market_mid']:.1%}  ·  "
                    f"{center_label} {center_f:.1f}°F{rh_tag}"
                ),
                "yes_bid":        bid_c,
                "yes_ask":        ask_c,
                "no_ask":         no_ask_c,
                "threshold":      float(row["threshold_f"]),
                "temp_f":         None,
                "day_h":          None,
                "gfs_h":          float(row["tmax_f_fcst"]),   # kept for drift check
                "model_mean_h":   center_f,                    # ensemble p50 or GFS tmax
                "model_spread_h": spread_f,                    # ensemble sd or None
                "ecmwf_h":        None,
                "nbm_h":          None,
                "hrrr_h":         None,
                "calib_h":        center_f,
                "bias_f":         None,
                "prob_estimate":  float(row["prob_estimate"]),
                "market_mid":     float(row["market_mid"]),
                "edge_raw":       float(row["edge_raw"]),
                "t_direction":    row["t_direction"],
            })

        with self._lock:
            self._model_picks   = picks
            self.last_model_utc = datetime.now(UTC)

        # P0.4: persist credibility verdict for each pick so gate thresholds can
        # be tuned post-hoc (was REJECT losing more than WARN? did ok picks win?).
        try:
            self._log_pick_credibility(picks)
        except Exception:
            pass  # never let logging block the refresh loop

    def _log_pick_credibility(self, picks: list[dict]) -> None:
        """Append one line per pick to data/logger/picks/DATE.jsonl with credibility verdict.
        Called after _refresh_model_picks so forecasts/ensemble stats are current.
        Rate-limited to ~GFS_REFRESH cadence by the forecast loop that calls us."""
        import json as _json
        log_dir = Path(__file__).parents[3] / "data" / "logger" / "picks"
        log_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.now(UTC).date().isoformat()
        log_path  = log_dir / f"{today_str}.jsonl"
        ts = datetime.now(UTC).isoformat()

        with self._lock:
            fc_snap  = {st: dict(days) for st, days in self._forecasts.items()}
            ens_snap = dict(self._ensemble_stats)

        with open(log_path, "a") as fh:
            for mp in picks:
                station  = mp.get("station", "?")
                occ_date = mp.get("occ_date", "")
                fc_day   = fc_snap.get(station, {}).get(occ_date, {})
                ens_stats = ens_snap.get(station, {})
                cred = _model_pick_credibility(mp, fc_day, ens_stats)
                record = {
                    "log_ts":             ts,
                    "ticker":             mp.get("ticker"),
                    "city":               mp.get("city"),
                    "settlement_date":    occ_date,
                    "action":             mp.get("action"),
                    "prob_estimate":      mp.get("prob_estimate"),
                    "market_mid":         mp.get("market_mid"),
                    "edge_raw":           mp.get("edge_raw"),
                    "prob_source":        mp.get("prob_source"),
                    "model_conf":         mp.get("model_conf"),
                    "is_same_day":        mp.get("is_same_day"),
                    "running_high":       mp.get("running_high"),
                    "credibility_tier":   cred.get("tier"),
                    "credibility_reasons": cred.get("reasons", []),
                }
                fh.write(_json.dumps(record) + "\n")

    def start(self) -> None:
        """Load all data synchronously, then kick off the background thread."""
        self._refresh_forecasts()
        self._refresh_contracts()
        self._backfill_today()        # seed trackers with today's full history
        try:
            self._refresh_metar()     # then add the current reading
        except Exception:
            pass                      # METAR outage must not block startup
        self._refresh_ensemble_stats()  # load any already-logged ensemble data
        try:
            self._refresh_model_picks()
        except Exception:
            pass

        def _metar_loop() -> None:
            # Fast loop: METAR + prices.  Must never block on GFS fetches.
            last_poll = time.monotonic()
            while True:
                time.sleep(30)
                now = time.monotonic()
                if now - last_poll >= self.poll_interval:
                    try:
                        self._refresh_metar()
                        self._refresh_contracts()
                    except Exception:
                        pass
                    try:
                        # Recompute picks from the fresh METAR/prices BEFORE checking
                        # transitions, so lock pushes fire even with no browser tab open.
                        # snapshot() is pure in-memory assembly (no network) and writes
                        # self._last_picks, which _check_transitions reads. Without this
                        # the notifier only saw picks refreshed by an open dashboard page.
                        self.snapshot()
                        self._check_transitions()
                    except Exception:
                        pass
                    last_poll = now

        def _forecast_loop() -> None:
            # Slow loop: GFS forecasts + model picks.
            # Runs independently so a slow Open-Meteo response never blocks METAR.
            last_forecast = time.monotonic()
            last_ensemble = time.monotonic()
            while True:
                time.sleep(60)
                now = time.monotonic()
                if now - last_forecast >= GFS_REFRESH:
                    try:
                        self._refresh_forecasts()
                    except Exception:
                        pass
                    try:
                        self._refresh_model_picks()
                    except Exception:
                        pass
                    last_forecast = now
                # Re-read ensemble parquets every 5 min.
                if now - last_ensemble >= 300:
                    try:
                        self._refresh_ensemble_stats()
                    except Exception:
                        pass
                    last_ensemble = now

        def _watchdog_loop() -> None:
            # Independent of the poll loops: if the last successful METAR fetch goes
            # stale, lock detection is blind and pushes can silently stop. Fire ONE
            # ntfy alert per episode (re-arm once the feed recovers) so the user knows
            # to restart — converting today's silent 100-min freeze into a loud alert.
            while True:
                time.sleep(60)
                try:
                    if self.last_metar_utc is None:
                        continue
                    stale_s = (datetime.now(UTC) - self.last_metar_utc).total_seconds()
                    if stale_s > METAR_STALE_ALERT_S:
                        if not self._feed_stale_alerted:
                            self._feed_stale_alerted = True
                            notify_feed_stale(stale_s / 60.0)
                    elif self._feed_stale_alerted:
                        self._feed_stale_alerted = False   # recovered → re-arm
                except Exception:
                    pass

        threading.Thread(target=_metar_loop,    daemon=True).start()
        threading.Thread(target=_forecast_loop, daemon=True).start()
        threading.Thread(target=_watchdog_loop, daemon=True).start()
