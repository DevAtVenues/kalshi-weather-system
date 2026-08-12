"""
Two-stage candidate pipeline: wide statistical sourcing, then deep per-pick vetting.

STAGE 1 (the funnel): every run_live cycle prices ~240 contracts across 20 cities.
The push gates are deliberately strict — good for not wiring bad trades to the
phone, bad for coverage: a pick that misses one gate by a hair disappears. Stage 1
re-reads the FULL signal universe from the latest run (no gate filtering), ranks by
edge, and keeps the top N (default 50) with each pick's gate outcome as an
annotation, not a filter.

STAGE 2 (the vet): for the top K candidates, run the checks a careful human does
before trading — verified against sources the pricing engine does NOT use:
  • live Kalshi metadata re-verify (strike type / floor / cap match what we priced)
  • climatology base rate for the exact contract (17y of settlement labels, ±10 DOY)
  • recent regime (last 7 actual highs; is the outcome we're shorting printing?)
  • multi-model agreement for the day (forecast archive: HRRR/ECMWF/NBM/GFS + NWS)
  • self-contradiction (BUY_NO on a bracket containing our own forecast center)
  • same-day quarantine (intraday taper falsified 2026-07-12; watch-only)
Each check emits OK / FLAG / RED. REDs split into FATAL (meta mismatch,
mean-in-bucket, longshot, implausible edge >2x proven ceiling — wrong-by-
construction, can't resolve in our favor → PASS) and CONDITIONAL (same-day
quarantine, model spread — can resolve on later runs: one alone → WATCH, two →
PASS). TRADE_SMALL needs zero REDs, ≤1 FLAG, edge ≥ 0.07, and edge inside the
proven ceiling. Verdicts are advisory — nothing here places trades.

The vet set is stratified by settlement day: day-ahead candidates are vetted
first (they're the actionable ones), same-day gets a small separate quota —
otherwise quarantined same-day monster edges monopolize the top-K by |edge|.

Usage:
  .venv/bin/python scripts/candidate_pipeline.py                 # latest run, top 50, vet 15
  .venv/bin/python scripts/candidate_pipeline.py --top 50 --vet 25 --no-api
Output: data/analysis/candidates/<run_ts>.md (+ .json), also printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalshi_weather.live.runner import CITY_CONFIGS, ENSEMBLE_CITIES  # noqa: E402
from kalshi_weather.dashboard.store import _OOS_CI                    # noqa: E402
from kalshi_weather.calibration.family_ev import (                    # noqa: E402
    family_maxes, family_probs, minimax_family_ev)
from kalshi_weather.ingest.nbm import latest_quantiles, nbm_prob      # noqa: E402

# One NBP download covers every station on the board — always ask for the full
# set so the per-cycle cache is complete on the first call.
_ALL_STATIONS = {c["nws_station"] for c in CITY_CONFIGS.values()}

SIGNALS   = ROOT / "data" / "signals" / "signals_log.jsonl"
GATE_LOG  = ROOT / "data" / "analysis" / "gate_log.jsonl"
FORECASTS = ROOT / "data" / "logger" / "forecasts" / "forecasts.jsonl"
LABELS    = ROOT / "data" / "raw" / "labels"
OUT_DIR   = ROOT / "data" / "analysis" / "candidates"

FWD_STATS = OUT_DIR / "forward_stats.json"

MAX_SPREAD_F   = 3.5    # same bar as run_live Gate 2
REGIME_LOOKBACK = 7
CLIM_DOY_WIN    = 10
FATAL_REDS = {"meta", "mean_in_bucket", "longshot", "implausible"}


# ── Stage 1: source the board ────────────────────────────────────────────────

def load_latest_run() -> pd.DataFrame:
    rows = []
    with open(SIGNALS) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    latest = df["run_ts"].max()
    return df[df["run_ts"] == latest].copy()


def join_gate_outcomes(sig: pd.DataFrame) -> pd.DataFrame:
    """Annotate each signal with its push-gate outcome from the same cycle (the gate
    log's run_ts differs from the signal log's by seconds)."""
    try:
        rows = [json.loads(line) for line in open(GATE_LOG)]
    except FileNotFoundError:
        sig["gate"] = "n/a"
        return sig
    g = pd.DataFrame(rows)
    if g.empty:
        sig["gate"] = "n/a"
        return sig
    sig_ts = pd.Timestamp(sig["run_ts"].iloc[0])
    g["dt"] = (pd.to_datetime(g["run_ts"]) - sig_ts).abs()
    g = g[g["dt"] < pd.Timedelta(minutes=10)]
    gate_map = {(r["ticker"], r["direction"]): r["gate"] for _, r in g.iterrows()}
    sig["gate"] = [gate_map.get((t, d), "not_gated")
                   for t, d in zip(sig["ticker"], sig["direction"])]
    return sig


def species_of(r) -> str:
    """Hypothesized SOURCE of a pick's edge. Graded separately because they are
    different strategies with different validation needs and capacity:
      short_bracket_no  — the measured STRUCTURAL overpricing of narrow B-type
                          brackets (+11.6¢ unconditional, historical backtest);
                          mostly not forecast alpha at all.
      fresh_divergence  — model updated recently and disagrees with the market:
                          the front-running species the timing study measured
                          (market converges to us 64%, we're right 72%).
      model_edge_other  — everything else (incl. stale reads; the stale_fight
                          column stays orthogonal so a stale short-bracket keeps
                          its structural label).
    Without these labels the graded record is an uninterpretable mix."""
    st = str(r.get("strike_type"))
    fl, cap = r.get("floor_strike"), r.get("cap_strike")
    if (st == "between" and str(r.get("direction")) == "BUY_NO"
            and pd.notna(fl) and pd.notna(cap) and float(cap) - float(fl) <= 1.0):
        return "short_bracket_no"
    age = r.get("prob_age_min")
    sf = r.get("stale_fight")
    stale = (sf is not None) and pd.notna(sf) and bool(sf)
    if age is not None and pd.notna(age) and float(age) <= 120.0 and not stale:
        return "fresh_divergence"
    return "model_edge_other"


def build_board(top: int) -> pd.DataFrame:
    sig = join_gate_outcomes(load_latest_run())
    sig = sig[sig["market_mid"].notna()].copy()          # need a tradeable price
    sig["abs_edge"] = sig["edge_raw"].abs()
    board = sig.sort_values("abs_edge", ascending=False).head(top).reset_index(drop=True)
    board["species"] = [species_of(r) for _, r in board.iterrows()]
    return board


# ── Stage 2: the checks ──────────────────────────────────────────────────────

def _labels(station: str) -> pd.DataFrame:
    frames = []
    for f in sorted((LABELS / station).glob("*.parquet")):
        try:
            frames.append(pd.read_parquet(f)[["date", "high"]])
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["date", "high"])
    df = pd.concat(frames, ignore_index=True).dropna(subset=["high"])
    df["date"] = pd.to_datetime(df["date"])
    # The label feed writes a PARTIAL row for the current day (morning-so-far high,
    # e.g. DEN 74 on a 95°F day). Not settlement truth — drop anything >= today so
    # neither the regime check nor verdict grading ever reads an unfinished day.
    return df[df["date"] < pd.Timestamp(date.today())]


def _settles_yes(strike_type: str, floor: float | None, cap: float | None,
                 high: float) -> bool:
    """Kalshi settlement on the integer CLI high — canonical empirically-derived
    rules (kalshi_weather.settlement; fixture-tested against settled markets)."""
    from kalshi_weather.settlement import settles_yes as _canon
    r = _canon(strike_type, floor, cap, high)
    return bool(r) if r is not None else False


def clim_check(station: str, sdate: str, strike_type: str,
               floor, cap) -> tuple[float | None, int, list[float], int]:
    """(climatological P(YES), n, last-7 actual highs, hits of YES in last 7)."""
    lab = _labels(station)
    if lab.empty:
        return None, 0, [], 0
    target = pd.Timestamp(sdate)
    doy = lab["date"].dt.dayofyear
    win = (doy - target.dayofyear).abs() <= CLIM_DOY_WIN
    hist = lab[win & (lab["date"] < target - pd.Timedelta(days=1))]
    p = float(np.mean([_settles_yes(strike_type, floor, cap, h) for h in hist["high"]])) \
        if len(hist) else None
    recent = lab[lab["date"] < target].sort_values("date").tail(REGIME_LOOKBACK)
    highs = recent["high"].tolist()
    hits = sum(_settles_yes(strike_type, floor, cap, h) for h in highs)
    return p, len(hist), highs, hits


def model_agreement(station: str, sdate: str) -> dict | None:
    """Latest archived multi-model snapshot for (station, settlement day)."""
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
    if best is None:
        return None
    vals = [best[m] for m in ("hrrr", "ecmwf", "nbm", "gfs") if best.get(m) is not None]
    return {"models": {m: best.get(m) for m in ("hrrr", "ecmwf", "nbm", "gfs", "nws")},
            "spread": (max(vals) - min(vals)) if len(vals) >= 2 else None}


def kalshi_verify(ticker: str) -> dict | None:
    try:
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}"
        req = urllib.request.Request(url, headers={"User-Agent": "plec-candidate-vet"})
        key = os.getenv("KALSHI_API_KEY")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        m = json.loads(urllib.request.urlopen(req, timeout=15).read())["market"]

        def _f(key):    # API returns "*_dollars" STRINGS (legacy int-cent keys are gone)
            v = m.get(key)
            return float(v) if v not in (None, "") else None

        return {"strike_type": m.get("strike_type"), "floor": m.get("floor_strike"),
                "cap": m.get("cap_strike"), "yes_bid": _f("yes_bid_dollars"),
                "yes_ask": _f("yes_ask_dollars"), "volume": _f("volume_fp"),
                "status": m.get("status")}
    except Exception:
        return None


@lru_cache(maxsize=64)
def _family_maxes_cached(station: str, sdate: str) -> dict:
    """One raw-member read per (station, day) per run — the vet prices many
    contracts against the same three family files."""
    return family_maxes(station, sdate)


def _forward_record(city, ctype: str, direction: str) -> str | None:
    """Graded forward record for this exact (city, type, direction), written by
    grade_candidates.py. Closes the validation-flag loop: the flag stays a FLAG
    (promotion is a human edit to ENSEMBLE_CITIES), but every board now shows
    the evidence accumulating for/against it instead of a bare 'not validated'."""
    try:
        doc = json.loads(FWD_STATS.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    rec = (doc.get("combos") or {}).get(f"{city}|{ctype}|{direction}")
    if not rec:
        return None
    s = (f"forward graded n={rec['n']}/{rec['days']}d "
         f"mean {rec['mean_ret']:+.2f} win {rec['win']:.0%} (mid-fill, advisory)")
    if rec.get("graduated"):
        s += " — MEETS graduation bar; human may promote"
    return s


def verdict_from(checks: list[tuple[str, str, str]], edge_raw: float) -> str:
    """Map vet checks to a verdict. FATAL reds are wrong-by-construction and can't
    resolve in our favor; CONDITIONAL reds (same_day quarantine, model spread) can
    clear on later runs — one alone is a WATCH, two are a PASS. A beyond-ceiling
    FLAG caps at WATCH: never TRADE_SMALL a magnitude the push gate distrusts."""
    reds = [c for c in checks if c[0] == "RED"]
    flags = [c for c in checks if c[0] == "FLAG"]
    if any(c[1] in FATAL_REDS for c in reds) or len(reds) >= 2:
        return "PASS"
    if reds:
        return "WATCH"
    if any(c[1] == "ceiling" for c in flags):
        return "WATCH"
    if len(flags) <= 1 and abs(edge_raw) >= 0.07:
        return "TRADE_SMALL"
    return "WATCH"


def vet_candidate(r: pd.Series, use_api: bool) -> dict:
    checks: list[tuple[str, str, str]] = []   # (severity OK/FLAG/RED, name, note)

    city, tdir = r.get("city"), str(r.get("strike_type"))
    station = (CITY_CONFIGS.get(city) or {}).get("nws_station")
    floor, cap = r.get("floor_strike"), r.get("cap_strike")
    floor = float(floor) if pd.notna(floor) else None
    cap = float(cap) if pd.notna(cap) else None
    model_p, mkt = float(r["prob_estimate"]), float(r["market_mid"])
    buy_no = r["direction"] == "BUY_NO"

    # 1. Standing quarantines. Conditional REDs — they lift when the forward
    # evidence arrives, so alone they read WATCH, never TRADE_SMALL.
    if bool(r.get("is_same_day")):
        checks.append(("RED", "same_day", "intraday pricing quarantined (taper refit "
                       "2026-07-12 unvalidated forward) — watch-only"))
    if tdir == "between" and r["direction"] == "BUY_YES":
        checks.append(("RED", "btype_yes_watch", "B-type BUY_YES watch-only (P1.7) "
                       "until 10 graded post-fix outcomes — mirror of run_live Gate 4"))

    # 2. Live metadata re-verify + book staleness. The board's mid is from the
    # run_live cycle (up to ~20 min old); the vet sees the CURRENT book, so it can
    # catch an edge that was really just the market moving.
    if use_api and station:
        meta = kalshi_verify(str(r["ticker"]))
        if meta:
            if meta["strike_type"] != tdir:
                checks.append(("RED", "meta", f"strike_type mismatch: priced as {tdir}, "
                               f"Kalshi says {meta['strike_type']}"))
            else:
                checks.append(("OK", "meta", f"verified {tdir} f={meta['floor']} c={meta['cap']}"))
            vol = meta.get("volume")
            if vol is not None and vol < 200:
                checks.append(("FLAG", "liquidity", f"volume {vol} — thin book"))
            bid, ask = meta.get("yes_bid"), meta.get("yes_ask")
            if bid and ask and 0 < bid <= ask <= 1:        # dollars; degenerate books skipped
                if ask - bid > 0.10:
                    checks.append(("FLAG", "spread", f"book {bid:.2f}/{ask:.2f} — wide; "
                                   "mid is fiction"))
                live_mid = (bid + ask) / 2
                if abs(live_mid - mkt) > 0.08:
                    checks.append(("FLAG", "stale_price", f"board mid {mkt:.2f} vs live "
                                   f"{live_mid:.2f} — edge was computed on a moved market"))
        else:
            checks.append(("FLAG", "meta", "Kalshi verify unavailable"))

    # 3. Validation boundary (same sets the push gate uses)
    ens_cfg = ENSEMBLE_CITIES.get(city)
    ctype = "B" if tdir == "between" else "T"
    if ens_cfg is None or ctype not in ens_cfg.get("types", set()):
        fr = _forward_record(city, ctype, str(r["direction"]))
        checks.append(("FLAG", "validation",
                       f"({city},{ctype}) not OOS-validated for trading; "
                       + (fr if fr else "no graded forward record yet")))
    ci = (_OOS_CI.get(city) or {}).get(ctype)
    ceiling = (ci[1] + 0.10) if (ci and ci[0] > 0) else 0.25
    abs_edge = abs(float(r["edge_raw"]))
    if abs_edge > 2 * ceiling:
        checks.append(("RED", "implausible", f"edge {r['edge_raw']:+.2f} is >2x the "
                       f"proven ceiling ({ceiling:.2f}) — calibration error until "
                       "proven otherwise"))
    elif abs_edge > ceiling:
        checks.append(("FLAG", "ceiling", f"edge {r['edge_raw']:+.2f} beyond proven "
                       f"range (ceiling {ceiling:.2f}) — likely calibration error"))

    # 4. Multi-model agreement
    ma = model_agreement(station, str(r["settlement_date"])) if station else None
    if ma:
        if ma["spread"] is not None and ma["spread"] > MAX_SPREAD_F:
            checks.append(("RED", "model_spread", f"models disagree ±{ma['spread']:.1f}°F "
                           f"{ma['models']}"))
        else:
            hrrr, p50 = ma["models"].get("hrrr"), r.get("ens_p50")
            if hrrr is not None and p50 is not None and abs(float(p50) - hrrr) > 2.5:
                checks.append(("FLAG", "ens_vs_hrrr", f"ensemble {p50:.1f} vs HRRR "
                               f"{hrrr:.1f} — edge rests on ensemble being right"))
            else:
                checks.append(("OK", "models", f"agree {ma['models']}"))

    # 5. Self-contradiction: shorting the bracket our own center sits in
    p50 = r.get("ens_p50")
    if buy_no and tdir == "between" and p50 is not None and floor is not None \
            and cap is not None and (floor - 0.5) <= float(p50) < (cap + 0.5):
        checks.append(("RED", "mean_in_bucket", f"BUY_NO but our center {p50:.1f} maps "
                       "INSIDE the bracket — self-contradicting"))

    # 6. Climatology + recent regime
    if station:
        clim_p, n, highs, hits = clim_check(station, str(r["settlement_date"]),
                                            tdir, floor, cap)
        if clim_p is not None:
            note = f"clim P(YES)={clim_p:.2f} (n={n}); last {len(highs)} highs {highs}"
            if abs(model_p - clim_p) > 0.30 and abs(mkt - clim_p) < abs(model_p - clim_p):
                checks.append(("FLAG", "climatology", note + " — market is closer to "
                               "climatology than we are"))
            else:
                checks.append(("OK", "climatology", note))
            if buy_no and hits >= 2:
                checks.append(("FLAG", "regime", f"outcome we're shorting hit {hits} of "
                               f"last {len(highs)} days"))
            if (not buy_no) and clim_p < 0.03 and mkt < 0.10:
                checks.append(("RED", "longshot", "buying a <3% climatology longshot — "
                               "favorite-longshot bias says these are overpriced"))

    verdict = verdict_from(checks, float(r["edge_raw"]))

    # 7. ADVISORY (log-only discipline, 2026-07-20): per-model-family EV from the
    # raw logged ensemble members. The blend can average a hostile family away —
    # the per-family minimum cannot (MSP T80: ICON EV −0.41 while the blend fired;
    # SAT B98.5: +EV under every family). Deliberately NOT a check: it must not
    # move verdicts until its own forward record is graded — the numbers recorded
    # here on every board ARE that accruing record.
    advisory: list[dict] = []
    fam_ev = None
    if station and tdir in ("greater", "less", "between"):
        try:
            fams = _family_maxes_cached(station, str(r["settlement_date"]))
            fp = family_probs(station, str(r["settlement_date"]), tdir, floor, cap,
                              fams=fams)
            fam_ev = minimax_family_ev(fp, str(r["direction"]), mkt)
            if fam_ev:
                det = ", ".join(
                    f"{f} p={fp[f]['p_yes']:.2f} ev{fam_ev['per_family_ev'][f]:+.2f} "
                    f"(n={fp[f]['n']} {fp[f]['init']})" for f in sorted(fp))
                tag = ("UNANIMOUS direction" if fam_ev["direction_unanimous"]
                       else "families SPLIT on direction")
                advisory.append({"name": "family_ev",
                                 "note": f"{tag}; worst-family EV "
                                         f"{fam_ev['worst_ev']:+.2f} "
                                         f"({fam_ev['worst_family']}) — {det} "
                                         "[raw grid basis, advisory]"})
            else:
                advisory.append({"name": "family_ev",
                                 "note": f"inconclusive — only {len(fp)} family "
                                         "file(s) cover this day (need ≥2)"})
        except Exception as e:          # advisory must never sink a board run
            advisory.append({"name": "family_ev", "note": f"unavailable: {e}"})

    # 8. ADVISORY (log-only): NBM percentile distribution. NOAA's calibrated
    # blend, statistically corrected TO THE STATION record — an independent,
    # professionally-calibrated distribution at (approximately) the settlement
    # sensor. Network fetch is once per NBM cycle (cached for the whole board).
    nbm_p = None
    if use_api and station and tdir in ("greater", "less", "between"):
        try:
            q = latest_quantiles(station, str(r["settlement_date"]),
                                 stations=_ALL_STATIONS)
            if q:
                nbm_p = nbm_prob(tdir, floor, cap, q)
            if nbm_p is not None:
                advisory.append({"name": "nbm_dist",
                                 "note": f"NBM p10/50/90 {q['p10']}/{q['p50']}/"
                                         f"{q['p90']} ({q.get('cycle', '?')}) → "
                                         f"P(YES)≈{nbm_p:.2f} vs model "
                                         f"{model_p:.2f} vs mkt {mkt:.2f}"})
            else:
                advisory.append({"name": "nbm_dist",
                                 "note": "no NBP MaxT quantiles for this day"})
        except Exception as e:          # advisory must never sink a board run
            advisory.append({"name": "nbm_dist", "note": f"unavailable: {e}"})

    # 9. ADVISORY (log-only): market-shrinkage posterior. The measured-skill
    # blend of model and market (w≈0.4 model; LODO beats both endpoints) — when
    # it collapses or flips the raw edge, the market's weight in the posterior
    # is saying most of that "edge" was our calibration error.
    es = r.get("edge_shrunk")
    if es is not None and pd.notna(es):
        es, er = float(es), abs(float(r["edge_raw"]))
        if es <= 0.0 < er:
            advisory.append({"name": "shrunk_edge",
                             "note": f"posterior FLIPS the edge: raw {er:+.2f} → "
                                     f"shrunk {es:+.2f} (P {model_p:.2f} → "
                                     f"{float(r.get('prob_shrunk')):.2f}) — market-"
                                     "weighted view says no edge here"})
        elif er >= 0.10 and es < er / 3:
            advisory.append({"name": "shrunk_edge",
                             "note": f"posterior collapses the edge: raw {er:+.2f} → "
                                     f"shrunk {es:+.2f} — mostly calibration error "
                                     "by the measured-skill blend"})

    # 10. ADVISORY (log-only): stale fight. run_live logs how long the model
    # prob has been static and how far the market moved against us meanwhile —
    # a growing edge under those conditions is the market disagreeing with a
    # stale read (the 2026-07-20 MSP evening pattern), not new signal.
    sf = r.get("stale_fight")
    if sf is not None and pd.notna(sf) and bool(sf):
        age, mv = r.get("prob_age_min"), r.get("mkt_move_adverse")
        age_s = f"{float(age):.0f} min" if age is not None and pd.notna(age) else "?"
        mv_s = f"{float(mv):+.2f}" if mv is not None and pd.notna(mv) else "?"
        advisory.append({"name": "stale_fight",
                         "note": f"model static {age_s} while market moved {mv_s} "
                                 "against us — edge growth is market disagreement, "
                                 "not new signal"})

    return {"ticker": r["ticker"], "direction": r["direction"],
            "settlement_date": str(r["settlement_date"]), "city": city,
            "station": station, "strike_type": tdir, "floor": floor, "cap": cap,
            "is_same_day": bool(r.get("is_same_day")),
            "model_p": model_p, "market_mid": mkt, "edge_raw": float(r["edge_raw"]),
            "gate": r.get("gate"), "verdict": verdict,
            "checks": [{"sev": s, "name": n, "note": t} for s, n, t in checks],
            "advisory": advisory, "family_ev": fam_ev, "nbm_p_yes": nbm_p}


# ── Report ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--vet", type=int, default=15,
                    help="day-ahead candidates to vet (the actionable ones)")
    ap.add_argument("--vet-same-day", type=int, default=5,
                    help="same-day candidates to vet (quarantined; grading signal only)")
    ap.add_argument("--no-api", action="store_true", help="skip live Kalshi re-verify")
    ap.add_argument("--scheduled", action="store_true",
                    help="launchd/cron invocation: apply the once-per-window slot guard")
    args = ap.parse_args()

    board = build_board(args.top)
    run_ts = board["run_ts"].iloc[0]
    # Stratify the vet set: TRUE day-ahead first (settlement date after the run's
    # local date — NOT the is_same_day flag: evening rows on today's nearly-settled
    # contracts have is_same_day=False past the 2 PM cutoff and would monopolize
    # the vet pool with stale-model-vs-settled-market phantoms, exactly like the
    # quarantined same-day monsters did in the morning).
    run_local_date = str((pd.Timestamp(run_ts) - pd.Timedelta(hours=5)).date())
    ahead_mask = board["settlement_date"].astype(str) > run_local_date
    vet_rows = pd.concat([board[ahead_mask].head(args.vet),
                          board[~ahead_mask].head(args.vet_same_day)])
    verdicts = [vet_candidate(r, use_api=not args.no_api)
                for _, r in vet_rows.iterrows()]

    lines = [f"# Candidate board — run {run_ts}",
             f"universe: {args.top} of latest run; vetted top {args.vet} day-ahead "
             f"+ top {args.vet_same_day} same-day", ""]
    lines.append("| # | ticker | dir | settle | model_p | mkt | edge | gate | verdict |")
    lines.append("|---|--------|-----|--------|---------|-----|------|------|---------|")
    vmap = {v["ticker"] + v["direction"]: v for v in verdicts}
    for i, r in board.iterrows():
        v = vmap.get(str(r["ticker"]) + str(r["direction"]))
        lines.append(f"| {i+1} | {r['ticker']} | {r['direction'][4:]} "
                     f"| {r['settlement_date']} | {r['prob_estimate']:.2f} "
                     f"| {r['market_mid']:.2f} | {r['edge_raw']:+.2f} | {r['gate']} "
                     f"| {v['verdict'] if v else '—'} |")
    lines.append("")
    for v in verdicts:
        lines.append(f"## {v['ticker']} {v['direction']} — **{v['verdict']}**")
        lines.append(f"model {v['model_p']:.2f} vs market {v['market_mid']:.2f} "
                     f"(edge {v['edge_raw']:+.2f}, gate: {v['gate']})")
        for c in v["checks"]:
            lines.append(f"- {c['sev']}: `{c['name']}` — {c['note']}")
        for a in v.get("advisory", []):
            lines.append(f"- ADV: `{a['name']}` — {a['note']}")
        lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = str(run_ts).replace(":", "").replace("+0000", "Z")[:17]
    (OUT_DIR / f"{stem}.md").write_text("\n".join(lines))
    # The full board goes into the JSON so the grading loop can measure the vet's
    # value against the RAW board (what did unvetted / suppressed rows return?).
    board_cols = ["ticker", "direction", "settlement_date", "city", "strike_type",
                  "floor_strike", "cap_strike", "prob_estimate", "market_mid",
                  "edge_raw", "gate", "is_same_day", "prob_source",
                  "hrrr_check_f", "hrrr_vs_ens_diff",
                  "prob_age_min", "mkt_move_adverse", "stale_fight",
                  "prob_shrunk", "edge_shrunk", "species"]
    board_rows = board[[c for c in board_cols if c in board.columns]].to_dict("records")
    from kalshi_weather.provenance import provenance
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(
        {"run_ts": str(run_ts), "provenance": provenance(),
         "verdicts": verdicts, "board": board_rows},
        indent=2, default=str))
    print("\n".join(lines))
    print(f"\nwrote {OUT_DIR / (stem + '.md')}")

    if args.scheduled:
        # An actionable verdict appearing on a scheduled board should reach the
        # phone — the user is not watching boards overnight. Deduped per
        # (day, ticker, direction) like the lock scanner.
        ts = [v for v in verdicts if v["verdict"] == "TRADE_SMALL"]
        if ts:
            state_p = ROOT / "logs" / "health" / "trade_small_pushed.json"
            try:
                state = json.loads(state_p.read_text())
            except (OSError, json.JSONDecodeError):
                state = {}
            today = str(date.today())
            pushed = set(state.get(today, []))
            fresh = [v for v in ts if f"{v['ticker']}:{v['direction']}" not in pushed]
            if fresh:
                from kalshi_weather.dashboard.notifications import notify_health
                body = "\n".join(
                    f"{v['ticker']} {v['direction']} model {v['model_p']:.2f} "
                    f"vs mkt {v['market_mid']:.2f} (edge {v['edge_raw']:+.2f})"
                    for v in fresh)
                notify_health(f"Board: {len(fresh)} TRADE_SMALL verdict(s)", body,
                              priority="high", tags="clipboard")
                state_p.parent.mkdir(parents=True, exist_ok=True)
                state_p.write_text(json.dumps({today: sorted(
                    pushed | {f"{v['ticker']}:{v['direction']}" for v in fresh})}))


if __name__ == "__main__":
    from kalshi_weather.preflight import preflight
    preflight("candidate_pipeline")   # refuse to run hollow
    if "--scheduled" in sys.argv:
        # launchd + any future cron entry can double-fire a window; same guard as
        # run_live. Manual runs skip this entirely — never block a human at the desk.
        from kalshi_weather.scheduling import acquire_slot, mark_and_release
        _state = ROOT / "logs" / "scheduler"
        _slot = acquire_slot(_state, "candidate_pipeline", min_gap_s=1800)
        if _slot is None:
            print("candidate_pipeline: ran recently or already running — skipping.")
            sys.exit(0)
        try:
            main()
        finally:
            mark_and_release(_state, "candidate_pipeline", _slot)
    else:
        main()
