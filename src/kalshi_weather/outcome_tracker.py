"""
Signal and trade outcome tracker.

Grades past signals from signals_log.jsonl against actual settlement temperatures
fetched from IEM (the same source Kalshi uses). Writes graded records to
data/outcomes/signal_outcomes.jsonl and updates trades.json in-place.

Contract settlement rules: canonical, empirically derived — see
kalshi_weather.settlement (between INCLUSIVE both ends, greater strict >,
less strict <; fixture-tested against settled markets).
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

import yaml

from kalshi_weather.ingest.labels import fetch_labels
from kalshi_weather.tz import UTC  # tz-conversion logic stays isolated in tz.py (HARD RULE 2)

_PROJECT_ROOT  = Path(__file__).parents[2]
_SIGNALS_LOG   = _PROJECT_ROOT / "data" / "signals" / "signals_log.jsonl"
_OUTCOMES_FILE = _PROJECT_ROOT / "data" / "outcomes" / "signal_outcomes.jsonl"
_TRADES_FILE   = _PROJECT_ROOT / "data" / "trades.json"
_STATIONS_CFG  = _PROJECT_ROOT / "config" / "stations.yaml"

# Min distinct settlement days for a block bootstrap to carry any weight. Below this
# the resampled interval is an artifact of the block count, not evidence of edge.
_MIN_BLOCKS = 10

_MONTH = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _load_stations() -> dict:
    with open(_STATIONS_CFG) as f:
        return yaml.safe_load(f)


def _station_for_city(city: str, stations: dict) -> str | None:
    cfg = stations.get(city, {})
    return cfg.get("nws_station")


def parse_ticker(ticker: str) -> dict | None:
    """
    Return settlement date, market_type, strike_type, floor, cap from a ticker.

    B{N}.5 → floor = N-0.5, cap = N+0.5  (matches API strikes N / N+1; the bracket
             is TWO integers wide, INCLUSIVE: YES iff N <= high <= N+1 — see
             kalshi_weather.settlement, derived from settled-market history)
    T{N}   → floor = None,  cap = N       (assumes "less"; the API strike_type
             overrides in grade_signals — a T can be greater or less)
    """
    m = re.match(
        r"KX(HIGH|LOW)[A-Z]+-(\d{2})([A-Z]{3})(\d{2})-(T|B)([\d.]+)$",
        ticker, re.IGNORECASE
    )
    if not m:
        return None
    market_type = "high" if m.group(1).upper() == "HIGH" else "low"
    yy, mon, dd = int(m.group(2)), m.group(3).upper(), int(m.group(4))
    month = _MONTH.get(mon)
    if not month:
        return None
    settlement_date = date(2000 + yy, month, dd)
    contract_type   = m.group(5).upper()
    threshold       = float(m.group(6))

    if contract_type == "T":
        return {
            "settlement_date": settlement_date,
            "market_type":     market_type,
            "strike_type":     "less",
            "floor_strike":    None,
            "cap_strike":      threshold,
        }
    else:  # B
        return {
            "settlement_date": settlement_date,
            "market_type":     market_type,
            "strike_type":     "between",
            "floor_strike":    threshold - 0.5,   # B83.5 → floor=83 (API strike)
            "cap_strike":      threshold + 0.5,   # B83.5 → cap=84; YES iff 83<=h<=84
        }


def yes_won(parsed: dict, actual_temp: float) -> bool:
    from kalshi_weather.settlement import settles_yes
    r = settles_yes(parsed["strike_type"], parsed.get("floor_strike"),
                    parsed.get("cap_strike"), actual_temp)
    return bool(r) if r is not None else False


def fetch_strike_map(series_set: set[str]) -> dict[str, dict]:
    """
    ticker -> {strike_type, floor_strike, cap_strike} from /markets?status=settled.

    The ticker alone cannot tell a "greater" T-contract ("X or above") from a
    "less" one ("X or below") — only the market's strike_type can. parse_ticker
    assumed every T was "less", inverting ~half of T-contract grades. This pulls
    the true strike_type so grading matches Kalshi settlement.
    """
    from kalshi_weather.ingest.kalshi import _get
    out: dict[str, dict] = {}
    for series in series_set:
        cursor: str | None = None
        while True:
            params = {"series_ticker": series, "status": "settled", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            try:
                data = _get("/markets", params)
            except Exception as e:
                print(f"  [warn] strike fetch failed {series}: {e}")
                break
            batch = data.get("markets", [])
            for m in batch:
                t = m.get("ticker")
                if not t:
                    continue
                out[t] = {
                    "strike_type":  m.get("strike_type"),
                    "floor_strike": m.get("floor_strike"),
                    "cap_strike":   m.get("cap_strike"),
                }
            cursor = data.get("cursor")
            if not cursor or not batch:
                break
    return out


def load_outcomes() -> list[dict]:
    if not _OUTCOMES_FILE.exists():
        return []
    records = []
    for line in _OUTCOMES_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _already_graded(outcomes: list[dict]) -> set[str]:
    """Return set of (ticker, run_ts) tuples already in the outcomes file."""
    return {(r["ticker"], r["run_ts"]) for r in outcomes}


def _fetch_actuals(
    station_dates: dict[str, set[date]],
) -> dict[tuple[str, date], dict]:
    """
    Fetch actual high and low for each (station, date) pair.
    Returns {(station, date): {"high": float|None, "low": float|None}}.
    """
    result: dict[tuple[str, date], dict] = {}
    by_year: dict[str, dict[int, set[date]]] = {}
    for station, dates in station_dates.items():
        for d in dates:
            by_year.setdefault(station, {}).setdefault(d.year, set()).add(d)

    current_year = date.today().year
    for station, year_map in by_year.items():
        for year, dates in year_map.items():
            try:
                # The current year's CLI cache is still accumulating — the parquet
                # written earlier this year ends at whatever date it was first pulled
                # and fetch_labels never re-fetches an existing file. Force a refresh
                # for the current year so newly-settled days get graded; past years
                # are complete and stay cached.
                df = fetch_labels(station, [year], force_refresh=(year == current_year))
                for _, row in df.iterrows():
                    d = row["date"]
                    if d in dates:
                        result[(station, d)] = {
                            "high": row.get("high"),
                            "low":  row.get("low"),
                        }
            except Exception as e:
                print(f"  [warn] label fetch failed {station} {year}: {e}")
    return result


def grade_signals(since: date | None = None) -> int:
    """
    Grade all ungraded past signals in signals_log.jsonl.
    Appends results to data/outcomes/signal_outcomes.jsonl.
    Returns number of newly graded signals.
    """
    if not _SIGNALS_LOG.exists():
        print("No signals log found.")
        return 0

    today      = date.today()
    stations   = _load_stations()
    existing   = load_outcomes()
    graded_ids = _already_graded(existing)

    # Read all signals
    signals: list[dict] = []
    for line in _SIGNALS_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            signals.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Filter: past settlement date, not yet graded, optionally after cutoff
    to_grade: list[dict] = []
    for sig in signals:
        key = (sig.get("ticker", ""), sig.get("run_ts", ""))
        if key in graded_ids:
            continue
        sdate_str = sig.get("settlement_date", "")
        if not sdate_str:
            continue
        try:
            sdate = date.fromisoformat(sdate_str)
        except ValueError:
            continue
        if sdate >= today:
            continue
        if since and sdate < since:
            continue
        to_grade.append(sig)

    if not to_grade:
        print("No new signals to grade.")
        return 0

    print(f"Grading {len(to_grade)} signals...")

    # Collect station+date pairs to fetch
    station_dates: dict[str, set[date]] = {}
    for sig in to_grade:
        parsed = parse_ticker(sig.get("ticker", ""))
        if not parsed:
            continue
        city    = sig.get("city", "")
        station = _station_for_city(city, stations)
        if station:
            station_dates.setdefault(station, set()).add(parsed["settlement_date"])

    actuals = _fetch_actuals(station_dates)

    # True strike types (greater/less/between) from settled-market metadata, so
    # "greater" T-contracts aren't mis-graded as "less".
    series_set = {t.split("-")[0] for s in to_grade if (t := s.get("ticker", "")) and "-" in t}
    strike_map = fetch_strike_map(series_set)
    print(f"  Strike metadata for {len(strike_map)} settled markets")

    # Grade each signal
    _OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    newly_graded = 0
    with open(_OUTCOMES_FILE, "a") as f:
        for sig in to_grade:
            parsed = parse_ticker(sig.get("ticker", ""))
            if not parsed:
                continue
            # Override assumed direction with the true strike_type when known.
            truth = strike_map.get(sig.get("ticker", ""))
            parsed["dir_source"] = "assumed_less"
            if truth and truth.get("strike_type"):
                st = truth["strike_type"]
                parsed["strike_type"] = st
                if truth.get("floor_strike") is not None:
                    parsed["floor_strike"] = float(truth["floor_strike"])
                if truth.get("cap_strike") is not None:
                    parsed["cap_strike"] = float(truth["cap_strike"])
                parsed["dir_source"] = "kalshi_meta"
            city    = sig.get("city", "")
            station = _station_for_city(city, stations)
            if not station:
                continue
            actual_info = actuals.get((station, parsed["settlement_date"]))
            if actual_info is None:
                continue

            actual_temp = (
                actual_info["high"] if parsed["market_type"] == "high"
                else actual_info["low"]
            )
            if actual_temp is None:
                continue

            yes_settled   = yes_won(parsed, actual_temp)
            signal_dir    = sig.get("direction", "")
            signal_correct = (
                (signal_dir == "BUY_YES" and yes_settled) or
                (signal_dir == "BUY_NO"  and not yes_settled)
            )

            outcome = {
                "run_ts":          sig["run_ts"],
                "ticker":          sig["ticker"],
                "city":            city,
                "station":         station,
                "settlement_date": str(parsed["settlement_date"]),
                "market_type":     parsed["market_type"],
                "strike_type":     parsed["strike_type"],
                "floor_strike":    parsed["floor_strike"],
                "cap_strike":      parsed["cap_strike"],
                "prob_estimate":   sig.get("prob_estimate"),
                "prob_raw":        sig.get("prob_raw", sig.get("prob_estimate")),
                "prob_source":     sig.get("prob_source"),
                "ens_p50":         sig.get("ens_p50"),
                "ens_center_err":  (                       # actual − ensemble center (P1.0 bias fix)
                    round(actual_temp - sig["ens_p50"], 1)
                    if sig.get("ens_p50") is not None else None
                ),
                "action_score":    sig.get("action_score"),
                "score_tier":      sig.get("score_tier"),
                "hours_to_settle": sig.get("hours_to_settle"),
                "ens_sd":          sig.get("ens_sd"),
                "nws_disagree":    sig.get("nws_disagree"),
                "market_mid":      sig.get("market_mid"),
                "edge_raw":        sig.get("edge_raw"),
                "direction":       signal_dir,
                "actionable":      sig.get("actionable", False),
                "liquid_atm":      sig.get("liquid_atm", False),
                "tmax_f_fcst":     sig.get("tmax_f_fcst"),
                "forecast_error_f": (
                    round(actual_temp - sig["tmax_f_fcst"], 1)
                    if sig.get("tmax_f_fcst") is not None else None
                ),
                "dir_source":      parsed.get("dir_source"),
                "actual_temp":     actual_temp,
                "yes_settled":     yes_settled,
                "signal_correct":  signal_correct,
                "graded_at":       datetime.now(UTC).isoformat(),
            }
            f.write(json.dumps(outcome) + "\n")
            newly_graded += 1

    print(f"Graded {newly_graded} signals → {_OUTCOMES_FILE}")
    return newly_graded


def grade_open_trades() -> int:
    """
    Update trades.json: for any trade with no outcome and a past settlement date,
    fetch the actual temperature and grade it.
    Returns count of newly graded trades.
    """
    if not _TRADES_FILE.exists():
        return 0

    today    = date.today()
    stations = _load_stations()

    with open(_TRADES_FILE) as f:
        data = json.load(f)
    trades = data.get("trades", [])

    to_grade = [
        t for t in trades
        if not t.get("outcome")
        and t.get("settlement_date")
        and date.fromisoformat(t["settlement_date"]) < today
    ]
    if not to_grade:
        print("No open trades to grade.")
        return 0

    # Collect station+date pairs
    station_dates: dict[str, set[date]] = {}
    for t in to_grade:
        city    = t.get("city", "")
        station = _station_for_city(city, stations)
        if station:
            sdate = date.fromisoformat(t["settlement_date"])
            station_dates.setdefault(station, set()).add(sdate)

    actuals = _fetch_actuals(station_dates)

    graded = 0
    for t in to_grade:
        city    = t.get("city", "")
        station = _station_for_city(city, stations)
        if not station:
            continue
        sdate       = date.fromisoformat(t["settlement_date"])
        actual_info = actuals.get((station, sdate))
        if not actual_info:
            continue

        ticker  = t.get("ticker", "")
        parsed  = parse_ticker(ticker)
        if not parsed:
            # Fall back to stored strike fields
            parsed = {
                "market_type":  "high",
                "strike_type":  t.get("strike_type", ""),
                "floor_strike": t.get("floor_strike"),
                "cap_strike":   t.get("cap_strike"),
            }

        market_type = parsed.get("market_type", "high")
        actual_temp = actual_info["high"] if market_type == "high" else actual_info["low"]
        if actual_temp is None:
            continue

        t["settlement_temp"] = actual_temp
        yes = yes_won(parsed, actual_temp)
        action = t.get("action", "")
        t["outcome"] = "won" if (
            (action == "BUY_YES" and yes) or
            (action == "BUY_NO"  and not yes)
        ) else "lost"
        print(
            f"  {ticker}: actual={actual_temp}°F  yes_settled={yes}  "
            f"action={action}  → {t['outcome']}"
        )
        graded += 1

    if graded:
        with open(_TRADES_FILE, "w") as f:
            json.dump({"trades": trades}, f, indent=2, default=str)
        print(f"Updated {graded} trade(s) in trades.json")

    return graded


def learnings_report(min_signals: int = 20) -> list[str]:
    """
    Auto-detect problems from graded outcomes and emit recommendations. This is the
    'identify where the problem lay + propose solutions' layer of the feedback loop.
    DETECTS and RECOMMENDS only — it never mutates the model (applying fixes stays
    human-reviewed, per the no-overfitting discipline). Returns the recommendation
    lines so a caller can persist/notify them.
    """
    from kalshi_weather.dashboard.store import _OOS_CI

    out = load_outcomes()
    recs: list[str] = []
    if not out:
        print("No graded outcomes."); return recs

    graded = [o for o in out if o.get("yes_settled") is not None]
    print(f"\n{'='*68}\n  LEARNINGS — auto-detected problems ({len(graded)} graded)\n{'='*68}")

    # 1) Calibration overconfidence (mid/high model prob)
    hi = [o for o in graded if (o.get("prob_estimate") or 0) >= 0.30]
    if hi:
        act = sum(1 for o in hi if o.get("yes_settled")) / len(hi)
        exp = sum(o["prob_estimate"] for o in hi) / len(hi)
        print(f"\n[calibration] model≥30%: predicts {exp:.0%} YES, actual {act:.0%} ({len(hi)} sigs)")
        if exp - act > 0.12:
            recs.append(f"OVERCONFIDENT on high-prob/YES side (says {exp:.0%}, real {act:.0%}) "
                        f"→ shade YES probs down; raise the bar for BUY_YES picks.")

    # 2) YES vs NO directional accuracy
    for d in ("BUY_YES", "BUY_NO"):
        s = [o for o in graded if o.get("direction") == d]
        if s:
            acc = sum(1 for o in s if o.get("signal_correct")) / len(s)
            print(f"[direction] {d}: {acc:.0%} correct ({len(s)} sigs)")
            if d == "BUY_YES" and acc < 0.45 and len(s) >= min_signals:
                recs.append(f"BUY_YES only {acc:.0%} correct ({len(s)}) → distrust YES signals until recalibrated.")

    # 3) Per-city forecast bias (actual − forecast)
    bias: dict[str, list] = {}
    for o in graded:
        e = o.get("forecast_error_f")
        if e is not None:
            bias.setdefault(o["city"], []).append(e)
    print(f"\n[forecast bias]  (actual − forecast; + = model runs COOL)")
    for city, errs in sorted(bias.items(), key=lambda x: -abs(sum(x[1])/len(x[1]))):
        if len(errs) < min_signals:
            continue
        mb = sum(errs) / len(errs)
        if abs(mb) >= 1.5:
            lean = "COOL" if mb > 0 else "WARM"
            print(f"   {city}: {mb:+.1f}°F over {len(errs)} days → runs {lean}")
            recs.append(f"{city} forecast runs {lean} by {abs(mb):.1f}°F → add per-station bias correction.")

    # 4) Realized edge vs OOS CI (regime divergence on validated cities)
    act = [o for o in graded if o.get("actionable")]
    bycity: dict[str, list] = {}
    for o in act:
        bycity.setdefault(o["city"], []).append(o)
    print(f"\n[CI divergence]  realized actionable edge vs 2025 OOS CI")
    for city, sigs in bycity.items():
        if len(sigs) < min_signals:
            continue
        # Direction-aware realized return per contract at entry price:
        #   BUY_YES: yes_settled − price ;  BUY_NO: price − yes_settled
        # (price ≈ market_mid). Averaged = the strategy's realized edge, NOT
        # an always-YES proxy.
        rets = []
        for o in sigs:
            y = 1 if o["yes_settled"] else 0
            p = o.get("market_mid")
            if p is None:
                continue
            rets.append((y - p) if o.get("direction") == "BUY_YES" else (p - y))
        if not rets:
            continue
        re_edge = sum(rets) / len(rets)
        ci = (_OOS_CI.get(city) or {}).get("T")  # validated reference
        flag = "REGIME DIVERGENCE" if (ci and ci[0] > 0 and re_edge < -0.03) else ""
        print(f"   {city}: realized {re_edge:+.2f} over {len(rets)}  (OOS T-CI {ci}) {flag}")
        if flag:
            recs.append(f"{city} was validated +EV but live realized edge {re_edge:+.2f} → re-validate before trusting.")

    print(f"\n{'─'*68}\n  RECOMMENDATIONS ({len(recs)})")
    for r in recs:
        print(f"  • {r}")
    if not recs:
        print("  (no problems crossing thresholds)")
    return recs


def _edge_contribution(r: dict) -> float:
    """Per-contract realized edge = (won?1:0) − market-implied prob of the side we took.
    Averaged over contracts this equals win-rate − mean implied = the strategy's realized
    edge, so a block bootstrap over these values gives an honest CI on that headline number."""
    mid = float(r.get("market_mid") or 0.0)
    implied_side = mid if r.get("direction") == "BUY_YES" else (1.0 - mid)
    won = 1.0 if r.get("signal_correct") else 0.0
    return won - implied_side


def _block_bootstrap_edge(rows: list[dict], n_boot: int = 5000, seed: int = 0):
    """Block bootstrap the realized edge, resampling whole SETTLEMENT DAYS with
    replacement (not individual contracts). Weather days are autocorrelated — a heat
    wave is one shared outcome across every contract that day, so contracts are NOT
    independent bets. Blocking by day is the charter-mandated way to keep the CI honest
    and expose the true effective sample size (distinct days, not contract count).

    Returns (point, lo95, hi95, n_contracts, n_days) or None."""
    import random
    from collections import defaultdict

    by_day: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_day[str(r.get("settlement_date"))].append(_edge_contribution(r))
    days = list(by_day.values())
    if not days:
        return None
    all_e = [e for d in days for e in d]
    point = sum(all_e) / len(all_e)
    rng = random.Random(seed)
    k = len(days)
    means: list[float] = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(k):
            sample.extend(days[rng.randrange(k)])
        if sample:
            means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(int(0.975 * len(means)), len(means) - 1)]
    return point, lo, hi, len(all_e), k


def forward_report(min_n: int = 1) -> None:
    """Forward-validation of the redesigned engine: reads graded outcomes and slices
    calibration + realized win-rate by the dimensions that tell us if the redesign
    works — ensemble-vs-rule pricing, score tier, and actionable/edge. Dedups to one
    row per contract (latest snapshot) so multi-cycle logging doesn't over-weight."""
    outs = load_outcomes()
    if not outs:
        print("No graded outcomes yet. Run grade_signals first (engine must have logged picks).")
        return

    # one row per ticker = its latest logged snapshot (the decision that stood)
    by_ticker: dict[str, dict] = {}
    for r in outs:
        k = r.get("ticker")
        if k and (k not in by_ticker or str(r.get("run_ts", "")) > str(by_ticker[k].get("run_ts", ""))):
            by_ticker[k] = r
    rows = [r for r in by_ticker.values()
            if r.get("prob_estimate") is not None and r.get("yes_settled") is not None]
    if len(rows) < min_n:
        print(f"Only {len(rows)} graded contracts — need ≥ {min_n}. Let the engine run and grade more.")
        return

    def _brier(rs):
        vs = [(float(r["prob_estimate"]) - (1.0 if r["yes_settled"] else 0.0)) ** 2 for r in rs]
        return sum(vs) / len(vs) if vs else float("nan")

    def _hit(rs):
        c = [r for r in rs if r.get("signal_correct") is not None]
        return (sum(1 for r in c if r["signal_correct"]) / len(c)) if c else float("nan")

    def _line(label, rs):
        if not rs:
            return
        print(f"  {label:<22} n={len(rs):<4} Brier={_brier(rs):.4f}  win-rate={_hit(rs):.1%}")

    print(f"\n{'='*64}\n  FORWARD VALIDATION — {len(rows)} graded contracts\n{'='*64}")
    print("\nOverall:")
    _line("all", rows)

    print("\nBy pricing source (is the ensemble better than the single-run rule?):")
    for src in ("ensemble", "rule"):
        _line(src, [r for r in rows if r.get("prob_source") == src])
    _line("(unlabeled/old log)", [r for r in rows if r.get("prob_source") not in ("ensemble", "rule")])

    print("\nBy score tier (does a higher action_score actually win more?):")
    for tier in ("strong", "watch", "low"):
        _line(tier, [r for r in rows if r.get("score_tier") == tier])

    print("\nActionable picks (edge≥thresh, liquid) — did we beat the price?:")
    act = [r for r in rows if r.get("actionable")]
    if act:
        # realized edge = our win-rate − mean market-implied prob of the side we took
        imp = []
        for r in act:
            mid = float(r.get("market_mid") or 0.0)
            imp.append(mid if r.get("direction") == "BUY_YES" else (1.0 - mid))
        realized = _hit(act) - (sum(imp) / len(imp))
        _line("actionable", act)
        print(f"  {'':<22} realized edge vs market = {realized:+.1%} "
              f"(win-rate − mean implied {sum(imp)/len(imp):.1%})")
        # Charter rule: never report an edge without its interval, and weather days are
        # autocorrelated → block-bootstrap by settlement day + report EFFECTIVE n (days).
        bs = _block_bootstrap_edge(act)
        if bs:
            pt, lo, hi, n_c, n_d = bs
            print(f"  {'':<22} 95% block-bootstrap CI = [{lo:+.1%}, {hi:+.1%}]  "
                  f"(effective n = {n_d} days, not {n_c} contracts)")
            # A block bootstrap resamples DAYS; with too few days the interval is an
            # artifact of the block count, not evidence. Refuse to call it significant.
            if n_d < _MIN_BLOCKS:
                print(f"  {'':<22} → NOT TRUSTWORTHY: only {n_d} independent days "
                      f"(< {_MIN_BLOCKS}). The {n_c} contracts are autocorrelated within "
                      f"those days; this is {n_d} shared outcomes, not {n_c} bets. Edge UNPROVEN.")
            elif lo <= 0.0 <= hi:
                print(f"  {'':<22} → interval includes zero: edge not yet statistically established.")
            else:
                print(f"  {'':<22} → interval excludes zero across {n_d} days: edge holds at 95%.")
        # Split by engine so the retired single-run rule is not credited to the ensemble.
        print("  by engine:")
        for src, lbl in (("ensemble", "ensemble (new)"), ("rule", "rule (retired)"),
                          (None, "unlabeled (old)")):
            sub = [r for r in act if (r.get("prob_source") == src if src else
                                      r.get("prob_source") not in ("ensemble", "rule"))]
            if not sub:
                continue
            simp = [(float(r.get("market_mid") or 0.0) if r.get("direction") == "BUY_YES"
                     else 1.0 - float(r.get("market_mid") or 0.0)) for r in sub]
            redge = _hit(sub) - (sum(simp) / len(simp))
            days = len({str(r.get("settlement_date")) for r in sub})
            print(f"    {lbl:<18} n={len(sub):<4} win={_hit(sub):.1%}  "
                  f"realized edge={redge:+.1%}  ({days} days)")
    else:
        print("  (none actionable yet)")

    print("\nReliability (predicted P(YES) vs realized YES frequency):")
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        hib = lo + 0.2
        b = [r for r in rows if lo <= float(r["prob_estimate"]) < hib]
        if b:
            realized_yes = sum(1 for r in b if r["yes_settled"]) / len(b)
            print(f"  P∈[{lo:.1f},{hib:.1f})  n={len(b):<4} predicted≈{lo+0.1:.2f}  realized={realized_yes:.2f}")

    # Ensemble-only reliability — separate from the stale rule/unlabeled rows that
    # dominate the table above and make the actually-traded path's calibration invisible.
    ens_rows = [r for r in rows if r.get("prob_source") == "ensemble"]
    if len(ens_rows) >= 10:
        print(f"\nEnsemble-path reliability (n={len(ens_rows)} rows — the traded path):")
        for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
            hib = lo + 0.2
            b = [r for r in ens_rows if lo <= float(r["prob_estimate"]) < hib]
            if b:
                realized_yes = sum(1 for r in b if r["yes_settled"]) / len(b)
                print(f"  P∈[{lo:.1f},{hib:.1f})  n={len(b):<4} predicted≈{lo+0.1:.2f}  realized={realized_yes:.2f}")

        # B-type vs T-type split — bracket exclusive-cap fix (2026-07-09) should
        # eliminate the P∈[0.4,0.6) B-type realized≈0 problem going forward.
        b_rows = [r for r in ens_rows if r.get("strike_type") == "between"]
        t_rows = [r for r in ens_rows if r.get("strike_type") in ("greater", "less")]
        if len(b_rows) >= 3 or len(t_rows) >= 3:
            print(f"\nEnsemble by contract type (between vs threshold):")
            for lbl, sub in [("B-type (between)", b_rows), ("T-type (≥/≤)", t_rows)]:
                if not sub:
                    continue
                sub_b = _brier(sub)
                sub_ry = sum(1 for r in sub if r["yes_settled"]) / len(sub)
                sub_mp = sum(float(r["prob_estimate"]) for r in sub) / len(sub)
                print(f"  {lbl}: n={len(sub):<4} Brier={sub_b:.4f}  RealYES={sub_ry:.2f}  MeanP={sub_mp:.3f}")
                for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
                    hib = lo + 0.2
                    bucket = [r for r in sub if lo <= float(r["prob_estimate"]) < hib]
                    if len(bucket) >= 3:
                        ry = sum(1 for r in bucket if r["yes_settled"]) / len(bucket)
                        print(f"    P∈[{lo:.1f},{hib:.1f}) n={len(bucket):<3} realized={ry:.2f}")

        # Per-city ensemble performance — which cities are calibrated, which aren't?
        by_city: dict[str, list] = {}
        for r in ens_rows:
            by_city.setdefault(r.get("city", "?"), []).append(r)
        if any(len(v) >= 5 for v in by_city.values()):
            print(f"\nEnsemble by city (n≥5):")
            print(f"  {'City':<5} {'N':>4}  {'Brier':>7}  {'Win%':>6}  {'RealYES%':>9}  {'MeanP':>7}  {'CenterErr':>10}")
            for city, rs in sorted(by_city.items(), key=lambda x: -len(x[1])):
                if len(rs) < 5:
                    continue
                b = _brier(rs)
                w = _hit(rs)
                real_yes = sum(1 for r in rs if r["yes_settled"]) / len(rs)
                mean_p   = sum(float(r["prob_estimate"]) for r in rs) / len(rs)
                # Per-city center error: day-averaged (not row-averaged) to avoid
                # N×18 inflation from multiple contracts on the same settlement day
                # all sharing the same ens_p50 / actual_temp measurement.
                day_errs: dict[str, list[float]] = {}
                for r in rs:
                    if r.get("ens_center_err") is not None:
                        day_errs.setdefault(r.get("settlement_date", ""), []).append(float(r["ens_center_err"]))
                day_means = [sum(v) / len(v) for v in day_errs.values()]
                c_str = f"{sum(day_means)/len(day_means):+.2f}°F ({len(day_means)}d)" if day_means else "    —"
                print(f"  {city:<5} {len(rs):>4}  {b:>7.4f}  {w:>6.1%}  {real_yes:>9.1%}  {mean_p:>7.3f}  {c_str:>10}")

    print(f"{'='*64}\n  (Brier: lower=better, 0.25=coinflip. Win-rate>50% and realized edge>0 = real edge.)\n{'='*64}")


def calibration_report(min_signals: int = 5) -> None:
    """Print a calibration + city performance report from graded signals."""
    outcomes = load_outcomes()
    if not outcomes:
        print("No graded outcomes yet. Run grade_signals first.")
        return

    total = len(outcomes)
    print(f"\n{'='*65}")
    print(f"  SIGNAL CALIBRATION REPORT  ({total} graded signals)")
    print(f"{'='*65}\n")

    # ── Calibration buckets ──────────────────────────────────────────
    buckets: dict[int, dict] = {i: {"n": 0, "correct": 0, "yes_count": 0} for i in range(10)}
    for o in outcomes:
        p = o.get("prob_estimate")
        if p is None:
            continue
        bucket = min(int(p * 10), 9)
        buckets[bucket]["n"] += 1
        if o.get("yes_settled"):
            buckets[bucket]["yes_count"] += 1
        if o.get("signal_correct"):
            buckets[bucket]["correct"] += 1

    print("Model Prob   Signals  Actual YES%  Expected YES%   Signal Acc%")
    print("-" * 65)
    for i, b in sorted(buckets.items()):
        if b["n"] < 1:
            continue
        lo, hi = i * 10, i * 10 + 10
        actual_yes = b["yes_count"] / b["n"] * 100
        expected   = (lo + hi) / 2
        acc        = b["correct"] / b["n"] * 100
        print(
            f"  {lo:2d}–{hi:2d}%   {b['n']:6d}    {actual_yes:6.1f}%        {expected:5.0f}%        {acc:5.1f}%"
        )

    # ── City performance (actionable only) ───────────────────────────
    actionable = [o for o in outcomes if o.get("actionable")]
    if not actionable:
        print("\nNo actionable signals graded yet.")
        return

    by_city: dict[str, list] = {}
    for o in actionable:
        by_city.setdefault(o["city"], []).append(o)

    print(f"\n{'─'*65}")
    print("  CITY PERFORMANCE  (actionable signals only)\n")
    print(f"  {'City':<6}  {'N':>4}  {'Model':>7}  {'Market':>7}  {'ActYES':>7}  {'EdgeReal':>9}  {'SigAcc':>7}")
    print(f"  {'─'*6}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}  {'─'*7}")

    rows = []
    for city, sigs in by_city.items():
        if len(sigs) < min_signals:
            continue
        n          = len(sigs)
        model_avg  = sum(s["prob_estimate"] or 0 for s in sigs) / n
        market_avg = sum(s["market_mid"]    or 0 for s in sigs) / n
        actual_yes = sum(1 for s in sigs if s["yes_settled"]) / n
        edge_real  = actual_yes - market_avg
        sig_acc    = sum(1 for s in sigs if s["signal_correct"]) / n
        rows.append((city, n, model_avg, market_avg, actual_yes, edge_real, sig_acc))

    for city, n, model, market, actual, edge, acc in sorted(rows, key=lambda r: -r[1]):
        print(
            f"  {city:<6}  {n:>4}  {model*100:6.1f}%  {market*100:6.1f}%  "
            f"{actual*100:6.1f}%  {edge*100:+8.1f}%  {acc*100:6.1f}%"
        )

    # ── Actual trade P&L ─────────────────────────────────────────────
    if _TRADES_FILE.exists():
        with open(_TRADES_FILE) as f:
            trades = json.load(f).get("trades", [])
        settled = [t for t in trades if t.get("outcome") in ("won", "lost")]
        if settled:
            wins   = sum(1 for t in settled if t["outcome"] == "won")
            losses = len(settled) - wins
            pnl_parts = []
            for t in settled:
                risked = t.get("dollars_risked", 0) or 0
                payout = t.get("dollars_payout", 0) or 0
                if t["outcome"] == "won":
                    pnl_parts.append(round(payout - risked, 2))
                else:
                    pnl_parts.append(-risked)
            total_pnl = round(sum(pnl_parts), 2)
            print(f"\n{'─'*65}")
            print(f"  ACTUAL TRADES: {wins}W / {losses}L  |  Net P&L: ${total_pnl:+.2f}")
