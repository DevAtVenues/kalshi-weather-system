"""
Step 5 — forward shadow validation for the price-path book (DORMANT PILOT).

A PAPER book, replayed daily from logged data: for each newly-complete UTC day
it assembles that day's panel (top-of-book + as-of model views), opens the
Step-2-gated entries, and runs EVERY exit policy through the Step-3 engine,
appending per-trade results to a forward shadow log. This is the ~40-day
forward record the original plan required — accruing at zero risk and zero
API cost (everything is read from the logger's own files), so an activation
decision, if the economics ever clear, starts from evidence instead of a gate
re-run.

Because every input is logged, a daily batch replay is *deterministically
identical* to having run the book live intraday — same entries, same exits,
same fills. That is why the shadow can be a daily job instead of a scheduled
intraday process.

A day D is processed once D+1's orderbook file is also complete (positions
entered late on D may exit on D+1). Rows carry mode="forward"; an optional
--backfill replays pre-build history as mode="replay", which the report
excludes by default (forward evidence is sacred — CLAUDE.md rule 4).

State : data/analysis/price_path/shadow_state.json
Log   : data/analysis/price_path/shadow_log.jsonl
Run   : .venv/bin/python scripts/price_path_shadow.py            (incremental)
        .venv/bin/python scripts/price_path_shadow.py --backfill (seed history)
Wired : scripts/daily_review.py calls run_and_report() every morning.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from kalshi_weather.pricepath.engine import (  # noqa: E402
    POLICIES, day_block_ci, gated_entries, simulate,
)
from build_price_path_panel import (  # noqa: E402
    _settlement_date_from_ticker, _top_of_book, assemble_panel, load_signals,
)

ROOT = Path(__file__).resolve().parents[1]
OB_DIR = ROOT / "data" / "logger" / "orderbook"
OUT_DIR = ROOT / "data" / "analysis" / "price_path"
STATE = OUT_DIR / "shadow_state.json"
LOG = OUT_DIR / "shadow_log.jsonl"

MIN_FORWARD_DAYS = 10          # below this the report prints UNPROVEN (charter)


def _load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        # First run: the forward record starts at build time, never earlier.
        return {"last_day": (date.today() - timedelta(days=2)).isoformat()}


def _day_book(d: date) -> pd.DataFrame | None:
    f = OB_DIR / f"{d.isoformat()}.parquet"
    if not f.exists():
        return None
    day = pd.read_parquet(
        f, columns=["snapshot_utc", "ticker", "side", "price_cents", "quantity"])
    return _top_of_book(day)


def process_day(d: date, sig: pd.DataFrame, mode: str) -> list[dict]:
    """Replay one settlement-of-entries day through every policy."""
    books = [_day_book(d), _day_book(d + timedelta(days=1))]
    if books[0] is None or books[0].empty:
        return []
    book = pd.concat([b for b in books if b is not None], ignore_index=True)
    book["snapshot_utc"] = pd.to_datetime(book["snapshot_utc"], utc=True)
    book["settlement_date_ob"] = book["ticker"].map(_settlement_date_from_ticker)
    panel = assemble_panel(book, sig)

    entries = gated_entries(panel)
    entries = entries[entries["snapshot_utc"].dt.date == d]
    rows: list[dict] = []
    for pol in POLICIES.values():
        trades = simulate(panel, pol, entries=entries)
        for r in trades.to_dict("records"):
            rows.append({"day": d.isoformat(), "mode": mode, **r})
    return rows


def _append(rows: list[dict]) -> int:
    """Append with (day, ticker, direction, policy) dedup — idempotent."""
    seen = set()
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            try:
                r = json.loads(line)
                seen.add((r["day"], r["ticker"], r["direction"], r["policy"]))
            except (json.JSONDecodeError, KeyError):
                continue
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(LOG, "a") as fh:
        for r in rows:
            key = (r["day"], r["ticker"], r["direction"], r["policy"])
            if key in seen:
                continue
            seen.add(key)
            fh.write(json.dumps(r) + "\n")
            n += 1
    return n


def run(backfill: bool = False) -> str:
    """Process every pending complete day. Returns a one-line summary."""
    state = _load_state()
    last = date.fromisoformat(state["last_day"])
    newest_complete = date.today() - timedelta(days=2)   # D+1 must be complete
    days: list[tuple[date, str]] = []
    if backfill:
        have = sorted(p.stem for p in OB_DIR.glob("*.parquet"))
        days += [(date.fromisoformat(s), "replay") for s in have
                 if date.fromisoformat(s) <= min(last, newest_complete)]
    d = last + timedelta(days=1)
    while d <= newest_complete:
        days.append((d, "forward"))
        d += timedelta(days=1)
    if not days:
        return "shadow book: up to date (no newly-complete days)."

    sig = load_signals()
    appended = 0
    processed = 0
    for d, mode in days:
        rows = process_day(d, sig, mode)
        appended += _append(rows)
        processed += 1
        if mode == "forward":
            state["last_day"] = d.isoformat()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state))
    return (f"shadow book: processed {processed} day(s), "
            f"appended {appended} paper trades.")


def report(include_replay: bool = False) -> str:
    if not LOG.exists():
        return "shadow book: no record yet."
    df = pd.DataFrame([json.loads(l) for l in LOG.read_text().splitlines()
                       if l.strip()])
    if not include_replay:
        df = df[df["mode"] == "forward"]
    if df.empty:
        return ("shadow book: forward record empty so far (accrues daily; "
                "use --include-replay for the historical replay view).")
    n_days = df["settlement_date"].nunique()
    badge = ("" if n_days >= MIN_FORWARD_DAYS
             else f"  [UNPROVEN: {n_days}/{MIN_FORWARD_DAYS} days]")
    lines = [f"PRICE-PATH SHADOW BOOK (dormant pilot, paper only){badge}",
             f"  {'policy':>14} {'n':>6} {'days':>4} {'PESS net':>9} "
             f"{'ci95':>18} {'OPT net':>9}"]
    for name, g in df.groupby("policy"):
        p = day_block_ci(g, "pnl_pess")
        o = day_block_ci(g, "pnl_opt")
        flag = "  <== CI>0 (escalate: re-run Step-2 study)" if p[1] > 0 else ""
        lines.append(f"  {name:>14} {p[3]:>6,} {p[4]:>4} {100*p[0]:>+8.2f} "
                     f"[{100*p[1]:+6.2f},{100*p[2]:+6.2f}] "
                     f"{100*o[0]:>+8.2f}{flag}")
    lines.append("  activation bar: pessimistic CI > 0 (charter; both fill "
                 "models reported, only PESS decides).")
    return "\n".join(lines)


def run_and_report() -> str:
    """Single entry point for daily_review: advance the book, then report."""
    return run() + "\n" + report()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="also replay all logged history (mode=replay)")
    ap.add_argument("--include-replay", action="store_true")
    args = ap.parse_args()
    print(run(backfill=args.backfill))
    print(report(include_replay=args.include_replay))
