"""
Step 1 of the price-path (mispricing) book — see docs/STATE_OF_THE_MODEL.md Part V.

Builds the joined price-path panel: one row per (ticker, orderbook snapshot),
carrying the executable top-of-book (bid / ask / spread / depth) joined AS-OF to
the most recent ensemble model view (model-P, ensemble dist, obs-so-far) known at
that instant.

No look-ahead, machine-enforced by construction: each snapshot is stamped with the
latest signals_log row whose run_ts <= snapshot_utc for the SAME ticker
(pd.merge_asof, direction="backward"). A snapshot can never see a model run that
had not happened yet.

Spine = orderbook snapshots (dense, the true executable book we log every few
minutes), NOT signals rows (sparse, fire only on run-live cadence). The price PATH
that Step 2 studies is the orderbook mid moving over time; the model view is the
slowly-updating signal attached to each point on that path.

Top-of-book convention mirrors scripts/validate_from_orderbook.py exactly:
    yes_bid = max(price_cents | side=="yes") / 100          # best resting YES buy
    yes_ask = (100 - max(price_cents | side=="no")) / 100   # buying YES == selling NO
Crossed/degenerate books (yes_ask < yes_bid) are flagged, not silently dropped.

Output: data/analysis/price_path/panel.parquet
Run:    .venv/bin/python scripts/build_price_path_panel.py
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import pandas as pd

from kalshi_weather.backtest import maker_fee, taker_fee

log = logging.getLogger("build_price_path_panel")

ROOT = Path(__file__).resolve().parents[1]
OB_DIR = ROOT / "data" / "logger" / "orderbook"
SIGNALS = ROOT / "data" / "signals" / "signals_log.jsonl"
OUT = ROOT / "data" / "analysis" / "price_path" / "panel.parquet"

# Model fields carried from the signals log onto each price-path point.
SIGNAL_COLS = [
    "ticker", "run_ts", "settlement_date", "city",
    "prob_estimate", "prob_raw", "prob_shrunk", "prob_source",
    "market_mid", "ens_p50", "ens_sd", "ens_center_shift",
    "running_high", "hours_to_settle", "local_hour", "is_same_day",
    "threshold_f", "strike_type", "direction", "contract_type",
    "edge_raw", "edge_shrunk", "nws_disagree", "stale_fight",
]

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _settlement_date_from_ticker(ticker: str) -> str | None:
    """KXHIGHNY-26AUG07-T96 -> '2026-08-07'. None if unparseable."""
    try:
        tok = ticker.split("-")[1]          # 26AUG07
        yy, mon, dd = int(tok[:2]), tok[2:5].upper(), int(tok[5:7])
        return f"20{yy:02d}-{_MONTHS[mon]:02d}-{dd:02d}"
    except (IndexError, KeyError, ValueError):
        return None


def _top_of_book(day: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse one day of raw depth rows into one row per (ticker, snapshot_utc)
    with the executable top of book on each side.

    day columns: snapshot_utc, ticker, side, price_cents, quantity
    """
    keys = ["ticker", "snapshot_utc"]
    sides = {}
    for side in ("yes", "no"):
        s = day[day["side"] == side]
        if s.empty:
            sides[side] = pd.DataFrame(columns=keys + ["best_c", "best_qty"])
            continue
        # best price level per (ticker, snapshot); carry the quantity resting there.
        idx = s.groupby(keys, sort=False)["price_cents"].idxmax()
        best = (s.loc[idx, keys + ["price_cents", "quantity"]]
                .rename(columns={"price_cents": "best_c", "quantity": "best_qty"}))
        sides[side] = best

    m = sides["yes"].merge(sides["no"], on=keys, how="inner",
                           suffixes=("_yes", "_no"))
    if m.empty:
        return m

    m["yes_bid"] = m["best_c_yes"] / 100.0
    m["yes_ask"] = (100 - m["best_c_no"]) / 100.0
    m["yes_mid"] = (m["yes_bid"] + m["yes_ask"]) / 2.0
    m["spread"] = m["yes_ask"] - m["yes_bid"]
    m["crossed"] = m["yes_ask"] < m["yes_bid"]
    # Depth available to actually transact at the touch:
    #   buying YES  == lifting the best NO offer  -> depth = no-side best qty
    #   selling YES == hitting the best YES bid   -> depth = yes-side best qty
    m["yes_bid_depth"] = m["best_qty_yes"]     # contracts you can SELL YES into
    m["yes_ask_depth"] = m["best_qty_no"]      # contracts you can BUY YES from
    return m[keys + ["yes_bid", "yes_ask", "yes_mid", "spread", "crossed",
                     "yes_bid_depth", "yes_ask_depth"]]


def build_book_panel() -> pd.DataFrame:
    """Top-of-book for every logged snapshot, across all daily files."""
    files = sorted(glob.glob(str(OB_DIR / "*.parquet")))
    if not files:
        raise SystemExit(f"no orderbook files under {OB_DIR}")
    parts = []
    for i, f in enumerate(files, 1):
        day = pd.read_parquet(
            f, columns=["snapshot_utc", "ticker", "side", "price_cents", "quantity"])
        tob = _top_of_book(day)
        if not tob.empty:
            parts.append(tob)
        log.info("  [%d/%d] %s -> %d book points", i, len(files),
                 Path(f).name, len(tob))
    book = pd.concat(parts, ignore_index=True)
    book["snapshot_utc"] = pd.to_datetime(book["snapshot_utc"], utc=True)
    book["settlement_date_ob"] = book["ticker"].map(_settlement_date_from_ticker)
    return book


def load_signals() -> pd.DataFrame:
    """signals_log.jsonl -> tidy frame of the model view over time."""
    rows = []
    with open(SIGNALS) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({c: d.get(c) for c in SIGNAL_COLS})
    sig = pd.DataFrame(rows)
    sig["run_ts"] = pd.to_datetime(sig["run_ts"], utc=True, errors="coerce")
    sig = sig.dropna(subset=["run_ts", "ticker"])
    return sig


def assemble_panel(book: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """AS-OF join model views onto book points + derived signal/econ columns.
    Shared by the full historical build (below) and the daily shadow replay
    (scripts/price_path_shadow.py) so there is exactly ONE join implementation."""
    book = book.sort_values("snapshot_utc")
    sig = sig.sort_values("run_ts")
    panel = pd.merge_asof(
        book, sig,
        left_on="snapshot_utc", right_on="run_ts",
        by="ticker", direction="backward",
    )

    # Prefer the signal's settlement_date; fall back to the ticker-parsed one.
    panel["settlement_date"] = panel["settlement_date"].fillna(
        panel["settlement_date_ob"])

    # Derived signal / economics columns.
    panel["signal_age_min"] = (
        (panel["snapshot_utc"] - panel["run_ts"]).dt.total_seconds() / 60.0)
    panel["has_model"] = panel["run_ts"].notna()
    # Core signal: model thinks YES is under(+)/over(-) priced vs the live mid.
    panel["divergence"] = panel["prob_estimate"] - panel["yes_mid"]

    # Round-trip cost on the 0-1 price scale (paid to enter AND exit):
    #   spread is paid once crossing in + once crossing out == the full spread.
    #   fees are charged per side.
    ask, bid = panel["yes_ask"], panel["yes_bid"]
    panel["fee_taker_rt"] = ask.map(taker_fee) + bid.map(taker_fee)
    panel["fee_maker_rt"] = ask.map(maker_fee) + bid.map(maker_fee)
    panel["round_trip_taker"] = panel["spread"] + panel["fee_taker_rt"]
    panel["round_trip_maker"] = panel["spread"] + panel["fee_maker_rt"]

    return panel


def build_panel() -> pd.DataFrame:
    log.info("collapsing orderbook depth to top-of-book ...")
    book = build_book_panel()
    log.info("book points: %d across %d tickers",
             len(book), book["ticker"].nunique())

    log.info("loading signals ...")
    sig = load_signals()
    log.info("signal rows: %d across %d tickers",
             len(sig), sig["ticker"].nunique())

    return assemble_panel(book, sig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    panel = build_panel()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)

    # Honest summary — what did we actually get?
    n = len(panel)
    with_model = int(panel["has_model"].sum())
    crossed = int(panel["crossed"].sum())
    log.info("\n=== price-path panel ===")
    log.info("rows (ticker x snapshot) : %d", n)
    log.info("distinct tickers         : %d", panel["ticker"].nunique())
    log.info("with a model view        : %d (%.1f%%)", with_model, 100 * with_model / n)
    log.info("crossed/degenerate books : %d (%.1f%%)", crossed, 100 * crossed / n)
    log.info("date span                : %s .. %s",
             panel["snapshot_utc"].min(), panel["snapshot_utc"].max())
    md = panel[panel["has_model"] & ~panel["crossed"]]
    if len(md):
        log.info("median spread (modeled, non-crossed): %.3f", md["spread"].median())
        log.info("median |divergence|                 : %.3f", md["divergence"].abs().median())
        log.info("median signal age (min)             : %.1f", md["signal_age_min"].median())
    log.info("wrote -> %s", out)


if __name__ == "__main__":
    main()
