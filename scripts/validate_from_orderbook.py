"""
Validate cities on the LOGGED ORDERBOOK instead of candles.

Why: Kalshi's public candle history only records trade PRINTS, which in thin
markets cluster at the extremes (0.005/0.995) — so candle-based validation finds
"no liquid ATM" for newer/smaller cities even when a real two-sided market existed.
The live logger (data/logger/orderbook/) captures resting DEPTH every snapshot, so
it can reconstruct a true at-the-money decision price the candles never show.

Decision price (mirrors ingest.kalshi.get_decision_prices): the latest orderbook
snapshot strictly BEFORE the contract's settlement window opens (LST midnight).
  yes_bid = max(price_cents | side == "yes")           # best resting YES buy
  yes_ask = 100 - max(price_cents | side == "no")       # buying YES = selling NO

Each city's EXACT deployed production rule is tested (leakage-safe: gaussian rules
train on 2022-23, isotonic on 2024 — neither sees 2026). Edge CI = 95% block
bootstrap, pessimistic fill.

  --promote   write cities whose CI lower bound > 0 (and >= MIN_TRADES) to
              data/models/oos_ci_overlay.json, which store.py merges into _OOS_CI.

Run: .venv/bin/python scripts/validate_from_orderbook.py [--promote]
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import cloudpickle
import pandas as pd

from kalshi_weather.align import build_aligned_dataset, threshold_contracts
from kalshi_weather.backtest import run_backtest
from kalshi_weather.config import get_station
from kalshi_weather.ingest.forecasts import fetch_forecasts
from kalshi_weather.ingest.kalshi import fetch_all_markets, _get
from kalshi_weather.ingest.labels import fetch_labels
from kalshi_weather.tz import lst_date_for_utc, settlement_window_utc

ROOT       = Path(__file__).parents[1]
OB_DIR     = ROOT / "data" / "logger" / "orderbook"
OVERLAY    = ROOT / "data" / "models" / "oos_ci_overlay.json"
PROD       = cloudpickle.load(open(ROOT / "data" / "models" / "production_rules.pkl", "rb"))

FCST_MODELS      = ["gfs_seamless"]
ATM_MIN, ATM_MAX = 0.15, 0.85
MAX_SPREAD       = 0.15
MIN_BID          = 0.01
MIN_TRADES       = 15          # below this the CI is too wide to act on


def fetch_settled_markets(series: str, since: date) -> pd.DataFrame:
    """
    Settled markets from the regular /markets endpoint (status=settled). Needed
    because fetch_all_markets() hits /historical/markets, which stops at Kalshi's
    ~2026-04-04 historical cutoff — the orderbook-logged June contracts are newer.
    Paginates most-recent-first and stops once contracts predate `since`.
    """
    out: list[dict] = []
    cursor: str | None = None
    while True:
        params = {"series_ticker": series, "status": "settled", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data  = _get("/markets", params)
        batch = data.get("markets", [])
        if not batch:
            break
        out.extend(batch)
        oldest = min((m.get("close_time", "") for m in batch), default="")
        cursor = data.get("cursor")
        # stop once this page's oldest close predates the window
        if not cursor or (oldest and oldest[:10] < since.isoformat()):
            break
    return pd.DataFrame(out)


def load_orderbook() -> pd.DataFrame:
    files = sorted(OB_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"], utc=True)
    return df


def decision_prices_from_ob(ob: pd.DataFrame, tickers: list[str], station: str) -> pd.DataFrame:
    """Best bid/ask at the latest snapshot before each contract's LST-midnight window."""
    rows: list[dict] = []
    sub = ob[ob["ticker"].isin(set(tickers))]
    for ticker, g in sub.groupby("ticker"):
        # settlement window start from the ticker's settlement date
        try:
            # ticker like KXHIGHTPHX-26JUN10-T102 → parse the date via close logic:
            # use the latest snapshot's date is unreliable; derive from settlement date.
            sdate = _ticker_sdate(ticker)
            win_start, _ = settlement_window_utc(sdate, station)
        except Exception:
            continue
        pre = g[g["snapshot_utc"] < win_start]
        if pre.empty:
            continue
        snap_t = pre["snapshot_utc"].max()
        snap   = pre[pre["snapshot_utc"] == snap_t]
        yes = snap[snap["side"] == "yes"]["price_cents"]
        no  = snap[snap["side"] == "no"]["price_cents"]
        if yes.empty or no.empty:
            continue
        yes_bid = yes.max() / 100.0
        yes_ask = (100 - no.max()) / 100.0
        if yes_ask < yes_bid:           # crossed/degenerate book
            continue
        rows.append({
            "ticker":            ticker,
            "decision_bid":      round(yes_bid, 4),
            "decision_ask":      round(yes_ask, 4),
            "decision_mid":      round((yes_bid + yes_ask) / 2, 4),
            "decision_ts":       snap_t,
        })
    return pd.DataFrame(rows)


_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
           "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

def _ticker_sdate(ticker: str) -> date:
    # KXHIGHTPHX-26JUN10-T102
    part = ticker.split("-")[1]
    yy, mon, dd = int(part[:2]), _MONTHS[part[2:5]], int(part[5:7])
    return date(2000 + yy, mon, dd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true",
                    help="write CI-positive cities to oos_ci_overlay.json")
    args = ap.parse_args()

    ob = load_orderbook()
    if ob.empty:
        print("No orderbook data logged yet.")
        return
    lo_d, hi_d = ob["snapshot_utc"].min().date(), ob["snapshot_utc"].max().date()
    print(f"Orderbook window: {lo_d} → {hi_d}  ({ob['ticker'].nunique()} tickers, {len(ob):,} levels)")
    today = datetime.now(timezone.utc).date()

    results: dict[str, dict] = {}
    for city, entry in PROD.items():
        series = entry["series"]
        rule   = entry["active_rule"]
        st     = get_station(city)
        nws, lat, lon = st["nws_station"], st["lat"], st["lon"]

        try:
            markets = fetch_settled_markets(series, lo_d)
        except Exception as exc:
            print(f"{city}: market fetch failed: {exc}")
            continue
        if markets.empty:
            continue

        labels    = fetch_labels(nws, [2026])
        forecasts = fetch_forecasts(nws, lat, lon, [2026], models=FCST_MODELS)
        aligned   = threshold_contracts(
            build_aligned_dataset(markets, labels, forecasts, station=nws, model=FCST_MODELS[0])
        )
        # settled, within the logged window
        sret = aligned[aligned["settlement_date"].apply(
            lambda d: hasattr(d, "year") and lo_d <= d < today)].reset_index(drop=True)
        if sret.empty:
            continue

        prices = decision_prices_from_ob(ob, sret["ticker"].tolist(), nws)
        if prices.empty:
            print(f"{city}: 0 contracts with a pre-window orderbook snapshot")
            continue
        m = sret.merge(prices, on="ticker", how="inner")
        m["last_price_dollars"] = m["decision_mid"]
        m["yes_bid_dollars"]    = m["decision_bid"]
        m["yes_ask_dollars"]    = m["decision_ask"]

        bid = pd.to_numeric(m["decision_bid"], errors="coerce")
        ask = pd.to_numeric(m["decision_ask"], errors="coerce")
        mid = pd.to_numeric(m["decision_mid"], errors="coerce")
        liq = m[(bid > MIN_BID) & ((ask - bid) < MAX_SPREAD) & mid.between(ATM_MIN, ATM_MAX)].reset_index(drop=True)
        if len(liq) < MIN_TRADES:
            print(f"{city}: {len(liq)} liquid-ATM (orderbook) — below MIN_TRADES={MIN_TRADES}, skip")
            continue

        res = run_backtest(liq, rule, rule_name=f"{city} OB", atm_min=ATM_MIN, atm_max=ATM_MAX)
        p = res.metrics_pess
        results[city] = {"n": p.get("n_trades", 0), "win": p.get("win_rate"),
                         "lo": p.get("edge_ci_low"), "hi": p.get("edge_ci_high")}
        print(f"{city}: trades={p.get('n_trades')} win={p.get('win_rate'):.1%} "
              f"CI=({p.get('edge_ci_low'):.3f},{p.get('edge_ci_high'):.3f})")

    # Summary + optional promotion
    print(f"\n{'═'*60}\n  ORDERBOOK-BASED 2026 OOS — deployed rule\n{'═'*60}")
    validated = {c: r for c, r in results.items() if r["lo"] is not None and r["lo"] > 0 and r["n"] >= MIN_TRADES}
    for c, r in results.items():
        tag = "  ✓ validated +EV" if c in validated else ""
        print(f"  {c:5} n={r['n']:>3} CI=({r['lo']:.3f},{r['hi']:.3f}){tag}")
    if not results:
        print("  (no city has enough orderbook history yet — accumulating)")

    if args.promote:
        overlay = {}
        if OVERLAY.exists():
            try: overlay = json.loads(OVERLAY.read_text())
            except Exception: overlay = {}
        for c, r in validated.items():
            overlay[c] = {"T": [round(r["lo"], 3), round(r["hi"], 3)], "B": None,
                          "source": "orderbook", "as_of": str(today), "n": r["n"]}
        OVERLAY.parent.mkdir(parents=True, exist_ok=True)
        OVERLAY.write_text(json.dumps(overlay, indent=2))
        print(f"\n  Promoted {len(validated)} city(ies) → {OVERLAY.name}: {list(validated)}")


if __name__ == "__main__":
    main()
