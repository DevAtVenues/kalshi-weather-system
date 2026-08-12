"""
Log trades to Google Sheets AND update the risk-layer state.

This is the single entry point where real fills/settlements are recorded, so it
also feeds RiskManager: "Open" trades register exposure against the correlation
cap, "Settled"/"Closed" trades register P/L against the daily circuit breaker.
Without this the breaker could never trip (nothing else calls record_fill/record_pnl).

Usage:
    .venv/bin/python scripts/log_to_sheets.py trades.json
    .venv/bin/python scripts/log_to_sheets.py  # reads from data/trades_to_log.json

First run opens a browser for Google OAuth. Token is cached; subsequent
runs are fully headless.

JSON format (list of trade objects):
[
  {
    "date":             "06/15/2026",
    "market":           "Miami High Temp, Jun 15 — Between 89.5-91.5°F (B91.5)",
    "position":         "No",
    "amount_committed": "$3.71",
    "entry_price":      "$0.53",
    "settlement_price": "$1.00",
    "fees":             "$0.08",
    "net_pl":           "$3.29",
    "status":           "Settled",
    "screenshot":       "",
    "notes":            "BUY NO. High reached 95°F, well above bracket ceiling."
  }
]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from kalshi_weather.sheets_logger import log_trades
from kalshi_weather.live.risk import RiskManager, parse_dollars, city_from_market

DEFAULT_INPUT = Path(__file__).parents[1] / "data" / "trades_to_log.json"
WATCHLIST     = Path(__file__).parents[1] / "data" / "analysis" / "watchlist.json"


def watchlist_entry_from_trade(trade: dict) -> dict | None:
    """Best-effort map of an Open trade onto a position_watch entry so the
    obs/price/thesis alerts arm automatically at fill time (2026-08-03: KPHL
    ran into a held NO's loss zone with an EMPTY watchlist — never again).
    Returns None when the market string can't be parsed; caller warns."""
    import re
    from datetime import datetime
    from kalshi_weather.live.runner import CITY_CONFIGS

    if str(trade.get("status", "")).lower() != "open":
        return None
    market = trade.get("market", "")
    city = city_from_market(market)
    cfg = CITY_CONFIGS.get(city) if city else None
    m = re.search(r"\((B[\d.]+|T\d+)\)", market)
    if not (cfg and m):
        return None
    suffix = m.group(1)
    try:
        sdate = datetime.strptime(trade["date"], "%m/%d/%Y").date()
    except (KeyError, ValueError):
        return None
    ticker = f"{cfg['series']}-{sdate.strftime('%y%b%d').upper()}-{suffix}"
    direction = "BUY_NO" if str(trade.get("position", "")).lower() == "no" else "BUY_YES"
    entry: dict = {
        "ticker": ticker, "direction": direction,
        "station": cfg["nws_station"], "settlement_date": str(sdate),
        "lst_offset_h": cfg.get("lst_offset", -5),
        "note": f"auto from fill {trade.get('entry_price','')}",
    }
    if suffix.startswith("B"):
        mid = float(suffix[1:])
        entry.update(strike_type="between", floor=int(mid - 0.5), cap=int(mid + 0.5))
    else:
        thr = int(suffix[1:])
        low = re.search(r"\bbelow\b|\bor lower\b|\bunder\b", market, re.I)
        entry.update(strike_type="less" if low else "greater", threshold=thr,
                     floor=None, cap=thr)
    return entry


def arm_watchlist(trades: list[dict]) -> None:
    entries = []
    try:
        entries = json.loads(WATCHLIST.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    known = {(e.get("ticker"), e.get("direction")) for e in entries}
    added = 0
    for t in trades:
        e = watchlist_entry_from_trade(t)
        if e is None:
            if str(t.get("status", "")).lower() == "open":
                print(f"  WARNING: could not arm position watch for: {t.get('market','?')}"
                      " — add data/analysis/watchlist.json entry by hand")
            continue
        if (e["ticker"], e["direction"]) in known:
            continue
        entries.append(e)
        known.add((e["ticker"], e["direction"]))
        added += 1
    if added:
        WATCHLIST.parent.mkdir(parents=True, exist_ok=True)
        WATCHLIST.write_text(json.dumps(entries, indent=1))
        print(f"  Armed position watch for {added} new position(s).")


def update_risk_state(trades: list[dict], risk: RiskManager) -> None:
    """Feed each logged trade into the risk layer.

    Open fill  -> record_fill (correlation-cap exposure for that city-date)
    Settled/Closed -> record_pnl (daily P/L; trips the circuit breaker on a bad day)
    """
    for t in trades:
        status = str(t.get("status", "")).strip().lower()
        committed = parse_dollars(t.get("amount_committed"))
        pl = parse_dollars(t.get("net_pl"))
        if status == "open" and committed:
            city = city_from_market(t.get("market", ""))
            if city is None:
                # Still record the exposure (under UNKNOWN) so the dollars are not
                # invisible to the risk layer — but an UNKNOWN fill is exempt from
                # its real city's correlation cap, so make the gap loud.
                city = "UNKNOWN"
                print(f"  ⚠️  could not parse a city from market string "
                      f"{t.get('market', '')!r} — exposure recorded under UNKNOWN; "
                      f"this fill will NOT count against its city's correlation cap.")
            entry = parse_dollars(t.get("entry_price")) or 0.0
            contracts = int(committed / entry) if entry > 0 else 0
            # record_fill normalizes the date to ISO so this exposure is visible to
            # size_trade's correlation cap (which keys by signal settlement_date).
            # Caveat: the sheet date is the TRADE date — for a day-ahead fill that
            # groups it one day early, which merges it with today's group (more
            # conservative, never less).
            risk.record_fill(city, str(t.get("date", "")), contracts, committed)
        elif status in ("settled", "closed") and pl is not None:
            risk.record_pnl(pl)
    print(risk.status_line())
    if risk.is_halted():
        print("⛔ CIRCUIT BREAKER TRIPPED — no more trades today. run_live pushes are suppressed.")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    if not path.exists():
        print(f"No trades file found at {path}")
        print("Pass a JSON file as argument or create data/trades_to_log.json")
        sys.exit(1)

    trades = json.loads(path.read_text())
    if not isinstance(trades, list):
        trades = [trades]

    print(f"Logging {len(trades)} trade(s) to Google Sheets…")
    log_trades(trades)
    risk = RiskManager()
    risk.reset_if_new_day()
    update_risk_state(trades, risk)
    arm_watchlist(trades)
    print("Done.")


if __name__ == "__main__":
    main()
