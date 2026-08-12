"""
Personal trade log — persistent in data/trades.json.

Grading rules (NO wins if):
  T-type "less"    (YES if high < cap)    : actual_high >= cap
  T-type "greater" (YES if high >= floor) : actual_high < floor
  B-type "between" (YES if floor<=h<cap)  : actual_high < floor OR actual_high >= cap
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

TRADES_PATH = Path(__file__).parents[3] / "data" / "trades.json"

CITY_DISPLAY = {
    "NYC": "New York City", "CHI": "Chicago", "MIA": "Miami",
    "PHX": "Phoenix", "BOS": "Boston", "SFO": "San Francisco",
    "LAX": "Los Angeles", "DEN": "Denver", "ATL": "Atlanta",
    "HOU": "Houston", "DCA": "Washington DC", "MSY": "New Orleans",
    "PHL": "Philadelphia", "SEA": "Seattle", "LAS": "Las Vegas",
    "MSP": "Minneapolis", "SAT": "San Antonio", "DAL": "Dallas",
    "AUS": "Austin", "OKC": "Oklahoma City",
}


def load_trades() -> list[dict]:
    if not TRADES_PATH.exists():
        return []
    with open(TRADES_PATH) as f:
        return json.load(f).get("trades", [])


def save_trades(trades: list[dict]) -> None:
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADES_PATH, "w") as f:
        json.dump({"trades": trades}, f, indent=2, default=str)


def add_trade(trade: dict) -> dict:
    trades = load_trades()
    if "id" not in trade:
        trade["id"] = "t" + uuid.uuid4().hex[:6]
    trades.append(trade)
    save_trades(trades)
    return trade


def _yes_wins(trade: dict, actual_high: float) -> bool:
    st     = trade.get("strike_type", "")
    floor_ = trade.get("floor_strike")
    cap_   = trade.get("cap_strike")
    if st == "less":
        return actual_high < float(cap_)
    if st == "greater":
        return actual_high >= float(floor_)
    if st == "between":
        return float(floor_) <= actual_high < float(cap_)
    return False


def grade_trade(trade_id: str, actual_high: float | None = None,
                outcome: str | None = None) -> bool:
    """
    Grade a trade by id. Pass either actual_high (auto-computes outcome) or
    a literal outcome string ("won"/"lost"). Returns True if found and updated.
    """
    trades = load_trades()
    for t in trades:
        if t["id"] == trade_id:
            if actual_high is not None:
                t["settlement_temp"] = actual_high
                yes = _yes_wins(t, actual_high)
                t["outcome"] = "won" if (
                    (t["action"] == "BUY_YES" and yes) or
                    (t["action"] == "BUY_NO"  and not yes)
                ) else "lost"
            elif outcome in ("won", "lost"):
                t["outcome"] = outcome
            save_trades(trades)
            return True
    return False


def auto_grade_from_market(trades: list[dict], market_snapshot: dict[str, dict]) -> list[dict]:
    """
    Grade open trades whose ticker appears in market_snapshot with extreme prices
    (yes_bid < 0.02 → NO effectively won; yes_ask > 0.98 → YES effectively won).
    market_snapshot: {ticker: {yes_bid: float, yes_ask: float}}
    Returns the updated trades list (also persists to disk if any changed).
    """
    changed = False
    for t in trades:
        if t.get("outcome") is not None:
            continue
        snap = market_snapshot.get(t["ticker"])
        if not snap:
            continue
        bid = snap.get("yes_bid", 0.5)
        ask = snap.get("yes_ask", 0.5)
        if bid < 0.02:
            # YES essentially zero → NO won
            new_outcome = "won" if t["action"] == "BUY_NO" else "lost"
        elif ask > 0.98:
            # YES essentially certain → YES won
            new_outcome = "won" if t["action"] == "BUY_YES" else "lost"
        else:
            continue
        t["outcome"] = new_outcome
        changed = True
    if changed:
        save_trades(trades)
    return trades


def compute_stats(trades: list[dict]) -> dict:
    settled  = [t for t in trades if t.get("outcome") in ("won", "lost")]
    pending  = [t for t in trades if t.get("outcome") not in ("won", "lost")]
    wins     = [t for t in settled if t["outcome"] == "won"]
    losses   = [t for t in settled if t["outcome"] == "lost"]

    win_rate = round(len(wins) / len(settled) * 100) if settled else 0

    # Dollar P&L — uses dollars_risked / dollars_payout fields
    def _pnl(t: dict) -> float | None:
        risked  = t.get("dollars_risked")
        payout  = t.get("dollars_payout")
        outcome = t.get("outcome")
        if risked is None:
            return None
        if outcome == "won":
            return round((payout or risked) - risked, 2)
        if outcome == "lost":
            return round(-risked, 2)
        return None

    pnl_values = [v for t in settled if (v := _pnl(t)) is not None]
    total_pnl  = round(sum(pnl_values), 2) if pnl_values else None

    # Earliest entry date for range display
    all_dates = [t.get("entry_date", "") for t in trades if t.get("entry_date")]
    date_range_start = min(all_dates) if all_dates else None

    # Results by settlement day (settled + pending)
    by_day: dict[str, dict[str, int]] = {}
    for t in trades:
        d = t.get("settlement_date", "?")
        by_day.setdefault(d, {"wins": 0, "losses": 0, "pending": 0})
        outcome = t.get("outcome")
        if outcome == "won":
            by_day[d]["wins"] += 1
        elif outcome == "lost":
            by_day[d]["losses"] += 1
        else:
            by_day[d]["pending"] += 1

    days = [
        {
            "date":     d,
            "wins":     r["wins"],
            "losses":   r["losses"],
            "pending":  r["pending"],
            "win_rate": round(r["wins"] / (r["wins"] + r["losses"]) * 100)
                        if (r["wins"] + r["losses"]) > 0 else None,
        }
        for d, r in sorted(by_day.items(), reverse=True)
    ]

    # Results by city (settled + pending)
    by_city: dict[str, dict] = {}
    for t in trades:
        city = t.get("city", "?")
        by_city.setdefault(city, {"wins": 0, "losses": 0, "pending": 0,
                                   "name": CITY_DISPLAY.get(city, city)})
        outcome = t.get("outcome")
        if outcome == "won":
            by_city[city]["wins"] += 1
        elif outcome == "lost":
            by_city[city]["losses"] += 1
        else:
            by_city[city]["pending"] += 1

    cities = sorted(
        [
            {
                "city":     c,
                "name":     r["name"],
                "wins":     r["wins"],
                "losses":   r["losses"],
                "pending":  r["pending"],
                "win_rate": round(r["wins"] / (r["wins"] + r["losses"]) * 100)
                            if (r["wins"] + r["losses"]) > 0 else None,
            }
            for c, r in by_city.items()
        ],
        key=lambda x: -(x["wins"] + x["losses"] + x["pending"]),
    )

    def _trade_row(t: dict) -> dict:
        return {
            "id":              t.get("id", ""),
            "ticker":          t.get("ticker", ""),
            "city":            t.get("city", ""),
            "city_name":       CITY_DISPLAY.get(t.get("city", ""), t.get("city", "")),
            "action":          t.get("action", ""),
            "settlement_date": t.get("settlement_date", ""),
            "dollars_risked":  t.get("dollars_risked"),
            "dollars_payout":  t.get("dollars_payout"),
            "outcome":         t.get("outcome"),
            "settlement_temp": t.get("settlement_temp"),
            "notes":           t.get("notes", ""),
        }

    # Pending list for display
    pending_list = [_trade_row(t) for t in pending]

    # All trades grouped by settlement date for day drill-down
    trades_by_day: dict[str, list] = {}
    for t in sorted(trades, key=lambda x: x.get("settlement_date", "")):
        d = t.get("settlement_date", "?")
        trades_by_day.setdefault(d, []).append(_trade_row(t))

    return {
        "win_rate":              win_rate,
        "wins":                  len(wins),
        "losses":                len(losses),
        "total_settled":         len(settled),
        "pending":               len(pending),
        "total_pnl":             total_pnl,
        "date_range_start":      date_range_start,
        "by_day":                days,
        "by_city":               cities,
        "pending_list":          pending_list,
        "trades_by_day":         trades_by_day,
    }
