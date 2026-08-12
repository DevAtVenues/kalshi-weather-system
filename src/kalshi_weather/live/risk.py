"""
Risk layer for Kalshi weather trading.

Sizes trades using fractional Kelly with four hard caps:
  - Per-trade cap:      max dollars committed to a single contract
  - Circuit breaker:    halts all sizing after daily loss threshold
  - Correlation cap:    max aggregate exposure per (city, settlement_date) group
  - Cluster cap:        max aggregate exposure per (REGION cluster, settlement_date)

Same-day same-city contracts are correlated (one heat-wave = many YES outcomes),
so the correlation cap prevents the per-trade cap from being gamed by spreading
one big bet across many bracket contracts. The cluster cap extends the same
logic across CITIES: same-day bets on Dallas + Austin + San Antonio ride one
Texas ridge — three per-city caps would quietly stack ~3x the intended exposure
to a single synoptic outcome (the CLAUDE.md correlation-caps mandate). Static
regional clusters are a v1 approximation of "same weather system".

Config:  data/risk_config.json   (user-editable; created with defaults on first run)
State:   data/risk_state.json    (auto-managed; resets each calendar day)

Known limitations (accepted, do not "fix" casually):
  - State writes are last-writer-wins (no file lock). Writers are the 4x/day cron,
    the engine loop, and manual trade logging — collisions are rare and the worst
    case is a lost exposure increment, never a lost halt (halt is re-read via
    reset_if_new_day before every cycle).
  - P/L counts toward the day it is LOGGED, not the day the contract settled. A
    same-day loss logged promptly trips the breaker same-day (the case that
    matters); logging yesterday's losses after midnight burdens today instead —
    conservative direction. Log trades promptly.

Usage:
    rm = RiskManager()
    rm.reset_if_new_day()          # call at top of each cron run
    sizing = rm.size_trade(signal_row)
    if sizing["contracts"] > 0:
        print(f"Buy {sizing['contracts']} contracts (${sizing['dollars']:.2f})")
    # After a fill is confirmed:
    rm.record_fill(city, settlement_date_str, sizing["contracts"], sizing["dollars"])
    # After a contract settles:
    rm.record_pnl(pnl_dollars)
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _iso_day(d: Any) -> str:
    """Normalize any settlement-date representation to 'YYYY-MM-DD'.

    Exposure keys are built from two independent sources — signal rows
    (date/Timestamp objects) and the trade sheet ("MM/DD/YYYY" strings). If each
    caller formats its own key, the correlation cap silently never binds across
    them, so normalization lives HERE at the single choke point.
    """
    if isinstance(d, datetime):          # includes pd.Timestamp
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    s = str(d).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s[:10] if fmt == "%Y-%m-%d" else s, fmt).date().isoformat()
        except ValueError:
            continue
    return s                              # unknown format: keep raw (still groups consistently)

_DATA_DIR   = Path(__file__).parents[3] / "data"
CONFIG_PATH = _DATA_DIR / "risk_config.json"
STATE_PATH  = _DATA_DIR / "risk_state.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "bankroll":            250.0,
    "kelly_fraction":      0.25,
    "per_trade_cap_pct":   0.015,
    "circuit_breaker_pct": 0.07,
    "corr_cap_pct":        0.05,
    "corr_cluster_cap_pct": 0.05,   # aggregate cap per (region cluster, date)
    "min_contracts":       1,
}

# Static regional clusters approximating "same synoptic system" for the
# cluster cap. Deliberately coarse and conservative: over-grouping can only
# REDUCE allowed size, never increase it. Every tradeable city code must map
# to exactly one cluster (pinned by tests); unknown cities become their own
# singleton cluster (same behavior as the per-city cap).
CORR_CLUSTERS: dict[str, set[str]] = {
    "northeast":       {"NYC", "PHL", "BOS", "DCA"},
    "southeast":       {"ATL", "MIA"},
    "southern_plains": {"DAL", "AUS", "SAT", "OKC", "HOU", "MSY"},
    "midwest":         {"CHI", "MSP"},
    "mountain_desert": {"DEN", "PHX", "LAS"},
    "california":      {"LAX", "SFO"},
    "pacific_nw":      {"SEA"},
}
_CITY_TO_CLUSTER: dict[str, str] = {
    city: name for name, cities in CORR_CLUSTERS.items() for city in cities
}


def cluster_of(city: str) -> str:
    return _CITY_TO_CLUSTER.get(city, f"city:{city}")


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_PATH
    if path.exists():
        with open(path) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    cfg = dict(DEFAULT_CONFIG)
    save_config(cfg, path)
    return cfg


def save_config(cfg: dict[str, Any], path: Path | None = None) -> None:
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def _load_state(path: Path | None = None) -> dict[str, Any]:
    path = path or STATE_PATH
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_state(state: dict[str, Any], path: Path | None = None) -> None:
    path = path or STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _fresh_state(today: str) -> dict[str, Any]:
    return {
        "date":         today,
        "daily_pnl":    0.0,
        "halted":       False,
        "open_exposure": {},    # {"CITY_YYYY-MM-DD": dollars_at_risk}
    }


class RiskManager:
    """
    Stateful risk manager.  One instance per cron run; state persists across runs
    via risk_state.json (resets automatically on calendar-day rollover).
    """

    def __init__(self, config_path: Path | None = None,
                 state_path: Path | None = None) -> None:
        # Optional path overrides give the demo executor an ISOLATED risk
        # store (play-money fills must never consume real cap headroom or
        # trip the real breaker). None = the shared production files.
        self._config_path = config_path
        self._state_path  = state_path
        self.cfg    = load_config(config_path)
        self._state = self._load_or_init()

    # ── State management ──────────────────────────────────────────────────────

    def _load_or_init(self) -> dict[str, Any]:
        today = date.today().isoformat()
        raw   = _load_state(self._state_path)
        if raw.get("date") != today:
            state = _fresh_state(today)
            _save_state(state, self._state_path)
            return state
        return raw

    def reset_if_new_day(self) -> bool:
        """Reset daily counters if the calendar date has changed. Returns True on rollover.

        Also re-reads state from disk so a long-running process (run_live_engine)
        picks up a circuit breaker tripped by another process (log_to_sheets.py)."""
        today = date.today().isoformat()
        disk = _load_state(self._state_path)
        if disk:
            self._state = disk
        if self._state.get("date") != today:
            self._state = _fresh_state(today)
            _save_state(self._state, self._state_path)
            return True
        return False

    def _persist(self) -> None:
        _save_state(self._state, self._state_path)

    # ── Sizing ────────────────────────────────────────────────────────────────

    def size_trade(self, signal: pd.Series) -> dict[str, Any]:
        """
        Return a sizing recommendation for one signal row from score_contracts().

        Return dict:
          contracts  int    number of contracts to trade (0 = do not trade)
          dollars    float  total cost at the execution price
          kelly_f    float  raw Kelly fraction before caps
          cap        str|None   binding cap: "per_trade" | "corr" | None
          blocked    str|None   reason if contracts == 0
        """
        if self._state["halted"]:
            return _blocked("circuit_breaker")

        bankroll   = float(self.cfg["bankroll"])
        kelly_frac = float(self.cfg["kelly_fraction"])
        trade_cap  = float(self.cfg["per_trade_cap_pct"]) * bankroll
        corr_cap   = float(self.cfg["corr_cap_pct"]) * bankroll

        q         = float(signal["prob_estimate"])
        mid       = float(signal["market_mid"])
        direction = str(signal["direction"])

        # Kelly fraction and per-contract cost depend on direction
        if direction == "BUY_YES":
            denom = 1.0 - mid
            if denom <= 0:
                return _blocked("price_at_limit")
            kelly_f            = (q - mid) / denom
            price_per_contract = mid
        else:  # BUY_NO
            if mid <= 0:
                return _blocked("price_at_limit")
            kelly_f            = (mid - q) / mid
            price_per_contract = 1.0 - mid

        if kelly_f <= 0:
            return {**_blocked("negative_kelly"), "kelly_f": round(kelly_f, 4)}

        # Fractional Kelly dollar target
        raw_dollars = kelly_f * kelly_frac * bankroll

        # Apply per-trade cap
        cap_reason  = None
        sized       = raw_dollars
        if sized > trade_cap:
            sized      = trade_cap
            cap_reason = "per_trade"

        # Apply correlation cap (remaining headroom for this city×date group)
        day       = _iso_day(signal["settlement_date"])
        corr_key  = f"{signal.get('city', '?')}_{day}"
        used      = float(self._state["open_exposure"].get(corr_key, 0.0))
        remaining = corr_cap - used
        if remaining <= 0:
            return {**_blocked(f"corr_cap_full"), "kelly_f": round(kelly_f, 4),
                    "cap": "corr"}
        if sized > remaining:
            sized      = remaining
            cap_reason = "corr"

        # Apply cluster cap: aggregate exposure across all SAME-CLUSTER cities
        # for this settlement date (exposure keys stay per city×date; the
        # cluster view is derived here at read time, so record_fill callers
        # are unchanged).
        cluster_cap = float(self.cfg.get("corr_cluster_cap_pct", 0.05)) * bankroll
        cl = cluster_of(str(signal.get("city", "?")))
        cl_used = 0.0
        for k, v in self._state["open_exposure"].items():
            kcity, _, kday = k.rpartition("_")
            if kday == day and cluster_of(kcity) == cl:
                cl_used += float(v)
        cl_remaining = cluster_cap - cl_used
        if cl_remaining <= 0:
            return {**_blocked("corr_cluster_full"), "kelly_f": round(kelly_f, 4),
                    "cap": "corr_cluster"}
        if sized > cl_remaining:
            sized      = cl_remaining
            cap_reason = "corr_cluster"

        # Convert to whole contracts (floor — never exceed budget)
        if price_per_contract <= 0:
            return _blocked("zero_price")
        contracts = math.floor(sized / price_per_contract)
        if contracts < int(self.cfg["min_contracts"]):
            return {**_blocked("below_min_contracts"), "kelly_f": round(kelly_f, 4),
                    "cap": cap_reason}

        actual_dollars = contracts * price_per_contract
        return {
            "contracts": contracts,
            "dollars":   round(actual_dollars, 2),
            "kelly_f":   round(kelly_f, 4),
            "cap":       cap_reason,
            "blocked":   None,
        }

    # ── State updates ─────────────────────────────────────────────────────────

    def record_fill(
        self,
        city: str,
        settlement_date: str,
        contracts: int,
        dollars_at_risk: float,
    ) -> None:
        """
        Call after a fill is confirmed to update the correlation-cap exposure tracker.
        dollars_at_risk = contracts × price_per_contract (what you paid, not potential profit).
        """
        key = f"{city}_{_iso_day(settlement_date)}"
        self._state["open_exposure"][key] = (
            float(self._state["open_exposure"].get(key, 0.0)) + dollars_at_risk
        )
        self._persist()

    def record_pnl(self, pnl_dollars: float) -> None:
        """
        Call when a position settles.  Updates daily P&L and trips the circuit
        breaker if the daily loss threshold is exceeded.
        """
        self._state["daily_pnl"] = float(self._state["daily_pnl"]) + pnl_dollars
        breaker = float(self.cfg["circuit_breaker_pct"]) * float(self.cfg["bankroll"])
        if self._state["daily_pnl"] <= -breaker and not self._state["halted"]:
            self._state["halted"] = True
        self._persist()

    # ── Accessors ─────────────────────────────────────────────────────────────

    def is_halted(self) -> bool:
        return bool(self._state["halted"])

    def daily_pnl(self) -> float:
        return float(self._state["daily_pnl"])

    def open_exposure(self) -> dict[str, float]:
        return dict(self._state["open_exposure"])

    def status_line(self) -> str:
        bankroll = float(self.cfg["bankroll"])
        breaker  = float(self.cfg["circuit_breaker_pct"]) * bankroll
        pnl      = float(self._state["daily_pnl"])
        halted   = self._state["halted"]
        status   = "HALTED ⛔" if halted else "Active"
        exp      = self._state["open_exposure"]
        exp_str  = (
            "  open: " + ", ".join(f"{k} ${v:.2f}" for k, v in exp.items())
            if exp else ""
        )
        return (
            f"Risk: bankroll ${bankroll:.0f}  |  daily P&L ${pnl:+.2f}"
            f"  |  CB at -${breaker:.0f}  |  {status}{exp_str}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blocked(reason: str) -> dict[str, Any]:
    return {"contracts": 0, "dollars": 0.0, "kelly_f": 0.0, "cap": None,
            "blocked": reason}


# Sheet-format money: "$3.71", "($4.49)" (= negative), "-", "". Used by the
# trade-logging entry point to feed real fills/P&L into the risk state.
def parse_dollars(s: Any) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if s in ("", "-"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


# Market-string city names as they appear in the trade log
# (e.g. "Miami High Temp, Jun 15 — Between 89.5-91.5°F"). Longest match wins
# so "New Orleans" can't be shadowed by a shorter name.
_MARKET_CITY_NAMES = {
    "new york": "NYC", "nyc": "NYC", "chicago": "CHI", "miami": "MIA",
    "phoenix": "PHX", "boston": "BOS", "san francisco": "SFO", "los angeles": "LAX",
    "denver": "DEN", "atlanta": "ATL", "houston": "HOU", "washington": "DCA",
    "new orleans": "MSY", "philadelphia": "PHL", "seattle": "SEA",
    "las vegas": "LAS", "minneapolis": "MSP", "san antonio": "SAT",
    "dallas": "DAL", "austin": "AUS", "oklahoma city": "OKC",
}


def city_from_market(market: str) -> str | None:
    m = market.lower()
    for name in sorted(_MARKET_CITY_NAMES, key=len, reverse=True):
        if name in m:
            return _MARKET_CITY_NAMES[name]
    return None
