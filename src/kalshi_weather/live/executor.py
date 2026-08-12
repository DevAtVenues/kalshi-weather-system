"""
Demo/live order executor — build-order step 5 (maker limit + auto-exit on demo).

Turns a FIRED pick from run_live's gate chain into a post-only maker limit
order, sized by the risk layer, and reconciles fills/settlements back into
risk state + the position watch. Exits are HOLD TO SETTLEMENT (the timing
study proved stops destroy the edge); the only "auto-exit" is order-level:
every order carries a server-side expiration_ts so an unfilled entry dies
quietly instead of resting stale.

Wiring (scripts/run_live.py):
    executor = maybe_executor()          # None unless KALSHI_EXEC=1 in .env
    executor.reconcile()                 # fills -> risk+watchlist, settles -> pnl
    push_forecast_picks(..., executor)   # FIRED picks -> execute_pick()

Isolation: in demo, the executor uses its OWN RiskManager backed by
data/execution/demo_risk_{config,state}.json (seeded from the real config on
first run), so play-money fills never consume the real correlation-cap
headroom or trip the real circuit breaker. In prod it would share the real
risk state — prod additionally requires the interlock in exchange.py.

State: data/execution/executor_state.json   (orders in flight, open positions)
Log:   data/execution/execution_log.jsonl   (every place/skip/fill/settle/cancel)
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from kalshi_weather.tz import UTC
from kalshi_weather.live.exchange import ExchangeError, KalshiExchange
from kalshi_weather.live.risk import RiskManager, CONFIG_PATH as _REAL_RISK_CONFIG

_ROOT = Path(__file__).parents[3]
EXEC_DIR   = _ROOT / "data" / "execution"
STATE_PATH = EXEC_DIR / "executor_state.json"
LOG_PATH   = EXEC_DIR / "execution_log.jsonl"
WATCHLIST  = _ROOT / "data" / "analysis" / "watchlist.json"
DEMO_RISK_CONFIG = EXEC_DIR / "demo_risk_config.json"
DEMO_RISK_STATE  = EXEC_DIR / "demo_risk_state.json"

ORDER_TTL_MIN = float(os.getenv("EXEC_ORDER_TTL_MIN", "120"))


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _fresh_state() -> dict[str, Any]:
    return {"orders": {}, "positions": {}, "last_fill_ts": 0, "seen_trades": []}


def maybe_executor() -> "Executor | None":
    """Build the executor iff execution is armed (KALSHI_EXEC=1 in .env).
    Misconfiguration is loud (health push) but never kills the signal run."""
    if os.getenv("KALSHI_EXEC") != "1":
        return None
    try:
        exchange = KalshiExchange()
    except ExchangeError as exc:
        print(f"  ⚠️  KALSHI_EXEC=1 but exchange unavailable: {exc}")
        try:
            from kalshi_weather.dashboard.notifications import notify_health
            notify_health("Executor misconfigured", str(exc))
        except Exception:
            pass
        return None
    if exchange.env == "demo":
        # Seed the demo risk config from the real one so demo sizing is
        # representative of real bankroll/caps, then keep them decoupled.
        if not DEMO_RISK_CONFIG.exists() and _REAL_RISK_CONFIG.exists():
            EXEC_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(_REAL_RISK_CONFIG, DEMO_RISK_CONFIG)
        risk = RiskManager(config_path=DEMO_RISK_CONFIG, state_path=DEMO_RISK_STATE)
    else:
        risk = RiskManager()          # prod shares the real risk state
    risk.reset_if_new_day()
    return Executor(exchange, risk)


class Executor:
    def __init__(self, exchange: KalshiExchange, risk: RiskManager,
                 state_path: Path = STATE_PATH, log_path: Path = LOG_PATH,
                 watchlist_path: Path = WATCHLIST) -> None:
        self.exchange = exchange
        self.risk = risk
        self._state_path = state_path
        self._log_path = log_path
        self._watchlist_path = watchlist_path
        self.state = self._load_state()

    # ── state / log plumbing ─────────────────────────────────────────────────

    def _load_state(self) -> dict[str, Any]:
        try:
            s = json.loads(self._state_path.read_text())
            return {**_fresh_state(), **s}
        except (OSError, json.JSONDecodeError):
            return _fresh_state()

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self.state, indent=1))

    def _log(self, event: str, **fields: Any) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": _now_iso(), "env": self.exchange.env, "event": event, **fields}
        with open(self._log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _push(self, title: str, body: str) -> None:
        """Fill/settle notices ride the health channel: they must reach the
        phone even under a risk halt (you need to KNOW what you hold)."""
        try:
            from kalshi_weather.dashboard.notifications import notify_health
            notify_health(title, body, priority="high", tags="moneybag")
        except Exception:
            pass

    def _held(self, ticker: str, direction: str) -> bool:
        if any(o["ticker"] == ticker and o["direction"] == direction
               for o in self.state["orders"].values()):
            return True
        pos = self.state["positions"].get(ticker)
        return bool(pos and pos["direction"] == direction)

    # ── entry ────────────────────────────────────────────────────────────────

    def execute_pick(self, row: Any) -> dict | None:
        """Place a post-only maker limit for one FIRED pick. Joins the current
        best bid of the side we buy (never crosses), sized by the risk layer at
        the mid (maker entry always costs <= that budget)."""
        ticker, direction = str(row["ticker"]), str(row["direction"])
        if self._held(ticker, direction):
            self._log("skip_duplicate", ticker=ticker, direction=direction)
            return None
        if self.risk.is_halted():
            self._log("skip_halted", ticker=ticker, direction=direction)
            return None

        sizing = self.risk.size_trade(row)
        if sizing["blocked"] or sizing["contracts"] < 1:
            self._log("skip_sizing", ticker=ticker, direction=direction,
                      blocked=sizing["blocked"], cap=sizing.get("cap"))
            return None

        yes_bid_c = int(round(float(row["yes_bid_dollars"]) * 100))
        yes_ask_c = int(round(float(row["yes_ask_dollars"]) * 100))
        if direction == "BUY_YES":
            side, price_c = "yes", yes_bid_c
        else:
            side, price_c = "no", 100 - yes_ask_c     # best resting NO bid
        if not (1 <= price_c <= 99):
            self._log("skip_price", ticker=ticker, direction=direction,
                      side=side, price_cents=price_c)
            return None

        expire = int(time.time() + ORDER_TTL_MIN * 60)
        try:
            order = self.exchange.create_order(
                ticker, side, sizing["contracts"], price_c, expiration_ts=expire)
        except ExchangeError as exc:
            self._log("order_rejected", ticker=ticker, direction=direction,
                      side=side, count=sizing["contracts"], price_cents=price_c,
                      error=str(exc))
            print(f"  ⚠️  order rejected {ticker} {direction}: {exc}")
            return None

        oid = order.get("order_id") or order.get("client_order_id") or ""
        # Everything reconcile() needs later — risk keys + watchlist strikes —
        # is captured NOW from the signal row, so a fill can be processed even
        # if the pick never recurs.
        meta = {
            "ticker": ticker, "direction": direction, "side": side,
            "count": int(sizing["contracts"]), "price_cents": price_c,
            "city": row.get("city"), "settlement_date": str(row["settlement_date"]),
            "strike_type": row.get("strike_type"),
            "threshold_f": _f(row.get("threshold_f")),
            "floor_strike": _f(row.get("floor_strike")),
            "cap_strike": _f(row.get("cap_strike")),
            "placed_ts": _now_iso(), "expire_ts": expire, "filled": 0,
            "prob_estimate": _f(row.get("prob_estimate")),
            "market_mid": _f(row.get("market_mid")),
            "edge_raw": _f(row.get("edge_raw")),
        }
        self.state["orders"][oid] = meta
        self._save_state()
        self._log("order_placed", order_id=oid, **{k: meta[k] for k in (
            "ticker", "direction", "side", "count", "price_cents", "city",
            "settlement_date", "prob_estimate", "market_mid", "edge_raw")},
            dollars=sizing["dollars"], kelly_f=sizing["kelly_f"], ttl_min=ORDER_TTL_MIN)
        print(f"  🛒 [{self.exchange.env}] resting {direction} {ticker}: "
              f"{sizing['contracts']} @ {price_c}c (TTL {ORDER_TTL_MIN:.0f}m)")
        return {"order_id": oid, **meta}

    # ── reconcile: fills -> positions/risk/watchlist, settles -> pnl ────────

    def reconcile(self) -> None:
        try:
            self._reconcile_fills()
        except ExchangeError as exc:
            print(f"  ⚠️  reconcile fills failed: {exc}")
        try:
            self._reconcile_order_status()
        except ExchangeError as exc:
            print(f"  ⚠️  reconcile orders failed: {exc}")
        try:
            self._reconcile_settlements()
        except ExchangeError as exc:
            print(f"  ⚠️  reconcile settlements failed: {exc}")
        if self.risk.is_halted() and self.state["orders"]:
            self._cancel_all("risk_halt")
        self._save_state()

    def _reconcile_fills(self) -> None:
        min_ts = max(0, int(self.state.get("last_fill_ts", 0)) - 60)
        fills = self.exchange.get_fills(min_ts=min_ts or None)
        seen = set(self.state.get("seen_trades", []))
        for f in fills:
            tid = f.get("trade_id") or f.get("fill_id") or ""
            oid = f.get("order_id") or ""
            if tid in seen or oid not in self.state["orders"]:
                continue
            meta = self.state["orders"][oid]
            count = int(f.get("count", 0))
            side = f.get("side", meta["side"])
            price_c = int(f.get("yes_price" if side == "yes" else "no_price",
                                meta["price_cents"]))
            cost = count * price_c / 100.0
            seen.add(tid)

            pos = self.state["positions"].setdefault(meta["ticker"], {
                "direction": meta["direction"], "side": meta["side"],
                "contracts": 0, "cost_dollars": 0.0, "city": meta["city"],
                "settlement_date": meta["settlement_date"],
            })
            pos["contracts"] += count
            pos["cost_dollars"] = round(pos["cost_dollars"] + cost, 4)
            meta["filled"] = int(meta.get("filled", 0)) + count

            self.risk.record_fill(meta["city"] or "UNKNOWN",
                                  meta["settlement_date"], count, cost)
            self._arm_watchlist(meta, price_c)
            self._log("fill", order_id=oid, trade_id=tid, ticker=meta["ticker"],
                      direction=meta["direction"], count=count,
                      price_cents=price_c, cost_dollars=round(cost, 4))
            self._push(f"{self.exchange.env.upper()} FILL: {meta['city'] or '?'}",
                       f"{meta['direction']} {meta['ticker']} x{count} @ {price_c}c "
                       f"(${cost:.2f})")
            created = f.get("created_time")
            if created:
                try:
                    ts = datetime.fromisoformat(
                        str(created).replace("Z", "+00:00")).timestamp()
                    self.state["last_fill_ts"] = max(
                        int(self.state.get("last_fill_ts", 0)), int(ts))
                except ValueError:
                    pass
            if meta["filled"] >= meta["count"]:
                del self.state["orders"][oid]
        self.state["seen_trades"] = sorted(seen)[-500:]

    def _reconcile_order_status(self) -> None:
        """Drop orders the exchange no longer holds (expired / canceled /
        executed-and-fills-recorded). Runs AFTER fills so nothing is lost."""
        for oid in list(self.state["orders"]):
            status = (self.exchange.get_order(oid).get("status") or "").lower()
            if status in ("canceled", "cancelled", "expired"):
                meta = self.state["orders"].pop(oid)
                self._log("order_gone", order_id=oid, status=status,
                          ticker=meta["ticker"], direction=meta["direction"],
                          filled=meta.get("filled", 0))
            elif status == "executed" and not self.state["orders"][oid].get("filled"):
                # Executed but its fill didn't appear yet — leave for next cycle.
                pass
            elif status == "executed":
                del self.state["orders"][oid]

    def _reconcile_settlements(self) -> None:
        if not self.state["positions"]:
            return
        setts = {s.get("ticker"): s for s in self.exchange.get_settlements()}
        for ticker in list(self.state["positions"]):
            s = setts.get(ticker)
            if not s:
                continue
            result = str(s.get("market_result", "")).lower()
            if result not in ("yes", "no"):
                continue        # voided/unknown — leave for manual review
            pos = self.state["positions"].pop(ticker)
            # P&L computed from OUR records (side + cost), never from the API's
            # revenue field — immune to its cents/centi-cents unit ambiguity.
            payout = float(pos["contracts"]) if result == pos["side"] else 0.0
            pnl = round(payout - pos["cost_dollars"], 4)
            self.risk.record_pnl(pnl)
            self._log("settlement", ticker=ticker, direction=pos["direction"],
                      contracts=pos["contracts"], result=result,
                      cost_dollars=pos["cost_dollars"], pnl_dollars=pnl)
            self._push(f"{self.exchange.env.upper()} SETTLE: {pos['city'] or '?'}",
                       f"{ticker} -> {result.upper()}  P&L ${pnl:+.2f}"
                       + ("  ⛔ breaker tripped" if self.risk.is_halted() else ""))

    def _cancel_all(self, reason: str) -> None:
        for oid in list(self.state["orders"]):
            meta = self.state["orders"][oid]
            try:
                self.exchange.cancel_order(oid)
            except ExchangeError as exc:
                print(f"  ⚠️  cancel {oid} failed: {exc}")
                continue
            del self.state["orders"][oid]
            self._log("order_canceled", order_id=oid, reason=reason,
                      ticker=meta["ticker"], direction=meta["direction"])

    # ── position watch arming (mirrors scripts/log_to_sheets.py shapes) ─────

    def _arm_watchlist(self, meta: dict, fill_price_c: int) -> None:
        from kalshi_weather.live.runner import CITY_CONFIGS   # lazy: heavy module
        cfg = CITY_CONFIGS.get(meta.get("city") or "")
        if cfg is None:
            print(f"  ⚠️  cannot arm position watch for {meta['ticker']} "
                  f"(unknown city {meta.get('city')!r}) — add watchlist entry by hand")
            return
        entry: dict[str, Any] = {
            "ticker": meta["ticker"], "direction": meta["direction"],
            "station": cfg["nws_station"],
            "settlement_date": meta["settlement_date"],
            "lst_offset_h": cfg.get("lst_offset", -5),
            "note": f"auto from {self.exchange.env} fill @{fill_price_c}c",
        }
        st = meta.get("strike_type")
        if st == "between" and meta.get("floor_strike") is not None:
            entry.update(strike_type="between",
                         floor=int(round(meta["floor_strike"])),
                         cap=int(round(meta["cap_strike"])))
        elif meta.get("threshold_f") is not None:
            thr = int(round(meta["threshold_f"]))
            entry.update(strike_type=("less" if st == "less" else "greater"),
                         threshold=thr, floor=None, cap=thr)
        else:
            print(f"  ⚠️  no strike info for {meta['ticker']} — watchlist not armed")
            return
        try:
            entries = json.loads(self._watchlist_path.read_text())
        except (OSError, json.JSONDecodeError):
            entries = []
        if any(e.get("ticker") == entry["ticker"]
               and e.get("direction") == entry["direction"] for e in entries):
            return
        entries.append(entry)
        self._watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        self._watchlist_path.write_text(json.dumps(entries, indent=1))
        print(f"  👁  position watch armed for {entry['ticker']} {entry['direction']}")


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v      # NaN -> None (json-safe)
