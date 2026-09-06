#!/usr/bin/env python3
"""
Reconciliation ledger — what the account DID vs what the system SAID it would do.

This is the instrument the system never had. Every reported number in this repo
(grade_candidates, validate_forward, the dashboard, the funnel table) is struck
at the board mid with no spread and no fees. Real fills pay the ask and pay
Kalshi's fee. Nothing anywhere compared the two, so a persistent gap between
"what we scored" and "what we banked" was structurally invisible.

It answers one question per week:

    claimed P&L  −  realized P&L  =  entry slippage + fees + model error

and splits the gap into those three parts, because they have different fixes:
  * entry slippage  → the gate/sizing/grading price basis is wrong (audit F3/F30)
  * fees            → never modelled on the live path at all
  * model error     → the probability itself was wrong; the only honest residual

Sources
  fills / settlements   Kalshi portfolio API (read-only client below)
  claimed decision      data/signals/signals_log.jsonl — the row that stood when
                        the fill happened (latest run_ts at or before fill time)

Usage
  # live (needs KALSHI_* env, see below)
  python scripts/reconcile_pnl.py --env prod
  # offline replay of a saved pull, or for testing
  python scripts/reconcile_pnl.py --fills fills.json --settlements settlements.json
  # save a pull for later / for sharing
  python scripts/reconcile_pnl.py --env prod --dump-raw out/

Env (same keys live/exchange.py already uses):
  KALSHI_PROD_KEY_ID / KALSHI_PROD_PRIVATE_KEY_PATH     for --env prod
  KALSHI_DEMO_KEY_ID / KALSHI_DEMO_PRIVATE_KEY_PATH     for --env demo

NOTE ON THE PROD INTERLOCK: live/exchange.py requires KALSHI_EXEC_ALLOW_PROD to
reach production, because that class can PLACE ORDERS. The client below is
read-only by construction — it implements GET and nothing else — so it does not
require the interlock and cannot be made to trade. Do not add write methods here.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

SIGNALS = ROOT / "data" / "signals" / "signals_log.jsonl"
OUT_DIR = ROOT / "data" / "analysis"

# Kalshi fee schedule: fees = ceil(0.07 x C x P x (1-P)) in cents, taker side only.
# Same 0.07 coefficient backtest.py:32 uses. VERIFY against a real statement before
# trusting the absolute level — the decomposition's shape does not depend on it,
# but the fee column does.
FEE_COEF = 0.07


def fee_cents(price_cents: int, count: int, is_taker: bool) -> float:
    """Kalshi trading fee for one fill, in cents. Makers are not charged."""
    if not is_taker:
        return 0.0
    p = price_cents / 100.0
    return math.ceil(FEE_COEF * count * p * (1.0 - p) * 100.0)


# ─────────────────────────────────────────────────────────────── read-only client

class ReadOnlyKalshi:
    """Signed GETs against the Kalshi portfolio API. No write methods. Ever."""

    BASES = {"demo": "https://demo-api.kalshi.co/trade-api/v2",
             "prod": "https://api.elections.kalshi.com/trade-api/v2"}
    PREFIX = "/trade-api/v2"
    MIN_GAP_S = 0.35                      # the WAF lesson recorded in exchange.py

    def __init__(self, env: str = "prod") -> None:
        import requests
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        self._requests, self._hashes, self._padding = requests, hashes, padding
        if env not in self.BASES:
            raise SystemExit(f"unknown env {env!r} (demo|prod)")
        self.base = self.BASES[env]
        pfx = "KALSHI_DEMO" if env == "demo" else "KALSHI_PROD"
        self.key_id = os.getenv(f"{pfx}_KEY_ID") or ""
        key_path = os.getenv(f"{pfx}_PRIVATE_KEY_PATH") or ""
        if not self.key_id or not key_path:
            raise SystemExit(
                f"{pfx}_KEY_ID / {pfx}_PRIVATE_KEY_PATH unset. Create a read key at "
                f"{'demo.kalshi.co' if env == 'demo' else 'kalshi.com'} settings, "
                "put both in .env, then re-run.")
        pem = Path(key_path).expanduser()
        if not pem.is_file():
            raise SystemExit(f"private key file not found: {pem}")
        self._key = serialization.load_pem_private_key(pem.read_bytes(), password=None)
        self._last = 0.0

    def _headers(self, path: str) -> dict:
        ts = str(int(time.time() * 1000))
        msg = f"{ts}GET{self.PREFIX}{path}".encode()
        sig = self._key.sign(
            msg,
            self._padding.PSS(mgf=self._padding.MGF1(self._hashes.SHA256()),
                              salt_length=self._padding.PSS.DIGEST_LENGTH),
            self._hashes.SHA256())
        return {"KALSHI-ACCESS-KEY": self.key_id,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "Content-Type": "application/json"}

    def _get(self, path: str, params: dict) -> dict:
        gap = self.MIN_GAP_S - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        for attempt in range(5):
            self._last = time.monotonic()
            r = self._requests.get(self.base + path, params=params,
                                   headers=self._headers(path), timeout=20)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise SystemExit(f"GET {path} -> {r.status_code}: {r.text[:300]}")
            return r.json() if r.text else {}
        raise SystemExit(f"GET {path}: retries exhausted")

    def paged(self, path: str, key: str, params: dict | None = None) -> list[dict]:
        """Follow Kalshi's cursor to exhaustion. The orderbook logger's missing
        pagination (audit F36) is why this is spelled out rather than assumed."""
        out: list[dict] = []
        cursor = None
        while True:
            p = dict(params or {}, limit=200)
            if cursor:
                p["cursor"] = cursor
            page = self._get(path, p)
            rows = page.get(key, []) or []
            out.extend(rows)
            cursor = page.get("cursor")
            if not cursor or not rows:
                return out


# ─────────────────────────────────────────────────────────────────────── loading

def load_signals(path: Path) -> dict[str, list[dict]]:
    """ticker -> decision rows, ascending by run_ts."""
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    if not path.exists():
        return by_ticker
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("ticker"):
            by_ticker[r["ticker"]].append(r)
    for rows in by_ticker.values():
        rows.sort(key=lambda r: str(r.get("run_ts", "")))
    return by_ticker


def _ts(v) -> datetime | None:
    """Kalshi sends either epoch seconds or RFC3339. Accept both, always UTC."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def decision_for(rows: list[dict], when: datetime | None) -> dict | None:
    """The signal row that STOOD when the trade happened — latest run_ts at or
    before the fill. Not the latest row overall: grading against a later snapshot
    is how forward_report ends up scoring a decision nobody made (audit F34/F18)."""
    if not rows:
        return None
    if when is None:
        return rows[-1]
    prior = [r for r in rows if (t := _ts(r.get("run_ts"))) is not None and t <= when]
    return prior[-1] if prior else None


# ───────────────────────────────────────────────────────────────── the ledger

def build_positions(fills: list[dict], settlements: list[dict],
                    signals: dict[str, list[dict]]) -> list[dict]:
    """One row per (ticker, side): what we paid, what we got, what we'd claimed."""
    agg: dict[tuple, dict] = {}
    for f in fills:
        ticker = f.get("ticker")
        side = (f.get("side") or "").lower()          # 'yes' | 'no'
        if not ticker or side not in ("yes", "no"):
            continue
        count = int(f.get("count") or 0)
        if count <= 0:
            continue
        price = int(f.get("yes_price") if side == "yes" else f.get("no_price") or 0)
        taker = bool(f.get("is_taker", True))
        signed = -1 if (f.get("action") or "buy").lower() == "sell" else 1
        k = (ticker, side)
        a = agg.setdefault(k, {"ticker": ticker, "side": side, "count": 0,
                               "cost_c": 0.0, "fee_c": 0.0, "first_fill": None})
        a["count"] += signed * count
        a["cost_c"] += signed * price * count
        a["fee_c"] += fee_cents(price, count, taker)
        t = _ts(f.get("created_time") or f.get("created_ts"))
        if t and (a["first_fill"] is None or t < a["first_fill"]):
            a["first_fill"] = t

    revenue: dict[str, float] = defaultdict(float)
    settled_at: dict[str, datetime | None] = {}
    for s in settlements:
        tk = s.get("ticker")
        if not tk:
            continue
        revenue[tk] += float(s.get("revenue") or 0.0)         # cents
        settled_at[tk] = _ts(s.get("settled_time") or s.get("settled_ts"))

    rows: list[dict] = []
    for (ticker, side), a in sorted(agg.items()):
        if a["count"] == 0:
            continue
        avg_price_c = a["cost_c"] / a["count"]
        rev_c = revenue.get(ticker, 0.0)
        realized_c = rev_c - a["cost_c"] - a["fee_c"]

        sig = decision_for(signals.get(ticker, []), a["first_fill"])
        claimed_c = expected_c = scored_price_c = None
        if sig is not None and sig.get("market_mid") is not None:
            mid = float(sig["market_mid"])
            # The price the system scored itself at, on the side we actually took.
            scored_price_c = (mid if side == "yes" else 1.0 - mid) * 100.0
            # claimed = the repo's own grader (grade_candidates.py:110): the REAL
            # outcome, priced at the mid it scored. This is the number the docs quote.
            won = rev_c > 0
            claimed_c = ((100.0 - scored_price_c) if won else -scored_price_c) * a["count"]
            # expected = the model's OWN expected value at decision time. Without
            # this, claimed − realized collapses to execution cost by construction
            # and the model is never held to account for its probability.
            if sig.get("prob_estimate") is not None:
                p_yes = float(sig["prob_estimate"])
                p_side = p_yes if side == "yes" else 1.0 - p_yes
                expected_c = (p_side * 100.0 - scored_price_c) * a["count"]

        rows.append({
            "ticker": ticker, "side": side, "count": a["count"],
            "avg_fill_price_c": round(avg_price_c, 2),
            "scored_price_c": None if scored_price_c is None else round(scored_price_c, 2),
            "cost_c": round(a["cost_c"], 2), "fee_c": round(a["fee_c"], 2),
            "revenue_c": round(rev_c, 2),
            "realized_c": round(realized_c, 2),
            "claimed_c": None if claimed_c is None else round(claimed_c, 2),
            "expected_c": None if expected_c is None else round(expected_c, 2),
            "slippage_c": None if scored_price_c is None else
                          round((avg_price_c - scored_price_c) * a["count"], 2),
            "first_fill": a["first_fill"].isoformat() if a["first_fill"] else None,
            "settled_at": (settled_at.get(ticker).isoformat()
                           if settled_at.get(ticker) else None),
            "matched": sig is not None,
            "city": (sig or {}).get("city"),
            "settlement_date": (sig or {}).get("settlement_date"),
            "prob_estimate": (sig or {}).get("prob_estimate"),
            "edge_raw": (sig or {}).get("edge_raw"),
            "run_ts": (sig or {}).get("run_ts"),
        })
    return rows


def week_of(row: dict) -> str:
    """ISO week of settlement (falls back to fill time, then 'unknown')."""
    stamp = row.get("settled_at") or row.get("first_fill")
    if row.get("settlement_date"):
        try:
            d = datetime.fromisoformat(str(row["settlement_date"])[:10])
            return f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"
        except ValueError:
            pass
    t = _ts(stamp)
    if t is None:
        return "unknown"
    return f"{t.isocalendar().year}-W{t.isocalendar().week:02d}"


def summarise(rows: list[dict]) -> dict:
    weeks: dict[str, dict] = {}
    for r in rows:
        w = weeks.setdefault(week_of(r), {
            "n": 0, "contracts": 0, "realized_c": 0.0, "claimed_c": 0.0,
            "expected_c": 0.0, "slippage_c": 0.0, "fee_c": 0.0,
            "unmatched": 0, "no_expected": 0})
        w["n"] += 1
        w["contracts"] += r["count"]
        w["realized_c"] += r["realized_c"]
        w["fee_c"] += r["fee_c"]
        if r["claimed_c"] is None:
            w["unmatched"] += 1
        else:
            w["claimed_c"] += r["claimed_c"]
            w["slippage_c"] += r["slippage_c"]
        if r["expected_c"] is None:
            w["no_expected"] += 1
        else:
            w["expected_c"] += r["expected_c"]
    for w in weeks.values():
        # Exact identity:
        #   expected − realized  =  model_error + slippage + fees
        # where model_error = expected − claimed  (the probability was wrong) and
        #       slippage + fees = claimed − realized  (execution cost).
        # Splitting it this way matters: claimed − realized alone is execution by
        # construction, so a two-way split can never indict the model.
        w["model_error_c"] = w["expected_c"] - w["claimed_c"]
        w["execution_c"] = w["claimed_c"] - w["realized_c"]
        w["gap_c"] = w["expected_c"] - w["realized_c"]
    return dict(sorted(weeks.items()))


def render(rows: list[dict], weeks: dict) -> str:
    L = []
    L.append("=" * 78)
    L.append("  RECONCILIATION LEDGER — claimed vs realized")
    L.append("=" * 78)
    if not rows:
        L.append("\n  No filled positions found. Nothing to reconcile.")
        L.append("  (If you expected trades: check the env/date range, and that")
        L.append("   fills are on the same account as the key you signed with.)")
        return "\n".join(L)

    matched = sum(1 for r in rows if r["matched"])
    L.append(f"\n  {len(rows)} filled positions · {matched} matched to a logged signal · "
             f"{len(rows) - matched} unmatched")
    if len(rows) - matched:
        L.append("  Unmatched = traded without a corresponding signals_log row "
                 "(manual trade, or the log was not running).")

    L.append("")
    L.append(f"  {'week':<10} {'n':>3} {'ctr':>5} {'expect$':>9} {'claim$':>9} "
             f"{'real$':>9} | {'model$':>9} {'slip$':>8} {'fees$':>7} {'GAP$':>9}")
    L.append(f"  {'-'*10} {'-'*3} {'-'*5} {'-'*9} {'-'*9} {'-'*9} | "
             f"{'-'*9} {'-'*8} {'-'*7} {'-'*9}")
    tot = defaultdict(float)
    for wk, w in weeks.items():
        L.append(f"  {wk:<10} {w['n']:>3} {w['contracts']:>5} "
                 f"{w['expected_c']/100:>+9.2f} {w['claimed_c']/100:>+9.2f} "
                 f"{w['realized_c']/100:>+9.2f} | {w['model_error_c']/100:>+9.2f} "
                 f"{w['slippage_c']/100:>+8.2f} {w['fee_c']/100:>7.2f} "
                 f"{w['gap_c']/100:>+9.2f}")
        for k in ("n", "contracts", "expected_c", "claimed_c", "realized_c",
                  "gap_c", "slippage_c", "fee_c", "model_error_c", "execution_c"):
            tot[k] += w[k]
    L.append(f"  {'-'*10} {'-'*3} {'-'*5} {'-'*9} {'-'*9} {'-'*9} | "
             f"{'-'*9} {'-'*8} {'-'*7} {'-'*9}")
    L.append(f"  {'TOTAL':<10} {int(tot['n']):>3} {int(tot['contracts']):>5} "
             f"{tot['expected_c']/100:>+9.2f} {tot['claimed_c']/100:>+9.2f} "
             f"{tot['realized_c']/100:>+9.2f} | {tot['model_error_c']/100:>+9.2f} "
             f"{tot['slippage_c']/100:>+8.2f} {tot['fee_c']/100:>7.2f} "
             f"{tot['gap_c']/100:>+9.2f}")

    L.append("")
    L.append("  expect$  what the model said it would make: its own probability at the")
    L.append("           price it scored — EV at decision time")
    L.append("  claim$   what the repo's grader reports: the REAL outcome, still priced")
    L.append("           at the scored mid with no spread or fees. Every performance")
    L.append("           number in docs/STATE_OF_THE_MODEL.md is this column")
    L.append("  real$    settlement revenue − what the fills actually cost − fees")
    L.append("")
    L.append("  GAP$ = expect$ − real$, split into three things with different fixes:")
    L.append("     model$  expect − claim: the probability was wrong")
    L.append("     slip$   what crossing the spread cost vs the scored mid")
    L.append("     fees$   Kalshi fees — never modelled anywhere on the live path")

    if tot["contracts"]:
        c = tot["contracts"]
        L.append("")
        L.append(f"  Per contract: expected {tot['expected_c']/c:+.2f}c · "
                 f"claimed {tot['claimed_c']/c:+.2f}c · realized {tot['realized_c']/c:+.2f}c")
        L.append(f"                model err {tot['model_error_c']/c:+.2f}c · "
                 f"slippage {tot['slippage_c']/c:+.2f}c · fees {tot['fee_c']/c:.2f}c")
        if tot["gap_c"]:
            share = lambda v: 100.0 * v / tot["gap_c"]
            L.append(f"  Gap attribution: model {share(tot['model_error_c']):.0f}% · "
                     f"slippage {share(tot['slippage_c']):.0f}% · "
                     f"fees {share(tot['fee_c']):.0f}%")
    if any(w["no_expected"] for w in weeks.values()):
        L.append("")
        L.append("  NOTE: some positions had no prob_estimate on the matched signal, so")
        L.append("  they contribute to claim$/real$ but not expect$ — model$ is a")
        L.append("  lower bound on those weeks.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=("demo", "prod"), default="prod",
                    help="which Kalshi account to read (default prod)")
    ap.add_argument("--fills", type=Path, help="offline: JSON file of fills")
    ap.add_argument("--settlements", type=Path, help="offline: JSON file of settlements")
    ap.add_argument("--signals", type=Path, default=SIGNALS)
    ap.add_argument("--since-days", type=int, default=180)
    ap.add_argument("--dump-raw", type=Path, help="save the raw API pull here")
    ap.add_argument("--json-out", type=Path,
                    default=OUT_DIR / "reconciliation.json")
    args = ap.parse_args()

    if bool(args.fills) != bool(args.settlements):
        ap.error("--fills and --settlements must be given together")

    if args.fills:
        fills = json.loads(args.fills.read_text())
        settlements = json.loads(args.settlements.read_text())
        fills = fills.get("fills", fills) if isinstance(fills, dict) else fills
        settlements = (settlements.get("settlements", settlements)
                       if isinstance(settlements, dict) else settlements)
    else:
        client = ReadOnlyKalshi(args.env)
        min_ts = int((datetime.now(timezone.utc)
                      - timedelta(days=args.since_days)).timestamp())
        print(f"  pulling fills + settlements from {args.env} "
              f"(last {args.since_days} days) …", file=sys.stderr)
        fills = client.paged("/portfolio/fills", "fills", {"min_ts": min_ts})
        settlements = client.paged("/portfolio/settlements", "settlements",
                                   {"min_ts": min_ts})
        print(f"  {len(fills)} fills · {len(settlements)} settlements", file=sys.stderr)
        if args.dump_raw:
            args.dump_raw.mkdir(parents=True, exist_ok=True)
            (args.dump_raw / "fills.json").write_text(json.dumps(fills, indent=1))
            (args.dump_raw / "settlements.json").write_text(json.dumps(settlements, indent=1))
            print(f"  raw pull saved to {args.dump_raw}", file=sys.stderr)

    signals = load_signals(args.signals)
    if not signals:
        print(f"  WARNING: no signals at {args.signals} — every position will be "
              "unmatched and only realized P&L is meaningful.", file=sys.stderr)

    rows = build_positions(fills, settlements, signals)
    weeks = summarise(rows)
    print(render(rows, weeks))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(),
         "fee_coef": FEE_COEF, "weeks": weeks, "positions": rows}, indent=1))
    print(f"\n  → {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
