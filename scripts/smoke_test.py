"""
End-to-end smoke test — the commit gate. No change lands without this passing.

pytest proves the units; this proves the SYSTEM: it imports every scheduled entry
point, exercises the real pipelines read-only against current data, and pins the
exact contracts that have already broken silently once. Run by .githooks/pre-commit
and pre-push (git config core.hooksPath .githooks). ~seconds, no network, no writes
outside logs/. Exit 0 = safe to land; anything else = the change breaks a job.

Contracts pinned (each one is a past incident):
  • every scheduled module imports (refactor/rename fallout)
  • fitted warming curve loads and β(h) is sane (falsified-taper class)
  • latest signals run is broad and non-null (hollow-run class)
  • candidate board Stage 1+2 produce verdicts in the allowed set (pipeline class)
  • ntfy titles/headers are latin-1 encodable (dead feed-stale-alert class)
  • .env resolves NTFY_TOPIC via package import (dead-push launchd class)
  • label parquets contain no partial today-row (settlement-leakage class)
  • preflight passes for every scheduled job (broken-environment class)
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FAILURES: list[str] = []


def check(name: str):
    def deco(fn):
        def run():
            try:
                fn()
                print(f"  ✅ {name}")
            except Exception as exc:
                print(f"  🔴 {name}: {exc}")
                FAILURES.append(f"{name}: {exc}")
        return run
    return deco


@check("scheduled entry points import")
def _imports():
    for mod in ("kalshi_weather.live.runner", "kalshi_weather.live.intraday",
                "kalshi_weather.dashboard.notifications", "kalshi_weather.preflight",
                "kalshi_weather.scheduling", "kalshi_weather.outcome_tracker",
                "kalshi_weather.ingest.fastobs",
                "candidate_pipeline", "grade_candidates", "health_check",
                "fast_obs_watch"):
        importlib.import_module(mod)


@check("preflight passes for every scheduled job")
def _preflights():
    from kalshi_weather import preflight as pf
    for job in pf._REQUIREMENTS:
        # inline re-implementation of the checks, without the sys.exit
        for kind, target, why in pf._REQUIREMENTS[job]:
            import os
            if kind == "env":
                assert os.getenv(target), f"{job}: env {target} unset ({why})"
            elif kind == "file":
                assert (ROOT / target).is_file(), f"{job}: missing {target} ({why})"


@check("fitted warming curve loads and is sane")
def _curve():
    from kalshi_weather.live.intraday import warming_beta
    noon, three = warming_beta(12.0), warming_beta(15.0)
    assert 0.5 <= noon <= 1.0, f"beta(12)={noon} outside sane range (fit broken?)"
    assert three < 0.3, f"beta(15)={three} — end-of-day should be near 0"
    assert noon > warming_beta(13.5) > three, "beta(h) not decreasing through the afternoon"


@check("latest signals run is broad and non-null")
def _signals():
    rows = [json.loads(l) for l in open(ROOT / "data/signals/signals_log.jsonl")
            if l.strip()]
    latest = max(r["run_ts"] for r in rows)
    run = [r for r in rows if r["run_ts"] == latest]
    cities = {r.get("city") for r in run}
    # Listing-aware (mirrors health_check): mornings have ONE settlement day
    # until Kalshi lists tomorrow — judge the fullest listed day, not the total.
    by_day: dict = {}
    for r in run:
        by_day[str(r.get("settlement_date"))] = by_day.get(str(r.get("settlement_date")), 0) + 1
    fullest = max(by_day.values()) if by_day else 0
    assert len(cities) >= 15 and fullest >= 100, \
        f"latest run thin: {len(run)} rows / {len(cities)} cities / days {by_day}"
    for key in ("ticker", "direction", "strike_type", "settlement_date",
                "prob_estimate", "edge_raw", "is_same_day"):
        frac = sum(r.get(key) is not None for r in run) / len(run)
        assert frac > 0.95, f"signals field '{key}' null in {100 * (1 - frac):.0f}% of rows"


@check("candidate pipeline Stage 1+2 produce valid verdicts (read-only)")
def _pipeline():
    import candidate_pipeline as cp
    board = cp.build_board(15)
    assert len(board) > 0, "Stage 1 produced an empty board"
    v = cp.vet_candidate(board.iloc[0], use_api=False)
    assert v["verdict"] in {"TRADE_SMALL", "WATCH", "PASS"}, v["verdict"]
    for key in ("ticker", "direction", "strike_type", "floor", "cap",
                "is_same_day", "checks", "settlement_date"):
        assert key in v, f"verdict payload lost field '{key}' (grading depends on it)"


@check("ntfy titles are header-safe (latin-1)")
def _ntfy_titles():
    from kalshi_weather.dashboard import notifications as nf
    titles = [c["title"] for c in nf._NTFY_CFG.values()]
    titles += ["Weather feed stale", "System health: 3 RED",
               "System health recovered", "Preflight failed: run_live"]
    for t in titles:
        t.encode("latin-1")           # raises = the push dies silently in prod
    assert nf._ascii_title("🔴 x") == "x"


@check(".env resolves push topic via package import alone")
def _env():
    import os
    import kalshi_weather  # noqa: F401  (import triggers the .env load)
    assert os.getenv("NTFY_TOPIC"), "NTFY_TOPIC unresolved — scheduled pushes dead"


@check("prob-calibration artifact is coherent (lead-keyed maps valid)")
def _prob_map():
    # A malformed/missing lead map fails SILENTLY (identity fallback) — if the
    # builder recorded evidence that ensemble@day-ahead shipped, the key must
    # actually exist, interpolate, and be monotone non-decreasing.
    from kalshi_weather.calibration.recalibrate import load_calibration, apply_prob_calibration
    maps = load_calibration()
    if not maps:
        return                      # no artifact yet — identity everywhere is valid
    if (maps.get("meta") or {}).get("ensemble_day_ahead_evidence"):
        m = (maps.get("sources") or {}).get("ensemble@day-ahead")
        assert m and m.get("x") and m.get("y"), "evidence recorded but map key missing"
        ys = m["y"]
        assert all(b >= a for a, b in zip(ys, ys[1:])), "lead map not monotone"
        cal = apply_prob_calibration(0.30, "ensemble", maps, lead="day-ahead")
        assert cal is not None and 0.0 < cal < 1.0
        # same-day ensemble must remain IDENTITY (measured well-calibrated)
        assert apply_prob_calibration(0.30, "ensemble", maps, lead="same-day") == 0.30


@check("prob-shrinkage artifact is coherent (weights in [0,1] + evidence)")
def _shrinkage():
    # Log-only consumer, but a malformed artifact silently nulls prob_shrunk on
    # every row (identity-style failure) — assert shape when it exists.
    from kalshi_weather.calibration.shrinkage import load_shrinkage, shrink_prob
    maps = load_shrinkage()
    if not maps:
        return                          # no artifact yet — Nones everywhere is valid
    for key, rec in (maps.get("weights") or {}).items():
        assert "@" in key, f"weight key {key!r} not source@lead"
        assert rec.get("w") is not None and 0.0 <= rec["w"] <= 1.0, f"bad w for {key}"
        assert rec.get("evidence"), f"{key} shipped without LODO evidence"
        src, lead = key.split("@", 1)
        p = shrink_prob(0.30, 0.60, src, lead, maps)
        assert p is not None and 0.0 < p < 1.0


@check("execution layer is safe: opt-in only + prod interlock enforced")
def _execution_safety():
    # The two properties that make automated order placement safe to keep in
    # the tree: (1) it stays OFF unless KALSHI_EXEC=1 is set deliberately,
    # (2) production can never be reached without the explicit interlock.
    import os
    from kalshi_weather.live.exchange import ExchangeError, KalshiExchange
    from kalshi_weather.live.executor import maybe_executor
    if os.getenv("KALSHI_EXEC") != "1":
        assert maybe_executor() is None, "executor armed without KALSHI_EXEC=1"
    if os.getenv("KALSHI_EXEC_ALLOW_PROD") is None:
        try:
            KalshiExchange(env="prod")
        except ExchangeError:
            pass
        else:
            raise AssertionError("PROD exchange constructed without the interlock")


@check("price-path pilot is dormant: engine runs offline, no exchange reach")
def _pricepath_dormant():
    # The pilot may live in the tree ONLY while it is provably paper: the
    # engine must run a synthetic lifecycle with no network, and its sources
    # must never reference the order-capable modules.
    import pandas as pd
    from kalshi_weather.backtest import taker_fee
    from kalshi_weather.pricepath.engine import POLICIES, gated_entries, simulate
    ts = pd.date_range("2026-01-01 12:00", periods=4, freq="10min", tz="UTC")
    mids = pd.Series([0.30, 0.40, 0.50, 0.60])
    panel = pd.DataFrame({
        "ticker": "T-SMOKE", "snapshot_utc": ts,
        "yes_bid": mids - 0.01, "yes_ask": mids + 0.01, "yes_mid": mids,
        "spread": 0.02, "crossed": False, "has_model": True,
        "signal_age_min": 5.0, "prob_estimate": 0.60,
        "divergence": 0.60 - mids, "settlement_date": "2026-01-01",
        "round_trip_taker": 0.02 + (mids + 0.01).map(taker_fee)
                            + (mids - 0.01).map(taker_fee),
    })
    assert len(gated_entries(panel)) == 1
    trades = simulate(panel, POLICIES["target_model"])
    assert len(trades) == 1 and trades.iloc[0]["exit_reason"] == "target_model"
    src = ROOT / "src" / "kalshi_weather" / "pricepath"
    for f in list(src.glob("*.py")) + [ROOT / "scripts" / "price_path_shadow.py"]:
        text = f.read_text()
        for bad in ("live.exchange", "live.executor", "KalshiExchange",
                    "create_order", "import requests"):
            assert bad not in text, f"{f.name} references {bad!r} — pilot not dormant"


@check("label parquets contain no partial today-row")
def _labels_clean():
    import pandas as pd
    year = date.today().year
    for f in (ROOT / "data/raw/labels").glob(f"*/{year}.parquet"):
        mx = pd.to_datetime(pd.read_parquet(f, columns=["date"])["date"]).max().date()
        assert mx < date.today(), f"{f.parent.name} has an unfinished-day row ({mx})"


if __name__ == "__main__":
    print("SMOKE TEST (commit gate)")
    for obj in list(globals().values()):
        if callable(obj) and getattr(obj, "__name__", "") == "run":
            obj()
    if FAILURES:
        print(f"\n🔴 SMOKE FAILED — {len(FAILURES)} broken contract(s). Do NOT land this change.")
        sys.exit(1)
    print("\n✅ smoke clean — change is safe to land")
