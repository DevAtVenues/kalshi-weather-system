# Kalshi Weather Trading System

A quantitative system for trading daily high- (and low-) temperature weather
contracts on [Kalshi](https://kalshi.com). The edge, where it exists, is a
**better-calibrated settlement probability than the market price** — measured
with confidence intervals, under both pessimistic and optimistic fill models,
and validated out-of-sample before any capital is committed.

> **📌 Authoritative overview:** For the current, detailed state of the system —
> architecture, live performance numbers, what's proven vs. unproven, and the
> forward roadmap — read **[`docs/STATE_OF_THE_MODEL.md`](docs/STATE_OF_THE_MODEL.md)**.
> That document is the single source of truth; this README is the short front door.
> Some older docs in `docs/` describe an earlier engine and carry a "SUPERSEDED"
> banner — defer to STATE_OF_THE_MODEL where they differ.

## Where this is headed (the trajectory)

**The end goal is a fully automated trading engine.** The system was deliberately
*piloted* as a forecast-and-signal engine and is now becoming that automated
trader. On real capital it is **notify-and-manual**: it computes and vets picks,
pushes them as phone notifications, logs advisory sizing under a hard risk layer,
and **a human places each real-money trade on Kalshi.** As of **2026-08-12** the
automation loop is closing: an execution layer places risk-sized, post-only maker
orders on **Kalshi's demo exchange** and reconciles fills and settlement P&L back
into the risk layer automatically — a pilot that runs until its forward record is
statistically decisive, so automated real capital is earned by evidence rather
than switched on. (`docs/STATE_OF_THE_MODEL.md` Part VI.)

## What it does today

1. **Prices contracts with a multi-model ensemble** — GEFS + ECMWF-ENS + ICON
   members from Open-Meteo — corrected for station grid-vs-sensor bias, dispersion,
   and center bias, then mixed with the official NWS forecast into a settlement
   probability.
2. **Fires a baseline signal** through an 8-gate stack (evidence-scaled edge
   bars, multi-model agreement, forecast stability, calibration ceilings).
3. **Vets every fire** through a second-stage externality check (climatology base
   rates, multi-model agreement, live re-verification, self-contradiction) that
   produces a TRADE_SMALL / WATCH / PASS verdict — **a fire is a baseline, not a
   trade.** Only vetted survivors, confirmed by an independent human check, are
   acted on.
4. **Logs proprietary data 24/7** — Kalshi orderbook depth (5-min) and Open-Meteo
   ensemble members (15-min) — neither of which the public APIs retain, building a
   dataset the free bots can't buy.
5. **Grades every settled pick** against NWS settlement truth and feeds the result
   back into weekly recalibration, bias corrections, and skill monitoring.
6. **Executes on demo** — fire-eligible picks become post-only maker limit orders
   on Kalshi's demo exchange, sized by the risk layer, with fills, settlements,
   and circuit-breaker halts reconciled automatically and every event audit-logged
   (opt-in; production is locked behind an explicit interlock).

## How edge is defined

Edge = a better-calibrated probability than the implied market price, by more than
fees + spread. Nothing else. Results are always reported with bootstrap confidence
intervals and under both pessimistic and optimistic fill models. Edge that survives
only the optimistic model is not real edge. **Current honest status:** the forecast
is genuinely skilled (~1.5°F MAE); a positive edge shows up in specific *vetted*
cities and structures, but is **not yet proven holistically** — see
`docs/STATE_OF_THE_MODEL.md` Part I/III for the numbers and caveats.

## Project structure

```
config/               Station config (lat/lon, NWS station, Kalshi series, LST offset)
data/                 Cached forecasts, labels, calibration maps, logged orderbook/ensembles
docs/                 STATE_OF_THE_MODEL.md (authoritative) + supporting/analysis docs
ops/                  Ops runbook + launchd job definitions
scripts/              Entry points — run_live, candidate_pipeline, daily_review, health_check, backtests
src/kalshi_weather/
  ingest/             Data fetchers: Open-Meteo forecasts, Kalshi API, NWS labels, METAR, NBM
  logger/             24/7 orderbook + ensemble-member capture (proprietary dataset)
  calibration/        Ensemble distribution, grid/center bias, dispersion, isotonic recalibration
  live/               Signal runner (pricing + gates), risk manager (Kelly, circuit breaker),
                      and the execution layer (signed exchange client + demo-pilot executor)
  monitor/            Intraday settled-outcome lock detection
  dashboard/          Read-only analytics view over picks/watchlist/outcomes
  settlement.py       Empirical Kalshi settlement conventions (fixture-tested)
  outcome_tracker.py  Grading + forward-validation feedback loop
  tz.py               UTC-only timestamp handling (the single timezone firewall)
tests/
  test_leakage.py     Fails the build if any feature timestamp >= decision time
  test_settlement_convention.py  Replays thousands of settled markets
```

## Data sources

| Source | Used for | Notes |
|--------|----------|-------|
| Open-Meteo ensemble + forecast APIs | Forecast features | Fetched by init time — no look-ahead; members logged live (retained ~3 days upstream) |
| NWS Daily Climate Report (CLI, via IEM) | Settlement labels | Exact station per contract rules; current unfinished day never cached |
| Kalshi API (OHLC + trades + live book) | Market prices | No historical orderbook depth — we log it ourselves |
| METAR / 5-min ASOS (IEM) | Intraday obs + lock detection | Whole-°C rounding margin applied |

ERA5 / Historical Weather API is **never** used as a feature — reanalysis leaks the future.

## Running it

```bash
# Generate signals for the current open markets
.venv/bin/python scripts/run_live.py

# Build + vet the candidate board
.venv/bin/python scripts/candidate_pipeline.py --scheduled

# Grade settled picks, refresh forward report + skill monitor
.venv/bin/python scripts/daily_review.py

# Backtest (deterministic lower bound; the ensemble edge is proven forward)
.venv/bin/python scripts/historical_backtest.py
```

Scheduled operation, health checks, and the change-safety gates are documented in
`ops/README.md`.

## Hard rules (never bypass)

1. Features, labels, and prices are kept in separate data sources — never crossed.
2. No look-ahead: a trade at time T uses only forecasts with `init_time < T`
   (machine-enforced by `tests/test_leakage.py`).
3. No survivorship: every settled contract is included, wins and losses.
4. Out-of-sample is sacred: final test set read exactly once with parameters frozen.
5. Fills are modeled under both pessimistic and optimistic assumptions.
6. Risk layer is mandatory before real capital: fractional Kelly ≤ 0.25×, 1–2%
   per-trade cap, daily circuit breaker.
7. All timestamps stored in UTC internally; timezone conversion isolated to `tz.py`.

Full operating principles are in [`docs/STATE_OF_THE_MODEL.md`](docs/STATE_OF_THE_MODEL.md).

## Stack

Python · pandas · event-driven backtest · Parquet caching · launchd scheduling ·
pinned deps (`pyproject.toml`).
