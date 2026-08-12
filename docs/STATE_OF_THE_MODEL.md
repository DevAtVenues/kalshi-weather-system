# State of the Model — Kalshi Weather Trading System

*Prepared as a technical briefing for a reader who has not written the code.*
*Snapshot: 2026-08-12 · branch `feat/ensemble-live-pricing` · HEAD `a221512`.*

This document has three jobs:

1. **Part I–IV — describe the system as it actually is today**, in technical
   detail, including where it is strong, where it is unproven, and where it is
   deliberately human-in-the-loop. Numbers are cited to the files that produce
   them so they can be re-derived.
2. **Part V — the price-path (mispricing) direction**: the plan as scoped, and —
   as of 2026-08-12 — the **executed verdict** of its designed GO/NO-GO gate
   (see the Verdict section at the end of Part V).
3. **Part VI — the execution layer**: the order-placement and demo-pilot
   machinery, built 2026-08-12, that closes the automation loop for the
   settlement book.

A note on tone, because it is the whole thesis: **this project's credibility is
its honesty.** The differentiator versus the free weather bots online is not a
secret signal — it is validation discipline, a real risk layer, and proprietary
data we log ourselves. So this document leads with what we *cannot* yet claim as
loudly as with what we can.

---

## Part I — Orientation

### What the system is

Each Kalshi contract is a yes/no bet on a US city's **settled daily high (or
low) temperature** — e.g. "NYC high ≥ 92°F" (a *threshold* / T-type) or "high
lands in 86–87°F" (a *bracket* / B-type). The market price of YES is effectively
the market's probability. Our job is to produce a **better-calibrated
probability** than that price and trade the gap when it clears fees + spread.

Critically — and this frames the whole trajectory: **the end goal is a fully
automated trading engine.** The system was deliberately *piloted* as a
forecast-and-signal engine, and is now in the process of becoming that automated
trader. On real capital it is still **notify-and-manual**: the pipeline computes
picks, pushes them as phone notifications (`ntfy`), logs an advisory position
size, and **a human places every real-money trade on Kalshi.** But as of
**2026-08-12 the order-placement code exists**: a signed execution layer
(Part VI) turns fire-eligible picks into risk-sized maker orders on Kalshi's
**demo exchange**, reconciling fills and settlement P&L back into the risk
layer automatically. The manual stage was a validation-phase choice, not the
destination — the risk scaffolding proven over the manual phase now governs the
demo pilot, and will govern automated real capital only when the pilot's
forward record earns it.

### The most important thing to understand first: a fire is not a trade

The single most common way to misread this system is to score *everything the
model fires* and conclude "the market beats it." That is true of the **raw
baseline** — and it is supposed to be. The model's fire is only step one. Every
fire is then put through an **externality / deep-dive check** — an automated
Stage-2 vet against sources the pricing engine never saw (climatology base
rates, multi-model agreement, self-contradiction, a live re-verify) **plus a
human independent verification** — and **we act only on the small subset that
survives.** The acted-upon subset performs materially better than the firehose.

The evidence for that, straight from `scripts/grade_candidates.py` (mid-fill,
see caveat) — win-rate climbs monotonically up the vetting funnel:

| Stage of the funnel | What it is | n | Win-rate |
|---|---|---|---|
| Raw / unvetted fires | the model's baseline output | 1,496 | 44% |
| WATCH (passed vet, conditional) | eligible, flagged | 64 | 55% |
| **TRADE_SMALL (vet says trade)** | **what we'd act on** | **5** | **80%** |
| PASS (vet rejected) | correctly discarded | 249 | 18% |

The vet lifts win-rate from ~44% raw to 80% on what it clears, and dumps losers
into PASS at 18%. **This is the actual product** — the deep-dive filter, not the
raw fire. (Caveat, stated plainly: TRADE_SMALL is only n=5 and all returns here
are struck at mid with no fees/spread — a directionally strong signal, not yet a
statistically settled one. More on this in Part III and Part II §5b.)

### The honest scorecard

*Figures refreshed 2026-08-11 from `model_skill_report.py`,
`validate_forward.py`, `grade_candidates.py`, `skill_monitor.py` (59-day window,
6,846 contracts).*

- **Forecast skill is real and improving.** Daily-high **MAE 1.52°F**, 95% CI
  [1.42, 1.63], over 1,144 city-days — better than the ~1.8°F of the prior
  window. There is now a small **cool bias of +0.46°F** [0.28, 0.63] (we forecast
  a touch low), concentrated in the known coastal settlement stations —
  **NYC −2.3°F, SFO −1.0°F (MAE 2.5), SEA +1.5°F** — which is edge source #3
  (settlement-sensor mechanics) in our own charter, a diagnosed and scoped fix,
  not a mystery. Interior cities are already at ~1°F.
- **The edge is real in specific pockets, not yet holistically.** Where enough
  settled history exists to check — specific cities and the `B|BUY_NO`
  short-bracket structure — the *vetted, acted-upon* picks show a positive,
  repeatable signal (funnel above; per-combo, e.g. SFO +0.173 / 68% win,
  DCA +0.169 / 65%, mid-fill). Pooled across *every* raw fire, the market prices
  better — but that pool is the unfiltered baseline, not the book we trade. The
  edge is **local, not global**, which is exactly why we trade so selectively.
- **The current RED monitor is a sample-size artifact on the baseline, not a
  broken model.** The live skill monitor (`skill_monitor.py`) is **RED**: over
  the trailing **14-settlement-day** window the ensemble **day-ahead** bucket
  scored Brier 0.0162 worse than the market (90% CI (−0.0204, −0.0125)). Two
  things to hold together: (a) this scores the *raw model output*, not the vetted
  subset; and (b) 14 autocorrelated weather days is a short, immature window on a
  model that has not had the calendar time to develop. It is the reason we stay
  conservative — not evidence the approach is wrong — and the expectation is that
  it improves as forward data accrues.

**In summary:** the model produces a baseline signal that, raw, prices about as
well as or a touch behind the market — as expected. What is actually traded is
the small subset that clears an automated externality vet and a human deep-dive,
and that subset hits at a materially higher rate. The edge is demonstrated in the
specific cities and structures with enough accumulated settled data; it is not
holistic yet, and the gaps are a time-and-data problem, not a broken model. The
system is built to widen what it acts on only as the evidence earns it — never to
treat the raw firehose as the finished product.

---

## Part II — Current architecture

### 1. Data architecture — three sources, never crossed

The cardinal rule (`CLAUDE.md` HARD RULES) is that **features, labels, and
prices are separate sources** and never leak into each other. Each has its own
ingestion path and cache.

| Source | Meaning | Ingestion | Cache |
|---|---|---|---|
| **FEATURES** | The forecast *as it existed at decision time* (no look-ahead) | `ingest/forecasts.py` (GFS/ECMWF point tmax); `logger/ensemble.py` (live ensemble **members**) | `data/raw/forecasts/…`; `data/logger/ensemble/{model}/{YYYYMMDD_HH}Z/{station}.parquet` |
| **LABELS** | The exact value Kalshi settles on | `ingest/labels.py` — NWS **Daily Climate Report (CLI)** via Iowa Environmental Mesonet | `data/raw/labels/{station}/{year}.parquet` (21 stations) |
| **PRICES** | Historical Kalshi book/quotes | `logger/orderbook.py` — live depth snapshots | `data/logger/orderbook/{date}.parquet` |

Two design consequences worth flagging to a technical reader:

- **We log our own data because the public APIs don't retain it.** Open-Meteo
  keeps individual ensemble *members* only ~3 days, and Kalshi's public API never
  exposes historical orderbook *depth*. The live logger
  (`logger/runner.py`, a 24/7 process) captures the orderbook every **5 min** and
  ensemble members every **15 min**, building a proprietary dataset from day one.
  This is the raw material the free bots cannot buy — and, as Part V explains,
  it is exactly the substrate a price-path strategy needs.
- **Leakage is machine-enforced.** The label feed never caches the current
  unfinished day (`labels.py`, `df["date"] < today`), so a partial "high so far"
  can't be mistaken for a settlement. A test suite (`tests/test_leakage.py`)
  *fails the build* if any feature timestamp is `>=` its decision time, and a
  static check fails the build if any module except `tz.py` imports a timezone
  library (timezone handling is the top leakage risk and is quarantined to one
  file).

### 2. The modeling core — from ensemble to probability

Live pricing runs through `live/runner.py` and `calibration/ensemble_dist.py`.
The pipeline for one contract:

1. **Fetch open markets** (`fetch_open_markets`) — parse T-type
   (`greater`/`less` threshold) and B-type (`between` bracket) contracts with
   their strikes and bid/ask/mid.
2. **Build the ensemble distribution** (`build_ensemble_context`) over the
   daily high from three model families — **GEFS + ECMWF-ENS + ICON**, pulled
   live from Open-Meteo's ensemble API. A **family-floor guard** refuses the
   whole fetch if any family drops below a member floor (GEFS 25 / ECMWF 40 /
   ICON 30). This guard exists because ECMWF silently returned *zero* members for
   weeks in July 2026 while the total count still looked plausible — a real
   incident that produced fake edge.
3. **Correct the members** on the *ensemble/analysis basis* (not the
   deterministic-forecast basis, which would double-count):
   - **Grid-vs-sensor bias** (`recent_grid_bias`): compares the gridded analysis
     high to the actual station high over ~14 days (robust median, soft-capped
     ±3°F). This is what corrects the known Central-Park (KNYC) cool bias in
     production — *note:* the older `sensor_drift.py` module referenced in some
     comments has been **removed**; `recent_grid_bias` is the live mechanism.
   - **Dispersion calibration** (`calibrate_dispersion`): shrinks the over-wide
     multi-model spread toward realized recent error (`target =
     0.6·realized + 0.4·ens_sd`). It only ever *shrinks*, never inflates.
   - **Center-bias** (`center_bias.center_shift`): a James-Stein shift toward a
     neutral 0°F prior, using only logged actual-minus-forecast errors.
4. **Apply the intraday floor**: for same-day contracts, members below the
   already-observed running high are physically impossible and are truncated.
5. **Compute the bracket probability** as a mixture:
   `P = 0.6·P_ensemble + 0.4·P_NWS`, where `P_ensemble` is the empirical fraction
   of corrected members landing in the YES interval and `P_NWS` is a Normal
   around the official NWS high. Both integrate over identical settlement bounds.

### 3. Calibration & recalibration

A raw model probability is passed through a **monotone isotonic recalibration**
(`calibration/recalibrate.py`) that maps "what the model says" to "what actually
happens," keyed by `prob_source` (ensemble vs rule) and lead bucket (same-day
0–14h vs day-ahead 14–40h). The map (`data/models/prob_calibration.json`, last
built 2026-08-03, n≈122k) is fit by `build_calibration_map.py`, which **only
emits a key when a time-split holdout proves it lowers Brier** on ≥150 samples.

Two disciplined choices a technical reader should note:

- **The ensemble base is deliberately left as identity.** Calibrating it made
  out-of-sample Brier *worse* (0.206 → 0.223), so we don't. Only the
  `ensemble@day-ahead` slice is calibrated, because it measurably over-predicts
  YES (bias ≈ +0.05). Calibrated probabilities are hard-clipped to [0.01, 0.97] —
  a mapped path may never assert near-certainty.
- **A market-shrinkage layer exists but is log-only.** `shrinkage.py` blends
  model-P and market-P in log-odds; the *fitted* weights are `w≈0.05` (same-day)
  and `w≈0.10` (day-ahead) — meaning the data currently says **the market
  carries most of the information** and our model should be shaded heavily toward
  it. This is computed and logged but not yet wired into gating. It is, bluntly,
  quantitative evidence consistent with the RED skill alarm.

### 4. Settlement mechanics

`settlement.py` is the single source of truth, derived empirically from **8,040
settled markets** and fixture-tested so the build fails if the rules drift:

- `between`: YES iff `floor ≤ high ≤ cap` — **inclusive both ends** (brackets are
  two integers wide).
- `greater`: YES iff `high > floor` (strict).
- `less`: YES iff `high < cap` (strict).

These conventions are derived empirically and pinned by a fixture test that
replays thousands of settled markets, so the build fails if the settlement
logic ever drifts from what Kalshi actually pays — the grading of every trade
rests on this, so it is verified against ground truth rather than assumed.

### 5. The decision pipeline — the gates

Pricing produces a probability and a signed edge (`edge = prob − market_mid`);
whether a pick is actually *pushed to the phone* is decided by a stack of
numbered gates in `scripts/run_live.py`. Each candidate's deciding gate is
written to a counterfactual `gate_log.jsonl` so gates can later be graded on
whether they filtered winners or losers.

| Gate | Suppresses |
|---|---|
| **−1** Validation boundary | Everything except ensemble-priced picks on an OOS-validated (city, type). Kills all rule-priced and unvalidated markets. |
| **0** Horizon | Targets starting > 36h out. |
| **1** Evidence-scaled min edge | Edge below a per-city bar (0.10 validated → 0.18 weak → 0.25 no-history), plus a same-day surcharge. |
| **2** Multi-model agreement | GFS/ECMWF/NBM spread > 3.5°F, or < 2 models answering (single-model outliers). |
| **2b** Forecast stability | NWS vs ensemble median disagreeing > 2.5°F, or a too-wide post-shrink spread. |
| **3** Evidence-scaled ceiling | Edges *too large* to be real (treated as calibration error, not mispricing). |
| **4** B-type BUY_YES watch | B-type YES picks until 10 graded outcomes clear a 15% YES rate (they historically over-fired). |
| **5** Same-day watch | All same-day picks are logged-but-not-pushed — the intraday warming taper was falsified live 2026-07-12. |

The takeaway: the system is currently **tuned to be conservative to a fault** —
several whole categories (rule-priced picks, same-day picks, B-type YES) are
suppressed pending forward proof. That is deliberate. It fires rarely.

### 5b. From baseline signal to acted-upon trade — the full funnel

**This is the part most external readers miss, and it is the core of how we
actually trade.** A gate-passing "fire" is a *baseline candidate*, not a
decision. It then goes through two further layers before any capital is
committed:

1. **Baseline fire** (Part II §2–5): the ensemble prices a contract, and the
   8-gate stack decides it clears the evidence-scaled bars. Output: a notification
   and a row on the candidate board. This is the level everything in Part III's
   *pooled* numbers is scored at — deliberately unfiltered, and therefore the
   *weakest* view of the system.

2. **Automated externality vet** (`scripts/candidate_pipeline.py`, Stage 2). Every
   fire is independently re-checked against sources the pricing engine never
   used: a **climatology base rate** (±10 days-of-year over 17 years of labels),
   **multi-model agreement** (HRRR/ECMWF/NBM/GFS + NWS), a **live Kalshi metadata
   re-verify**, a **self-contradiction check** (is it a BUY_NO on a bracket that
   contains our own forecast center?), and a same-day quarantine. Each check emits
   OK / FLAG / RED, producing a verdict: **TRADE_SMALL** (clean + real edge),
   **WATCH** (conditional flags), or **PASS** (fatal problem → discard).

3. **Human independent verification** (the deep dive). Before acting, the trader
   verifies the survivor through a *different code path than the one that priced
   it* — direct per-model fetches, climatology sanity, a member-level stress test
   — never confirming a pick via the same engine that generated it. Only then is
   it traded, at small size under the Part II §6 risk caps.

The empirical payoff of this funnel (from `grade_candidates.py`, mid-fill):

| Layer | n | Win-rate | Mean return (mid-fill) |
|---|---|---|---|
| Raw / unvetted fires (layer 1) | 1,496 | 44% | +0.072 |
| WATCH (survived vet, conditional) | 64 | 55% | +0.069 |
| **TRADE_SMALL (vet clears to trade)** | **5** | **80%** | **+0.109** |
| PASS (vet rejected — the losers) | 249 | 18% | +0.062 |

Two honest caveats that must travel with these numbers: **(a)** TRADE_SMALL is
only n=5 — the vet is *extremely* selective right now, by design, so the top of
the funnel is not yet statistically powered; and **(b)** all returns are struck
at **mid, with no fees or spread** — optimistic. What the table *does* establish,
even at small n, is the **shape**: vetting monotonically raises win-rate and the
rejected bucket is correctly the worst. That shape is the argument that the
externality check adds real value beyond the baseline fire.

### 6. Risk layer

`live/risk.py` (`RiskManager`) sizes with **fractional (quarter) Kelly** under
hard caps. Current configured parameters (`data/risk_config.json`):

| Parameter | Value |
|---|---|
| Bankroll | **$3,000** |
| Kelly fraction | 0.25 |
| Per-trade cap | 1.5% (≈ $45) |
| Daily circuit breaker | 7% (≈ −$210) → halt |
| Correlation cap (per city×date) | 5% (≈ $150) |
| Cluster cap (per synoptic region×date) | 5% (≈ $150) |

Correlation clusters (e.g. northeast {NYC,PHL,BOS,DCA}, southern-plains
{DAL,AUS,SAT,OKC,HOU,MSY}) exist because same-day bets across one weather system
are *not* independent and would otherwise stack ~3× the intended exposure. The
circuit breaker makes a 30–40% day structurally impossible.

**Wiring status:** on the manual (real-capital) path, sizing output is
*advisory* (logged, printed on the pick card), and the circuit breaker is fed
P&L only from **manually logged trades** (via the Google-Sheets logger) — if a
human doesn't log fills, the breaker never trips. That was the accepted
limitation of a notify-and-manual system, and it is exactly the loop the
Part VI execution layer closes: on the demo pilot, fills and settlement P&L
flow into the risk state automatically, a breaker trip cancels every resting
order, and each fill auto-arms the position watch — no human logging required.

### 7. Outcome tracking — the feedback flywheel

`outcome_tracker.py` grades every settled pick against IEM settlement truth
(deferring, never guessing, when a T-contract's direction isn't yet knowable),
and writes per-pick outcomes that feed back into: (a) the weekly isotonic
recalibration refit, (b) the center-bias correction, (c) the watch gates. A
daily driver (`daily_review.py`, 10:00 local) runs the whole loop: health check
→ replay drift check → grade → staleness watchdog → weekly refit → forward
report → skill monitor → candidate grading. **Every model change remains
human-reviewed** — the grader recommends, it never auto-mutates the model.

### 8. Validation machinery

- **Backtest** (`backtest.py` + `scripts/historical_backtest.py`): event-driven,
  chronological, with **both fill models computed on every trade** — pessimistic
  (taker: worst plausible price + slippage + `0.07·p·(1−p)` fee) and optimistic
  (maker: mid, quarter fee). Confidence intervals come from a **block bootstrap
  in 30-day blocks** to respect heat-wave autocorrelation, plus an **effective-N**
  adjustment. Historical calibration is walk-forward (fit only on years *before*
  the settlement year).
- **A candid limitation:** there is **no historical ensemble backtest**, because
  members aren't archived. The historical backtest uses the *deterministic*
  forecast as an explicit **lower bound**; the real ensemble engine can only be
  proven **forward**, on the data we're logging now.
- **Forward validation** (`validate_forward.py` → `forward_report`): Brier /
  win-rate / realized edge sliced by source and city, with a **day-blocked
  bootstrap** and an honesty gate that prints "UNPROVEN" below 10 independent
  settlement days.
- **Replay & provenance** (`provenance.py`, `replay_check.py`): every signal row
  is stamped with `code_sha` / `calib_hash` / `curve_hash`, and a pre-push hook
  re-derives recent production output through current code and **blocks the push**
  on unacknowledged model drift.

### 9. Operations

- **Scheduling — production runs 24/7 on a cloud VPS.** The always-on
  production host is a DigitalOcean droplet running the full stack under
  **systemd** (`plec-*` units; the health check was ported to monitor them on
  2026-08-05). The repo-tracked **launchd** plists (11 jobs) are the canonical
  job definitions and run the identical stack on the development machine as a
  redundant mirror; a flock+marker slot guard makes the parallel schedulers
  double-fire-safe. Jobs: the 24/7 logger; `run_live` every 30 min; the
  candidate pipeline 4×/day; `daily_review` at 10:00; an **hourly** health
  check; plus intraday watchers (position-watch 30 min, lock-scanner 10 min,
  fast-obs-watch 5 min).
- **Change safety:** git hooks run pytest + a smoke test on every commit, plus
  the replay check on push; entry points call a `preflight()` that exits loudly
  rather than run hollow; a health check verifies *outputs* (not just processes)
  and pushes on RED.
- **One housekeeping item, noted for completeness:** the systemd unit files
  live on the droplet and are not yet mirrored into source control — the
  tracked launchd plists are the reference job definitions. A copy-back is
  planned; it does not affect what runs.

### 10. Live vs advisory vs manual — the honesty map

- **Live (gates/prices what fires):** ensemble pricing + all three corrections,
  isotonic recalibration, settlement conventions, the 8 push gates, risk-halt
  notification suppression.
- **Advisory (accrues evidence, does not yet act):** market-shrinkage posterior,
  action-score/tiers, same-day picks, B-type YES, candidate "graduation" bar,
  all daily-review diagnostics.
- **Manual (not automated):** real-capital trade execution and its risk
  P&L/exposure (fed only by human-logged trades), and every model/threshold
  change. Demo-exchange execution is now automated end-to-end (Part VI) —
  the pilot that earns real-capital automation.

---

## Part III — What we can and cannot claim today

The one distinction that governs this whole section: **the baseline fire vs. the
acted-upon trade** (Part II §5b). Pooled numbers score the baseline firehose;
the funnel numbers score what we actually trade. Both are shown; neither is
hidden.

**Can claim, with error bars:**
- A skilled daily-high forecast — **MAE 1.52°F** [1.42, 1.63] over 1,144
  city-days — with a small, *diagnosed* cool bias (+0.46°F) concentrated in the
  coastal settlement stations (NYC/SFO/SEA), i.e. a known settlement-mechanics
  fix, not a mystery. Re-derivable from `model_skill_report.py`.
- Correct, empirically-verified settlement mechanics over 8,040 markets.
- A real, enforced discipline layer (leakage tests, walk-forward, block
  bootstrap, replay gate, risk caps) that a second person can reproduce.
- A growing proprietary dataset (orderbook depth + ensemble members) no
  competitor can purchase.
- **That the vetting funnel adds value**: acted-upon (TRADE_SMALL) win-rate 80%
  vs 44% raw, with rejected picks correctly worst at 18% (Part II §5b). The
  *shape* is robust even though the top-of-funnel *level* is small-n.

**Can claim in pockets — the important nuance:**
- A **localized** edge on the *vetted* subset. Specific cities and the `B|BUY_NO`
  short-bracket structure show a positive, repeatable signal — the ~8–9 combos
  clearing our advisory bar (SFO +0.173 / 68% win, DCA +0.169 / 65%, etc.). This
  is why the picks we *act on* are so narrow: they are the pockets the data has
  earned.

**Cannot claim yet — stated as plainly as the wins:**
- A **holistic** edge on the *raw* output. Pooled over every fire, the market
  prices better (`model_skill_report`: model Brier 0.11 vs market 0.03; the
  day-ahead skill monitor is RED at −0.0162, CI excluding zero). That pool is the
  unfiltered baseline, not the traded book — and it is reported here in full,
  because the discipline is to surface weaknesses, not bury them.
- **Statistical** significance on the acted-upon edge. The cost-aware forward cut
  (`validate_forward`: actionable +3.7%, CI [−2.8%, +9.3%], eff. n=10 days) still
  spans zero, and TRADE_SMALL is n=5. The direction is right; the sample is not
  yet decisive.
- That the pocket combos are profitable *net of costs*. Returns are struck at
  **mid, no fees/spread**; autocorrelation inflates them. A strong research
  signal and the basis for our selectivity — not yet a booked P&L.

The intellectually honest position: **the vetted subset we actually trade shows a
real, local edge; the raw baseline does not beat the market holistically, and it
isn't meant to.** The gap between them is the value of the externality vet, and
the reason the aggregate looks weak is overwhelmingly a *time-and-data* problem —
the model hasn't had the calendar to develop across every city and structure,
and the vetted funnel is too new to be statistically powered. That is precisely
why we pick so exclusively today. The machinery is built to expand what we act on
only as the evidence earns it, and to stop us declaring a broad
edge before it does.

---

## Part IV — Why the current design has a ceiling

Everything above optimizes one quantity: **P(settlement outcome)**, held to
expiry. The validated edge, if it materializes, is a *terminal-probability*
edge — be better-calibrated than the price about where the temperature lands,
then wait. This is crowded (every weather bot forecasts settlements) and, as the
current window shows, hard to beat *holistically* on day-ahead — even where we
win in specific validated pockets.

There is a second, largely untapped quantity in the same data: **how the price
*moves* between listing and settlement.** That is the subject of Part V.

---

## Part V — New direction: a price-path (mispricing) book

> **Status (2026-08-12): the plan below was executed through its designed
> GO/NO-GO gate.** Steps 1–2 were built and run; the Step-2 gate returned
> **NO-GO**, killing the *live-trading* question in an afternoon exactly as
> the plan priced it — settlement book unaffected, and the underlying signal
> confirmed and redirected to where it pays. Steps 3–5 have since been built
> anyway as a **dormant paper pilot**: the full mark-to-market engine,
> exit-policy bake-off, and a daily forward shadow book that accrues the
> evidence an activation decision would need — at zero risk and zero cost.
> The plan is kept below as the record of what was tested and how; the
> outcome is in the **Verdict** section at the end of this Part.

### The idea

Today we ask *"where will the temperature settle?"* and hold to expiry. The new
book asks a different question: *"which contracts are underpriced right now and
likely to rise, so we can buy low and sell into the rise — banking the profit
even if the contract ultimately settles NO?"*

Formally, the current edge is:

> `P_model(settle) − P_market > costs` → trade → **hold to settlement.**

The new edge is:

> `E[P_market(t+Δ)] − P_market(t) − round_trip_costs > 0` → buy → **sell at the
> peak, outcome be damned.**

This is a forecast of the **price path**, not the terminal outcome. It is a
genuinely different strategy — closer to short-horizon statistical arbitrage
than to weather forecasting — and it is **additive**: the settlement book keeps
running unchanged.

### Why it is feasible with what we already have

We are not starting from theory. Our own timing study (n≈708, 15 days) measured
that **the ensemble front-runs the market**: the market price drifts *toward* our
ensemble over time (market→us 64%, and when they disagree our side is right
72%). Restated in price-path language, that *is* a directional price forecast —
when our ensemble sits above the market, the market is expected to rise to meet
it. And we already log, at 5-minute resolution, the exact data needed to test
this: the full orderbook path joined to the contemporaneous ensemble. **The
signal and the substrate both already exist.** The work is turning "the market
eventually agrees with us" into "the price will be X¢ higher in N hours,
reliably enough to clear a round trip."

### How it differs operationally (and why that matters)

| | Settlement book (today) | Price-path book (new) |
|---|---|---|
| Predicts | Terminal outcome | Price trajectory |
| Exit | Hold to expiry | **Actively sell at/near a peak** |
| Costs | One-way | **Round-trip (spread ×2, fees ×2)** |
| P&L | Realized at settlement | **Marked-to-market intra-life** |
| Latency sensitivity | Low (hours to settle) | **High (must hit the exit window)** |
| Hardest decision | Entry | **Exit** |

### Workplan

The same vertical-slice discipline as the original build: prove it on **one
city, daily highs first**, add **lows** only if they come cheaply, under both
fill models, with block-bootstrapped CIs.

- **Step 1 — Build the joined price-path panel.** For each target market, join
  the logged orderbook path (`mid/bid/ask/depth` over time) to the contemporaneous
  ensemble-P and obs-so-far at each snapshot. *Deliverable:* a clean panel table.
  *Effort:* a few hours — it's mostly a join over data we already collect.
- **Step 2 — Descriptive convergence study (the GO/NO-GO gate).** No trading.
  Measure empirically: when ensemble-P and market-P diverge, how fast and how far
  does the market close the gap? What's the half-life? **How often is the expected
  move larger than the round-trip cost (2× spread + 2× fees)** on our actual, thin
  weather books? *Deliverable:* a decision memo with distributions and CIs.
  *Effort:* a few hours on the panel from Step 1. **If the moves don't clear the
  round trip, the strategy is dead here — for an afternoon, not a build cycle.**
- **Step 3 — Mark-to-market engine.** Extend the harness (which today only
  realizes at settlement) to open, mark, and *close mid-life* under an explicit
  exit policy, with **pessimistic exit fills** held to the same standard as
  entries. *Effort:* a few hours reusing the existing fill/bootstrap code.
- **Step 4 — Exit-policy bake-off.** Test exit rules head-to-head:
  fixed-horizon, target-price, trailing-peak, "convergence complete" (sell when
  the market reaches our ensemble), and time-stop. This is where we learn whether
  the *exit* — the hard half — is capturable on the data we have so far.
  *Effort:* a few hours, same day as Steps 1–3 if we pair on it.
- **Step 5 — Forward validation as a brand-new edge.** Its own block-bootstrap
  CI, its own walk-forward, both fill models — **no borrowing credibility from
  the settlement work.** This is the *only* step bound by the calendar: it needs
  live forward days to accumulate. *Effort:* ~**40 days** of data for full
  statistical confidence — but see the phased go-live below; careful trading
  starts well before day 40.

### Timeline — the build is fast; only verification is calendar-bound

The engineering is *not* the long pole. Steps 1–4 reuse machinery we already
have (the logger, settlement conventions, fill models, block bootstrap), so the
whole analysis-and-harness build is realistically **~1 focused day of paired
work**, not weeks. What cannot be compressed is *live data accumulation* — the
market only prints so many independent days per week.

So the timeline is phased, and trading starts far earlier than "fully proven":

| Phase | When | What happens |
|---|---|---|
| **Build + GO/NO-GO** | Day 1 | Steps 1–4 in a focused session. Step 2 tells us within hours whether convergence moves clear the round trip. If not, we stop here. |
| **Careful-trading window** | Once Step 2 clears + a short live sanity check | Begin placing **small, careful** trades on the strongest, most liquid convergence setups — under the existing risk caps, both fill models, human-verified — *while data accrues*. This is a live-but-tiny probe, not full deployment. |
| **Full statistical verification** | ~**40 days** of forward data | Step 5's block-bootstrapped CI clears zero under the pessimistic fill model. Only then does size scale beyond the careful probe. |

To be explicit about the timeline: **we do not wait 40 days to touch it.**
Day 40 is when the edge is *statistically confirmed*; careful trading can begin
as soon as the descriptive study (Step 2) shows the economics are there and a
brief live check confirms exits actually fill. This mirrors how the settlement
book already graduates positions — watch-only → tiny → small → size — rather than
flipping a switch at a single proof point.

### Risks (in rough priority order)

1. **The exit is the whole game, and it's the harder call.** "Shoots up then
   loses, but we already banked it" only works if we *reliably exit before the
   reversal* — two correct decisions per trade, and the exit is the one we've
   never modeled. A momentum strategy that misses exits is just a slower loss.
2. **Round-trip cost may swallow the move.** Spread paid twice on thin books.
   Step 2 exists precisely to kill the idea fast if the economics don't clear.
3. **Adverse selection near settlement.** The same front-running that helps us
   early works *against* us late, when flow is better-informed. The exit must
   land in the window where we're still the smart money.
4. **Effective sample size collapses.** Intraday snapshots are heavily
   autocorrelated; naïve n will badly overstate significance. Block bootstrap
   becomes even more essential.
5. **Can we even sell?** Exiting means resting an offer or crossing a 3-deep
   book. We must confirm on logged depth that exits fill without giving the edge
   back.
6. **Latency / the automation question.** Path trading is time-sensitive in a
   way settlement-holding is not. The current system is **notify-and-manual**;
   catching exit windows by hand may not be fast enough, which would force a
   real **order-placement + automated-risk** build — a significant new
   surface, and a risk-layer escalation we should scope deliberately, not drift
   into.
7. **The signal may already be priced / decaying.** The 15-day front-running
   study is itself a small sample; the convergence may be weaker or faster than
   measured. Step 2 re-measures it on far more data before we commit.

### Effort / resourcing summary

- Steps 1–4 are **~1 focused day of paired work**, reusing the existing logger,
  settlement, and bootstrap machinery — the build is genuinely cheap.
- Step 5 is **calendar, not labor** — ~40 forward days to *statistical*
  confidence, but careful live trading starts well before that (see timeline).
- The one item that could balloon scope is **execution automation** (Risk #6),
  which we would only build *after* the careful-trading window shows the edge is
  real and the manual pace is the binding constraint. It should be costed
  separately when we get there.

### Decision gates (so this can be killed cheaply)

- **After Step 2 (Day 1):** if expected moves don't reliably exceed round-trip
  cost → **stop.** An afternoon spent, strategy disproven, settlement book
  unaffected.
- **Before the careful-trading window:** a brief live check must confirm exits
  actually fill on the logged book without giving the edge back — if we can't
  sell, we don't start.
- **After Step 4:** if no exit policy beats "hold to settlement" net of costs
  under the pessimistic fill model → **stop** (or keep it watch-only).
- **Scaling past the careful probe:** size up beyond tiny only once the ~40-day
  forward record clears a block-bootstrapped CI above zero under both fill
  models — the identical bar the settlement book must clear.

### Verdict — the gate was run, and it worked (2026-08-11 → 12)

Steps 1–2 were executed exactly as scoped, in about a day
(`scripts/build_price_path_panel.py`, `scripts/price_path_convergence_study.py`,
`scripts/price_path_stress_test.py`; all re-runnable).

**Step 1 delivered and validated.** 92.6M logged depth rows collapsed to a
1.2M-point top-of-book panel across 7,390 markets, 80% carrying a
contemporaneous model view, machine-checked for look-ahead (zero violations)
and validated against an independent price source (0.9928 correlation between
the orderbook-derived mid and the signal log's own market mid).

**Step 2 verdict: NO-GO — with the signal itself *confirmed*.** The market
does drift toward our ensemble, monotonically in the size of the disagreement —
the timing study's front-running finding, re-confirmed on ~1.2M snapshots
instead of 708 observations. But the convergence is far too *slow* to trade as
an intraday round trip: even the largest disagreements (~40¢) close at roughly
**½¢ per hour**. Over 30–240-minute horizons the realized move (≈+0.1¢) cannot
cover a round trip on these books, and every honest execution model loses
about the same ~4¢/contract: pessimistic taker **−4.2 to −4.5¢** at every
horizon; a maker-entry hybrid **−4.4¢ per filled trade** (48% fill rate —
adverse selection cancels the spread saved); deep-books-only **−3.8¢**. All on
52–53 settlement days with day-block bootstrapped CIs tightly below zero. The
only positive variant is the fully optimistic maker-both model (+1.6¢), which
is pure spread capture — and by this project's charter, edge that survives
only the optimistic model is not edge.

Three things this outcome bought, because a clean kill is a *product* of this
system, not a failure of it:

1. **The decision gate did its job at the advertised price.** The plan's own
   words were "if the moves don't clear the round trip, the strategy is dead
   here — for an afternoon, not a build cycle." That is precisely what
   happened. No capital was ever at risk; the settlement book ran unchanged
   throughout.
2. **The validation discipline stopped a false positive — again.** The first
   run of the study showed a spurious "+3¢, GO" result. The mandated
   stress-test pass traced it to a subtle dataframe-alignment bug in the
   forward join; the corrected result was then reproduced by a *second,
   independent* join implementation to within 0.04¢. This is the same
   discipline layer that caught the ECMWF feed dropout and the
   settlement-convention error before they could cost money — the
   differentiator we claim, doing exactly what we claim, on the exact occasion
   it was needed.
3. **The signal survives — in the book it always belonged to.** "The market
   comes to our ensemble over days" is exactly what the settlement book's
   hold-to-expiry design harvests, without ever paying the intraday round
   trip. The re-confirmation *strengthens* the settlement book's premise. The
   panel builder remains a validated, reusable research substrate that grows
   daily with the logger, so this gate can be re-run essentially for free if a
   faster-converging market regime ever appears — 52 days of one summer is the
   evidence we have today, not the final word on every regime.

**The redirect:** with the intraday path ruled out, the automation effort this
Part motivated (Risk #6) moved to the settlement book — and was built the next
day. That is Part VI.

**And the machinery was completed anyway — as a dormant pilot.** Steps 3–5
were subsequently built in full (`pricepath/engine.py`,
`price_path_exit_bakeoff.py`, `price_path_shadow.py`): a mark-to-market
engine with an eight-policy exit library, a bake-off harness, and a daily
forward **shadow book** that paper-trades every gated entry under every exit
policy from logged data (a batch replay that is deterministically identical
to having traded live, since every input is logged). The bake-off over the
full 53-day panel confirms the Step-2 verdict across the whole exit-policy
space — no policy's pessimistic CI clears zero (best: 4-hour horizon at
−3.18¢ [−3.61, −2.78]; the stop-loss grades among the worst, the empirical
receipt for hold-to-settlement). The pilot is **provably dormant**: it can
never place an order or touch the network (pinned by the commit-gate smoke
test), has deliberately **no activation flag**, and costs nothing to run.
If the forward shadow record ever clears the pessimistic bar — regime
change, tighter books — activation is a conscious wiring decision made from
accumulated evidence, not a rebuild.

---

## Part VI — Closing the automation loop: the execution layer + demo pilot

**Built 2026-08-12** (`live/exchange.py`, `live/executor.py`,
`tests/test_execution.py`). This is the first order-capable code in the
system, and it converts the trajectory promised in Part I — notify-and-manual
as a validation phase, automation as the destination — into running machinery.

### What it does

- A **fire-eligible pick** (one that has cleared the full Part II gate stack)
  becomes a **post-only maker limit order** resting at the current touch —
  never crossing the spread, never paying taker fees — **sized by the same
  quarter-Kelly risk layer** (per-trade, correlation, and cluster caps) that
  has governed advisory sizing all along.
- Every order carries a **server-side expiration** (default 2h), so an
  unfilled entry dies on its own. Exits are **hold-to-settlement** — our own
  timing study showed stops destroy this edge, so the only "auto-exit" is at
  the order level, never the position level.
- Each cycle **reconciles automatically**: fills feed the risk layer's
  exposure tracking and auto-arm the intraday position watch; settlements feed
  P&L; a circuit-breaker trip **cancels every resting order**. Every event —
  place, skip, reject, fill, settle, cancel — is appended to an audit log
  (`data/execution/execution_log.jsonl`).

### Safety model (deliberately paranoid)

- Execution is **opt-in** (an explicit environment flag) and defaults to
  **Kalshi's demo exchange** (play money). Production is unreachable without a
  second, explicit interlock variable — and both properties are pinned by the
  commit-gate smoke test, so a change that weakened them could not land.
- The demo pilot runs against an **isolated risk state**, seeded from the real
  configuration: demo fills can never consume real correlation-cap headroom or
  trip the real-capital breaker while manual trading continues in parallel.
- Execution failures can never take down the signal pipeline (fully
  exception-isolated), and 17 dedicated tests cover the signing, the
  interlocks, sizing, fills, settlement math, and halt behavior.

### The pilot, and how it graduates

The demo pilot's job is to convert the research-grade record (mid-fill, no
fees — Part II §5b's honest caveat) into a **booked, fee-real, fill-real
track record** under identical discipline. It has no fixed end date: **it runs
until the forward numbers are statistically decisive**, and size follows
evidence — never the other way around. Two expectations, stated plainly:

- **The record will accrue slowly at first, by design.** The gate stack is
  tuned conservative (Part II §5), so fires are rare; a thin early sample with
  wide error bars is the expected shape of month one, not a warning sign.
- **The numbers should improve with time and data**, for the same structural
  reasons as Part III: every settled day feeds the weekly recalibration,
  bias corrections, and per-city validation that decide what is allowed to
  fire — the machinery that widens the book only as evidence earns it.

Real-capital automation happens only when the demo record clears the same bar
everything else in this project must clear: a block-bootstrapped confidence
interval above zero under the pessimistic fill model. Until then, the pilot
accumulates exactly the evidence that decision needs.

---

## Appendix — where to look in the repo

| Concern | Primary files |
|---|---|
| Live pricing & gates | `src/kalshi_weather/live/runner.py`, `scripts/run_live.py` |
| Ensemble engine | `src/kalshi_weather/calibration/ensemble_dist.py`, `center_bias.py` |
| Calibration | `calibration/recalibrate.py`, `build_calibration_map.py`, `data/models/prob_calibration.json` |
| Settlement truth | `src/kalshi_weather/settlement.py` (+ `tests/test_settlement_convention.py`) |
| Risk | `src/kalshi_weather/live/risk.py`, `data/risk_config.json` |
| Execution (demo pilot) | `src/kalshi_weather/live/exchange.py`, `live/executor.py`, `tests/test_execution.py`, `data/execution/` |
| Price-path study (Part V verdict) | `scripts/build_price_path_panel.py`, `price_path_convergence_study.py`, `price_path_stress_test.py` |
| Price-path dormant pilot (Steps 3–5) | `src/kalshi_weather/pricepath/engine.py`, `scripts/price_path_exit_bakeoff.py`, `scripts/price_path_shadow.py`, `tests/test_pricepath.py` |
| Grading & feedback | `src/kalshi_weather/outcome_tracker.py`, `scripts/daily_review.py` |
| Backtest & forward | `backtest.py`, `scripts/historical_backtest.py`, `scripts/validate_forward.py`, `scripts/skill_monitor.py` |
| Leakage / provenance | `tests/test_leakage.py`, `provenance.py`, `scripts/replay_check.py` |
| Data logger | `src/kalshi_weather/logger/` |
| Plain-language background | `docs/walkthrough.md`, `docs/how-the-model-works.md` — *both predate the current ensemble engine (see their banners); this document's Part II is authoritative* |

*All performance figures in this document were refreshed **2026-08-11** and are
re-derivable from the cited scripts (`model_skill_report.py`,
`validate_forward.py`, `grade_candidates.py`, `skill_monitor.py`). The trailing
windows move daily, so re-run them before quoting numbers externally — the
figures here are a same-day snapshot, not a fixed record.*
