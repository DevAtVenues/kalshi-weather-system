# How the Model Works — The Steps

*Plain-language companion to [`STATE_OF_THE_MODEL.md`](STATE_OF_THE_MODEL.md),
which is authoritative for figures and status.*

A walkthrough of how we turn raw weather forecasts into a probability we can bet
against the market — and, just as important, how we decide which of those bets to
actually place. Each contract is a yes/no bet on a city's daily high (e.g. "NYC
high ≥ 92°F", or "high lands in 86–87°F"). The market price of YES is effectively
the market's probability. Our job is to produce a *better-calibrated* probability
and trade the gap — but only where the gap survives a second, independent check.

> **Where this is headed.** The end goal is a fully automated trading engine. The
> system was deliberately *piloted* as a forecast-and-signal engine and is now in
> transition toward that. Today it is **notify-and-manual**: it computes and vets
> picks and a human places each trade.

---

## Step 1 — Build an ensemble, not a single forecast

This is the most important idea in the model. Instead of trusting one "best
guess," we pull **dozens of forecast runs across three model families** — **GEFS
(the US ensemble), ECMWF-ENS (the European ensemble), and ICON (the German
one)** — from Open-Meteo, at decision time only (never a run that didn't exist
yet — no look-ahead).

Each run makes slightly different assumptions about the starting state of the
atmosphere, so together they trace out the *range* of plausible outcomes:

```
Each dot = one model run's predicted high for tomorrow in Philadelphia:

  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92°F
   .   .   ..  ...  ....  .....  ......  .....  ....  ...  ..  .   .

  (this spread of dots is the "ensemble" — a distribution, not one number)
```

A safety check runs first: if any family comes back with too few members
(GEFS < 25, ECMWF < 40, ICON < 30), we **refuse the whole fetch** rather than
trade on a thin, unreliable ensemble. This guard exists because ECMWF once
silently returned *zero* members for weeks while the total still looked
plausible — and that produced fake edge.

---

## Step 2 — Correct the members for the settlement sensor

Raw models are computed on a geographic **grid**. But Kalshi settles on a
specific physical thermometer — NYC settles at **Central Park**, not JFK — and
the grid and the sensor don't always agree. We apply three corrections to the
ensemble members (not to a single number):

1. **Grid-vs-sensor bias.** We compare the model's gridded high to the *actual*
   station high over roughly the last 14 days, take a robust median of the gap,
   cap it at ±3°F, and subtract it out. This is what removes the known Central
   Park cool bias. (It's learned fresh from recent data — an older hard-coded
   version, `sensor_drift.py`, has been removed.)
2. **Dispersion calibration.** Three disagreeing model families are usually
   *too* wide. We shrink the spread toward the error the station has actually
   shown recently (`target = 0.6·realized + 0.4·ensemble spread`). It only ever
   **shrinks, never inflates.**
3. **Center-bias.** A small, statistically-shrunk nudge (James-Stein, toward a
   neutral zero) that corrects any residual lean using only our own logged
   forecast-vs-actual errors.

For **same-day** contracts there's a fourth step: if the day's high has *already*
reached, say, 87°F, any ensemble member predicting 85°F is physically impossible,
so we truncate those away.

---

## Step 3 — Turn the ensemble into a probability

Now the question **"Will the high be 88°F or above?"** has a direct answer:
**what fraction of the corrected members land in the winning range?**

```
  If 18 of ~90 corrected members land at 88 or above → P_ensemble = 20%
```

We then **blend that with the official NWS forecast** — a Normal curve centered
on the NWS high — to get the final probability:

```
  P(YES) = 0.6 · P_ensemble  +  0.4 · P_NWS
```

The two views anchor each other: the ensemble supplies the shape of the
uncertainty, the NWS number supplies an authoritative center. Both are read
against the **exact settlement rules** (verified against 8,040 past markets):
*between* is inclusive of both ends, *greater* is strictly above the floor,
*less* is strictly below the cap.

---

## Step 4 — Recalibrate: "when we say 70%, does it happen 70% of the time?"

Even a good probability can be systematically over- or under-confident, so we
pass it through a learned correction curve (**isotonic recalibration**) built
from our own graded history — separately for each *lead time* (same-day vs.
day-ahead), because a day-out forecast behaves differently from a few-hours-out
one.

Two disciplined details:

- **We only correct where correcting provably helps.** The ensemble's *base* is
  deliberately left **untouched** — when we tried calibrating it, the
  out-of-sample score got *worse*, so we don't. Only the day-ahead ensemble
  slice, which measurably over-predicts, gets a correction.
- **We never assert certainty.** Final probabilities are clipped to the range
  1%–97% — the model is not allowed to say "100%."

---

## Step 5 — A fire is a *baseline*, not a trade (the part most people miss)

Everything above produces a **baseline signal**. It is *not* yet a decision. This
is the core of how we actually trade, and the reason our headline numbers look
modest while our *acted-on* numbers look much better.

Every signal passes through three filters before any money moves:

1. **The gates.** An 8-check stack decides whether the signal even clears the
   bar — enough edge (scaled to how well that city is validated), the model
   families agree (spread ≤ 3.5°F), the forecast is stable, and the edge isn't
   *implausibly* large (which usually means a calibration error, not real
   mispricing). Whole categories are currently suppressed pending proof. It fires
   rarely, on purpose.

2. **The externality vet.** Every fire is independently re-checked against
   sources the pricing engine never used: a **climatology base rate** (how often
   this outcome happens historically), **multi-model agreement**, a **live
   re-verification** of the market, and a **self-contradiction check** (are we
   betting NO on a bracket that contains our own forecast?). The result is a
   verdict: **TRADE_SMALL**, **WATCH**, or **PASS** (discard).

3. **A human deep-dive.** Before acting, a person re-verifies the survivor
   through a *different code path* than the one that priced it — never confirming
   a pick with the same engine that produced it.

The payoff of this funnel is real — win-rate climbs as you move up it:

```
  Raw / unvetted fires ........ 44% win   (the baseline firehose)
  WATCH (passed the vet) ...... 55% win
  TRADE_SMALL (vet says trade)  80% win   ← what we actually act on
  PASS (vet rejected) ......... 18% win   (correctly thrown out)
```

Two honest caveats that always travel with those numbers: **TRADE_SMALL is only a
handful of picks so far** (n=5 — not yet statistically powered), and these
returns are struck at mid-price with **no fees or spread** (optimistic). What the
funnel *does* establish, even so, is the **shape**: vetting lifts the win-rate and
correctly dumps the losers. That shape is the argument that the vet adds value.

---

## What we can honestly say today

- **The forecast is genuinely skilled.** Recent daily-high error is **1.52°F**
  (95% CI [1.42, 1.63]), with a small **cool bias (+0.46°F)** concentrated in the
  hard coastal settlement stations (NYC −2.3, SFO −1.0, SEA +1.5) — a *diagnosed*
  settlement-sensor problem, not a mystery.
- **The edge is local, not holistic — yet.** Scored across *every raw fire*, the
  market currently prices better than we do (our day-ahead skill monitor is
  RED). The positive edge shows up in the **vetted pockets** — specific cities and
  the short-bracket BUY_NO structure — not across the board. That's exactly why
  we trade so selectively.
- **The gaps are a time-and-data problem, not a broken model.** The edge can only
  be proven forward, one settled day at a time, and the vetted funnel is still
  too new to be statistically settled.

---

## And then — sizing and risk

Only after all three filters is a trade sized, using **one-quarter of the Kelly
criterion** (deliberately conservative) under hard caps on a **$3,000** bankroll:

```
  Per trade:   ≤ 1.5% of bankroll        (≈ $45)
  Per day:     circuit breaker at 7%     (≈ −$210) → stop trading that day
  Correlation: same-weather-system city bets are capped together (≈ $150)
```

A 30–40% day is **structurally impossible** by these rules — not "we'd try to
stop it," the math makes it impossible.

The headline principle throughout: **we never claim certainty.** Credibility comes
from honest, out-of-sample-validated probabilities with real error bars — and
from acting only on the small set of bets that survive an independent second
look — not from a secret signal.
