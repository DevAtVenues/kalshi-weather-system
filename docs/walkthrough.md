# Weather Trading — A Walkthrough

*Plain-language companion to [`STATE_OF_THE_MODEL.md`](STATE_OF_THE_MODEL.md),
which is authoritative for figures and status.*

> **Where this is headed.** The end goal is a fully automated trading engine. The
> system was deliberately *piloted* as a forecast-and-signal engine and is now in
> transition toward that. Today it is **notify-and-manual**: it computes and vets
> picks, and a human places each trade.

---

## What are we actually doing?

There's a financial exchange called **Kalshi** — think of it like a stock market,
but instead of buying shares of Apple, you're betting on yes/no questions about
the real world.

One category of questions is **daily weather**. Specifically: will a city's high
temperature hit a certain number tomorrow?

An example question on Kalshi right now:

```
"Will New York City's high temperature be 88°F or above on July 10th?"
```

You can buy YES or NO. If you buy YES for 40 cents and it hits 88°F, you get $1.
If it doesn't, you lose your 40 cents.

The price — that 40 cents — is essentially **what the crowd thinks the odds are.**
40 cents = the market thinks there's a 40% chance of YES.

---

## Our job, in one sentence

> Find the cases where we think the real probability is meaningfully different
> from what the market is pricing — **and where that gap survives an independent
> check** — and bet accordingly.

We're not predicting the weather perfectly. We just need to be **more right than
the market price** often enough, on the trades we actually place, to make money
after fees.

---

## Where does the market's price come from?

The market price is set by thousands of people buying and selling — traders,
weather hobbyists, algos, whoever. Their collective opinion becomes the price.

The price is usually pretty good. But it's not perfect. And markets specifically
have blind spots around:

- **Systematic forecast bias** (a model consistently runs hot in June in a city)
- **Settlement mechanics** (the thermometer Kalshi actually uses isn't always the
  obvious one — NYC settles at Central Park, not JFK)

Those blind spots are where we try to live.

---

## The data we use

Three ingredients, kept completely separate:

```
┌─────────────────────────────────────────────────────┐
│  1. THE FORECAST                                    │
│     Dozens of weather model runs (an "ensemble")    │
│     across three families — GEFS, ECMWF, ICON       │
│     Tells us the range of plausible outcomes        │
│     for tomorrow's high.                            │
├─────────────────────────────────────────────────────┤
│  2. THE SETTLEMENT TRUTH                            │
│     Official NWS climate reports — what actually    │
│     happened at the exact station Kalshi uses.      │
│     Used to measure past errors, never to predict.  │
├─────────────────────────────────────────────────────┤
│  3. THE MARKET PRICE                                │
│     What Kalshi is trading at right now.            │
│     The number we're trying to beat.                │
└─────────────────────────────────────────────────────┘
```

---

## Why many forecasts instead of one?

This is the most important idea in the whole model.

When a weather model runs, it makes thousands of tiny assumptions about the
starting state of the atmosphere. Slightly different assumptions → slightly
different outcomes. So instead of pretending one "best guess" is the truth, we
pull **dozens of runs across three model families** — **GEFS** (the US ensemble),
**ECMWF-ENS** (the European ensemble), and **ICON** (the German one).

The result looks like this:

```
Each dot = one model run's prediction for tomorrow's high in Philadelphia:

                         ← most runs land here →

  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92°F
   .   .   ..  ...  ....  .....  ......  .....  ....  ...  ..  .   .

  (this spread of dots is the "ensemble" — a distribution, not one number)
```

Now the question **"Will the high be 88°F or above?"** becomes:

```
What fraction of the corrected members land at 88 or above?

  If ~18 of ~90 members land there → our probability ≈ 20%
  If the market is pricing it at 40% → the market is overconfident
  → we lean NO (buy NO / sell YES)
```

This is much smarter than trusting a single forecast that might just be an
outlier. A safety check runs first: if any family returns too few members
(GEFS < 25, ECMWF < 40, ICON < 30), we **refuse the whole fetch** rather than
trade a thin ensemble — a guard added after ECMWF once silently returned *zero*
members for weeks while the total still looked plausible.

---

## The correction layer — this is our edge

Raw model forecasts are computed on a geographic **grid** — a mesh of points
across the country. But the thermometer Kalshi actually settles on is a specific
physical sensor at a specific location.

Those two things don't always agree.

```
  Weather model grid point            Actual settlement sensor
        (what the model sees)               (what Kalshi pays on)

            ╔═══╗
            ║   ║  ← grid cell covers a big area
            ║ X ║  ← Central Park sensor is here
            ║   ║     but the grid "sees" the whole cell
            ╚═══╝

  The size of this gap differs by city and by season — under a degree for most,
  but larger for a fog-bound city like San Francisco in summer.
  It's a systematic error, so we measure it (from the last ~14 days of grid-vs-
  station data) and subtract it out — then shrink the over-wide spread toward the
  error the station actually shows.
```

We do this per station, learned fresh from recent data. It's automatic, specific
to each sensor, and it's the kind of thing the market doesn't fully account for.

---

## The per-city picture — measured out-of-sample (the historical proof)

This is the honest test. We took three years of forecasts vs. actual settlements
(21,630 matched days across 20 cities), **trained the correction on 2022–2023 and
scored it on 2024 — a year the model had never seen.** No peeking. That's how you
know a correction actually works instead of just fitting the past.

Here's how far off our corrected forecast was, per city, on that unseen 2024 data:

```
  Average error on unseen 2024 settlements (°F). Lower = better.

  WE NAIL THESE                MIDDLE                    HARDEST
  ─────────────────            ─────────────────         ─────────────────
  LAX          0.96            San Antonio  1.71         Seattle    2.44
  Atlanta      1.30            Chicago      1.76         SFO        2.52
  Houston      1.34            Dallas       1.89         Denver     2.73
  Philadelphia 1.40            Minneapolis  1.89
  Miami        1.49            Austin       1.97         (marine layer +
  DC           1.52            Phoenix      2.01          mountains — the
  New Orleans  1.53            Las Vegas    2.09          genuinely hard
  NYC          1.55            OKC          2.18          forecast problems)
  Boston       1.55

  POOLED: 1.79°F average error across 7,315 unseen city-days (historical, 2024).
```

The pattern is real and physical: **coastal marine-layer cities (SFO, Seattle)
and mountain cities (Denver) are the hard ones** — the fog and terrain make the
settlement sensor genuinely hard to predict. The flat interior and LA basin we
forecast tightly.

### And it holds up on the recent live window

That historical test established the skill; the recent forward window confirms it.
Over the last ~59 days (1,144 city-days), the corrected forecast is running at:

```
  Recent daily-high error:  1.52°F   95% CI [1.42, 1.63]
  Small COOL bias:         +0.46°F   95% CI [0.28, 0.63]  (we forecast a touch low)
       → concentrated in the coastal stations: NYC −2.3, SFO −1.0, SEA +1.5
       → a diagnosed settlement-sensor issue, not a mystery; the scoped next fix.
```

## But is ~1.5°F actually good?

On its own it's just a number. Here's the yardstick — the same unseen 2024 days,
scored against honest baselines anyone can reproduce:

```
  Method                           Average error
  ──────────────────────────────   ─────────────
  Guess the seasonal normal          6.2°F    ← a "no-skill" baseline
  Guess yesterday's temperature      4.7°F
  Raw model, uncorrected             1.85°F
  OUR MODEL (calibrated)             1.79°F   ◀ ~3.4× sharper than no-skill
```

So the forecast is genuinely skilled — about **3.4× closer than guessing the
seasonal average**, in the range professional next-day forecasts achieve. One
honest note: the raw model is already good (1.85°F), so calibration adds only a
modest bump to the *average*. Its real job isn't shaving that average — it's
correcting the **specific stations** the market misreads and getting the
**probabilities** right, which is what actually prices a contract.

---

## Fine-tuning our confidence — the spread

The center of our forecast is accurate. The other half of a good probability is
the **spread** — how confident to be. Raw weather ensembles are *always* slightly
off on this; tightening them is a standard step every serious forecasting group
does. What's unusual is that **we actually measure ours** — logging 100+ live
ensemble members every day and checking them against real outcomes, which the free
bots never do.

The direction of the miss is the reassuring one: on very stable desert/tropical
cities we come out a touch **too cautious** (the three families disagree with each
other more than the weather actually moves), never overconfident. Being a little
too cautious just means we pass on some marginal trades and bet a little smaller —
it can never make us bet too big on a bad read. The dangerous failure mode,
overconfidence, is the one we've ruled out.

---

## A fire is not a trade — the vetting funnel

This is the part most people miss, and it's the core of how we actually trade.
Everything above produces a **baseline signal** — not a decision. Every signal
runs through three filters before any money moves:

```
  1. THE GATES        An 8-check stack: enough edge (scaled to how well the
                      city is validated), the model families agree, the forecast
                      is stable, and the edge isn't implausibly large (which
                      usually means calibration error, not real mispricing).

  2. EXTERNALITY VET  Each fire re-checked against sources the pricing engine
                      never used — climatology base rate, multi-model agreement,
                      a live market re-verify, a self-contradiction check.
                      Verdict: TRADE_SMALL / WATCH / PASS.

  3. HUMAN DEEP-DIVE  A person re-verifies the survivor through a DIFFERENT code
                      path than the one that priced it — never confirming a pick
                      with the same engine that produced it. Only then do we act.
```

The payoff is real — win-rate climbs as you move up the funnel:

```
  Raw / unvetted fires ........ 44% win   (the baseline firehose)
  WATCH (passed the vet) ...... 55% win
  TRADE_SMALL (vet says trade)  80% win   ← what we actually act on
  PASS (vet rejected) ......... 18% win   (correctly thrown out)
```

Two honest caveats always travel with those numbers: **TRADE_SMALL is only a
handful of picks so far** (n=5 — not yet statistically powered), and the returns
are struck at mid-price with **no fees or spread** (optimistic). What the funnel
*does* establish is the **shape**: vetting lifts the win-rate and dumps the losers.
That shape is the argument that the second check adds real value beyond the raw
signal.

---

## The risk layer — how we keep it controlled

Every bet is sized by the **Kelly criterion** — a mathematically optimal way to
size bets on your edge. We use **one quarter** of what Kelly recommends,
deliberately conservative, on a **$3,000 bankroll**.

```
  HARD LIMITS (on a $3,000 bankroll):
  ┌──────────────────────────────────────────────────────┐
  │  Per trade:    max 1.5% of bankroll     ≈ $45        │
  │  Per day:      circuit breaker at 7%    ≈ −$210      │
  │                → system stops trading for that day    │
  │  Correlation:  same-weather-system city bets are     │
  │                capped together          ≈ $150       │
  └──────────────────────────────────────────────────────┘

  A 30–40% day is structurally impossible by these rules —
  not "we'd try to stop it," the math makes it impossible.
```

---

## Where we stand

```
  WHAT WE KNOW (strong, out-of-sample):
  ─────────────────────────────────────
  Forecast skill:  1.79°F on 7,315 unseen 2024 days (historical OOS),
                   confirmed at 1.52°F on the recent ~59-day live window.
                   Small cool bias (+0.46°F) on coastal stations — diagnosed.

  WHAT'S HONEST ABOUT THE EDGE:
  ─────────────────────────────────────
  Raw / pooled signal:  the market currently prices BETTER than our raw
                        output (the day-ahead skill monitor is RED). That
                        firehose is the baseline, not the book we trade.
  Vetted subset:        the picks we actually act on (specific cities +
                        short-bracket NO structures) show a positive, local
                        edge — not holistic yet, and not fee-proven yet.

  WHAT'S NEXT:
  ─────────────────────────────────────
  → Coastal settlement-sensor fix (NYC/SFO/SEA) — highest leverage, diagnosed
  → Per-city spread calibration (EMOS) — built + measured, switch on after
    the live record proves it
  → Let the forward log accrue — the only cure for a small sample
  → The new price-path (mispricing) book — see STATE_OF_THE_MODEL Part V
```

---

## The timeline

```
  Today: 2026-08-11
    • ~59 days of live logging across 20 cities (100+ members/day)
    • Coastal fix diagnosed; ensemble engine is the production engine
    • Full statistical verification of the traded edge needs the forward
      log to keep accruing (~40 more days) — but careful, small trading
      begins earlier, as each niche's record earns it.
```

Edge in prediction markets isn't invented — it's validated forward, one day at a
time. We can't rush the calendar. What we *can* do is keep the model clean before
those days accrue, so we're not relearning mistakes in the sample we'll present as
our proof.

---

## The one-sentence version

> "The forecast is genuinely good — ~1.5°F average error across 20 cities. Raw,
> our probabilities roughly match or slightly trail the market; what we actually
> trade is the small vetted subset that clears an independent check, and that hits
> at a much higher rate. The edge is real in specific pockets, not holistic yet —
> a time-and-data problem — and the end goal is to automate the whole loop."
