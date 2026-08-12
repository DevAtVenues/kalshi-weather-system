# Timing & Edge-Realization Analysis — 2026-07-20

**Data:** 71,244 graded signal snapshots with market price at observation time
(data/outcomes/signal_outcomes.jsonl), settlement dates 2026-07-05 → 2026-07-19
(**15 days with price coverage — all mid-July, one summer regime**). Every P&L
figure = one contract traded in the signal's direction at the observed mid,
minus taker fee. 95% CIs block-bootstrapped over settlement_date.
Reproduce/re-run: `.venv/bin/python scripts/timing_report.py`.

## Findings

### 1. Model edge is real information (monotone in realized P&L)
| logged edge | realized P&L/contract | CI |
|---|---|---|
| 0–5¢ | +0.5¢ | [−0.9, +1.9] |
| 5–10¢ | +2.3¢ | [+1.1, +4.2] |
| 10–15¢ | +3.4¢ | [+1.5, +6.7] |
| 15–25¢ | +8.8¢ | [+5.9, +13.2] |
| ≥25¢ | +12.5¢ | [+8.6, +17.2] |

Realized P&L ≈ ⅓–½ of logged edge (the rest is model error), but strictly
increasing: **bigger logged edge ⇒ bigger realized edge**. Sizing should scale
with the edge bucket, not treat all fires equally.

### 2. The ensemble front-runs the market (updates morning-edge FIRST READ)
On 708 ticker-events where the ensemble disagreed with the market by ≥5¢ and we
observed ≥6h of subsequent trajectory: the market moved **toward** our number
63.7% [59, 68] of the time, and the outcome sided with our model 72.3% [66, 77].
The retired rule source shows no such effect (49% — coin flip). The earlier
"we don't front-run" read (35% beat) was 5 days of mixed-source data.
Implication: **when the ensemble disagrees, place early as maker** — waiting
means paying the convergence we predicted.

### 3. Where the P&L lives: BUY_NO between, and almost nowhere else
| species (edge ≥10¢) | P&L/contract | CI | win% |
|---|---|---|---|
| **BUY_NO between** | **+15.7¢** | [+13.5, +18.5] | 75.7% |
| BUY_YES greater | +11.4¢ | [−3.5, +31.9] | 25.3% |
| BUY_NO greater / less | −5¢ | not distinguishable from 0 | |
| BUY_YES between | −7.1¢ | [−9.5, −3.7] | 6.8% |
| BUY_YES less | −6.6¢ | [−11.2, −1.3] | 4.1% |

BUY_YES between/less are **confirmed money-losers** even at ≥10¢ logged edge —
the YES over-prediction problem survives in the realized P&L. (P1.7's BUY_YES
suppression gate is vindicated; note BUY_YES *greater* is the only YES species
with a positive point estimate — wide CI, watch not trade.)

### 4. Timing windows (edge ≥10¢, local wall-clock at the station)
| window | P&L/contract | CI |
|---|---|---|
| settlement-day 12:00–18:00 (afternoon) | +12.5¢ | [+6.9, +16.0] |
| prev-day 06:00–12:00 (morning) | +9.1¢ | [+6.6, +14.0] |
| prev-day 12:00–18:00 (afternoon) | +9.0¢ | [+8.0, +10.5] |
| settlement-day morning + overnight | +8.5¢ | wide |
| prev-day evening | +5.4¢ | [+3.5, +8.2] |
| settlement-day after 18:00 | ~0 | dead |

Ensemble-only, the settlement-day afternoon window is **+20.4¢ [12.6, 24.8]**
(shorting brackets the realized high has left behind) and ensemble BUY_NO
intraday overall is +22.3¢ vs +13.5¢ day-ahead. Best-entry study on 982 winners:
cheapest NO entry clustered prev-day morning (37%) and settlement-day
06:00–12:00 — i.e., **enter day-ahead when the ensemble first disagrees; add
intraday noon–18:00 as brackets die**. Caution: same-day picks are Gate 5
watch-only; that falsification was about BUY_YES warming upside — this data
argues for forward-validating same-day **BUY_NO** separately.

### 5. Stop-losses destroy the edge — hold to settlement
Simulated ensemble BUY_NO-between entries (n=540): hold-to-settlement
**+16.0¢ [+13.8, +18.3]**. Adverse-move stops at +10/15/20/25/30¢ collapse it
to −0.4…+3.8¢ — intraday mid swings stop out mostly eventual winners
(173 of 248 stops at +10¢ would have won). Losers do drift against us (95% had
rising YES mids) but not separably from winners' noise at snapshot cadence.
**Risk is managed by sizing and correlation caps, never by exits.**

### 6. Execution: spreads are not the constraint
14 days of orderbook snapshots, mids 10–90¢: median spread 1–3¢ at every hour;
worst at 10:00 ET (new-market repricing, mean 5.5¢, only 72% ≤5¢). Maker entry
is nearly free; taker is fine when edge ≥10¢. Avoid initiating 10:00–10:30 ET.

### 7. The binding constraint is UPTIME, not math
Jul 19: orderbook logger had continuous coverage only ~12:00–14:00 ET; run-live
completed 3 of ~48 intended cycles. 9 of 15 settlement dates have <10 distinct
run timestamps. The laptop sleeping (lid closed on battery) blinds both
proprietary datasets and delays picks through the exact windows above. The
Jul 6–9 full-cadence burst is what made this analysis possible at all.

## Sizing (bankroll now $3,000, updated in data/risk_config.json)
Quarter-Kelly on the workhorse species (win ~76%, NO cost ~55¢) ≈ 8–11% of
bankroll — the caps bind first, which is correct at n=15 days. Ladder:
- **A-trades** (ensemble BUY_NO between, edge ≥15¢, prime window): full
  per-trade cap ($45).
- **B-trades** (edge 10–15¢, or off-window): half cap (~$20–25).
- Everything else: no trade. Correlated same-system exposure ≤ $150 (5%).

## Caveats
- 15 July days, one regime (summer heat, mostly one CONUS pattern sequence);
  effective n is much smaller than row counts suggest. The short-bracket edge
  was already known to be mostly structural and 2025-heavy.
- Mid-fill assumption (±1–2¢ per §6); no queue/adverse-selection modeling.
- Snapshot cadence is irregular (§7) — trajectory stats undersample overnight.

## Actions taken 2026-07-20
- `scripts/timing_report.py` — re-runs all of the above on growing data.
- `data/risk_config.json` bankroll 2000 → 3000.

## Proposed next (needs owner sign-off)
1. Fix uptime (keep plugged in + pmset, or move loggers to a small VM).
2. Wire a lead-window multiplier + species filter into action_score.
3. Forward-validate same-day BUY_NO as its own Gate-5 track.
4. Re-run `timing_report.py` weekly; distrust any table whose CI includes 0.
