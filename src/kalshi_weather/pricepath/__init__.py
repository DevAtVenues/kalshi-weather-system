"""
Price-path (mispricing) book — Steps 3-5 machinery. DORMANT PILOT.

Status: the Step-2 GO/NO-GO gate (2026-08-11) returned NO-GO for live intraday
trading — convergence toward the model is real but ~0.5c/hr, far below the
~4c round-trip cost. This package is the full Steps 3-5 build kept in PILOT:

  Step 3  engine.py                      mark-to-market engine + exit policies
  Step 4  scripts/price_path_exit_bakeoff.py   policy bake-off over the panel
  Step 5  scripts/price_path_shadow.py   daily forward shadow book (paper only)

It runs daily (via daily_review), accumulating a forward, per-policy shadow
record from logged data — so if the economics ever clear (regime change,
tighter books), activation is a decision, not a build.

HARD GUARDRAILS (pinned by tests/test_pricepath.py and the smoke gate):
  - This package NEVER imports the exchange/executor and NEVER performs
    network I/O. It reads logged parquet/jsonl and writes only under
    data/analysis/price_path/.
  - There is deliberately NO activation flag here. Activating this book means
    consciously wiring engine decisions to live/executor.py behind its own
    interlock — a separate, explicit change that must clear the forward
    shadow record's block-bootstrap bar first (charter: both fill models).
"""
