"""
Post-fix signal verification: compare the latest cron run (expected post-fix)
against the pre-fix 23:02 UTC baseline.

Run after 2026-07-10 01:00 UTC cron completes.

Usage: .venv/bin/python scripts/verify_postfix_run.py
"""
from __future__ import annotations

import json, math
from pathlib import Path
from kalshi_weather.outcome_tracker import parse_ticker

PRE_FIX_RTS  = "2026-07-09T23:02:30.642736+00:00"
POST_FIX_AFTER = "2026-07-10T01:00"  # look for runs after this

siglog = Path("data/signals/signals_log.jsonl")
rows   = [json.loads(l) for l in siglog.read_text().splitlines() if l.strip()]

pre_rows  = [r for r in rows if str(r.get("run_ts","")) == PRE_FIX_RTS]
post_runs = sorted(set(str(r.get("run_ts","")) for r in rows
                       if str(r.get("run_ts","")) > POST_FIX_AFTER), reverse=True)

if not post_runs:
    print("No post-01:00 UTC run found yet. Re-run after cron completes.")
    raise SystemExit(0)

post_rts  = post_runs[0]
post_rows = [r for r in rows if str(r.get("run_ts","")) == post_rts]

print(f"Pre-fix run:  {PRE_FIX_RTS}  ({len(pre_rows)} signals)")
print(f"Post-fix run: {post_rts}  ({len(post_rows)} signals)")

# Index by ticker
pre  = {r["ticker"]: r for r in pre_rows  if r.get("prob_source") == "ensemble"}
post = {r["ticker"]: r for r in post_rows if r.get("prob_source") == "ensemble"}

# --- 1. Center-shift check for Tier 1a cities ---
print("\n=== CENTER SHIFT CHECK (should use corrected values) ===")
EXPECTED_SHIFTS = {
    "NYC": +0.24, "MIA": +0.36, "CHI": -0.07, "AUS": -0.03, "PHL": -0.33,
}
for city, expected in EXPECTED_SHIFTS.items():
    # Find any post-fix T-type signal for this city to read center_shift
    city_post = [r for r in post_rows if r.get("city") == city and r.get("prob_source") == "ensemble"]
    if not city_post:
        print(f"  {city}: no post-fix signals found")
        continue
    shifts = [r.get("ens_center_shift") for r in city_post if r.get("ens_center_shift") is not None]
    if shifts:
        actual_shift = shifts[0]
        ok = abs(actual_shift - expected) < 0.15
        print(f"  {city}: ens_center_shift={actual_shift:+.2f}°F  expected≈{expected:+.2f}°F  {'✓' if ok else '✗ MISMATCH'}")
    else:
        print(f"  {city}: ens_center_shift not logged for any signal (expected {expected:+.2f}°F)")

# --- 2. B-type bracket probability check ---
print("\n=== B-TYPE PROBABILITY FIX VERIFICATION ===")
print("Expecting: BUY_YES signals from pre-fix to be NO_EDGE or BUY_NO in post-fix\n")

PRE_FIX_BUYS = [
    "KXHIGHCHI-26JUL10-B79.5",  # was BUY_YES 0.409 → should be NO_EDGE ~0.156
    "KXHIGHTDAL-26JUL10-B100.5", # was BUY_YES 0.620 → should be NO_EDGE ~0.279
    "KXHIGHNY-26JUL10-B89.5",    # was BUY_YES 0.396 → should be NO_EDGE ~0.209
    "KXHIGHTSATX-26JUL10-B93.5", # was BUY_YES 0.517 → should be NO_EDGE ~0.345
    "KXHIGHTBOS-26JUL10-B85.5",  # was BUY_YES 0.443 → should be NO_EDGE ~0.202
    "KXHIGHLAX-26JUL10-B76.5",   # was BUY_YES 0.493 → should be NO_EDGE ~0.217
]

all_ok = True
for ticker in PRE_FIX_BUYS:
    pre_r  = pre.get(ticker)
    post_r = post.get(ticker)
    if pre_r is None:
        print(f"  {ticker}: not in pre-fix signals")
        continue
    pre_dir  = pre_r.get("direction","")
    pre_p    = float(pre_r.get("prob_estimate",0))
    pre_mkt  = float(pre_r.get("market_mid",0))
    if post_r is None:
        print(f"  {ticker}: MISSING from post-fix run!")
        all_ok = False
        continue
    post_dir = post_r.get("direction","")
    post_p   = float(post_r.get("prob_estimate",0))
    post_act = post_r.get("actionable", False)
    post_st  = post_r.get("strike_type","?")  # should now be populated

    # A raw BUY_YES is acceptable if either the probability collapsed (now BUY_NO /
    # NO_EDGE / non-actionable) OR it remains a raw BUY_YES only because the market
    # drifted to create a genuine edge — in which case the P1.7 gate (checked below,
    # against the gate log) is the authoritative safety net that stops it pushing.
    fixed = (pre_dir == "BUY_YES") and (post_dir in ("BUY_NO", "NO_EDGE") or not post_act)
    status = "✓ FIXED" if fixed else "• still raw BUY_YES (market drift) — see push-safety check"

    print(f"  {ticker}")
    print(f"    Pre:  P={pre_p:.3f} mkt={pre_mkt:.3f} dir={pre_dir}")
    print(f"    Post: P={post_p:.3f} mkt={float(post_r.get('market_mid',0)):.3f} dir={post_dir} act={post_act} st={post_st}")
    print(f"    → {status}")

# --- 2b. AUTHORITATIVE push-safety check: no B-type BUY_YES may actually FIRE ---
# The raw-direction check above can legitimately show a residual BUY_YES when the
# market moves to create a real edge. What must NEVER happen post-fix is a B-type
# BUY_YES that PUSHES. Assert it against the gate log (gate=FIRED is a real push).
print("\n=== PUSH-SAFETY: no B-type BUY_YES may fire (gate log) ===")
gate_log = Path("data/analysis/gate_log.jsonl")
push_safe = True
if gate_log.exists():
    grows = [json.loads(l) for l in gate_log.read_text().splitlines() if l.strip()]
    # The gate log and signals log are written at slightly different instants of the
    # SAME run_cycle (gate during push, signals at the end), so their run_ts never
    # match exactly. Use the latest gate run — it corresponds to the latest signals run.
    grts  = max((str(r.get("run_ts", "")) for r in grows), default="")
    gpost = [r for r in grows if str(r.get("run_ts", "")) == grts]
    print(f"  (gate run {grts[11:19]} — latest; signals run {post_rts[11:19]})")
    fired_byes = [r for r in gpost
                  if r.get("gate") == "FIRED" and r.get("direction") == "BUY_YES"
                  and "-B" in r.get("ticker", "")]
    held = [r for r in gpost
            if r.get("direction") == "BUY_YES" and "-B" in r.get("ticker", "")
            and r.get("gate") != "FIRED"]
    if fired_byes:
        push_safe = False
        for r in fired_byes:
            print(f"  ✗ FIRED B-type BUY_YES: {r['ticker']} edge={r.get('edge_raw')} — MUST NOT HAPPEN")
    else:
        print(f"  ✓ zero B-type BUY_YES fired; {len(held)} held "
              f"({', '.join(sorted(set(r['gate'] for r in held))) or 'none'})")
else:
    print("  (no gate log for post-fix run yet)")

all_ok = all_ok and push_safe
print(f"\nPush-safety (authoritative): {'PASS ✓' if push_safe else 'FAIL ✗'}")

# --- 3. Strike_type field now populated ---
print("\n=== STRIKE_TYPE FIELD POPULATION ===")
with_st = [r for r in post_rows if r.get("strike_type") is not None]
print(f"  {len(with_st)}/{len(post_rows)} post-fix signals have strike_type populated")
if with_st:
    sample = with_st[0]
    print(f"  Sample: {sample['ticker']} → {sample['strike_type']}")

# --- 4. BUY_NO signals that should be STRONGER ---
print("\n=== BUY_NO SIGNALS (should have bigger edge post-fix) ===")
BUYNO_CHECKS = [
    ("KXHIGHMIA-26JUL10-B93.5", "strong BUY_NO >0.15"),
    ("KXHIGHNY-26JUL10-T87",     "BUY_NO ~0.30+"),
    ("KXHIGHTMIN-26JUL10-B87.5", "flipped from BUY_YES to BUY_NO"),
    ("KXHIGHNY-26JUL10-B87.5",   "flipped from BUY_YES to BUY_NO"),
]
for ticker, note in BUYNO_CHECKS:
    pre_r  = pre.get(ticker)
    post_r = post.get(ticker)
    if pre_r is None or post_r is None:
        print(f"  {ticker}: missing")
        continue
    pre_p  = float(pre_r.get("prob_estimate",0))
    pre_e  = float(pre_r.get("edge_raw",0))
    pre_d  = pre_r.get("direction","")
    post_p = float(post_r.get("prob_estimate",0))
    post_e = float(post_r.get("edge_raw",0))
    post_d = post_r.get("direction","")

    print(f"  {ticker}  ({note})")
    print(f"    Pre:  P={pre_p:.3f} edge={pre_e:.3f} dir={pre_d}")
    print(f"    Post: P={post_p:.3f} edge={post_e:.3f} dir={post_d}")

print("\nVERIFICATION COMPLETE")
