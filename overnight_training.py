# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
Overnight full-pipeline training.

Trains all three regime specialists sequentially, then runs the meta-controller
OOS evaluation. Designed to be left running unattended — prints timestamped
phase banners and total elapsed time at the end.

Order (sideways → bear → bull) matches the current investigation priority:
sideways and bear are the ones that most need retraining under the new
directional HMM labels. Bull is last because it's the most stable.

Usage:
  python V2/SAC/overnight_training.py

Tee to a log file if you want a reviewable transcript:
  python V2/SAC/overnight_training.py 2>&1 | tee overnight_$(date +%Y%m%d_%H%M%S).log

Skip a phase by setting env vars:
  SKIP_HMM=1 SKIP_SIDEWAYS=1 python V2/SAC/overnight_training.py
"""
import os
import sys
import time
from datetime import datetime

from _shared import (
    SYMBOLS, PERIOD, INTERVAL, CASH,
    ensure_hmms_fitted, train_regime,
)


def banner(title):
    print("\n" + "#" * 72)
    print(f"#  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  {title}")
    print("#" * 72 + "\n", flush=True)


def phase(name, fn):
    """Run a phase with timing + a banner; return elapsed seconds."""
    banner(f"START — {name}")
    t0 = time.time()
    fn()
    elapsed = time.time() - t0
    banner(f"DONE  — {name}  ({elapsed/60:.1f} min)")
    return elapsed


def main():
    overall_t0 = time.time()
    banner(f"OVERNIGHT TRAINING PIPELINE  —  started {datetime.now().isoformat()}")

    totals = {}

    if not os.environ.get("SKIP_HMM"):
        totals["hmm_fit"] = phase("HMM fit (cached, skips if present)",
                                  ensure_hmms_fitted)

    if not os.environ.get("SKIP_SIDEWAYS"):
        totals["sideways"] = phase(
            "TRAIN sideways (exposure 0.80, 1M steps)",
            lambda: train_regime(
                regime="sideways", exposure_target=0.80, timesteps=1_000_000,
                description="Sideways agent: lite-bull, ride mild drift"))

    if not os.environ.get("SKIP_BEAR"):
        totals["bear"] = phase(
            "TRAIN bear (exposure 0.20, 1M steps)",
            lambda: train_regime(
                regime="bear", exposure_target=0.20, timesteps=1_000_000,
                description="Bear agent: defensive, preserve capital"))

    if not os.environ.get("SKIP_BULL"):
        totals["bull"] = phase(
            "TRAIN bull (exposure 0.95, 2M steps)",
            lambda: train_regime(
                regime="bull", exposure_target=0.95, timesteps=2_000_000,
                description="Bull agent: stay invested, ride momentum"))

    if not os.environ.get("SKIP_META"):
        banner("START — META-CONTROLLER OOS EVAL")
        t0 = time.time()
        from eval_meta import (
            load_trained_models, meta_controller_eval, print_portfolio_summary)
        models = load_trained_models()
        agent_rets, bh_rets, usage, symbols_done = meta_controller_eval(
            SYMBOLS, models, PERIOD, INTERVAL, CASH)
        totals["meta_eval"] = time.time() - t0
        print_portfolio_summary(symbols_done, agent_rets, bh_rets, usage)
        banner(f"DONE  — META-CONTROLLER OOS EVAL  ({totals['meta_eval']/60:.1f} min)")

    total_elapsed = time.time() - overall_t0
    banner("PIPELINE COMPLETE")
    print(f"  Total wall time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)")
    print(f"  Per-phase breakdown:")
    for name, secs in totals.items():
        print(f"    {name:>12s}: {secs/60:>6.1f} min")
    print(f"  Finished at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
