# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
Orchestrator: runs the full pipeline end-to-end.

  1. Fit per-symbol HMMs (cached — skips symbols already persisted).
  2. Train bull / bear / sideways specialists sequentially.
  3. Run meta-controller OOS evaluation.

To iterate on a single specialist instead, run its dedicated scripts:
  python bull/train.py       python bull/eval.py
  python bear/train.py       python bear/eval.py
  python sideways/train.py   python sideways/eval.py
  python eval_meta.py
"""
from _shared import ensure_hmms_fitted, train_regime

ensure_hmms_fitted()

train_regime(regime="bull",     exposure_target=0.95, timesteps=2_000_000,
             description="Bull agent: stay invested, ride momentum")
train_regime(regime="bear",     exposure_target=0.20, timesteps=1_000_000,
             description="Bear agent: defensive, preserve capital")
train_regime(regime="sideways", exposure_target=0.80, timesteps=1_000_000,
             description="Sideways agent: lite-bull, ride mild drift")

print("\n" + "=" * 60)
print("  ALL REGIME AGENTS TRAINED — running meta-controller eval")
print("=" * 60)

from _shared import SYMBOLS, PERIOD, INTERVAL, CASH
from eval_meta import load_trained_models, meta_controller_eval, print_portfolio_summary

models = load_trained_models()
agent_rets, bh_rets, usage, symbols_done = meta_controller_eval(
    SYMBOLS, models, PERIOD, INTERVAL, CASH)
print_portfolio_summary(symbols_done, agent_rets, bh_rets, usage)
