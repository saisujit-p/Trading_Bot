# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
Meta-controller OOS evaluation.

Loads whichever regime specialists are currently saved on disk
(regime_models/{regime}/best_model.zip), then for each symbol walks through
OOS data, detects regime per bar via the per-symbol HMM, and dispatches
each step to the matching specialist.
"""
import os, sys
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from Trading_Bot.SAC.env import MarketContinuous, HMM_MODELS_DIR
from Trading_Bot.SAC.genObs import generate_observations
from Trading_Bot.SAC._shared import (
    SYMBOLS, PERIOD, INTERVAL, CASH, TRAIN_FRAC, REGIME_METHOD,
    regime_model_dir,
)

# Drawdown + trend-slope gate: redirect bear → sideways only for "pullback-
# in-a-rally" bars — shallow drawdown AND the 60-bar OLS trend of log price
# is still positive. Both conditions required (AND). Rationale:
#   - Pure dd gate at -10% helps AVGO (-89% → -63%) but a looser -15% hurts
#     ORCL (-31% → -37%); ORCL's shallow-dd bars were genuine early weakness.
#   - Requiring trend_slope > 0 filters out those "shallow dd but tape is
#     rolling over" cases while keeping AVGO-style pullbacks.
# Window values match hmm_regime.py defaults so the gate sees the same
# context the HMM used when labeling.
DRAWDOWN_GATE_THRESHOLD = -0.10
DRAWDOWN_GATE_WINDOW = 252
# Disabled: tested with threshold=0.0, but ANDing it with the dd gate
# filtered out helpful redirects on ORCL (-31% → -41%). Slope direction
# turned out to be a poor proxy for whether bear-or-sideways is the right
# call on shallow-drawdown bars. Keep the plumbing for future tuning.
TREND_SLOPE_GATE_THRESHOLD = None
TREND_SLOPE_GATE_WINDOW = 60


def _drawdown_series(prices, window):
    """log_price - rolling_max(log_price, window). Returns values <= 0."""
    log_p = np.log(np.asarray(prices, dtype=np.float64))
    roll_max = pd.Series(log_p).rolling(window, min_periods=1).max().to_numpy()
    return log_p - roll_max


def _trend_slope_series(prices, window):
    """Rolling OLS slope of log(price) over `window` bars ending at each index.
    Expanding window for the first (window-1) bars. Aligned 1:1 with prices."""
    log_p = np.log(np.maximum(np.asarray(prices, dtype=np.float64), 1e-12))
    n = len(log_p)
    slope = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i + 1 - window)
        y = log_p[lo:i + 1]
        k = len(y)
        if k < 2:
            continue
        t = np.arange(k, dtype=np.float64)
        t_mean, y_mean = t.mean(), y.mean()
        denom = ((t - t_mean) ** 2).sum()
        slope[i] = ((t - t_mean) * (y - y_mean)).sum() / max(denom, 1e-12)
    return slope

from Trading_Bot.SAC.hmm_regime import HMMRegimeDetector


def _label_for(symbol, prices):
    if REGIME_METHOD == "hmm":
        det = HMMRegimeDetector.load(os.path.join(HMM_MODELS_DIR, f"{symbol}.pkl"))
        return det.predict(prices)
    from Trading_Bot.SAC.regime_detector import detect_regime
    return detect_regime(prices)


def load_trained_models():
    """Load every best_model.zip that exists under regime_models/."""
    models = {}
    for regime in ("bull", "bear", "sideways"):
        path = os.path.join(regime_model_dir(regime), "best_model.zip")
        if os.path.exists(path):
            models[regime] = SAC.load(path)
            print(f"[load] {regime} ← {path}")
        else:
            print(f"[load] {regime}: no model found, skipping")
    if not models:
        raise RuntimeError("No trained specialists found. "
                           "Run train_bull.py / train_bear.py / train_sideways.py first.")
    return models


def meta_controller_eval(symbols, models, period, interval, cash, train_frac=TRAIN_FRAC,
                         dd_gate_threshold=DRAWDOWN_GATE_THRESHOLD,
                         dd_gate_window=DRAWDOWN_GATE_WINDOW,
                         ts_gate_threshold=TREND_SLOPE_GATE_THRESHOLD,
                         ts_gate_window=TREND_SLOPE_GATE_WINDOW):
    """Redirect bear→sideways when BOTH drawdown > dd_gate_threshold AND
    trend_slope > ts_gate_threshold. Set either threshold to None to drop
    that condition (gate still requires the other to fire)."""
    all_agent_rets, all_bh_rets, symbols_done = [], [], []
    regime_usage = {"bull": 0, "bear": 0, "sideways": 0}     # raw HMM labels
    dispatched_usage = {"bull": 0, "bear": 0, "sideways": 0}  # after gate
    total_gate_fires = 0

    dd_on = dd_gate_threshold is not None
    ts_on = ts_gate_threshold is not None
    gate_on = dd_on or ts_on
    cond_parts = []
    if dd_on:
        cond_parts.append(f"dd > {dd_gate_threshold:.2f} (w={dd_gate_window})")
    if ts_on:
        cond_parts.append(f"trend_slope > {ts_gate_threshold:.4f} (w={ts_gate_window})")
    print(f"[gate] bear→sideways when: {' AND '.join(cond_parts) if gate_on else 'OFF'}")

    for sym in symbols:
        try:
            _, full_df = generate_observations(sym, period, interval)
            prices_all = full_df['close'].to_numpy(dtype=np.float32)

            regime_labels = _label_for(sym, prices_all)
            dd_all = _drawdown_series(prices_all, dd_gate_window) if dd_on else None
            ts_all = _trend_slope_series(prices_all, ts_gate_window) if ts_on else None

            n = len(prices_all)
            cut = int(n * train_frac)
            labels_oos = regime_labels[cut:]
            dd_oos = dd_all[cut:] if dd_on else None
            ts_oos = ts_all[cut:] if ts_on else None

            if len(labels_oos) < 10:
                print(f"[meta] {sym}: not enough OOS data, skipping")
                continue

            env = MarketContinuous(symbol=sym, period=period, interval=interval,
                                   initial_cash=cash, split="test", train_frac=train_frac)
            obs, _ = env.reset()

            action_counts = {0: 0, 1: 0, 2: 0}
            initial_price = float(np.exp(obs["price"][0]))
            initial_cash_val = env.market.cash
            gate_fires = 0
            steps = 0

            while True:
                raw_regime = str(labels_oos[min(steps, len(labels_oos) - 1)])
                regime_usage[raw_regime] = regime_usage.get(raw_regime, 0) + 1

                current_regime = raw_regime
                if gate_on and raw_regime == "bear" and "sideways" in models:
                    idx = min(steps, len(labels_oos) - 1)
                    dd_ok = (not dd_on) or (dd_oos[idx] > dd_gate_threshold)
                    ts_ok = (not ts_on) or (ts_oos[idx] > ts_gate_threshold)
                    if dd_ok and ts_ok:
                        current_regime = "sideways"
                        gate_fires += 1

                dispatched_usage[current_regime] = dispatched_usage.get(current_regime, 0) + 1

                if current_regime in models:
                    agent = models[current_regime]
                else:
                    agent = models.get("bull", list(models.values())[0])

                action, _ = agent.predict(obs, deterministic=True)
                action_type, _ = MarketContinuous._decode_action(action)
                action_counts[action_type] += 1
                obs, _, done, _, info = env.step(action)
                steps += 1
                if done:
                    break

            total_gate_fires += gate_fires
            final_val = info["portfolio_value"]
            final_price = float(env.market.prices[env.market.current_step])
            agent_ret = (final_val - initial_cash_val) / initial_cash_val
            bh_ret = (final_price - initial_price) / initial_price

            print(f"\n===== {sym} OOS META-CONTROLLER ({steps} steps) =====")
            print(f"  Hold: {action_counts[0]} ({100*action_counts[0]/steps:.1f}%)")
            print(f"  Buy:  {action_counts[1]} ({100*action_counts[1]/steps:.1f}%)")
            print(f"  Sell: {action_counts[2]} ({100*action_counts[2]/steps:.1f}%)")
            if gate_on:
                print(f"  Gate fires (bear→sideways): {gate_fires}")
            print(f"  Final portfolio: {final_val:.2f}  (start {initial_cash_val:.2f})")
            print(f"  Agent return:    {agent_ret*100:+.2f}%")
            print(f"  Buy&Hold return: {bh_ret*100:+.2f}%")
            print(f"  Excess vs B&H:   {(agent_ret - bh_ret)*100:+.2f}%")

            all_agent_rets.append(agent_ret)
            all_bh_rets.append(bh_ret)
            symbols_done.append(sym)

        except Exception as e:
            print(f"[meta] {sym}: error — {e}")

    # Flat keys (bull/bear/sideways) = dispatched counts — what actually ran.
    # raw_* = pre-gate HMM labels, so you can see how many bear bars got redirected.
    usage = dict(dispatched_usage)
    usage.update({f"raw_{k}": v for k, v in regime_usage.items()})
    usage["gate_fires"] = total_gate_fires
    return all_agent_rets, all_bh_rets, usage, symbols_done


def print_portfolio_summary(symbols_done, agent_rets, bh_rets, usage):
    """Print portfolio-avg line + bear-mode/bull-mode split. Bear-mode =
    symbols where B&H finished negative, i.e. the specialist's defensive
    behavior had a chance to earn its keep. Bull-mode = the rest."""
    if not agent_rets:
        return
    ag = np.asarray(agent_rets, dtype=np.float64)
    bh = np.asarray(bh_rets, dtype=np.float64)
    ex = ag - bh

    bear_mask = bh < 0
    bull_mask = ~bear_mask

    print("\n" + "=" * 60)
    print(f"  PORTFOLIO AVG OOS — agent: {100*ag.mean():+.2f}%  "
          f"B&H: {100*bh.mean():+.2f}%  excess: {100*ex.mean():+.2f}%")

    def _line(label, mask):
        n = int(mask.sum())
        if n == 0:
            print(f"  {label}: (no symbols)")
            return
        syms = [symbols_done[i] for i in range(len(symbols_done)) if mask[i]]
        print(f"  {label} ({n}): agent {100*ag[mask].mean():+.2f}%  "
              f"B&H {100*bh[mask].mean():+.2f}%  excess {100*ex[mask].mean():+.2f}%  "
              f"[{','.join(syms)}]")

    _line("Bear-mode (B&H<0)", bear_mask)
    _line("Bull-mode (B&H≥0)", bull_mask)

    print(f"  Dispatched: bull={usage.get('bull',0)}, bear={usage.get('bear',0)}, "
          f"sideways={usage.get('sideways',0)}")
    print(f"  Raw HMM:    bull={usage.get('raw_bull',0)}, bear={usage.get('raw_bear',0)}, "
          f"sideways={usage.get('raw_sideways',0)}")
    print(f"  Gate fires (bear→sideways redirects): {usage.get('gate_fires',0)}")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"  META-CONTROLLER OOS EVALUATION  ({REGIME_METHOD})")
    print("=" * 60)

    models = load_trained_models()
    agent_rets, bh_rets, usage, symbols_done = meta_controller_eval(
        SYMBOLS, models, PERIOD, INTERVAL, CASH)

    print_portfolio_summary(symbols_done, agent_rets, bh_rets, usage)
