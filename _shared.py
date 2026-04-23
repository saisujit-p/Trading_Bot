# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
Shared config and helpers for the three regime-specialist training scripts
(train_bull.py, train_bear.py, train_sideways.py) and eval_meta.py.
"""
import os, sys, traceback
import numpy as np

from env import HMM_MODELS_DIR
from genObs import generate_observations

# Each regime has its own folder under V2/SAC/: models save there directly,
# tensorboard/eval logs go under {regime}/logs/. Resolved from this file's
# path so runs land in the right place regardless of CWD.
_SAC_DIR = os.path.dirname(os.path.abspath(__file__))


def regime_dir(regime):
    return os.path.join(_SAC_DIR, regime)


def regime_model_dir(regime):
    """Where best_model.zip and sac_{regime}_final.zip live."""
    return regime_dir(regime)


def regime_log_dir(regime):
    """Where tensorboard events + eval callback logs go."""
    return os.path.join(regime_dir(regime), "logs")

from hmm_regime import HMMRegimeDetector


# ===== GLOBAL CONFIG =====
SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "AVGO", "ORCL", "AMD", "NFLX", "ADBE", "CRM", "INTC", "QCOM",
]
PERIOD, INTERVAL, CASH = "10y", "1d", 100
SEED = 42
REGIME_METHOD = "hmm"            # "hmm" or "sma"
HMM_N_STATES = 3                 # bear / sideways / bull
TRAIN_FRAC = 0.8

EVAL_FREQ = 10_000
PATIENCE_EVALS = 8


# Per-regime reward shaping. exposure_penalty_coef is multiplied by
# abs(exposure - target) each step; tx_cost_weight multiplies the per-trade
# cost penalty. Tuned so bear punishes over-investment + churn hardest;
# sideways now mirrors bull's churn weight since its raised target (0.80)
# makes it behave more like a lite-bull than a hedge.
REGIME_REWARD = {
    "bull":     {"exposure_penalty_coef": 0.01, "tx_cost_weight": 1.0},
    "bear":     {"exposure_penalty_coef": 0.05, "tx_cost_weight": 2.0},
    "sideways": {"exposure_penalty_coef": 0.03, "tx_cost_weight": 1.0},
}


def linear_schedule(initial_lr: float, final_lr: float = 1e-5):
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule


def fit_hmms(symbols=SYMBOLS, period=PERIOD, interval=INTERVAL,
             train_frac=TRAIN_FRAC, n_states=HMM_N_STATES, seed=SEED,
             force=False):
    """
    Fit one HMM per symbol on its training-portion price history.
    Skips symbols that already have a persisted model unless force=True.
    """
    os.makedirs(HMM_MODELS_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print(f"  FITTING HMMs  (n_states={n_states}, train_frac={train_frac})")
    print("=" * 60)

    for sym in symbols:
        out_path = os.path.join(HMM_MODELS_DIR, f"{sym}.pkl")
        if not force and os.path.exists(out_path):
            print(f"[hmm] {sym}: cached, skipping (force=False)")
            continue
        try:
            _, full_df = generate_observations(sym, period, interval)
            prices_full = full_df['close'].to_numpy(dtype=np.float64)
            n = len(prices_full)
            cut = int(n * train_frac)
            prices_train = prices_full[:cut]
            if len(prices_train) < 300:
                print(f"[hmm] {sym}: only {len(prices_train)} train bars, skipping")
                continue
            det = HMMRegimeDetector(n_states=n_states, seed=seed)
            det.fit(prices_train)
            det.save(out_path)
            print(f"[hmm] {sym} fitted  ({len(prices_train)} train bars)")
            det.describe()
        except Exception as e:
            print(f"[hmm] {sym}: fit failed — {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stdout)


def ensure_hmms_fitted():
    """Call at the top of each train_*.py. Only fits symbols missing a .pkl."""
    if REGIME_METHOD == "hmm":
        fit_hmms(force=False)


def train_regime(regime, exposure_target, timesteps, description):
    """
    Train a single SAC specialist on regime-filtered data. Saves best-by-eval
    checkpoint to regime_models/{regime}/best_model.zip and final weights to
    regime_models/{regime}/sac_{regime}_final.zip.
    """
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.monitor import Monitor
    from env import MarketContinuous

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    reward_cfg = REGIME_REWARD.get(regime, {"exposure_penalty_coef": 0.02, "tx_cost_weight": 1.0})

    print("\n" + "=" * 60)
    print(f"  TRAINING {regime.upper()} AGENT  ({REGIME_METHOD})")
    print(f"  {description}")
    print(f"  exposure_target={exposure_target}, timesteps={timesteps:,}")
    print(f"  reward: exposure_penalty={reward_cfg['exposure_penalty_coef']}, "
          f"tx_cost_weight={reward_cfg['tx_cost_weight']}")
    print("=" * 60)

    save_dir = regime_model_dir(regime)
    log_dir = regime_log_dir(regime)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_env = MarketContinuous(
        symbols=SYMBOLS, period=PERIOD, interval=INTERVAL,
        initial_cash=CASH, split="train", train_frac=TRAIN_FRAC,
        regime=regime, exposure_target=exposure_target,
        regime_method=REGIME_METHOD, **reward_cfg)

    try:
        eval_env = Monitor(MarketContinuous(
            symbols=SYMBOLS, period=PERIOD, interval=INTERVAL,
            initial_cash=CASH, split="test", train_frac=TRAIN_FRAC,
            regime=regime, exposure_target=exposure_target,
            regime_method=REGIME_METHOD, **reward_cfg))
    except RuntimeError:
        print(f"[warn] {regime}: not enough OOS regime segments, using train for eval")
        eval_env = Monitor(MarketContinuous(
            symbols=SYMBOLS, period=PERIOD, interval=INTERVAL,
            initial_cash=CASH, split="train", train_frac=TRAIN_FRAC,
            regime=regime, exposure_target=exposure_target,
            regime_method=REGIME_METHOD, **reward_cfg))

    n_eval_episodes = min(len(train_env._pool), 15)

    early_stop = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=PATIENCE_EVALS, min_evals=5, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        callback_after_eval=early_stop,
        verbose=1,
    )

    model = SAC(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        buffer_size=500_000,
        learning_starts=1_000,
        batch_size=512,
        train_freq=4,
        gradient_steps=4,
        ent_coef="auto",
        learning_rate=linear_schedule(3e-4),
        gamma=0.99,
        tau=0.005,
        tensorboard_log=log_dir,
        device="cuda",
        seed=SEED,
    )

    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=eval_callback,
    )
    model.save(os.path.join(save_dir, f"sac_{regime}_final"))
    print(f"\n[{regime}] training complete. "
          f"best_model.zip and sac_{regime}_final.zip saved to {save_dir}/")


def eval_regime(regime, exposure_target, use_final=False):
    """
    Evaluate a trained regime specialist on regime-filtered OOS test data.
    Walks every segment in the test pool deterministically, reports per-segment
    action distribution + return vs buy-and-hold, and aggregates.
    """
    from stable_baselines3 import SAC
    from env import MarketContinuous

    model_name = f"sac_{regime}_final.zip" if use_final else "best_model.zip"
    model_path = os.path.join(regime_model_dir(regime), model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Train the {regime} specialist first: "
            f"python train_{regime}.py"
        )

    print("\n" + "=" * 60)
    print(f"  EVAL {regime.upper()} SPECIALIST on OOS {regime} segments")
    print(f"  model: {model_path}")
    print("=" * 60)

    model = SAC.load(model_path)

    reward_cfg = REGIME_REWARD.get(regime, {"exposure_penalty_coef": 0.02, "tx_cost_weight": 1.0})

    env = MarketContinuous(
        symbols=SYMBOLS, period=PERIOD, interval=INTERVAL,
        initial_cash=CASH, split="test", train_frac=TRAIN_FRAC,
        regime=regime, exposure_target=exposure_target,
        regime_method=REGIME_METHOD, **reward_cfg)

    pool = env._pool
    agent_rets, bh_rets, avg_exposures = [], [], []
    total_counts = {0: 0, 1: 0, 2: 0}
    total_steps = 0
    total_exposure_sum = 0.0  # step-weighted numerator for overall avg exposure

    for i in range(len(pool)):
        # Force deterministic segment selection by swapping in directly.
        sym_id, prices, features = pool[i]
        from tradingsys import SimpleTradingEnv
        env.current_symbol = sym_id
        env.market = SimpleTradingEnv(prices, features,
                                      initial_cash=env._initial_cash,
                                      exposure_target=env._exposure_target,
                                      exposure_penalty_coef=env._exposure_penalty_coef,
                                      tx_cost_weight=env._tx_cost_weight)
        obs = env.market.reset()

        counts = {0: 0, 1: 0, 2: 0}
        initial_price = float(np.exp(obs["price"][0]))
        initial_cash_val = env.market.cash
        steps = 0
        info = {}
        exposure_sum = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action_type, _ = MarketContinuous._decode_action(action)
            counts[action_type] += 1
            obs, _, done, _, info = env.step(action)
            steps += 1
            # Realized exposure after this step: equity / total portfolio value.
            cur_price = float(env.market.prices[env.market.current_step])
            port_val = env.market.cash + env.market.shares * cur_price
            exposure_sum += (env.market.shares * cur_price) / max(port_val, 1e-8)
            if done:
                break

        final_val = info["portfolio_value"]
        final_price = float(env.market.prices[env.market.current_step])
        agent_ret = (final_val - initial_cash_val) / initial_cash_val
        bh_ret = (final_price - initial_price) / initial_price
        avg_exposure = exposure_sum / max(steps, 1)

        print(f"\n----- {sym_id} ({steps} bars) -----")
        print(f"  Hold: {counts[0]} ({100*counts[0]/steps:.1f}%)  "
              f"Buy: {counts[1]} ({100*counts[1]/steps:.1f}%)  "
              f"Sell: {counts[2]} ({100*counts[2]/steps:.1f}%)")
        print(f"  Avg exposure: {avg_exposure:.2f} (target {exposure_target:.2f})")
        print(f"  Agent: {agent_ret*100:+.2f}%   B&H: {bh_ret*100:+.2f}%   "
              f"Excess: {(agent_ret-bh_ret)*100:+.2f}%")

        agent_rets.append(agent_ret)
        bh_rets.append(bh_ret)
        avg_exposures.append(avg_exposure)
        for k in counts:
            total_counts[k] += counts[k]
        total_steps += steps
        total_exposure_sum += exposure_sum

    if agent_rets:
        overall_exposure = total_exposure_sum / max(total_steps, 1)
        bh_avg = float(np.mean(bh_rets))
        # Exposure-adjusted benchmark: what overall_exposure × B&H would earn.
        expected_from_exposure = overall_exposure * bh_avg
        print("\n" + "=" * 60)
        print(f"  {regime.upper()} AGG ({len(agent_rets)} segments, {total_steps} bars)")
        print(f"  Hold: {100*total_counts[0]/total_steps:.1f}%  "
              f"Buy: {100*total_counts[1]/total_steps:.1f}%  "
              f"Sell: {100*total_counts[2]/total_steps:.1f}%")
        print(f"  Avg exposure: {overall_exposure:.2f} (target {exposure_target:.2f})")
        print(f"  Agent avg: {100*np.mean(agent_rets):+.2f}%   "
              f"B&H avg: {100*bh_avg:+.2f}%   "
              f"Excess: {100*(np.mean(agent_rets)-bh_avg):+.2f}%")
        print(f"  Exposure-adjusted bench ({overall_exposure:.2f}×B&H): "
              f"{100*expected_from_exposure:+.2f}%   "
              f"vs-bench: {100*(np.mean(agent_rets)-expected_from_exposure):+.2f}%")
        print("=" * 60)
