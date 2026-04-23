# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
import os, sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from tradingsys import SimpleTradingEnv
from genObs import generate_observations

from regime_detector import detect_regime as detect_regime_sma, split_by_regime
from hmm_regime import HMMRegimeDetector


# Default directory where per-symbol HMMs are persisted
HMM_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hmm_models")


def _label_prices(symbol, prices, method="sma"):
    """
    Dispatch regime labeling to the chosen detector.
      - "sma": uses regime_detector.detect_regime (stateless).
      - "hmm": loads V2/SAC/hmm_models/{symbol}.pkl and decodes the series.
    """
    if method == "sma":
        return detect_regime_sma(prices)
    if method == "hmm":
        path = os.path.join(HMM_MODELS_DIR, f"{symbol}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"HMM model for {symbol} not found at {path}. "
                "Fit HMMs first via training.py's fit_hmms() step."
            )
        det = HMMRegimeDetector.load(path)
        return det.predict(prices)
    raise ValueError(f"unknown regime method: {method}")


def _load_symbol(symbol, period, interval, train_frac, split, start=None, end=None):
    if start is not None:
        import pandas as pd
        warm_start = (pd.Timestamp(start) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
        obs_df, full_df = generate_observations(symbol, period, interval,
                                                start=warm_start, end=end)
        obs_df = obs_df.loc[obs_df.index >= pd.Timestamp(start, tz=obs_df.index.tz)]
    else:
        obs_df, full_df = generate_observations(symbol, period, interval, start=start, end=end)

    prices_full = full_df.loc[obs_df.index, 'close'].to_numpy(dtype=np.float32)
    features_full = obs_df.to_numpy(dtype=np.float32)

    n = len(prices_full)
    cut = int(n * train_frac)
    if split == "train":
        prices, features = prices_full[:cut], features_full[:cut]
    elif split == "test":
        prices, features = prices_full[cut:], features_full[cut:]
    else:
        prices, features = prices_full, features_full
    return prices, features, list(obs_df.columns)


def _load_symbol_regime(symbol, period, interval, train_frac, split,
                        target_regime, min_segment_length=30,
                        start=None, end=None, regime_method="sma"):
    """
    Load data for a symbol, filtered to contiguous segments of a market regime.
    Uses SMA or HMM detector depending on `regime_method`.
    """
    if start is not None:
        import pandas as pd
        warm_start = (pd.Timestamp(start) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
        obs_df, full_df = generate_observations(symbol, period, interval,
                                                start=warm_start, end=end)
        obs_df = obs_df.loc[obs_df.index >= pd.Timestamp(start, tz=obs_df.index.tz)]
    else:
        obs_df, full_df = generate_observations(symbol, period, interval, start=start, end=end)

    prices_all = full_df.loc[obs_df.index, 'close'].to_numpy(dtype=np.float32)
    features_all = obs_df.to_numpy(dtype=np.float32)

    # Label regimes on FULL series (needs complete history for accurate SMAs /
    # consistent HMM state sequence).
    regime_labels = _label_prices(symbol, prices_all, method=regime_method)

    n = len(prices_all)
    cut = int(n * train_frac)
    if split == "train":
        prices = prices_all[:cut]
        features = features_all[:cut]
        labels = regime_labels[:cut]
    elif split == "test":
        prices = prices_all[cut:]
        features = features_all[cut:]
        labels = regime_labels[cut:]
    else:
        prices, features, labels = prices_all, features_all, regime_labels

    segments = split_by_regime(prices, features, labels, target_regime, min_segment_length)
    return segments, list(obs_df.columns)


class MarketContinuous(gym.Env):
    """
    Continuous-action wrapper around SimpleTradingEnv, fed by the
    feature matrix from genObs.generate_observations.

    If `symbols` is given, each reset() randomly picks one symbol from the
    pool — this forces the policy to learn transferable patterns rather
    than memorize a single price path.
    """

    def __init__(self, symbol="AAPL", symbols=None, period="2y", interval="1h",
                 initial_cash=10000, split="all", train_frac=0.8,
                 start=None, end=None, regime=None, exposure_target=0.9,
                 regime_method="sma",
                 exposure_penalty_coef=0.02, tx_cost_weight=1.0):
        super().__init__()

        self._initial_cash = initial_cash
        self._exposure_target = exposure_target
        self._exposure_penalty_coef = exposure_penalty_coef
        self._tx_cost_weight = tx_cost_weight
        self._regime_method = regime_method
        symbol_list = list(symbols) if symbols else [symbol]
        self._pool = []
        feature_names = None

        for sym in symbol_list:
            try:
                if regime is not None:
                    # Regime-filtered: split each symbol into regime segments
                    segments, cols = _load_symbol_regime(
                        sym, period, interval, train_frac, split,
                        target_regime=regime, min_segment_length=30,
                        start=start, end=end, regime_method=regime_method)
                    if feature_names is None:
                        feature_names = cols
                    elif cols != feature_names:
                        print(f"[env] skipping {sym}: feature schema mismatch")
                        continue
                    for i, (seg_p, seg_f) in enumerate(segments):
                        self._pool.append((f"{sym}_{regime}_{i}", seg_p, seg_f))
                else:
                    # Standard: full price series per symbol
                    prices, features, cols = _load_symbol(
                        sym, period, interval, train_frac, split, start, end)
                    if len(prices) < 50:
                        print(f"[env] skipping {sym}: only {len(prices)} bars")
                        continue
                    if feature_names is None:
                        feature_names = cols
                    elif cols != feature_names:
                        print(f"[env] skipping {sym}: feature schema mismatch")
                        continue
                    self._pool.append((sym, prices, features))
            except Exception as e:
                print(f"[env] skipping {sym}: {e}")

        if not self._pool:
            raise RuntimeError(
                f"No usable data for regime={regime} from {symbol_list}. "
                "This regime may have too few contiguous segments (min 30 days).")

        self.feature_names = feature_names
        n_features = self._pool[0][2].shape[1]

        sym0, prices0, features0 = self._pool[0]
        self.current_symbol = sym0
        self.market = SimpleTradingEnv(prices0, features0, initial_cash=initial_cash,
                                      exposure_target=exposure_target,
                                      exposure_penalty_coef=exposure_penalty_coef,
                                      tx_cost_weight=tx_cost_weight)

        self.observation_space = spaces.Dict({
            "features": spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32),
            "price":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "Cash":     spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "Shares":   spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "exposure": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        if regime:
            print(f"[env] regime={regime} ({regime_method}) pool ({split}): "
                  f"{len(self._pool)} segments from {len(symbol_list)} symbols")
        elif len(self._pool) > 1:
            print(f"[env] multi-symbol pool ({split}): {[s for s,_,_ in self._pool]}")

    @staticmethod
    def _decode_action(action):
        raw_type = float(action[0])
        raw_fraction = float(action[1])

        # Wide Hold band (60% of action range) — reduce over-trading.
        if raw_type < -0.6:
            action_type = 2  # Sell
        elif raw_type > 0.6:
            action_type = 1  # Buy
        else:
            action_type = 0  # Hold

        fraction = (raw_fraction + 1.0) / 2.0
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return action_type, fraction

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if len(self._pool) > 1:
            idx = int(self.np_random.integers(0, len(self._pool)))
            sym, prices, features = self._pool[idx]
            self.current_symbol = sym
            self.market = SimpleTradingEnv(prices, features,
                                          initial_cash=self._initial_cash,
                                          exposure_target=self._exposure_target,
                                          exposure_penalty_coef=self._exposure_penalty_coef,
                                          tx_cost_weight=self._tx_cost_weight)
        observation = self.market.reset()
        return observation, {}

    def step(self, action):
        action_type, fraction = self._decode_action(action)
        observation, port_val, reward, done = self.market.step((action_type, fraction))

        if not np.isfinite(reward):
            raise ValueError(f"Non-finite reward detected: {reward}")
        for key, value in observation.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Non-finite observation detected for {key}: {value}")

        return observation, reward, done, False, {"portfolio_value": port_val}
