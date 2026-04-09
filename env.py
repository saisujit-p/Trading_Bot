# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from tradingsys import SimpleTradingEnv
from genObs import generate_observations


class MarketContinuous(gym.Env):
    """
    Continuous-action wrapper around SimpleTradingEnv, fed by the
    feature matrix from genObs.generate_observations.
    """

    def __init__(self, symbol="AAPL", period="2y", interval="1h", initial_cash=10000,
                 split="all", train_frac=0.8, start=None, end=None):
        """
        split : "all" | "train" | "test"
            "train" -> first `train_frac` of the data
            "test"  -> last  `1 - train_frac` of the data
        """
        super().__init__()

        # If a date window is requested, fetch ~3 extra years of history first so
        # the rolling indicators / 500-bar z-scores are warm by the time we hit `start`.
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

        self.market = SimpleTradingEnv(prices, features, initial_cash=initial_cash)
        self.feature_names = list(obs_df.columns)
        n_features = features.shape[1]

        self.observation_space = spaces.Dict({
            "features": spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32),
            "price":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "Cash":     spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "Shares":   spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    @staticmethod
    def _decode_action(action):
        raw_type = float(action[0])
        raw_fraction = float(action[1])

        # Wider Hold band (60% of action range) — Hold is the default prior.
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
