# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
Live deployment environment.

Mirrors the observation format produced by MarketContinuous (see env.py) so a
trained model can be loaded and run against real-time data without retraining.

Usage
-----
    from stable_baselines3 import PPO
    from deployment_env import LiveMarket

    live = LiveMarket(symbol="AAPL", interval="1h", initial_cash=100)
    model = PPO.load("Nigesh_AAPL_V2")

    obs = live.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, info = live.step(action)
        print(info)
        live.wait_for_next_bar()
"""

import time
import numpy as np

from genObs import generate_observations
from env import MarketContinuous


class LiveMarket:
    """
    Thin live-trading wrapper. Not a gym.Env (no reward, no done) — just produces
    observations in the exact shape MarketContinuous emits and tracks a paper
    portfolio so model.predict() can be called in a loop.

    Plug a real broker into `execute_order` to go from paper -> live.
    """

    # How much history to pull each refresh. Must be long enough that the
    # rolling 500-bar z-scores in genObs are warm.
    DEFAULT_HISTORY = {
        "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
        "1h": "2y", "1d": "10y",
    }

    _BAR_SECONDS = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "1d": 86400,
    }

    def __init__(self, symbol="AAPL", interval="1h", initial_cash=100.0,
                 history_period=None, paper=True):
        self.symbol = symbol
        self.interval = interval
        self.paper = paper
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.shares = 0.0
        self.history_period = history_period or self.DEFAULT_HISTORY.get(interval, "2y")

        self._last_bar_time = None
        self._feature_names = None
        self._refresh()

    # ------------------------------------------------------------------ data
    def _refresh(self):
        """Re-pull history and recompute the feature matrix. Call on every new bar."""
        obs_df, full_df = generate_observations(
            self.symbol, period=self.history_period, interval=self.interval
        )
        if obs_df.empty:
            raise RuntimeError("generate_observations returned no rows — check data source.")

        self._obs_df = obs_df
        self._full_df = full_df
        self._feature_names = list(obs_df.columns)
        self._latest_features = obs_df.iloc[-1].to_numpy(dtype=np.float32)
        self._latest_price = float(full_df["close"].to_numpy()[-1])
        self._last_bar_time = obs_df.index[-1]

    # ------------------------------------------------------------ observation
    def _build_obs(self):
        """Match MarketContinuous.observation_space exactly."""
        return {
            "features": self._latest_features.astype(np.float32),
            "price":    np.array([np.log(self._latest_price)], dtype=np.float32),
            "Cash":     np.array([self.cash], dtype=np.float32),
            "Shares":   np.array([self.shares], dtype=np.float32),
        }

    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0.0
        self._refresh()
        return self._build_obs()

    # ----------------------------------------------------------------- action
    def step(self, action):
        """
        Pull the latest bar, decode the action, (paper-)execute, return new obs.
        Returns (obs, info).
        """
        self._refresh()
        action_type, fraction = MarketContinuous._decode_action(action)
        executed = self.execute_order(action_type, fraction, self._latest_price)

        port_val = self.cash + self.shares * self._latest_price
        info = {
            "time": self._last_bar_time,
            "price": self._latest_price,
            "action": ["HOLD", "BUY", "SELL"][action_type],
            "fraction": fraction,
            "executed": executed,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": port_val,
        }
        return self._build_obs(), info

    def execute_order(self, action_type, fraction, price):
        """
        Paper execution. Override / replace this method to wire in a real broker
        (Alpaca, IBKR, Binance, ...). Should mutate self.cash / self.shares.
        """
        if action_type == 1:  # BUY
            spend = self.cash * fraction
            qty = spend / price
            self.cash -= spend
            self.shares += qty
            return {"side": "buy", "qty": qty, "price": price}
        if action_type == 2:  # SELL
            qty = self.shares * fraction
            self.cash += qty * price
            self.shares -= qty
            return {"side": "sell", "qty": qty, "price": price}
        return {"side": "hold"}

    # ------------------------------------------------------------------ timing
    def wait_for_next_bar(self):
        """Sleep until the next bar should be available."""
        secs = self._BAR_SECONDS.get(self.interval, 3600)
        time.sleep(secs + 5)  # cushion for the data source to publish


# =====================================================================
# Alpaca paper-trading subclass
# =====================================================================
#
# Requires:  pip install alpaca-py
# Env vars:  ALPACA_API_KEY, ALPACA_SECRET_KEY  (from your paper account
#            at https://app.alpaca.markets/paper/dashboard/overview)
#
class AlpacaLiveMarket(LiveMarket):
    """
    LiveMarket that routes orders through the Alpaca paper-trading API and
    syncs cash/shares from the broker on every step (so the model's obs
    reflects the actual paper account state, not a local guess).
    """

    def __init__(self, symbol="AAPL", interval="1h",
                 api_key=None, secret_key=None, **kwargs):
        import os
        from alpaca.trading.client import TradingClient

        api_key = api_key or os.environ.get("ALPACA_API_KEY")
        secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError(
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars (paper account)."
            )

        self.client = TradingClient(api_key, secret_key, paper=True)

        # Pull starting cash from the broker, not the user — paper accounts
        # default to $100k and that's what the model is actually trading.
        acct = self.client.get_account()
        kwargs.setdefault("initial_cash", float(getattr(acct, "cash", 0) or 0))

        super().__init__(symbol=symbol, interval=interval, **kwargs)
        self._sync_account()

    # ------------------------------------------------------------------ sync
    def _sync_account(self):
        """Refresh self.cash / self.shares from the live paper account."""
        acct = self.client.get_account()
        self.cash = float(getattr(acct, "cash", 0) or 0)
        try:
            pos = self.client.get_open_position(self.symbol)
            self.shares = float(getattr(pos, "qty", 0) or 0)
        except Exception:
            # No open position -> alpaca raises; treat as zero.
            self.shares = 0.0

    # ----------------------------------------------------------------- order
    def execute_order(self, action_type, fraction, price):
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        if action_type == 0 or fraction <= 0:
            self._sync_account()
            return {"side": "hold"}

        if action_type == 1:  # BUY — spend `fraction` of available cash
            notional = round(self.cash * fraction, 2)
            if notional < 1.0:  # Alpaca minimum notional is $1
                self._sync_account()
                return {"side": "buy", "skipped": "below min notional"}
            req = MarketOrderRequest(
                symbol=self.symbol,
                notional=notional,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )

        else:  # SELL — sell `fraction` of current position
            qty = round(self.shares * fraction, 6)
            if qty <= 0:
                self._sync_account()
                return {"side": "sell", "skipped": "no position"}
            req = MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )

        order = self.client.submit_order(req)
        # Resync after submission so obs reflects new cash/shares.
        # (Market orders fill near-instantly during RTH; otherwise this
        # will reflect the pending state until the order fills.)
        self._sync_account()
        return {
            "side": "buy" if action_type == 1 else "sell",
            "order_id": str(getattr(order, "id", "")),
            "status": str(getattr(order, "status", "")),
        }


# --------------------------------------------------------------------- demo
if __name__ == "__main__":
    from stable_baselines3 import PPO

    USE_ALPACA = True  # flip to False for pure local paper

    if USE_ALPACA:
        live = AlpacaLiveMarket(symbol="AAPL", interval="1h",api_key="PKFXBMAB2Y73763VJKZR2IKBUB", secret_key="9rrYVoyrR2Jpvu5L2emnJLUVqjSpkhXZ1KdghJJvVpFw")
    else:
        live = LiveMarket(symbol="AAPL", interval="1h", initial_cash=100.0)

    model = PPO.load("Nigesh_AAPL_V2")

    obs = live.reset()
    print(f"Loaded {len(live._feature_names or [])} features. "
          f"Latest bar: {live._last_bar_time}  "
          f"Cash=${live.cash:.2f}  Shares={live.shares}")

    # One-shot trade against the latest bar.
    action, _ = model.predict(obs, deterministic=True)
    obs, info = live.step(action)
    print(info)

    # Continuous loop (uncomment to run live):
    # while True:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, info = live.step(action)
    #     print(info)
    #     live.wait_for_next_bar()
