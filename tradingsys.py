import numpy as np


class SimpleTradingEnv:
    def __init__(self, prices, features, initial_cash=10000,
                 transaction_cost_pct=0.003, slippage_pct=0.0005,
                 start_invested=False, exposure_target=0.9,
                 exposure_penalty_coef=0.02, tx_cost_weight=1.0):
        self.prices = np.asarray(prices, dtype=np.float32)
        self.features = np.asarray(features, dtype=np.float32)
        if self.prices.ndim != 1 or len(self.prices) < 2:
            raise ValueError("prices must be a 1D array with at least 2 entries")
        if self.features.ndim != 2 or len(self.features) != len(self.prices):
            raise ValueError("features must be 2D and row-aligned with prices")
        self.n_features = self.features.shape[1]
        self.initial_cash = initial_cash
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.slippage_pct = float(slippage_pct)
        self.start_invested = bool(start_invested)
        self.exposure_target = float(exposure_target)
        self.exposure_penalty_coef = float(exposure_penalty_coef)
        self.tx_cost_weight = float(tx_cost_weight)
        self.reset()

    def reset(self):
        self.cash = float(self.initial_cash)
        self.shares = 0.0
        self.current_step = 0
        if self.start_invested:
            price0 = max(float(self.prices[0]), 1e-8)
            self.shares = self.cash / price0
            self.cash = 0.0
        return self._get_state()

    def _get_state(self):
        price = max(float(self.prices[self.current_step]), 1e-8)
        cash = max(float(self.cash), 1e-8)
        shares = max(float(self.shares), 0.0)
        feats = self.features[self.current_step].astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Exposure fraction in [0, 1]: lets the policy directly see whether it
        # is above / below the regime target without having to infer it from
        # log(cash) + log(shares) + log(price).
        equity = shares * price
        port_val = cash + equity
        exposure = float(equity / max(port_val, 1e-8))

        return {
            "features": feats,
            "price": np.array([np.log(price)], dtype=np.float32),
            "Cash": np.array([np.log(cash)], dtype=np.float32),
            "Shares": np.array([np.log(shares + 1.0)], dtype=np.float32),
            "exposure": np.array([exposure], dtype=np.float32),
        }

    def step(self, action):
        action_type = int(action[0])
        fraction = float(action[1])
        price = float(self.prices[self.current_step])

        prev_value = self.cash + self.shares * price
        prev_shares = self.shares
        invalid_action = False
        trade_cost = 0.0

        fill_step = min(self.current_step + 1, len(self.prices) - 1)
        fill_price = float(self.prices[fill_step])

        if action_type == 0:  # HOLD
            pass
        elif action_type == 1:  # BUY
            if self.cash > 0 and fraction > 0.01:
                spend = fraction * self.cash
                execution_price = fill_price * (1.0 + self.slippage_pct)
                transaction_cost = spend * self.transaction_cost_pct
                trade_cost = transaction_cost
                spend_after_cost = max(spend - transaction_cost, 0.0)
                self.shares += spend_after_cost / max(execution_price, 1e-8)
                self.cash -= spend
            else:
                invalid_action = True
        elif action_type == 2:  # SELL
            if self.shares > 0 and fraction > 0.01:
                shares_to_sell = fraction * self.shares
                execution_price = fill_price * (1.0 - self.slippage_pct)
                gross_sale_value = shares_to_sell * execution_price
                transaction_cost = gross_sale_value * self.transaction_cost_pct
                trade_cost = transaction_cost
                self.cash += max(gross_sale_value - transaction_cost, 0.0)
                self.shares -= shares_to_sell
            else:
                invalid_action = True

        self.cash = max(float(self.cash), 0.0)
        self.shares = max(float(self.shares), 0.0)

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        next_price = float(self.prices[self.current_step])
        new_value = self.cash + self.shares * next_price

        if new_value <= 0:
            done = True

        # Per-step: raw portfolio log-return
        reward = float(np.log(max(new_value, 1e-8) / max(prev_value, 1e-8)))

        # Transaction cost penalty (tx_cost_weight lets bear punish churn harder).
        reward -= self.tx_cost_weight * trade_cost / max(prev_value, 1e-8)

        # Two-sided exposure penalty: punish deviation from regime target in both
        # directions so the agent converges to the intended allocation instead of
        # always maxing out (the old one-sided +coef*(exp-target) rewarded fullness).
        exposure = (self.shares * next_price) / max(new_value, 1e-8)
        reward -= self.exposure_penalty_coef * abs(exposure - self.exposure_target)

        if invalid_action:
            reward -= 0.01

        reward = float(np.clip(reward, -1.0, 1.0))

        return self._get_state(), float(new_value), reward, done
