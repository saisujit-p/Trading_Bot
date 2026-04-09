import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from env import MarketContinuous

SYMBOL, PERIOD, INTERVAL, CASH = "AAPL", "10y", "1d", 100

# ---------- TRAIN ENV (first 80%) ----------
train_env = MarketContinuous(symbol=SYMBOL, period=PERIOD, interval=INTERVAL,
                             initial_cash=CASH, split="train", train_frac=0.8)

model = PPO(
    "MultiInputPolicy",
    train_env,
    verbose=1,
    ent_coef=0.005,
    policy_kwargs=dict(net_arch=[128, 128]),
)

model.learn(total_timesteps=500000,progress_bar=True)
model.save("Nigesh_AAPL_V2")


# ---------- EVAL HELPER ----------
def evaluate(env, model, label):
    obs, _ = env.reset()
    action_counts = {0: 0, 1: 0, 2: 0}
    initial_price = float(np.exp(obs["price"][0]))
    initial_cash = env.market.cash
    steps = 0
    info = {"portfolio_value": initial_cash}

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action_type, _ = MarketContinuous._decode_action(action)
        action_counts[action_type] += 1
        obs, _, done, _, info = env.step(action)
        steps += 1
        if done:
            break

    final_val = info["portfolio_value"]
    final_price = float(env.market.prices[env.market.current_step])
    agent_ret = (final_val - initial_cash) / initial_cash
    bh_ret = (final_price - initial_price) / initial_price

    print(f"\n===== {label} ({steps} steps) =====")
    print(f"  Hold: {action_counts[0]} ({100*action_counts[0]/steps:.1f}%)")
    print(f"  Buy:  {action_counts[1]} ({100*action_counts[1]/steps:.1f}%)")
    print(f"  Sell: {action_counts[2]} ({100*action_counts[2]/steps:.1f}%)")
    print(f"  Final portfolio: {final_val:.2f}  (start {initial_cash:.2f})")
    print(f"  Agent return:    {agent_ret*100:+.2f}%")
    print(f"  Buy&Hold return: {bh_ret*100:+.2f}%")
    print(f"  Excess vs B&H:   {(agent_ret - bh_ret)*100:+.2f}%")


# ---------- IN-SAMPLE ----------
evaluate(train_env, model, "IN-SAMPLE (train 80%)")

# ---------- OUT-OF-SAMPLE ----------
test_env = MarketContinuous(symbol=SYMBOL, period=PERIOD, interval=INTERVAL,
                            initial_cash=CASH, split="test", train_frac=0.8)
evaluate(test_env, model, "OUT-OF-SAMPLE (test 20%)")
