import gymnasium as gym
from stable_baselines3 import PPO
from env import Market
from data import load_data

new_env=Market(load_data("AAPL","1d","1h"))
model = PPO.load("Nigesh_AAPL_V1", env=new_env)
done = False
print(new_env.market.prices)

obs, _ = new_env.reset()
'''
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, info = new_env.step(action)
    print(f"Action: {action}, Reward: {reward}, Portfolio Value: {info['portfolio_value']}")
'''

Bear_market=Market([300, 297, 294, 296, 291, 288, 290])
n_obs,_=Bear_market.reset()
done=False
while not done:
    action, _ = model.predict(n_obs)
    n_obs, reward, done, _, info = Bear_market.step(action)
    print(f"Action: {action}, Reward: {reward}, Portfolio Value: {info['portfolio_value']}")
