from gymnasium.envs.registration import register
register(id='X3_uav-V0', entry_point='env:X3UavRl')
import gymnasium as gym
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
# import numpy as np
# import matplotlib.pyplot as plt

env = gym.make('X3_uav-V0')

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs/SAC/")
model.learn(total_timesteps=200_000, log_interval=10)
model.save("sac_x3_uav")

#log the results
# action_log = []
# observation_log = []
# reward_log = []

del model # remove to demonstrate saving and loading

model = SAC.load("sac_x3_uav")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    # action_log.append(action)
    obs, reward, terminated, truncated, info = env.step(action)
    # observation_log.append(obs)
    # reward_log.append(reward)
    if terminated or truncated:
        obs, info = env.reset()
        # observation_log.append(obs)

# fig, axes = plt.subplots(7, 1, figsize=(10, 14), constrained_layout=True)
# for i in range(7):
#     axes[i].plot(observation_log[:, i])
#     axes[i].set_ylabel(f"obs[{i}]")
# axes[-1].set_xlabel("step")
# fig.suptitle("Observations over time")