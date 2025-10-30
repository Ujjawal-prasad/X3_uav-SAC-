from gymnasium.envs.registration import register
register(id='X3_uav-V0', entry_point='env:X3UavRl')
import gymnasium as gym
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

env = gym.make('X3_uav-V0')
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs/SAC/")
model.learn(total_timesteps=200_000, log_interval=10)
model.save("sac_x3_uav")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_x3_uav")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
