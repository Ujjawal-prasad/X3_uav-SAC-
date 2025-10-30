from gymnasium.envs.registration import register
register(id='X3_uav-V0', entry_point='env:X3UavRl')
import gymnasium as gym
import asyncio
from mavsdk import System

env = gym.make('X3_uav-V0')
obs, info = env.reset()
print (f"starting_observation:{obs}")
episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()
    # print("taking_actions....")
    obs , reward, terminated, truncated, info = env.step(action)
    
    total_reward+=reward
    episode_over = terminated or truncated

print(f"episode finished! Total reward: {total_reward}")
env.close()
