import gym
import numpy as np
from stable_baselines3 import SAC
from bicycle_env.bicycle_env import BicycleEnv

# 创建 BicycleEnv 环境
env = BicycleEnv()

# 加载训练好的模型
model = SAC.load("logs/driver_10000_steps", env=env)

# 设置测试参数
num_episodes = 10
max_steps_per_episode = 100

# 测试循环
for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0
    success = False
    
    for step in range(max_steps_per_episode):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        env.render()

        if terminated or truncated:
            success = info.get("is_success", False)
            break
    
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Success = {success}")

# 关闭环境
env.close()