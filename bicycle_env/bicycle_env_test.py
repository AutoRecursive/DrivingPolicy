import jax.numpy as jnp
import matplotlib.pyplot as plt
from bicycle_env import BicycleEnv

# 创建自行车环境
env = BicycleEnv()

# 设置目标状态
goal = jnp.array([10.0, 10.0, 0.0, 1.0, 2.0])
env.set_goal(goal)

# 重置环境并获取初始观测
observation = env.reset()
env.render()

# 循环执行动作
for _ in range(100):
    action = env.action_space.sample()  # 随机采样动作
    observation, reward, done, info = env.step(action)
    
    print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    print("---")
    
    env.render()
    
    if done:
        observation = env.reset()

# 关闭环境
env.close()