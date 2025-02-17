import gymnasium as gym
import numpy as np
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from bicycle_env.bicycle_env import BicycleEnv

# class RewardCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         self.episode_rewards = []
#         self.episode_lengths = []
#         self.episode_success = []
        
#     def _on_step(self):
#         # 获取每个环境的信息
#         for info in self.locals['infos']:
#             if 'episode' in info.keys():
#                 self.episode_rewards.append(info['episode']['r'])
#                 self.episode_lengths.append(info['episode']['l'])
#                 self.episode_success.append(info['is_success'])
#                 # 计算最近10个episode的统计信息
#                 recent_rewards = self.episode_rewards[-10:]
#                 recent_success = self.episode_success[-10:]
#                 avg_reward = sum(recent_rewards) / len(recent_rewards)
#                 success_rate = sum(recent_success) / len(recent_success)
#                 print(f"Episode {len(self.episode_rewards)}")
#                 print(f"  Average reward over last 10 episodes: {avg_reward:.2f}")
#                 print(f"  Success rate over last 10 episodes: {success_rate:.2%}")
#         return True

# 设置随机种子
set_random_seed(42)

# 创建向量化环境，使用 DummyVecEnv 而不是 SubprocVecEnv
n_envs = 4  # 减少并行环境数量
env = make_vec_env(
    BicycleEnv,
    n_envs=n_envs,
    vec_env_cls=DummyVecEnv,
    seed=42
)

# 确保日志目录存在
os.makedirs("logs", exist_ok=True)

try:
    model = PPO.load("logs/ppo_driver_latest", env=env)
    print("Model loaded successfully.")
except:
    print("Model not found. Training new model.")
    # PPO hyperparams:
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,  # 减小每次更新前收集的步数
        batch_size=64,
        n_epochs=10,   # 每批数据的训练轮数
        gamma=0.99,    # 折扣因子
        gae_lambda=0.95,  # GAE lambda 参数
        clip_range=0.2,   # PPO clip 参数
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 512],  # policy network
                vf=[512, 512]   # value network
            )
        ),
        device="auto"  # 自动选择设备
    )

# 创建检查点回调和奖励回调
checkpoint_callback = CheckpointCallback(
    save_freq=5000,  # 更频繁地保存检查点
    save_path='./logs/',
    name_prefix='ppo_driver',
    save_replay_buffer=False
)

# reward_callback = RewardCallback()

try:
    # 训练模型
    total_timesteps = int(5e5)  # 减少总训练步数，先测试稳定性
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        log_interval=10,
    )
    # 保存最终模型
    model.save("logs/ppo_driver_final")
except Exception as e:
    print(f"Training interrupted with error: {e}")
    # 发生错误时也保存模型
    model.save("logs/ppo_driver_interrupted")
    raise e 