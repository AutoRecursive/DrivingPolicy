import gymnasium as gym
import numpy as np

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.noise import NormalActionNoise

from bicycle_env.bicycle_env import BicycleEnv

# Create 4 artificial transitions per real transition
n_sampled_goal = 10
env = BicycleEnv()

try:
  model = SAC.load("logs/driver_100000_steps", env=env)
  print("Model loaded successfully.")
except:
  print("Model not found. Training new model.")
  # SAC hyperparams:
  model = SAC(
      "MultiInputPolicy",
      env,
      replay_buffer_class=HerReplayBuffer,
      replay_buffer_kwargs=dict(
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy="future",
      ),
      verbose=1,
      buffer_size=int(1e6),
      learning_rate=1e-3,
      gamma=0.95,
      batch_size=256,
      policy_kwargs=dict(net_arch=[64, 64]),
      gradient_steps=1,
  )


checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/',
                                         name_prefix='driver')

model.learn(int(1e5), log_interval=10, callback=checkpoint_callback)
model.save("trained_model")
