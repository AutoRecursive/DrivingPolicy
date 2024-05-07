import numpy as np


class NaiveReward:
    def __init__(self, horizon, dt):
        # 在这里初始化奖励函数所需f的参数或属性
        self.tolerances = [0.5, 0.5, 0.1, 0.1, 0.1]
        self.horizon = horizon
        self.dt = dt

    def calculate_reward(self, state, action, next_state, goal_state):
        # 计算状态与目标状态之间的差异
        dx = np.abs(next_state[..., 0] - goal_state[..., 0])
        dy = np.abs(next_state[..., 1] - goal_state[..., 1])
        dnx = np.abs(next_state[..., 2] - goal_state[..., 2])
        dny = np.abs(next_state[..., 3] - goal_state[..., 3])
        dv = np.abs(next_state[..., 4] - goal_state[..., 4])

        # 计算状态与目标状态之间的距离
        distance = np.sqrt(dx**2 + dy**2 + dnx**2 + dny**2 + dv**2)

        # 检查是否达到目标状态
        reached_goal = (dx <= self.tolerances[0]) & (dy <= self.tolerances[1])

        # 计算奖励
        reward = np.where(reached_goal, self.horizon *
                          self.dt * 10, - distance * self.dt)
        # reward = -distance * self.dt

        return reward
