import numpy as np


class NaiveReward:
    def __init__(self, horizon, dt):
        # 在这里初始化奖励函数所需f的参数或属性
        self.tolerances = [0.5, 0.5,  0.05, 0.05, 1.0]
        self.weights = [1, 1, 0., 0., 1]

        self.horizon = horizon
        self.dt = dt

    def reached_goal(self, state, goal):
         # 计算状态与目标状态之间的差异
         # state 和 goal 现在都是 [x_local, y_local, v] 格式
        dx = np.abs(state[..., 0] - goal[..., 0])
        dy = np.abs(state[..., 1] - goal[..., 1])
        dv = np.abs(state[..., 2] - goal[..., 2])
        return (dx <= self.tolerances[0]) & (dy <= self.tolerances[1]) & (dv <= self.tolerances[4])

    def calculate_reward(self, action, next_state, goal_state, info):
        # 检查是否达到目标状态
        reached_goal = self.reached_goal(next_state, goal_state) 
        
        # 计算基础奖励
        effort_coef = 0.01
        base_reward = np.where(reached_goal, self.dt *
                          self.horizon, -self.dt * effort_coef if isinstance(action, dict) else -action[..., 1]**2 * self.dt * effort_coef)
        
        # # 添加对 y 坐标超出范围的惩罚
        # y_limit = 3.75/2
        # y = next_state[..., 1]
        # y_penalty = np.where(np.abs(y) > y_limit, -5.0 * self.dt, 0.0)  # 使用 np.where 来处理数组情况
        y_penalty = 0.0
        reward = base_reward + y_penalty
        
        return reward
