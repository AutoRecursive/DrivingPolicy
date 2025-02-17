import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from .naive_reward import NaiveReward
import os


class BicycleEnv(gym.Env):
    def __init__(self):
        super(BicycleEnv, self).__init__()
        self.a_range = [-4, 2]

        # 定义观测空间 
        # observation: [x_local, y_local, v, delta] (目标在自车坐标系下的相对位置，自车速度和方向盘角度)
        # goal: [x_local, y_local, v] (目标在自车坐标系下的相对位置和目标速度)
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(low=np.array(jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.pi/4])),
                                       high=np.array(
                                           jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.pi/4])),
                                       dtype=np.float32),
                desired_goal=spaces.Box(low=np.array(jnp.array([-jnp.inf, -jnp.inf, -jnp.inf])),
                                        high=np.array(
                                            jnp.array([jnp.inf, jnp.inf, jnp.inf])),
                                        dtype=np.float32),
                achieved_goal=spaces.Box(low=np.array(jnp.array([-jnp.inf, -jnp.inf, -jnp.inf])),
                                         high=np.array(
                                             jnp.array([jnp.inf, jnp.inf, jnp.inf])),
                                         dtype=np.float32)
            )
        )

        # 定义动作空间
        self.action_space = spaces.Box(low=np.array(jnp.array([self.a_range[0], -1.0])),
                                       high=np.array(jnp.array([self.a_range[1], 1.0])),
                                       dtype=np.float32)

        # 初始化状态
        self.state = None
        self.state_history = []

        # 设置自行车参数
        self.dt = 0.1  # 时间步长
        self.L = 1.0   # 自行车轴距
        self.W = 2.0   # 自行车宽度
        self.H = 5.0   # 自行车长度
        self.fig, self.ax = plt.subplots()
        self.patch = None
        self.goal_patch = None

        self.max_steps = 20

        self.reward_func = NaiveReward(self.max_steps, self.dt)
        self.goal = None  # 初始化目标状态为 None
        self.max_v = 11.1
        
        # 添加图片保存相关的属性
        self.episode_count = 0  # 用于生成唯一的文件名
        self.save_dir = "trajectory_plots"  # 保存图片的目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def set_goal(self, goal):
        self.goal = goal

    def global_to_local(self, global_state, ego_state):
        """将全局坐标系下的状态转换到自车坐标系下"""
        # ego_state: [x, y, nx, ny, v, delta]
        # global_state: [x, y, nx, ny, v]
        
        # 提取ego的位置和方向
        ego_x, ego_y = ego_state[0:2]
        ego_nx, ego_ny = ego_state[2:4]
        
        # 计算ego坐标系的旋转矩阵
        theta = jnp.arctan2(ego_ny, ego_nx)
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        R = jnp.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
        
        # 计算相对位置
        dx = global_state[0] - ego_x
        dy = global_state[1] - ego_y
        local_pos = jnp.dot(R, jnp.array([dx, dy]))
        
        # 转换目标的朝向（如果需要）
        if len(global_state) > 2:
            target_nx, target_ny = global_state[2:4]
            local_dir = jnp.dot(R, jnp.array([target_nx, target_ny]))
            
            # 组合局部坐标系下的状态
            if len(global_state) > 4:
                # 如果有速度信息
                return jnp.array([local_pos[0], local_pos[1], local_dir[0], local_dir[1], global_state[4]])
            else:
                return jnp.array([local_pos[0], local_pos[1], local_dir[0], local_dir[1]])
        
        return jnp.array([local_pos[0], local_pos[1]])

    def get_obs(self):
        """获取当前观测"""
        # 将目标转换到自车坐标系下
        goal_local = self.global_to_local(self.goal, self.state)
        
        # 构建observation
        obs = jnp.array([goal_local[0], goal_local[1], self.state[4], self.state[5]])  # [x_local, y_local, v, delta]
        achieved_goal = jnp.array([0., 0., self.state[4]])  # 当前位置在自身坐标系下为原点
        desired_goal = jnp.array([goal_local[0], goal_local[1], self.goal[4]])  # 目标在自身坐标系下的位置和速度
        
        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }

    def reset(self, seed=None, options=None):
        # 重置环境状态
        super().reset(seed=seed)

        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        x = 0.0
        y = 0.0
        nx = 1.0
        ny = 0.0
        v = self.np_random.uniform(low=0.0, high=self.max_v)
        delta = 0.0
        self.state = jnp.array([x, y, nx, ny, v, delta])
        self.state_history = [self.state]
        self.steps = 0
        self.episode_count += 1  # 每次reset时增加episode计数

        self.goal = self.inference_goal_state()

        if options is not None:
            goal = options.get('goal')
            if goal is not None:
                self.set_goal(goal)

        return self.get_obs(), {}

    def sample_goal_state(self, max_steps):
        state = self.state.copy()
        for _ in range(max_steps):
            action = self.action_space.sample()  # 随机选择动作
            next_state = self.simulate_step(state, action)  # 使用模拟步骤更新状态
            state = next_state
        return state[:5]  # 返回目标状态的前5个分量
    
    def inference_goal_state(self):
        # 使用随机策略模拟一段时间来生成目标
        state = self.state.copy()
        horizon_steps = int(self.max_steps / 2)  # 使用一半的最大步数作为预测范围
        
        # 存储所有模拟状态
        simulated_states = []
        
        # 模拟多次随机轨迹，选择其中一个终点作为目标
        num_trajectories = 5
        final_states = []
        
        for _ in range(num_trajectories):
            current_state = state.copy()
            for _ in range(horizon_steps):
                # 生成随机动作，但限制在合理范围内
                action = self.action_space.sample()
                # 模拟一步
                current_state = self.simulate_step(current_state, action)
            final_states.append(current_state)
        
        # 随机选择一个终点状态作为目标
        goal_idx = self.np_random.integers(0, len(final_states))
        goal_state = final_states[goal_idx][:5]  # 只取前5个状态分量作为目标
        
        # 确保目标状态的朝向为单位向量
        nx, ny = goal_state[2:4]
        norm = np.sqrt(nx**2 + ny**2)
        goal_state = goal_state.at[2:4].set(jnp.array([nx/norm, ny/norm]))
        
        return goal_state


    def simulate_step(self, state, action):
        # 根据当前状态和动作计算下一个状态
        acceleration, gamma = action

        # 更新状态(使用简化的自行车模型动力学方程)
        x, y, nx, ny, v, delta = state
        v = max(0., v)
        theta = jnp.arctan2(ny, nx)
        x += v * nx * self.dt
        y += v * ny * self.dt
        theta += v * jnp.tan(delta) / self.L * self.dt
        nx = jnp.cos(theta)
        ny = jnp.sin(theta)
        v += acceleration * self.dt
        delta += gamma * self.dt

        next_state = jnp.array([x, y, nx, ny, v, delta])
        return next_state

    def step(self, action):
        # 执行动作并返回下一个状态、奖励、是否完成和信息
        next_state = self.simulate_step(self.state, action)
        self.state = next_state
        self.state_history.append(self.state)

        # 获取观测
        obs = self.get_obs()
        achieved_goal = obs['achieved_goal']
        desired_goal = obs['desired_goal']
        
        # 计算奖励和完成状态
        reward = self.compute_reward(achieved_goal, desired_goal, info=[action])
        self.steps += 1
        is_success = reward > 0
        terminated = self.steps >= self.max_steps or is_success
        truncated = terminated

        # 创建信息字典
        info = {
            'is_success': is_success,
            'steps': self.steps,
            'state_history': self.state_history,
            'is_terminated': terminated
        }
        self.last_info = info

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        # 清空坐标轴
        self.ax.clear()

        # 绘制车道线
        y_limit = 3.75/2
        state_history_np = jnp.array(self.state_history)
        x_min = state_history_np[:, 0].min() - self.H
        x_max = state_history_np[:, 0].max() + self.H
        self.ax.hlines([y_limit, -y_limit], x_min, x_max, colors='gray', linestyles='dashed', label='Lane Boundaries')

        # 绘制自行车的轨迹
        self.ax.plot(state_history_np[:, 0], state_history_np[:, 1], 'b-')

        # 绘制自行车的当前位置
        self.ax.plot(self.state[0], self.state[1], 'ro')
        self.ax.scatter(self.goal[0], self.goal[1])
        # 获取当前状态下自行车边界框的顶点坐标
        x_corners, y_corners = self._get_bicycle_bbox(self.state)

        # 如果边界框不存在,则创建一个新的边界框
        if self.patch is None:
            self.patch = patches.Polygon(xy=np.column_stack((np.asarray(x_corners), np.asarray(
                y_corners))), closed=True, edgecolor='r', facecolor='r', alpha=0.3)
            self.ax.add_patch(self.patch)
        else:
            # 如果边界框已存在,则更新其顶点坐标
            self.patch.set_xy(np.column_stack(
                (np.asarray(x_corners), np.asarray(y_corners))))
            self.ax.add_patch(self.patch)
        

        goal_x_corners, goal_y_corners = self._get_bicycle_bbox(self.goal)

        # 如果边界框不存在,则创建一个新的边界框
        if self.goal_patch is None:
            self.goal_patch = patches.Polygon(xy=np.column_stack((np.asarray(goal_x_corners), np.asarray(
                goal_y_corners))), closed=True, edgecolor='b', facecolor='b', alpha=0.3)
            self.ax.add_patch(self.goal_patch)
        else:
            # 如果边界框已存在,则更新其顶点坐标
            self.goal_patch.set_xy(np.column_stack(
                (np.asarray(goal_x_corners), np.asarray(goal_y_corners))))
            self.ax.add_patch(self.goal_patch)

        # 设置合适的xlim和ylim
        max_range = max(
            np.abs(state_history_np[:, :2]).max() + self.H, self.W) * 1.1
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Bicycle Trajectory')
        self.ax.set_aspect('equal')

        # 刷新画布
        self.fig.canvas.draw()
        plt.pause(0.001)
        
        # 在episode结束时保存图片
        if self.steps >= self.max_steps or (hasattr(self, 'last_info') and self.last_info.get('is_success', False)):
            # 生成文件名，包含episode编号和是否成功的信息
            success_str = "success" if hasattr(self, 'last_info') and self.last_info.get('is_success', False) else "timeout"
            filename = f"{self.save_dir}/trajectory_ep{self.episode_count}_{success_str}.png"
            plt.savefig(filename)
            print(f"Saved trajectory plot to {filename}")

    def _get_bicycle_bbox(self, state):
        x, y, nx, ny = state[:4]

        # 计算后轴中心相对于自行车几何中心的偏移量
        dx = -self.H * 1 / 4 * nx
        dy = -self.H * 1 / 4 * ny

        # 计算自行车边界框的四个顶点坐标
        x_corners = jnp.array([
            x + dx + self.H * nx - self.W / 2 * ny,
            x + dx + self.H * nx + self.W / 2 * ny,
            x + dx + self.W / 2 * ny,
            x + dx - self.W / 2 * ny
        ])
        y_corners = jnp.array([
            y + dy + self.H * ny + self.W / 2 * nx,
            y + dy + self.H * ny - self.W / 2 * nx,
            y + dy - self.W / 2 * nx,
            y + dy + self.W / 2 * nx
        ])
        return x_corners, y_corners

    def compute_reward(self, achieved_goal, desired_goal, info):
        # 计算奖励
        action = info[0]
        reward = self.reward_func.calculate_reward(
            action, achieved_goal, desired_goal, info)
        return reward
