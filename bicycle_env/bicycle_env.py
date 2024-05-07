import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from .naive_reward import NaiveReward


class BicycleEnv(gym.Env):
    def __init__(self):
        super(BicycleEnv, self).__init__()

        # 定义观测空间 s=[x,y,nx,ny,v,delta]
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(low=np.array(jnp.array([-jnp.inf, -jnp.inf, -1.0, -1.0, -jnp.inf, -jnp.pi/4])),
                                       high=np.array(
                                           jnp.array([jnp.inf, jnp.inf, 1.0, 1.0, jnp.inf, jnp.pi/4])),
                                       dtype=np.float32),
                desired_goal=spaces.Box(low=np.array(jnp.array([-jnp.inf, -jnp.inf, -1.0, -1.0, -jnp.inf])),
                                        high=np.array(
                                            jnp.array([jnp.inf, jnp.inf, 1.0, 1.0, jnp.inf])),
                                        dtype=np.float32),
                achieved_goal=spaces.Box(low=np.array(jnp.array([-jnp.inf, -jnp.inf, -1.0, -1.0, -jnp.inf])),
                                         high=np.array(
                                             jnp.array([jnp.inf, jnp.inf, 1.0, 1.0, jnp.inf])),
                                         dtype=np.float32)
            )
        )

        # 定义动作空间
        self.action_space = spaces.Box(low=np.array(jnp.array([-4, -1.0])),
                                       high=np.array(jnp.array([2, 1.0])),
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
        self.max_steps = 20

        self.reward_func = NaiveReward(self.max_steps, self.dt)
        self.goal = None  # 初始化目标状态为 None
        self.max_v = 11.1

    def set_goal(self, goal):
        self.goal = goal

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

        if self.goal is None:
            # 通过动态采样获取可达目标状态
            max_steps = self.max_steps  # 每个采样轨迹的最大步数
            goal_state = self.sample_goal_state(max_steps)
            self.goal = goal_state

        observation = {
            'observation': self.state,
            'desired_goal': self.goal,
            'achieved_goal': self.state[:5]
        }

        if options is not None:
            goal = options.get('goal')
            if goal is not None:
                self.set_goal(goal)
                observation['desired_goal'] = self.goal

        reset_info = {}  # 创建一个空字典或包含任何你想传递的重置信息的字典

        return observation, reset_info

    def sample_goal_state(self, max_steps):
        state = self.state.copy()
        for _ in range(max_steps):
            action = self.action_space.sample()  # 随机选择动作
            next_state = self.simulate_step(state, action)  # 使用模拟步骤更新状态
            state = next_state
        return state[:5]  # 返回目标状态的前5个分量

    def simulate_step(self, state, action):
        # 根据当前状态和动作计算下一个状态
        acceleration, gamma = action

        # 更新状态(使用简化的自行车模型动力学方程)
        x, y, nx, ny, v, delta = state
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
        achieved_goal = next_state[:5]
        reward = self.compute_reward(achieved_goal, self.goal, info=[action])
        self.state = next_state
        self.state_history.append(self.state)

        self.steps += 1
        terminated = self.steps >= self.max_steps or reward > -self.dt
        truncated = terminated  # 在这个例子中,我们将 truncated 设置为与 terminated 相同的值

        observation = {
            'observation': self.state,
            'desired_goal': self.goal,
            'achieved_goal': achieved_goal
        }
        # 创建信息字典
        info = {
            'steps': self.steps,
            'state_history': self.state_history
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        # 清空坐标轴
        self.ax.clear()

        # 绘制自行车的轨迹
        state_history_np = jnp.array(self.state_history)
        self.ax.plot(state_history_np[:, 0], state_history_np[:, 1], 'b-')

        # 绘制自行车的当前位置
        self.ax.plot(self.state[0], self.state[1], 'ro')

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

    def _get_bicycle_bbox(self, state):
        x, y, nx, ny, _, _ = state

        # 计算后轴中心相对于自行车几何中心的偏移量
        dx = -self.H * 3 / 4 * nx
        dy = -self.H * 3 / 4 * ny

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
        state = self.state
        action = info[0]
        next_state = achieved_goal
        reward = self.reward_func.calculate_reward(
            state, action, next_state, desired_goal)
        return reward
