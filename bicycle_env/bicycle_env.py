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
        self.a_range = [-4, 2]

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

        self.goal =  self.inference_goal_state()

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
    
    def inference_goal_state(self):
        horizon = self.max_steps * self.dt / 2
        
        # 计算速度范围
        v_min = max(0., self.state[4] + horizon * self.a_range[0])
        v_max = self.state[4] + horizon * self.a_range[1]
        v_range = [v_min, v_max]
        v_l_range = [-1, 1]
        
        # 计算位置范围
        x_min = self.state[0] + v_min * horizon
        x_max = self.state[0] + v_max * horizon
        y_min = self.state[1] + v_l_range[0] * horizon
        y_max = self.state[1] + v_l_range[1] * horizon
        
        # 创建目标状态范围
        goal_low = jnp.array([x_min, y_min, 1, 0, v_min])
        goal_high = jnp.array([x_max, y_max, 1, 0, v_max])
        
        # 在目标状态范围内随机选择一个目标状态
        goal_state = self.np_random.uniform(low=goal_low, high=goal_high)
        
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
        achieved_goal = next_state[:5]
        reward = self.compute_reward(achieved_goal, self.goal, info=[action])
        self.state = next_state
        self.state_history.append(self.state)

        self.steps += 1
        is_success = reward > 0
        terminated = self.steps >= self.max_steps or is_success
        truncated = terminated  # 在这个例子中,我们将 truncated 设置为与 terminated 相同的值

      
        observation = {
            'observation': self.state,
            'desired_goal': self.goal,
            'achieved_goal': achieved_goal
        }
        # 创建信息字典
        info = {
            'is_success': is_success,  # 是否成功到达目标
            'steps': self.steps,
            'state_history': self.state_history,
        }
        # if terminated:
        #     for i, t  in enumerate(self.reward_func.tolerances):
        #         self.reward_func.tolerances[i] = (1.0 - 1e-3) *  self.reward_func.tolerances[i]
        #         self.reward_func.tolerances[i] = max(0.1, self.reward_func.tolerances[i])

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        # 清空坐标轴
        self.ax.clear()

        # 绘制自行车的轨迹
        state_history_np = jnp.array(self.state_history)
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
        state = self.state
        action = info[0]
        next_state = achieved_goal
        reward = self.reward_func.calculate_reward(
            state, action, next_state, desired_goal)
        return reward
