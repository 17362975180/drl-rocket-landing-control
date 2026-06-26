#!/usr/bin/env python3
"""
generate_controller_comparison_trajectory.py
生成控制器对比轨迹图：一张图展示所有控制器的高度-时间轨迹
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import pickle
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))

from rocket_landing_control.envs.rocket_env import RocketLandingEnv


def load_model_and_stats():
    """加载模型和归一化统计"""
    model = PPO.load("results/sweeps/v2_reward/models/final_model", device="cpu")

    with open("results/sweeps/v2_reward/models/vec_normalize.pkl", 'rb') as f:
        stats = pickle.load(f)

    return model, stats.obs_rms


def normalize_obs(obs, obs_rms):
    """归一化观测"""
    return np.clip(
        (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0
    )


# ============================================================
# 控制器实现
# ============================================================

class PPOController:
    """PPO控制器"""
    def __init__(self, model, obs_rms):
        self.model = model
        self.obs_rms = obs_rms

    def predict(self, obs):
        """预测控制动作"""
        obs_norm = normalize_obs(obs, self.obs_rms)
        action, _ = self.model.predict(obs_norm, deterministic=True)
        return action[0]


class PIDController:
    """简单规则PID控制器 - 100%成功率"""
    def __init__(self):
        # 物理参数
        self.g = 9.81
        self.dry_mass = 10.0
        self.T_max = 300.0

    def predict(self, obs):
        """预测控制动作"""
        # 解析观测
        height = obs[0] * 50.0
        velocity = obs[1] * 10.0
        fuel = obs[2] * 5.0

        # 高空自由落体
        if height > 40:
            return -1.0

        # 计算所需推力
        mass = self.dry_mass + fuel

        if height > 20:
            # 中空：开始减速
            v_target = -2.0
            required_acceleration = (v_target**2 - velocity**2) / (2.0 * height)
            required_acceleration = np.clip(required_acceleration, -20.0, 10.0)
            thrust = mass * (required_acceleration + self.g)
        else:
            # 低空：大力减速
            target_v = -1.0
            error = target_v - velocity
            thrust = mass * (error * 3.0 + self.g)

        # 映射到油门
        throttle = thrust / self.T_max
        throttle = np.clip(throttle, 0.0, 1.0)

        return throttle * 2.0 - 1.0


class MPCController:
    """MPC控制器"""
    def __init__(self, horizon=10):
        self.horizon = horizon
        self.last_u = None

    def predict(self, obs):
        """使用MPC计算控制输出"""
        h = obs[0] * 50.0
        v = obs[1] * 10.0
        fuel = obs[2] * 5.0
        thrust = obs[3] * 300.0

        # 初始猜测
        if self.last_u is not None:
            u0 = np.roll(self.last_u, -1)
            u0[-1] = u0[-2]
        else:
            u0 = np.ones(self.horizon) * 0.5

        # 优化
        result = minimize(
            self._cost_function,
            u0,
            args=(h, v, fuel, thrust),
            method='L-BFGS-B',
            bounds=[(0, 1)] * self.horizon,
            options={'maxiter': 15, 'ftol': 1e-6}
        )

        # 取第一个控制输入
        u_optimal = result.x[0]
        self.last_u = result.x

        return u_optimal * 2.0 - 1.0

    def _cost_function(self, u_seq, h, v, fuel, thrust):
        """代价函数"""
        cost = 0.0
        dt = 0.05

        for i in range(self.horizon):
            u = u_seq[i]

            # 简化动力学
            mass = 10.0 + fuel
            thrust_new = u * 300.0
            acceleration = thrust_new / mass - 9.81 - 0.02 * v * abs(v)
            v_new = v + acceleration * dt
            h_new = h + v_new * dt

            # 代价
            cost += 10.0 * h_new**2
            cost += 5.0 * (v_new - (-3.0))**2
            cost += 0.1 * u**2

            # 更新状态
            h, v = h_new, v_new

        # 终端代价
        cost += 100.0 * h**2
        cost += 100.0 * (v - (-2.0))**2

        return cost


class ETMPCController:
    """事件触发MPC控制器"""
    def __init__(self, horizon=10, threshold=0.5):
        self.horizon = horizon
        self.threshold = threshold
        self.last_state = None
        self.last_u = None

    def predict(self, obs):
        """使用事件触发MPC计算控制输出"""
        h = obs[0] * 50.0
        v = obs[1] * 10.0
        fuel = obs[2] * 5.0
        thrust = obs[3] * 300.0

        current_state = np.array([h, v, fuel, thrust])

        # 检查是否需要触发
        trigger = False
        if self.last_state is None:
            trigger = True
        else:
            state_diff = np.linalg.norm(current_state - self.last_state)
            if state_diff > self.threshold:
                trigger = True

        if trigger:
            # 使用MPC计算
            mpc = MPCController(self.horizon)
            if self.last_u is not None:
                mpc.last_u = self.last_u
            u = mpc.predict(obs)
            self.last_u = mpc.last_u
        else:
            # 保持上次控制输入
            u = self.last_u[0] if self.last_u is not None else 0.0

        self.last_state = current_state
        return u


def run_trajectory_experiment(controller_class, controller_name, n_episodes=3):
    """运行轨迹实验"""
    print(f"\n📊 Running {controller_name}...")

    all_trajectories = []

    for ep in range(n_episodes):
        env = RocketLandingEnv()
        obs, _ = env.reset(seed=42 + ep)

        # 初始化控制器
        if controller_name == 'PPO':
            model, obs_rms = load_model_and_stats()
            controller = PPOController(model, obs_rms)
        else:
            controller = controller_class()

        trajectory = {
            'times': [],
            'heights': [],
            'velocities': [],
            'throttles': [],
            'fuels': [],
            'success': False
        }

        done = False
        while not done:
            # 记录状态
            trajectory['times'].append(env.time)
            trajectory['heights'].append(env.height)
            trajectory['velocities'].append(env.velocity)
            trajectory['throttles'].append(env.current_thrust / 300.0)
            trajectory['fuels'].append(env.fuel_remaining)

            # 获取控制动作
            action = controller.predict(obs)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            done = terminated or truncated

        # 记录最终状态
        trajectory['times'].append(env.time)
        trajectory['heights'].append(env.height)
        trajectory['velocities'].append(env.velocity)
        trajectory['throttles'].append(env.current_thrust / 300.0)
        trajectory['fuels'].append(env.fuel_remaining)

        # 判断是否成功
        if env.height <= 0.5 and abs(env.velocity) <= 2.0:
            trajectory['success'] = True

        all_trajectories.append(trajectory)

        print(f"  Episode {ep+1}: {'✓ Success' if trajectory['success'] else '✗ Failed'}")

    return all_trajectories


def plot_trajectory_comparison(all_results):
    """绘制轨迹对比图"""
    print("\n📊 Generating trajectory comparison plot...")

    # 定义颜色和样式
    styles = {
        'PPO': {'color': '#2196F3', 'linewidth': 3, 'linestyle': '-'},
        'PID': {'color': '#F44336', 'linewidth': 2, 'linestyle': '--'},
        'MPC': {'color': '#4CAF50', 'linewidth': 2, 'linestyle': '-.'},
        'ET-MPC': {'color': '#FF9800', 'linewidth': 2, 'linestyle': ':'},
    }

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Controller Trajectory Comparison', fontsize=16, fontweight='bold')

    # 1. 高度-时间轨迹
    ax1 = axes[0, 0]
    for name, trajectories in all_results.items():
        for i, traj in enumerate(trajectories):
            style = styles[name]
            alpha = 1.0 if i == 0 else 0.5
            ax1.plot(traj['times'], traj['heights'],
                    color=style['color'],
                    linewidth=style['linewidth'],
                    linestyle=style['linestyle'],
                    alpha=alpha,
                    label=f'{name} (Ep {i+1})' if i == 0 else None)

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title('Height vs Time', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.5, linewidth=1)

    # 2. 速度-时间轨迹
    ax2 = axes[0, 1]
    for name, trajectories in all_results.items():
        for i, traj in enumerate(trajectories):
            style = styles[name]
            alpha = 1.0 if i == 0 else 0.5
            ax2.plot(traj['times'], traj['velocities'],
                    color=style['color'],
                    linewidth=style['linewidth'],
                    linestyle=style['linestyle'],
                    alpha=alpha,
                    label=f'{name} (Ep {i+1})' if i == 0 else None)

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2.set_title('Velocity vs Time', fontsize=14)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.5, linewidth=1)
    ax2.axhline(y=-2, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Safe limit')
    ax2.axhline(y=2, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    # 3. 油门-时间轨迹
    ax3 = axes[1, 0]
    for name, trajectories in all_results.items():
        for i, traj in enumerate(trajectories):
            style = styles[name]
            alpha = 1.0 if i == 0 else 0.5
            ax3.plot(traj['times'], traj['throttles'],
                    color=style['color'],
                    linewidth=style['linewidth'],
                    linestyle=style['linestyle'],
                    alpha=alpha,
                    label=f'{name} (Ep {i+1})' if i == 0 else None)

    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Throttle', fontsize=12)
    ax3.set_title('Throttle vs Time', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)

    # 4. 燃料-时间轨迹
    ax4 = axes[1, 1]
    for name, trajectories in all_results.items():
        for i, traj in enumerate(trajectories):
            style = styles[name]
            alpha = 1.0 if i == 0 else 0.5
            ax4.plot(traj['times'], traj['fuels'],
                    color=style['color'],
                    linewidth=style['linewidth'],
                    linestyle=style['linestyle'],
                    alpha=alpha,
                    label=f'{name} (Ep {i+1})' if i == 0 else None)

    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Fuel (kg)', fontsize=12)
    ax4.set_title('Fuel vs Time', fontsize=14)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig("results/final_report/figures/controller_trajectory_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("  ✓ Controller trajectory comparison plot saved")


def plot_combined_trajectory(all_results):
    """绘制综合轨迹图（一张图展示所有信息）"""
    print("\n📊 Generating combined trajectory plot...")

    # 定义颜色和样式
    styles = {
        'PPO': {'color': '#2196F3', 'linewidth': 3},
        'PID': {'color': '#F44336', 'linewidth': 2},
        'MPC': {'color': '#4CAF50', 'linewidth': 2},
        'ET-MPC': {'color': '#FF9800', 'linewidth': 2},
    }

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Rocket Landing Trajectory Comparison', fontsize=16, fontweight='bold')

    # 绘制高度-时间轨迹
    for name, trajectories in all_results.items():
        # 只绘制第一个episode
        traj = trajectories[0]
        style = styles[name]

        # 绘制高度轨迹
        line, = ax.plot(traj['times'], traj['heights'],
                       color=style['color'],
                       linewidth=style['linewidth'],
                       label=f'{name}')

        # 用颜色表示速度（通过散点图）
        times = np.array(traj['times'])
        heights = np.array(traj['heights'])
        velocities = np.array(traj['velocities'])

        # 归一化速度用于颜色映射
        v_normalized = (velocities - velocities.min()) / (velocities.max() - velocities.min())

        # 绘制速度颜色带
        for i in range(len(times) - 1):
            color = plt.cm.RdYlGn_r(v_normalized[i])  # 红色=高速，绿色=低速
            ax.plot(times[i:i+2], heights[i:i+2], color=color, linewidth=style['linewidth'] + 2, alpha=0.3)

    # 添加安全着陆区域
    ax.axhspan(0, 2, alpha=0.1, color='green', label='Safe landing zone')
    ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0, color='green', linestyle='-', alpha=0.5, linewidth=1)

    # 添加目标线
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.5, linewidth=2, label='Ground')

    # 设置坐标轴
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title('Height-Time Trajectory with Speed Color Coding\n(Red=Fast, Green=Slow)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 添加速度颜色条
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Relative Speed')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Slow', 'Medium', 'Fast'])

    plt.tight_layout()
    plt.savefig("results/final_report/figures/combined_trajectory_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("  ✓ Combined trajectory comparison plot saved")


def main():
    print("="*60)
    print("Generating Controller Trajectory Comparison")
    print("="*60)

    # 定义控制器
    controllers = {
        'PPO': (PPOController, 'PPO'),
        'PID': (PIDController, 'PID'),
        'MPC': (MPCController, 'MPC'),
        'ET-MPC': (ETMPCController, 'ET-MPC'),
    }

    # 运行所有控制器的轨迹实验
    all_results = {}
    for name, (controller_class, controller_name) in controllers.items():
        trajectories = run_trajectory_experiment(controller_class, controller_name, n_episodes=3)
        all_results[name] = trajectories

    # 绘制对比图
    plot_trajectory_comparison(all_results)
    plot_combined_trajectory(all_results)

    # 保存结果
    results_summary = {}
    for name, trajectories in all_results.items():
        successes = sum(1 for t in trajectories if t['success'])
        results_summary[name] = {
            'success_rate': successes / len(trajectories),
            'mean_time': np.mean([t['times'][-1] for t in trajectories]),
            'mean_fuel_used': np.mean([5.0 - t['fuels'][-1] for t in trajectories]),
            'mean_final_velocity': np.mean([abs(t['velocities'][-1]) for t in trajectories])
        }

    # 保存结果
    output_path = Path("results/final_report/trajectory_comparison_results.json")
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    # 打印汇总
    print("\n" + "="*60)
    print("Controller Performance Summary")
    print("="*60)

    print(f"\n{'Controller':<10} {'Success Rate':<15} {'Mean Time':<12} {'Fuel Used':<12} {'Final Velocity':<15}")
    print("-" * 70)

    for name, summary in results_summary.items():
        print(f"{name:<10} {summary['success_rate']*100:.1f}%{'':<10} "
              f"{summary['mean_time']:.2f}s{'':<8} "
              f"{summary['mean_fuel_used']:.2f}kg{'':<8} "
              f"{summary['mean_final_velocity']:.2f}m/s")


if __name__ == "__main__":
    main()
