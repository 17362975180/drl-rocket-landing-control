#!/usr/bin/env python3
"""
生成各实验的轨迹对比图：
- 大实验2：安全机制对比轨迹图
- 大实验3：综合扰动典型轨迹图
- 大实验5：奖励消融轨迹对比图
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import pickle

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from rocket_landing_control.envs.rocket_env import RocketLandingEnv
from rocket_landing_control.envs.rocket_env_safe import RocketLandingEnvSafe

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def normalize_obs(obs, obs_rms):
    """归一化观测"""
    return np.clip(
        (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0
    )


def load_model_and_stats():
    """加载模型和归一化统计"""
    model = PPO.load("results/sweeps/v2_reward/models/final_model", device="cpu")
    with open("results/sweeps/v2_reward/models/vec_normalize.pkl", 'rb') as f:
        stats = pickle.load(f)
    return model, stats.obs_rms


def run_episode(model, obs_rms, env):
    """运行一个episode，返回轨迹数据"""
    obs, _ = env.reset()

    heights = []
    velocities = []
    throttles = []
    fuels = []

    done = False
    while not done:
        normed = normalize_obs(obs, obs_rms)
        action, _ = model.predict(normed, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        heights.append(float(obs[0]) * 50.0)
        velocities.append(float(obs[1]) * 10.0)
        fuels.append(float(obs[2]) * 5.0)
        throttle = (float(action[0]) + 1.0) / 2.0
        throttles.append(throttle)

    return {
        'heights': np.array(heights),
        'velocities': np.array(velocities),
        'throttles': np.array(throttles),
        'fuels': np.array(fuels),
        'info': info
    }


def generate_safety_comparison_trajectory():
    """生成大实验2：安全机制对比轨迹图"""
    print("生成安全机制对比轨迹图...")

    model, obs_rms = load_model_and_stats()

    # 基础环境
    env_basic = RocketLandingEnv()
    trajectory_basic = run_episode(model, obs_rms, env_basic)

    # 安全环境
    env_safe = RocketLandingEnvSafe()
    trajectory_safe = run_episode(model, obs_rms, env_safe)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    time_basic = np.arange(len(trajectory_basic['heights'])) * 0.05
    time_safe = np.arange(len(trajectory_safe['heights'])) * 0.05

    axes[0, 0].plot(time_basic, trajectory_basic['heights'], 'b-', label='Without Safety', linewidth=2)
    axes[0, 0].plot(time_safe, trajectory_safe['heights'], 'r--', label='With Safety', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Height (m)')
    axes[0, 0].set_title('Height Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time_basic, trajectory_basic['velocities'], 'b-', label='Without Safety', linewidth=2)
    axes[0, 1].plot(time_safe, trajectory_safe['velocities'], 'r--', label='With Safety', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity Trajectory Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle=':', alpha=0.5)

    axes[1, 0].plot(time_basic, trajectory_basic['throttles'], 'b-', label='Without Safety', linewidth=2)
    axes[1, 0].plot(time_safe, trajectory_safe['throttles'], 'r--', label='With Safety', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Throttle')
    axes[1, 0].set_title('Throttle Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.05, 1.05)

    axes[1, 1].plot(time_basic, trajectory_basic['fuels'], 'b-', label='Without Safety', linewidth=2)
    axes[1, 1].plot(time_safe, trajectory_safe['fuels'], 'r--', label='With Safety', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Fuel (kg)')
    axes[1, 1].set_title('Fuel Consumption Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Experiment 2: Safety Mechanism Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/final_report/figures/safety_trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> safety_trajectory_comparison.png OK")


class PerturbedEnv(RocketLandingEnv):
    """带扰动的环境"""

    def __init__(self, perturbation_config=None):
        super().__init__()
        self.perturbation_config = perturbation_config or {}
        self.thrust_efficiency = 1.0
        self.sensor_noise_std = 0.0
        self.action_delay_steps = 0
        self.action_buffer = []

        if 'gravity_offset' in self.perturbation_config:
            offset = self.perturbation_config['gravity_offset']
            self.g += offset

        if 'thrust_efficiency' in self.perturbation_config:
            self.thrust_efficiency = self.perturbation_config['thrust_efficiency']

        if 'sensor_noise_std' in self.perturbation_config:
            self.sensor_noise_std = self.perturbation_config['sensor_noise_std']

        if 'action_delay_steps' in self.perturbation_config:
            self.action_delay_steps = self.perturbation_config['action_delay_steps']

    def step(self, action):
        # 动作延迟
        if self.action_delay_steps > 0:
            self.action_buffer.append(action.copy())
            if len(self.action_buffer) > self.action_delay_steps:
                delayed_action = self.action_buffer.pop(0)
            else:
                delayed_action = np.zeros_like(action)
        else:
            delayed_action = action

        # 推力效率
        original_T_max = self.T_max
        self.T_max *= self.thrust_efficiency
        obs, reward, terminated, truncated, info = super().step(delayed_action)
        self.T_max = original_T_max

        # 传感器噪声
        if self.sensor_noise_std > 0:
            noise = np.random.normal(0, self.sensor_noise_std, obs.shape)
            obs = obs + noise

        return obs, reward, terminated, truncated, info


def generate_robustness_trajectory():
    """生成大实验3：综合扰动典型轨迹图"""
    print("生成综合扰动典型轨迹图...")

    model, obs_rms = load_model_and_stats()

    # 标准场景
    env_standard = RocketLandingEnv()
    trajectory_standard = run_episode(model, obs_rms, env_standard)

    # 综合扰动场景
    env_perturbed = PerturbedEnv({
        'gravity_offset': 0.5,
        'thrust_efficiency': 0.9,
        'sensor_noise_std': 0.01,
        'action_delay_steps': 1
    })
    trajectory_perturbed = run_episode(model, obs_rms, env_perturbed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    time_std = np.arange(len(trajectory_standard['heights'])) * 0.05
    time_pert = np.arange(len(trajectory_perturbed['heights'])) * 0.05

    axes[0, 0].plot(time_std, trajectory_standard['heights'], 'b-', label='Standard', linewidth=2)
    axes[0, 0].plot(time_pert, trajectory_perturbed['heights'], 'r--', label='Combined Perturbation', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Height (m)')
    axes[0, 0].set_title('Height: Standard vs Combined Perturbation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time_std, trajectory_standard['velocities'], 'b-', label='Standard', linewidth=2)
    axes[0, 1].plot(time_pert, trajectory_perturbed['velocities'], 'r--', label='Combined Perturbation', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity: Standard vs Combined Perturbation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle=':', alpha=0.5)

    axes[1, 0].plot(time_std, trajectory_standard['throttles'], 'b-', label='Standard', linewidth=2)
    axes[1, 0].plot(time_pert, trajectory_perturbed['throttles'], 'r--', label='Combined Perturbation', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Throttle')
    axes[1, 0].set_title('Throttle: Standard vs Combined Perturbation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.05, 1.05)

    axes[1, 1].plot(time_std, trajectory_standard['fuels'], 'b-', label='Standard', linewidth=2)
    axes[1, 1].plot(time_pert, trajectory_perturbed['fuels'], 'r--', label='Combined Perturbation', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Fuel (kg)')
    axes[1, 1].set_title('Fuel: Standard vs Combined Perturbation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Experiment 3: Robustness - Combined Perturbation Trajectory', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/final_report/figures/robustness_trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> robustness_trajectory_comparison.png OK")


def generate_ablation_trajectory():
    """生成大实验5：奖励消融轨迹对比图"""
    print("生成奖励消融轨迹对比图...")

    variants = ['full', 'no_fuel', 'no_smooth', 'no_safety', 'no_success', 'basic']
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    labels = ['Full Reward', 'No Fuel Penalty', 'No Smooth Penalty',
              'No Safety Penalty', 'No Success Bonus', 'Basic Only']

    model, obs_rms = load_model_and_stats()
    trajectories = {}

    for variant in variants:
        model_path = f"results/final_report/ablation/{variant}/model"
        if Path(model_path + ".zip").exists():
            try:
                ablation_model = PPO.load(model_path, device="cpu")
                env = RocketLandingEnv()
                trajectory = run_episode(ablation_model, obs_rms, env)
                trajectories[variant] = trajectory
                print(f"  {variant} OK")
            except Exception as e:
                print(f"  {variant} FAIL: {e}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, (variant, trajectory) in enumerate(trajectories.items()):
        time = np.arange(len(trajectory['heights'])) * 0.05
        color = colors[i]
        label = labels[i]

        axes[0, 0].plot(time, trajectory['heights'], color=color, label=label, linewidth=1.5, alpha=0.8)
        axes[0, 1].plot(time, trajectory['velocities'], color=color, label=label, linewidth=1.5, alpha=0.8)
        axes[1, 0].plot(time, trajectory['throttles'], color=color, label=label, linewidth=1.5, alpha=0.8)
        axes[1, 1].plot(time, trajectory['fuels'], color=color, label=label, linewidth=1.5, alpha=0.8)

    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Height (m)')
    axes[0, 0].set_title('Height Trajectory Comparison')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity Trajectory Comparison')
    axes[0, 1].legend(loc='lower left', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle=':', alpha=0.5)

    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Throttle')
    axes[1, 0].set_title('Throttle Comparison')
    axes[1, 0].legend(loc='upper right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.05, 1.05)

    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Fuel (kg)')
    axes[1, 1].set_title('Fuel Consumption Comparison')
    axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Experiment 5: Reward Ablation Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/final_report/figures/ablation_trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> ablation_trajectory_comparison.png OK")


def main():
    print("=" * 60)
    print("生成各实验轨迹对比图")
    print("=" * 60)

    Path("results/final_report/figures").mkdir(parents=True, exist_ok=True)

    generate_safety_comparison_trajectory()
    generate_robustness_trajectory()
    generate_ablation_trajectory()

    print("\n" + "=" * 60)
    print("所有轨迹对比图生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
