#!/usr/bin/env python3
"""
强化学习算法对比实验
包含：PPO/SAC/TD3 训练、评估、鲁棒性测试、泛化性测试、轨迹对比图
"""

import sys
import os
import json
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.rocket_env import RocketLandingEnv

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 带扰动的环境
# ============================================================

class PerturbedEnv(RocketLandingEnv):
    def __init__(self, perturbation_config=None):
        super().__init__()
        self.perturbation_config = perturbation_config or {}
        self.thrust_efficiency = 1.0
        self.sensor_noise_std = 0.0
        self.action_delay_steps = 0
        self.action_buffer = []

        if 'gravity_offset' in self.perturbation_config:
            self.g += self.perturbation_config['gravity_offset']
        if 'thrust_efficiency' in self.perturbation_config:
            self.thrust_efficiency = self.perturbation_config['thrust_efficiency']
        if 'sensor_noise_std' in self.perturbation_config:
            self.sensor_noise_std = self.perturbation_config['sensor_noise_std']
        if 'action_delay_steps' in self.perturbation_config:
            self.action_delay_steps = self.perturbation_config['action_delay_steps']

    def step(self, action):
        if self.action_delay_steps > 0:
            self.action_buffer.append(action.copy())
            if len(self.action_buffer) > self.action_delay_steps:
                delayed_action = self.action_buffer.pop(0)
            else:
                delayed_action = np.zeros_like(action)
        else:
            delayed_action = action

        original_T_max = self.T_max
        self.T_max *= self.thrust_efficiency
        obs, reward, terminated, truncated, info = super().step(delayed_action)
        self.T_max = original_T_max

        if self.sensor_noise_std > 0:
            noise = np.random.normal(0, self.sensor_noise_std, obs.shape)
            obs = obs + noise

        return obs, reward, terminated, truncated, info


# ============================================================
# 训练函数
# ============================================================

def train_algorithm(algorithm_name, total_timesteps=300000):
    """训练单个RL算法"""
    print(f"\n{'='*60}")
    print(f"训练算法: {algorithm_name}")
    print(f"{'='*60}")

    env = RocketLandingEnv()
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 选择算法
    if algorithm_name == "PPO":
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            verbose=1, device="cpu"
        )
    elif algorithm_name == "SAC":
        model = SAC(
            "MlpPolicy", env,
            learning_rate=3e-4, buffer_size=100000, batch_size=256,
            gamma=0.99, tau=0.005, ent_coef='auto',
            verbose=1, device="cpu"
        )
    elif algorithm_name == "TD3":
        model = TD3(
            "MlpPolicy", env,
            learning_rate=3e-4, buffer_size=100000, batch_size=256,
            gamma=0.99, tau=0.005,
            verbose=1, device="cpu"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # 保存模型
    model_dir = Path(__file__).parent / "models" / algorithm_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / "model"))
    env.save(str(model_dir / "vec_normalize.pkl"))

    print(f"模型已保存到: {model_dir}")
    return model, env


# ============================================================
# 评估函数
# ============================================================

def evaluate_algorithm(algorithm_name, n_episodes=100):
    """评估单个RL算法"""
    print(f"\n评估算法: {algorithm_name}")

    model_dir = Path(__file__).parent / "models" / algorithm_name

    # 加载模型
    if algorithm_name == "PPO":
        model = PPO.load(str(model_dir / "model"), device="cpu")
    elif algorithm_name == "SAC":
        model = SAC.load(str(model_dir / "model"), device="cpu")
    elif algorithm_name == "TD3":
        model = TD3.load(str(model_dir / "model"), device="cpu")

    env = RocketLandingEnv()
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), env)
    env.training = False
    env.norm_reward = False

    results = {
        'success': 0, 'crash': 0, 'timeout': 0,
        'fuel_used': [], 'landing_time': [], 'final_velocity': [],
        'throttle_delta': []
    }

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        throttles = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            raw_action = float(action[0][0]) if hasattr(action, '__len__') else float(action)
            throttle = (raw_action + 1.0) / 2.0
            throttles.append(throttle)

        info = info[0] if isinstance(info, list) else info
        if info.get('success', False):
            results['success'] += 1
            results['fuel_used'].append(info.get('fuel_used', 0))
            results['landing_time'].append(info.get('time', 0))
            results['final_velocity'].append(abs(info.get('velocity', 0)))
        elif info.get('crash', False):
            results['crash'] += 1
        else:
            results['timeout'] += 1

        if len(throttles) > 1:
            delta = np.mean(np.abs(np.diff(throttles)))
            results['throttle_delta'].append(delta)

    stats = {
        'success_rate': results['success'] / n_episodes,
        'crash_rate': results['crash'] / n_episodes,
        'timeout_rate': results['timeout'] / n_episodes,
        'mean_fuel_used': np.mean(results['fuel_used']) if results['fuel_used'] else 0,
        'mean_landing_time': np.mean(results['landing_time']) if results['landing_time'] else 0,
        'mean_final_velocity': np.mean(results['final_velocity']) if results['final_velocity'] else 0,
        'mean_throttle_delta': np.mean(results['throttle_delta']) if results['throttle_delta'] else 0,
        'n_episodes': n_episodes
    }

    return stats


# ============================================================
# 鲁棒性测试
# ============================================================

def run_robustness_test(algorithm_name, n_episodes=30):
    """对RL算法进行鲁棒性测试"""
    print(f"\n鲁棒性测试: {algorithm_name}")

    model_dir = Path(__file__).parent / "models" / algorithm_name

    if algorithm_name == "PPO":
        model = PPO.load(str(model_dir / "model"), device="cpu")
    elif algorithm_name == "SAC":
        model = SAC.load(str(model_dir / "model"), device="cpu")
    elif algorithm_name == "TD3":
        model = TD3.load(str(model_dir / "model"), device="cpu")

    with open(str(model_dir / "vec_normalize.pkl"), 'rb') as f:
        stats = pickle.load(f)
    obs_rms = stats.obs_rms

    def normalize_obs(obs):
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

    scenarios = {
        'Standard': {},
        'Gravity+0.5': {'gravity_offset': 0.5},
        'Gravity-0.5': {'gravity_offset': -0.5},
        'ThrustEff_0.9': {'thrust_efficiency': 0.9},
        'ThrustEff_0.8': {'thrust_efficiency': 0.8},
        'SensorNoise_0.01': {'sensor_noise_std': 0.01},
        'SensorNoise_0.02': {'sensor_noise_std': 0.02},
        'ActionDelay_1': {'action_delay_steps': 1},
        'ActionDelay_2': {'action_delay_steps': 2},
        'Combined': {'gravity_offset': 0.3, 'thrust_efficiency': 0.9, 'sensor_noise_std': 0.01}
    }

    results = {}
    for scenario_name, config in scenarios.items():
        env = PerturbedEnv(config)
        success = 0
        fuel_list = []
        velocity_list = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                normed = normalize_obs(obs)
                action, _ = model.predict(normed, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            if info.get('success', False):
                success += 1
                fuel_list.append(info.get('fuel_used', 0))
                velocity_list.append(abs(info.get('velocity', 0)))

        results[scenario_name] = {
            'success_rate': success / n_episodes,
            'mean_fuel': np.mean(fuel_list) if fuel_list else 0,
            'mean_velocity': np.mean(velocity_list) if velocity_list else 0,
            'n_episodes': n_episodes
        }

    return results


# ============================================================
# 泛化性测试
# ============================================================

def run_generalization_test(algorithm_name, n_episodes=30):
    """对RL算法进行泛化性测试"""
    print(f"\n泛化性测试: {algorithm_name}")

    model_dir = Path(__file__).parent / "models" / algorithm_name

    if algorithm_name == "PPO":
        model = PPO.load(str(model_dir / "model"), device="cpu")
    elif algorithm_name == "SAC":
        model = SAC.load(str(model_dir / "model"), device="cpu")
    elif algorithm_name == "TD3":
        model = TD3.load(str(model_dir / "model"), device="cpu")

    with open(str(model_dir / "vec_normalize.pkl"), 'rb') as f:
        stats = pickle.load(f)
    obs_rms = stats.obs_rms

    def normalize_obs(obs):
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

    scenarios = {
        'InDist-Standard': {'initial_height': 50.0, 'initial_velocity': 0.0},
        'InDist-HeightVar': {'initial_height_range': (40.0, 60.0)},
        'InDist-VelVar': {'initial_velocity_range': (-2.0, 2.0)},
        'OutDist-LowHeight': {'initial_height': 25.0, 'initial_velocity': 0.0},
        'OutDist-HighHeight': {'initial_height': 80.0, 'initial_velocity': 0.0},
        'OutDist-UpVel': {'initial_height': 50.0, 'initial_velocity': 5.0},
        'OutDist-DownVel': {'initial_height': 50.0, 'initial_velocity': -5.0},
        'OutDist-LightMass': {'dry_mass_offset': -3.0},
        'OutDist-HeavyMass': {'dry_mass_offset': 5.0},
        'OutDist-LowFuel': {'initial_fuel': 2.0},
        'OutDist-HighFuel': {'initial_fuel': 8.0}
    }

    results = {}
    for scenario_name, config in scenarios.items():
        env = RocketLandingEnv()

        if 'initial_height' in config:
            env.initial_height = config['initial_height']
        if 'initial_velocity' in config:
            env.initial_velocity = config['initial_velocity']
        if 'initial_height_range' in config:
            h_min, h_max = config['initial_height_range']
            env.initial_height = np.random.uniform(h_min, h_max)
        if 'initial_velocity_range' in config:
            v_min, v_max = config['initial_velocity_range']
            env.initial_velocity = np.random.uniform(v_min, v_max)
        if 'dry_mass_offset' in config:
            env.dry_mass += config['dry_mass_offset']
        if 'initial_fuel' in config:
            env.initial_fuel = config['initial_fuel']

        success = 0
        fuel_list = []
        velocity_list = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                normed = normalize_obs(obs)
                action, _ = model.predict(normed, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            if info.get('success', False):
                success += 1
                fuel_list.append(info.get('fuel_used', 0))
                velocity_list.append(abs(info.get('velocity', 0)))

        results[scenario_name] = {
            'success_rate': success / n_episodes,
            'mean_fuel': np.mean(fuel_list) if fuel_list else 0,
            'mean_velocity': np.mean(velocity_list) if velocity_list else 0,
            'n_episodes': n_episodes
        }

    return results


# ============================================================
# 轨迹录制
# ============================================================

def record_trajectory(algorithm_name):
    """录制单个算法的轨迹"""
    model_dir = Path(__file__).parent / "models" / algorithm_name

    if algorithm_name == "PPO":
        model = PPO.load(str(model_dir / "model"), device="cpu")
    elif algorithm_name == "SAC":
        model = SAC.load(str(model_dir / "model"), device="cpu")
    elif algorithm_name == "TD3":
        model = TD3.load(str(model_dir / "model"), device="cpu")

    env = RocketLandingEnv()
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()
    heights, velocities, throttles, fuels = [], [], [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        raw_obs = obs[0] if hasattr(obs, '__len__') else obs
        heights.append(float(raw_obs[0]) * 50.0)
        velocities.append(float(raw_obs[1]) * 10.0)
        fuels.append(float(raw_obs[2]) * 5.0)
        throttle = (float(action[0][0]) + 1.0) / 2.0 if hasattr(action, '__len__') else (float(action) + 1.0) / 2.0
        throttles.append(throttle)

    return {
        'heights': np.array(heights),
        'velocities': np.array(velocities),
        'throttles': np.array(throttles),
        'fuels': np.array(fuels)
    }


# ============================================================
# 生成图表
# ============================================================

def generate_trajectory_comparison():
    """生成所有RL算法的轨迹对比图"""
    print("\n生成轨迹对比图...")

    algorithms = ['PPO', 'SAC', 'TD3']
    colors = ['blue', 'red', 'green']
    labels = ['PPO', 'SAC', 'TD3']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, algo in enumerate(algorithms):
        try:
            trajectory = record_trajectory(algo)
            time = np.arange(len(trajectory['heights'])) * 0.05

            axes[0, 0].plot(time, trajectory['heights'], color=colors[i], label=labels[i], linewidth=2)
            axes[0, 1].plot(time, trajectory['velocities'], color=colors[i], label=labels[i], linewidth=2)
            axes[1, 0].plot(time, trajectory['throttles'], color=colors[i], label=labels[i], linewidth=2)
            axes[1, 1].plot(time, trajectory['fuels'], color=colors[i], label=labels[i], linewidth=2)
        except Exception as e:
            print(f"  {algo} 轨迹录制失败: {e}")

    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Height (m)')
    axes[0, 0].set_title('Height Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity Trajectory Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle=':', alpha=0.5)

    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Throttle')
    axes[1, 0].set_title('Throttle Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.05, 1.05)

    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Fuel (kg)')
    axes[1, 1].set_title('Fuel Consumption Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('RL Algorithm Comparison: Trajectory', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent / 'figures' / 'trajectory_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> trajectory_comparison.png OK")


def generate_robustness_chart(robustness_by_scenario):
    """生成鲁棒性测试柱状图"""
    print("生成鲁棒性测试图...")

    scenarios = list(robustness_by_scenario.keys())
    algorithms = list(robustness_by_scenario[scenarios[0]].keys())
    n_scenarios = len(scenarios)
    n_algos = len(algorithms)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(n_scenarios)
    width = 0.25
    colors = ['blue', 'red', 'green']

    for i, algo in enumerate(algorithms):
        success_rates = [robustness_by_scenario[s][algo]['success_rate'] for s in scenarios]
        ax.bar(x + i * width, success_rates, width, label=algo, color=colors[i], alpha=0.8)

    ax.set_xlabel('Robustness Scenario')
    ax.set_ylabel('Success Rate')
    ax.set_title('RL Algorithm Robustness Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent / 'figures' / 'robustness_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> robustness_comparison.png OK")


def generate_generalization_chart(generalization_by_scenario):
    """生成泛化性测试图"""
    print("生成泛化性测试图...")

    scenarios = list(generalization_by_scenario.keys())
    algorithms = list(generalization_by_scenario[scenarios[0]].keys())
    n_scenarios = len(scenarios)
    n_algos = len(algorithms)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(n_scenarios)
    width = 0.25
    colors = ['blue', 'red', 'green']

    for i, algo in enumerate(algorithms):
        success_rates = [generalization_by_scenario[s][algo]['success_rate'] for s in scenarios]
        ax.bar(x + i * width, success_rates, width, label=algo, color=colors[i], alpha=0.8)

    ax.set_xlabel('Generalization Scenario')
    ax.set_ylabel('Success Rate')
    ax.set_title('RL Algorithm Generalization Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent / 'figures' / 'generalization_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> generalization_comparison.png OK")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("强化学习算法对比实验")
    print("=" * 60)

    results_dir = Path(__file__).parent / "results"
    figures_dir = Path(__file__).parent / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ['PPO', 'SAC', 'TD3']

    # 1. 评估所有算法
    print("\n" + "=" * 60)
    print("阶段1: 评估所有算法")
    print("=" * 60)

    eval_results = {}
    for algo in algorithms:
        try:
            stats = evaluate_algorithm(algo, n_episodes=100)
            eval_results[algo] = stats
            print(f"  {algo}: {stats['success_rate']*100:.1f}% success")
        except Exception as e:
            print(f"  {algo} 评估失败: {e}")

    with open(str(results_dir / 'evaluation.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)

    # 2. 鲁棒性测试
    print("\n" + "=" * 60)
    print("阶段2: 鲁棒性测试")
    print("=" * 60)

    robustness_results = {}
    for algo in algorithms:
        try:
            results = run_robustness_test(algo, n_episodes=30)
            robustness_results[algo] = results
        except Exception as e:
            print(f"  {algo} 鲁棒性测试失败: {e}")

    # 转换格式
    robustness_by_scenario = {}
    for algo, scenarios in robustness_results.items():
        for scenario, stats in scenarios.items():
            if scenario not in robustness_by_scenario:
                robustness_by_scenario[scenario] = {}
            robustness_by_scenario[scenario][algo] = stats

    with open(str(results_dir / 'robustness.json'), 'w') as f:
        json.dump(robustness_by_scenario, f, indent=2)

    # 3. 泛化性测试
    print("\n" + "=" * 60)
    print("阶段3: 泛化性测试")
    print("=" * 60)

    generalization_results = {}
    for algo in algorithms:
        try:
            results = run_generalization_test(algo, n_episodes=30)
            generalization_results[algo] = results
        except Exception as e:
            print(f"  {algo} 泛化性测试失败: {e}")

    generalization_by_scenario = {}
    for algo, scenarios in generalization_results.items():
        for scenario, stats in scenarios.items():
            if scenario not in generalization_by_scenario:
                generalization_by_scenario[scenario] = {}
            generalization_by_scenario[scenario][algo] = stats

    with open(str(results_dir / 'generalization.json'), 'w') as f:
        json.dump(generalization_by_scenario, f, indent=2)

    # 4. 生成图表
    print("\n" + "=" * 60)
    print("阶段4: 生成图表")
    print("=" * 60)

    generate_trajectory_comparison()
    if robustness_by_scenario:
        generate_robustness_chart(robustness_by_scenario)
    if generalization_by_scenario:
        generate_generalization_chart(generalization_by_scenario)

    print("\n" + "=" * 60)
    print("RL算法对比实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
