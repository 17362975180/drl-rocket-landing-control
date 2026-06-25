#!/usr/bin/env python3
"""
传统控制器对比实验
包含：PPO/PID/MPC/ET-MPC 评估、鲁棒性测试、泛化性测试、轨迹对比图
"""

import sys
import os
import json
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.rocket_env import RocketLandingEnv

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 控制器定义
# ============================================================

class PPOController:
    """PPO控制器"""
    def __init__(self, model_path):
        self.model = PPO.load(model_path, device="cpu")
        with open(str(Path(model_path).parent / "vec_normalize.pkl"), 'rb') as f:
            stats = pickle.load(f)
        self.obs_rms = stats.obs_rms

    def predict(self, obs):
        normed = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10.0, 10.0)
        action, _ = self.model.predict(normed, deterministic=True)
        return float(action[0]) if hasattr(action, '__len__') else float(action)


class PIDController:
    """PID控制器 - 基于物理规则"""
    def __init__(self, dry_mass=10.0, g=9.81, T_max=300.0):
        self.dry_mass = dry_mass
        self.g = g
        self.T_max = T_max

    def predict(self, obs):
        height = float(obs[0]) * 50.0
        velocity = float(obs[1]) * 10.0
        fuel = float(obs[2]) * 5.0

        if height > 40:
            return -1.0  # 自由下落

        mass = self.dry_mass + fuel

        if height > 20:
            v_target = -2.0
            required_acceleration = (v_target**2 - velocity**2) / (2.0 * max(height, 0.1))
            required_acceleration = np.clip(required_acceleration, -20.0, 10.0)
            thrust = mass * (required_acceleration + self.g)
        else:
            target_v = -1.0
            error = target_v - velocity
            thrust = mass * (error * 3.0 + self.g)

        throttle = np.clip(thrust / self.T_max, 0.0, 1.0)
        return throttle * 2.0 - 1.0


class MPCController:
    """MPC控制器"""
    def __init__(self, horizon=10):
        self.horizon = horizon
        self.last_u = None

    def predict(self, obs):
        h = float(obs[0]) * 50.0
        v = float(obs[1]) * 10.0
        fuel = float(obs[2]) * 5.0

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
            args=(h, v, fuel),
            method='L-BFGS-B',
            bounds=[(0, 1)] * self.horizon,
            options={'maxiter': 15, 'ftol': 1e-6}
        )

        u_optimal = result.x[0]
        self.last_u = result.x

        return u_optimal * 2.0 - 1.0

    def _cost_function(self, u_seq, h, v, fuel):
        """代价函数"""
        cost = 0.0
        dt = 0.05

        for i in range(self.horizon):
            u = u_seq[i]
            mass = 10.0 + fuel
            thrust = u * 300.0
            acceleration = thrust / mass - 9.81 - 0.02 * v * abs(v)
            v = v + acceleration * dt
            h = h + v * dt

            cost += 10.0 * h**2
            cost += 5.0 * (v - (-3.0))**2
            cost += 0.1 * u**2

        # 终端代价
        cost += 100.0 * h**2
        cost += 100.0 * (v - (-2.0))**2

        return cost


class ETMPCController:
    """事件触发MPC控制器"""
    def __init__(self, horizon=10, threshold=0.5):
        self.mpc = MPCController(horizon)
        self.threshold = threshold
        self.last_state = None
        self.last_action = -1.0
        self.trigger_count = 0
        self.total_count = 0

    def predict(self, obs):
        self.total_count += 1

        if self.last_state is None:
            state_changed = True
        else:
            state_changed = np.any(np.abs(obs - self.last_state) > self.threshold)

        if state_changed:
            action = self.mpc.predict(obs)
            self.last_action = action
            self.last_state = obs.copy()
            self.trigger_count += 1
        else:
            action = self.last_action

        return action

    def get_trigger_rate(self):
        return self.trigger_count / max(self.total_count, 1)


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
# 评估函数
# ============================================================

def evaluate_controller(controller_name, controller, n_episodes=100, env=None):
    """评估单个控制器"""
    print(f"\n评估控制器: {controller_name}")

    if env is None:
        env = RocketLandingEnv()

    results = {
        'success': 0, 'crash': 0, 'timeout': 0,
        'fuel_used': [], 'landing_time': [], 'final_velocity': [],
        'throttle_delta': []
    }

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        throttles = []

        while not done:
            action = controller.predict(obs)
            throttle = (action + 1.0) / 2.0
            throttles.append(throttle)

            action_array = np.array([action], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action_array)
            done = terminated or truncated

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

def run_robustness_test(controller_name, controller, n_episodes=30):
    """对控制器进行鲁棒性测试"""
    print(f"\n鲁棒性测试: {controller_name}")

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
                action = controller.predict(obs)
                action_array = np.array([action], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action_array)
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

def run_generalization_test(controller_name, controller, n_episodes=30):
    """对控制器进行泛化性测试"""
    print(f"\n泛化性测试: {controller_name}")

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
                action = controller.predict(obs)
                action_array = np.array([action], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action_array)
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

def record_trajectory(controller_name, controller):
    """录制单个控制器的轨迹"""
    env = RocketLandingEnv()
    obs, _ = env.reset()

    heights, velocities, throttles, fuels = [], [], [], []
    done = False

    while not done:
        action = controller.predict(obs)
        throttle = (action + 1.0) / 2.0
        throttles.append(throttle)

        action_array = np.array([action], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action_array)
        done = terminated or truncated

        heights.append(float(obs[0]) * 50.0)
        velocities.append(float(obs[1]) * 10.0)
        fuels.append(float(obs[2]) * 5.0)

    return {
        'heights': np.array(heights),
        'velocities': np.array(velocities),
        'throttles': np.array(throttles),
        'fuels': np.array(fuels)
    }


# ============================================================
# 生成图表
# ============================================================

def generate_trajectory_comparison(controllers):
    """生成所有控制器的轨迹对比图"""
    print("\n生成轨迹对比图...")

    colors = ['blue', 'red', 'green', 'orange']
    labels = list(controllers.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, (name, controller) in enumerate(controllers.items()):
        try:
            trajectory = record_trajectory(name, controller)
            time = np.arange(len(trajectory['heights'])) * 0.05

            axes[0, 0].plot(time, trajectory['heights'], color=colors[i], label=labels[i], linewidth=2)
            axes[0, 1].plot(time, trajectory['velocities'], color=colors[i], label=labels[i], linewidth=2)
            axes[1, 0].plot(time, trajectory['throttles'], color=colors[i], label=labels[i], linewidth=2)
            axes[1, 1].plot(time, trajectory['fuels'], color=colors[i], label=labels[i], linewidth=2)
        except Exception as e:
            print(f"  {name} 轨迹录制失败: {e}")

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

    plt.suptitle('Controller Comparison: Trajectory', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent / 'figures' / 'trajectory_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> trajectory_comparison.png OK")


def generate_robustness_chart(robustness_by_scenario):
    """生成鲁棒性测试柱状图"""
    print("生成鲁棒性测试图...")

    scenarios = list(robustness_by_scenario.keys())
    controllers = list(robustness_by_scenario[scenarios[0]].keys())
    n_scenarios = len(scenarios)
    n_ctrls = len(controllers)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(n_scenarios)
    width = 0.2
    colors = ['blue', 'red', 'green', 'orange']

    for i, ctrl in enumerate(controllers):
        success_rates = [robustness_by_scenario[s][ctrl]['success_rate'] for s in scenarios]
        ax.bar(x + i * width, success_rates, width, label=ctrl, color=colors[i], alpha=0.8)

    ax.set_xlabel('Robustness Scenario')
    ax.set_ylabel('Success Rate')
    ax.set_title('Controller Robustness Comparison')
    ax.set_xticks(x + width * (n_ctrls - 1) / 2)
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
    controllers = list(generalization_by_scenario[scenarios[0]].keys())
    n_scenarios = len(scenarios)
    n_ctrls = len(controllers)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(n_scenarios)
    width = 0.2
    colors = ['blue', 'red', 'green', 'orange']

    for i, ctrl in enumerate(controllers):
        success_rates = [generalization_by_scenario[s][ctrl]['success_rate'] for s in scenarios]
        ax.bar(x + i * width, success_rates, width, label=ctrl, color=colors[i], alpha=0.8)

    ax.set_xlabel('Generalization Scenario')
    ax.set_ylabel('Success Rate')
    ax.set_title('Controller Generalization Comparison')
    ax.set_xticks(x + width * (n_ctrls - 1) / 2)
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
    print("传统控制器对比实验")
    print("=" * 60)

    results_dir = Path(__file__).parent / "results"
    figures_dir = Path(__file__).parent / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 初始化控制器
    ppo_controller = PPOController("results/sweeps/v2_reward/models/final_model")
    pid_controller = PIDController()
    mpc_controller = MPCController()
    et_mpc_controller = ETMPCController()

    controllers = {
        'PPO': ppo_controller,
        'PID': pid_controller,
        'MPC': mpc_controller,
        'ET-MPC': et_mpc_controller
    }

    # 1. 评估所有控制器
    print("\n" + "=" * 60)
    print("阶段1: 评估所有控制器")
    print("=" * 60)

    eval_results = {}
    for name, controller in controllers.items():
        try:
            stats = evaluate_controller(name, controller, n_episodes=100)
            eval_results[name] = stats
            print(f"  {name}: {stats['success_rate']*100:.1f}% success")
        except Exception as e:
            print(f"  {name} 评估失败: {e}")

    with open(str(results_dir / 'evaluation.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)

    # 2. 鲁棒性测试
    print("\n" + "=" * 60)
    print("阶段2: 鲁棒性测试")
    print("=" * 60)

    robustness_results = {}
    for name, controller in controllers.items():
        try:
            results = run_robustness_test(name, controller, n_episodes=30)
            robustness_results[name] = results
        except Exception as e:
            print(f"  {name} 鲁棒性测试失败: {e}")

    robustness_by_scenario = {}
    for ctrl, scenarios in robustness_results.items():
        for scenario, stats in scenarios.items():
            if scenario not in robustness_by_scenario:
                robustness_by_scenario[scenario] = {}
            robustness_by_scenario[scenario][ctrl] = stats

    with open(str(results_dir / 'robustness.json'), 'w') as f:
        json.dump(robustness_by_scenario, f, indent=2)

    # 3. 泛化性测试
    print("\n" + "=" * 60)
    print("阶段3: 泛化性测试")
    print("=" * 60)

    generalization_results = {}
    for name, controller in controllers.items():
        try:
            results = run_generalization_test(name, controller, n_episodes=30)
            generalization_results[name] = results
        except Exception as e:
            print(f"  {name} 泛化性测试失败: {e}")

    generalization_by_scenario = {}
    for ctrl, scenarios in generalization_results.items():
        for scenario, stats in scenarios.items():
            if scenario not in generalization_by_scenario:
                generalization_by_scenario[scenario] = {}
            generalization_by_scenario[scenario][ctrl] = stats

    with open(str(results_dir / 'generalization.json'), 'w') as f:
        json.dump(generalization_by_scenario, f, indent=2)

    # 4. 生成图表
    print("\n" + "=" * 60)
    print("阶段4: 生成图表")
    print("=" * 60)

    generate_trajectory_comparison(controllers)
    if robustness_by_scenario:
        generate_robustness_chart(robustness_by_scenario)
    if generalization_by_scenario:
        generate_generalization_chart(generalization_by_scenario)

    print("\n" + "=" * 60)
    print("传统控制器对比实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
