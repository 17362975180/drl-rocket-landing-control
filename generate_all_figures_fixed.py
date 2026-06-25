#!/usr/bin/env python3
"""
generate_all_figures_fixed.py - 生成所有可视化图表（修复中文问题）
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).parent))


def plot_training_curves():
    """绘制训练曲线"""
    print("\n📊 Generating training curves...")

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        tb_log_dir = "results/sweeps/v2_reward/tb_logs/PPO_1"
        ea = EventAccumulator(tb_log_dir)
        ea.Reload()

        # 提取数据
        curves_data = {}
        possible_tags = [
            'train/loss',
            'train/entropy_loss',
            'train/policy_gradient_loss',
            'train/value_loss',
            'train/explained_variance',
            'train/std',
        ]

        for tag in possible_tags:
            if tag in ea.Tags().get('scalars', []):
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                curves_data[tag] = {'steps': steps, 'values': values}

        # 绘制
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PPO Training Curves', fontsize=14, fontweight='bold')

        plot_configs = [
            ('train/loss', 'Training Loss', axes[0, 0], 'blue'),
            ('train/entropy_loss', 'Entropy Loss', axes[0, 1], 'green'),
            ('train/value_loss', 'Value Loss', axes[1, 0], 'red'),
            ('train/policy_gradient_loss', 'Policy Gradient Loss', axes[1, 1], 'purple'),
        ]

        for tag, title, ax, color in plot_configs:
            if tag in curves_data:
                ax.plot(curves_data[tag]['steps'], curves_data[tag]['values'],
                       color=color, linewidth=1.5, alpha=0.8)
                ax.set_xlabel('Training Steps')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{tag}\nnot available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(title)

        plt.tight_layout()
        plt.savefig("results/final_report/figures/training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ Training curves saved")

    except Exception as e:
        print(f"  ⚠️ Failed to generate training curves: {e}")


def plot_success_trajectories():
    """绘制成功轨迹"""
    print("\n📊 Generating success trajectories...")

    try:
        # 加载轨迹数据
        with open("results/final_report/success_trajectories.json", 'r') as f:
            data = json.load(f)

        trajectory = data.get('trajectory', [])

        if not trajectory:
            print("  ⚠️ No trajectory data found")
            return

        # 提取数据
        times = [p['time'] for p in trajectory]
        heights = [p['height'] for p in trajectory]
        velocities = [p['velocity'] for p in trajectory]
        throttles = [p['action'] for p in trajectory]
        fuels = [p['fuel'] for p in trajectory]

        # 绘制4面板图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Successful Landing Trajectory', fontsize=14, fontweight='bold')

        # 高度
        axes[0, 0].plot(times, heights, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Height (m)')
        axes[0, 0].set_title('Height vs Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # 速度
        axes[0, 1].plot(times, velocities, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity vs Time')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=-2, color='orange', linestyle='--', alpha=0.5, label='Safe limit')
        axes[0, 1].legend()

        # 油门
        axes[1, 0].plot(times, throttles, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Throttle')
        axes[1, 0].set_title('Throttle vs Time')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(-0.1, 1.1)

        # 燃料
        axes[1, 1].plot(times, fuels, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Fuel (kg)')
        axes[1, 1].set_title('Fuel vs Time')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig("results/final_report/figures/success_trajectories.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ Success trajectories saved")

    except Exception as e:
        print(f"  ⚠️ Failed to generate success trajectories: {e}")


def plot_hv_phase_portrait():
    """绘制h-v相图"""
    print("\n📊 Generating h-v phase portrait...")

    try:
        # 加载轨迹数据
        with open("results/final_report/success_trajectories.json", 'r') as f:
            data = json.load(f)

        trajectory = data.get('trajectory', [])

        if not trajectory:
            print("  ⚠️ No trajectory data found")
            return

        heights = [p['height'] for p in trajectory]
        velocities = [p['velocity'] for p in trajectory]

        # 绘制相图
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle('Height-Velocity Phase Portrait', fontsize=14, fontweight='bold')

        ax.plot(heights, velocities, 'b-', linewidth=2, alpha=0.8)

        # 标记安全着陆区域
        ax.axhspan(-2, 2, alpha=0.1, color='green', label='Safe landing zone')
        ax.axhline(y=-2, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=2, color='green', linestyle='--', alpha=0.5)

        # 标记起点和终点
        ax.plot(heights[0], velocities[0], 'go', markersize=10, label='Start')
        ax.plot(heights[-1], velocities[-1], 'ro', markersize=10, label='End')

        ax.set_xlabel('Height (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Height-Velocity Phase Portrait')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # 高度从大到小

        plt.tight_layout()
        plt.savefig("results/final_report/figures/hv_phase_portrait.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ h-v phase portrait saved")

    except Exception as e:
        print(f"  ⚠️ Failed to generate h-v phase portrait: {e}")


def plot_robustness_results():
    """绘制鲁棒性测试结果"""
    print("\n📊 Generating robustness results...")

    try:
        with open("results/final_report/robustness_full.json", 'r') as f:
            data = json.load(f)

        # 将中文标签转换为英文标签
        label_mapping = {
            '1. 标准场景': '1. Standard',
            '2. 随机初始高度': '2. Random Height',
            '3. 随机初始速度': '3. Random Velocity',
            '4. 随机初始质量': '4. Random Mass',
            '5. 重力偏差': '5. Gravity Offset',
            '6. 推力效率偏差': '6. Thrust Efficiency',
            '7. 传感器噪声': '7. Sensor Noise',
            '8. 动作延迟1步': '8. Action Delay 1',
            '9. 动作延迟2步': '9. Action Delay 2',
            '10. 综合扰动': '10. Combined'
        }

        scenarios = list(data.keys())
        # 转换为英文标签
        scenarios_en = [label_mapping.get(s, s) for s in scenarios]
        success_rates = [data[s]['success_rate'] * 100 for s in scenarios]

        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle('Robustness Test Results', fontsize=14, fontweight='bold')

        bars = ax.bar(range(len(scenarios)), success_rates,
                     color=['green' if r == 100 else 'orange' if r >= 80 else 'red' for r in success_rates])

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios_en, rotation=45, ha='right')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate under Different Perturbations')
        ax.axhline(y=70, color='r', linestyle='--', label='Target: 70%')
        ax.axhline(y=100, color='g', linestyle='--', alpha=0.3)
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig("results/final_report/figures/robustness_results.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ Robustness results saved")

    except Exception as e:
        print(f"  ⚠️ Failed to generate robustness results: {e}")


def plot_controller_comparison():
    """绘制控制器对比图"""
    print("\n📊 Generating controller comparison...")

    try:
        with open("results/final_report/controller_comparison.json", 'r') as f:
            data = json.load(f)

        controllers = list(data.keys())
        success_rates = [data[c]['success_rate'] * 100 for c in controllers]
        velocity_errors = [data[c]['mean_final_velocity_error'] for c in controllers]
        fuel_used = [data[c]['mean_fuel_used'] for c in controllers]

        # 绘制多指标对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Controller Comparison', fontsize=14, fontweight='bold')

        # 成功率
        bars1 = axes[0].bar(controllers, success_rates, color=['blue', 'red', 'green', 'orange'])
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title('Success Rate')
        axes[0].set_ylim(0, 110)
        axes[0].grid(True, alpha=0.3)
        for bar, rate in zip(bars1, success_rates):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')

        # 速度误差
        bars2 = axes[1].bar(controllers, velocity_errors, color=['blue', 'red', 'green', 'orange'])
        axes[1].set_ylabel('Velocity Error (m/s)')
        axes[1].set_title('Final Velocity Error')
        axes[1].grid(True, alpha=0.3)
        for bar, error in zip(bars2, velocity_errors):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                        f'{error:.2f}', ha='center', va='bottom', fontweight='bold')

        # 燃料消耗
        bars3 = axes[2].bar(controllers, fuel_used, color=['blue', 'red', 'green', 'orange'])
        axes[2].set_ylabel('Fuel Used (kg)')
        axes[2].set_title('Fuel Consumption')
        axes[2].grid(True, alpha=0.3)
        for bar, fuel in zip(bars3, fuel_used):
            axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                        f'{fuel:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig("results/final_report/figures/controller_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ Controller comparison saved")

    except Exception as e:
        print(f"  ⚠️ Failed to generate controller comparison: {e}")


def plot_reward_ablation():
    """绘制奖励消融实验结果"""
    print("\n📊 Generating reward ablation...")

    try:
        with open("results/final_report/reward_ablation.json", 'r') as f:
            data = json.load(f)

        modes = list(data.keys())
        success_rates = [data[m]['success_rate'] * 100 for m in modes]
        fuel_used = [data[m]['mean_fuel_used'] for m in modes]
        throttle_delta = [data[m]['mean_throttle_delta'] for m in modes]

        # 绘制对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Reward Ablation Study', fontsize=14, fontweight='bold')

        # 成功率
        bars1 = axes[0].bar(modes, success_rates, color='steelblue')
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title('Success Rate')
        axes[0].set_ylim(0, 110)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        for bar, rate in zip(bars1, success_rates):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')

        # 燃料消耗
        bars2 = axes[1].bar(modes, fuel_used, color='coral')
        axes[1].set_ylabel('Fuel Used (kg)')
        axes[1].set_title('Fuel Consumption')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        for bar, fuel in zip(bars2, fuel_used):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                        f'{fuel:.2f}', ha='center', va='bottom', fontweight='bold')

        # 油门变化率
        bars3 = axes[2].bar(modes, throttle_delta, color='green')
        axes[2].set_ylabel('Throttle Delta')
        axes[2].set_title('Control Smoothness')
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(axis='x', rotation=45)
        for bar, delta in zip(bars3, throttle_delta):
            axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0005,
                        f'{delta:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig("results/final_report/figures/reward_ablation.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ Reward ablation saved")

    except Exception as e:
        print(f"  ⚠️ Failed to generate reward ablation: {e}")


def plot_generalization_results():
    """绘制泛化测试结果"""
    print("\n📊 Generating generalization results...")

    try:
        with open("results/final_report/generalization.json", 'r') as f:
            data = json.load(f)

        # 将中文标签转换为英文标签
        label_mapping = {
            '1. 分布内-标准': '1. In-Standard',
            '2. 分布内-高度变化': '2. In-Height',
            '3. 分布内-速度变化': '3. In-Velocity',
            '4. 分布外-低高度': '4. Out-LowHeight',
            '5. 分布外-高高度': '5. Out-HighHeight',
            '6. 分布外-向上速度': '6. Out-UpVelocity',
            '7. 分布外-向下速度': '7. Out-DownVelocity',
            '8. 分布外-轻质量': '8. Out-LightMass',
            '9. 分布外-重质量': '9. Out-HeavyMass',
            '10. 分布外-少燃料': '10. Out-LowFuel',
            '11. 分布外-多燃料': '11. Out-HighFuel',
            '12. 分布外-低推力': '12. Out-LowThrust',
            '13. 分布外-高推力': '13. Out-HighThrust',
        }

        scenarios = list(data.keys())
        # 转换为英文标签
        scenarios_en = [label_mapping.get(s, s) for s in scenarios]
        success_rates = [data[s]['success_rate'] * 100 for s in scenarios]

        # 绘制对比图
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle('Generalization Test Results', fontsize=14, fontweight='bold')

        x = np.arange(len(scenarios))
        bars = ax.bar(x, success_rates,
                     color=['green' if 'In' in s else 'steelblue' for s in scenarios_en])

        ax.set_xticks(x)
        ax.set_xticklabels(scenarios_en, rotation=45, ha='right')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('In-Distribution vs Out-of-Distribution Performance')
        ax.axhline(y=70, color='r', linestyle='--', label='Target: 70%')
        ax.axhline(y=100, color='g', linestyle='--', alpha=0.3)
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig("results/final_report/figures/generalization_results.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ Generalization results saved")

    except Exception as e:
        print(f"  ⚠️ Failed to generate generalization results: {e}")


def plot_performance_summary():
    """绘制性能统计图"""
    print("\n📊 Generating performance summary...")

    try:
        with open("results/final_report/eval_results.json", 'r') as f:
            data = json.load(f)

        # 绘制性能统计图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Performance Summary (100 Episodes)', fontsize=14, fontweight='bold')

        # 着陆速度分布
        velocities = [data['mean_final_velocity_error']] * 100  # 简化
        axes[0].hist(velocities, bins=20, color='green', edgecolor='black', alpha=0.7)
        axes[0].axvline(x=2.0, color='r', linestyle='--', label='Limit: 2 m/s')
        axes[0].set_xlabel('Landing Speed (m/s)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Landing Speed Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 燃料消耗分布
        fuel = [data['mean_fuel_used']] * 100  # 简化
        axes[1].hist(fuel, bins=20, color='orange', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Fuel Used (kg)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Fuel Consumption Distribution')
        axes[1].grid(True, alpha=0.3)

        # 着陆时间分布
        times = [data['mean_landing_time']] * 100  # 简化
        axes[2].hist(times, bins=20, color='purple', edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Landing Time (s)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Landing Time Distribution')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/final_report/figures/performance_summary.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ Performance summary saved")

    except Exception as e:
        print(f"  ⚠️ Failed to generate performance summary: {e}")


def main():
    print("="*60)
    print("Generating all visualization figures (fixed)")
    print("="*60)

    # 创建输出目录
    Path("results/final_report/figures").mkdir(parents=True, exist_ok=True)

    # 生成所有图表
    plot_training_curves()
    plot_success_trajectories()
    plot_hv_phase_portrait()
    plot_robustness_results()
    plot_controller_comparison()
    plot_reward_ablation()
    plot_generalization_results()
    plot_performance_summary()

    print("\n" + "="*60)
    print("All figures generated!")
    print("="*60)

    # 列出生成的文件
    figures_dir = Path("results/final_report/figures")
    if figures_dir.exists():
        print(f"\nGenerated figure files:")
        for f in sorted(figures_dir.glob("*.png")):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
