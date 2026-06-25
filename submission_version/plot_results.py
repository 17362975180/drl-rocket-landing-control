"""
plot_results.py - 可视化脚本
生成训练曲线、轨迹图、h-v相图等。

用法:
    python plot_results.py --result-dir results/sweeps/lr3e4_s42
    python plot_results.py --eval-dir results/final_eval
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_curves(eval_history_path, output_dir):
    """从quick_eval_history.json绘制训练曲线"""
    if not os.path.exists(eval_history_path):
        print(f"  WARNING: No eval history found: {eval_history_path}")
        return

    with open(eval_history_path) as f:
        history = json.load(f)

    if len(history) < 2:
        print(f"  WARNING: Not enough data points ({len(history)})")
        return

    steps = [e.get("step", i * 5000) for i, e in enumerate(history)]
    success_rates = [e["success_rate"] for e in history]
    crash_rates = [e["crash_rate"] for e in history]
    v_errors = [e["mean_final_velocity_error"] for e in history]
    h_errors = [e["mean_final_height_error"] for e in history]
    fuels = [e["mean_fuel_used"] for e in history]
    rewards = [e["mean_episode_reward"] for e in history]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training Progress", fontsize=14)

    # Success rate
    ax = axes[0, 0]
    ax.plot(steps, success_rates, 'g-o', linewidth=2, markersize=4)
    ax.fill_between(steps, success_rates, alpha=0.2, color='green')
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Crash rate
    ax = axes[0, 1]
    ax.plot(steps, crash_rates, 'r-o', linewidth=2, markersize=4)
    ax.fill_between(steps, crash_rates, alpha=0.2, color='red')
    ax.set_ylabel("Crash Rate")
    ax.set_title("Crash Rate")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Velocity error
    ax = axes[0, 2]
    ax.plot(steps, v_errors, 'b-o', linewidth=2, markersize=4)
    ax.set_ylabel("Mean |v_final| (m/s)")
    ax.set_title("Final Velocity Error")
    ax.grid(True, alpha=0.3)

    # Height error
    ax = axes[1, 0]
    ax.plot(steps, h_errors, 'm-o', linewidth=2, markersize=4)
    ax.set_ylabel("Mean |h_final| (m)")
    ax.set_title("Final Height Error")
    ax.grid(True, alpha=0.3)

    # Fuel used
    ax = axes[1, 1]
    ax.plot(steps, fuels, 'c-o', linewidth=2, markersize=4)
    ax.set_ylabel("Mean Fuel Used")
    ax.set_title("Fuel Consumption")
    ax.grid(True, alpha=0.3)

    # Episode reward
    ax = axes[1, 2]
    ax.plot(steps, rewards, 'y-o', linewidth=2, markersize=4)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Episode Reward")
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Training Steps")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved to: {fig_path}")


def plot_trajectory(trajectory_data, output_dir, prefix=""):
    """绘制轨迹图"""
    if not trajectory_data or "trajectory" not in trajectory_data:
        print(f"  WARNING: No trajectory data")
        return

    traj = trajectory_data["trajectory"]
    if not traj:
        return

    t = [p["time"] for p in traj]
    h = [p["height"] for p in traj]
    v = [p["velocity"] for p in traj]
    thrust = [p["thrust"] for p in traj]
    fuel = [p.get("fuel", np.nan) for p in traj]
    mass = [p.get("mass", np.nan) for p in traj]
    rewards = [p.get("reward", np.nan) for p in traj]

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    terminal = trajectory_data.get("terminal_reason", trajectory_data.get("terminal", "unknown"))
    fig.suptitle(f"Trajectory ({terminal})", fontsize=14)

    # 高度
    ax = axes[0, 0]
    ax.plot(t, h, 'b-', linewidth=2)
    ax.set_ylabel("Height (m)")
    ax.set_title("Height vs Time")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 速度
    ax = axes[0, 1]
    ax.plot(t, v, 'r-', linewidth=2)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity vs Time")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 推力/控制输入
    ax = axes[1, 0]
    ax.plot(t, thrust, 'g-', linewidth=2)
    ax.set_ylabel("Thrust (N)")
    ax.set_title("Control Input")
    ax.grid(True, alpha=0.3)

    # h-v 相图
    ax = axes[1, 1]
    ax.plot(h, v, 'purple', linewidth=2, alpha=0.7)
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("h-v Phase Portrait")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    ax = axes[2, 0]
    ax.plot(t, fuel, 'tab:orange', linewidth=2, label="Fuel")
    ax.plot(t, mass, 'tab:brown', linewidth=2, label="Mass")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("kg")
    ax.set_title("Fuel and Mass")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t, rewards, 'tab:blue', linewidth=1.5, label="Step reward")
    if "reward_breakdown" in traj[-1]:
        keys = sorted(traj[-1]["reward_breakdown"].keys())
        for key in keys:
            values = [p.get("reward_breakdown", {}).get(key, 0.0) for p in traj]
            if any(abs(x) > 1e-9 for x in values):
                ax.plot(t, values, linewidth=1.0, alpha=0.65, label=key)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Decomposition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        if ax not in (axes[1, 1], axes[2, 0], axes[2, 1]):
            ax.set_xlabel("Time (s)")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{prefix}trajectory.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Trajectory saved to: {fig_path}")


def plot_hv_phase(trajectory_data, output_dir, prefix=""):
    """绘制h-v相图"""
    if not trajectory_data or "trajectory" not in trajectory_data:
        return

    traj = trajectory_data["trajectory"]
    if not traj:
        return

    h = [p["height"] for p in traj]
    v = [p["velocity"] for p in traj]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(h, v, 'purple', linewidth=2, alpha=0.7)
    ax.scatter(h[0], v[0], color='green', s=100, zorder=5, label='Start')
    ax.scatter(h[-1], v[-1], color='red', s=100, zorder=5, label='End')

    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("h-v Phase Portrait")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{prefix}hv_phase.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  h-v phase portrait saved to: {fig_path}")


def plot_eval_summary(eval_results_path, output_dir):
    """绘制评估统计图"""
    if not os.path.exists(eval_results_path):
        print(f"  WARNING: No eval results found: {eval_results_path}")
        return

    with open(eval_results_path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Evaluation Summary", fontsize=14)

    # 终止原因饼图
    ax = axes[0]
    counts = results.get("terminal_reason_counts", {})
    labels = []
    sizes = []
    colors = []
    color_map = {"success": "green", "crash": "red", "timeout": "orange", "out_of_bounds": "purple"}

    for reason, count in counts.items():
        if count > 0:
            labels.append(f"{reason}\n({count})")
            sizes.append(count)
            colors.append(color_map.get(reason, "gray"))

    if sizes:
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title("Terminal Reasons")

    # 关键指标柱状图
    ax = axes[1]
    metrics = {
        "Success\nRate": results.get("success_rate", 0),
        "Crash\nRate": results.get("crash_rate", 0),
        "Timeout\nRate": results.get("timeout_rate", 0),
    }
    bars = ax.bar(metrics.keys(), metrics.values(), color=['green', 'red', 'orange'], alpha=0.8)
    ax.set_ylabel("Rate")
    ax.set_title("Key Metrics")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "eval_summary.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Eval summary saved to: {fig_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training and evaluation results")
    parser.add_argument("--result-dir", type=str, default=None,
                        help="Directory containing training results")
    parser.add_argument("--eval-dir", type=str, default=None,
                        help="Directory containing evaluation results")
    parser.add_argument("--trajectory", type=str, default=None,
                        help="Path to trajectory JSON file")
    return parser.parse_args()


def main():
    args = parse_args()

    # 处理结果目录
    result_dir = args.result_dir or "."
    eval_dir = args.eval_dir or result_dir
    os.makedirs(result_dir, exist_ok=True)

    print(f"Plotting results from: {result_dir}")

    # 训练曲线
    eval_history_path = os.path.join(result_dir, "quick_eval_history.json")
    if os.path.exists(eval_history_path):
        plot_training_curves(eval_history_path, result_dir)

    # 轨迹图
    if args.trajectory and os.path.exists(args.trajectory):
        with open(args.trajectory) as f:
            traj_data = json.load(f)
        plot_trajectory(traj_data, result_dir)
        plot_hv_phase(traj_data, result_dir)
    else:
        # 尝试从eval目录加载
        for name in ["success_trajectories.json", "fail_trajectories.json"]:
            traj_path = os.path.join(eval_dir, name)
            if os.path.exists(traj_path):
                with open(traj_path) as f:
                    traj_data = json.load(f)
                prefix = name.replace("_trajectories.json", "_")
                plot_trajectory(traj_data, result_dir, prefix)
                plot_hv_phase(traj_data, result_dir, prefix)

    # 评估统计
    eval_results_path = os.path.join(eval_dir, "eval_results.json")
    if os.path.exists(eval_results_path):
        plot_eval_summary(eval_results_path, result_dir)

    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
