#!/usr/bin/env python3
"""
demo.py - 生成火箭着陆动画 demo.mp4

用法:
    python -m rocket_landing_control.visualization.demo --model saved_models/ppo_rocket_v7.zip --output results/demo.mp4
    python -m rocket_landing_control.visualization.demo --model results/sweeps/progress_v18/models/final_model.zip --episodes 3
"""
import argparse
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rocket_landing_control.envs.rocket_env import RocketLandingEnv


def create_env():
    """创建环境"""
    return RocketLandingEnv()


def load_normalization_stats(stats_path):
    """加载归一化统计"""
    if stats_path and Path(stats_path).exists():
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        return stats.obs_rms
    return None


def normalize_obs(obs, obs_rms):
    """归一化观测"""
    if obs_rms is not None:
        return np.clip(
            (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0
        )
    return obs


def run_episode(model, env, obs_rms=None):
    """运行一个episode，记录轨迹数据"""
    obs, info = env.reset()

    # 归一化观测
    obs_norm = normalize_obs(obs, obs_rms)

    # 保存初始状态
    trajectory = {
        "time": [float(env.time)],
        "height": [float(env.height)],
        "velocity": [float(env.velocity)],
        "thrust": [float(env.current_thrust)],
        "fuel": [float(env.fuel_remaining)],
        "action": [],
        "reward_breakdown": [],
    }

    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    max_steps = 600  # 安全限制
    terminal_reason = "none"
    success = False

    while not (done or truncated) and step_count < max_steps:
        action, _ = model.predict(obs_norm, deterministic=True)

        # 执行动作
        raw_obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)

        # 归一化观测
        obs_norm = normalize_obs(raw_obs, obs_rms)

        step_count += 1

        # 记录轨迹
        trajectory["time"].append(float(env.time))
        trajectory["height"].append(float(env.height))
        trajectory["velocity"].append(float(env.velocity))
        trajectory["thrust"].append(float(env.current_thrust))
        trajectory["fuel"].append(float(env.fuel_remaining))

        # 处理action
        action_val = action[0] if hasattr(action, '__len__') else action
        trajectory["action"].append(float(action_val))

        # 获取reward_breakdown
        if hasattr(env, '_reward_breakdown'):
            trajectory["reward_breakdown"].append(env._reward_breakdown.copy())
        else:
            trajectory["reward_breakdown"].append({})

    # 判断终止原因
    final_h = env.height
    final_v = env.velocity
    if done and final_h <= 0.01 and abs(final_v) <= 2.0:
        terminal_reason = "success"
        success = True
    elif done and final_h <= 0.01 and abs(final_v) > 2.0:
        terminal_reason = "crash"
    elif done and abs(final_v) > 50.0:
        terminal_reason = "velocity_exceeded"
    elif truncated:
        terminal_reason = "timeout"
    else:
        terminal_reason = "unknown"

    trajectory["total_reward"] = float(total_reward)
    trajectory["terminal_reason"] = terminal_reason
    trajectory["success"] = success

    print(f"  Steps: {step_count}, Terminal: {terminal_reason}, Success: {success}")

    return trajectory


def create_animation(trajectories, output_path, fps=20):
    """创建火箭着陆动画"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Rocket Landing Demo - PPO Agent", fontsize=14, fontweight="bold")

    # 颜色方案
    colors = {
        "height": "#2196F3",
        "velocity": "#FF5722",
        "thrust": "#4CAF50",
        "fuel": "#FFC107",
        "ground": "#795548",
        "rocket": "#607D8B",
        "flame": "#FF9800",
        "success": "#4CAF50",
        "crash": "#F44336",
    }

    for idx, traj in enumerate(trajectories):
        # 清除并设置子图
        for ax in axes.flat:
            ax.clear()

        n_steps = len(traj["time"]) - 1
        success = traj["success"]

        # ===== 左上：火箭动画 =====
        ax_rocket = axes[0, 0]
        ax_rocket.set_xlim(-2, 2)
        ax_rocket.set_ylim(-5, 55)
        ax_rocket.set_aspect("equal")
        ax_rocket.set_xlabel("X (m)")
        ax_rocket.set_ylabel("Height (m)")
        ax_rocket.set_title(f"Rocket Trajectory (Episode {idx+1})")
        ax_rocket.grid(True, alpha=0.3)

        # 地面
        ax_rocket.fill_between([-2, 2], [-5, -5], [0, 0], color=colors["ground"], alpha=0.3)
        ax_rocket.axhline(y=0, color=colors["ground"], linewidth=2, linestyle="-")

        # 火箭主体（矩形）
        rocket_body = plt.Rectangle((-0.3, 0), 0.6, 3, fill=True,
                                     facecolor=colors["rocket"], edgecolor="black", linewidth=1.5)
        ax_rocket.add_patch(rocket_body)

        # 火焰
        flame, = ax_rocket.plot([], [], color=colors["flame"], linewidth=3, alpha=0.8)

        # 轨迹线
        trail, = ax_rocket.plot([], [], color=colors["height"], linewidth=1, alpha=0.5, linestyle="--")

        # 信息文本
        info_text = ax_rocket.text(0.02, 0.98, "", transform=ax_rocket.transAxes,
                                    verticalalignment="top", fontsize=9,
                                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # ===== 右上：高度和速度 =====
        ax_hv = axes[0, 1]
        ax_hv.set_xlabel("Time (s)")
        ax_hv.set_ylabel("Height (m)", color=colors["height"])
        ax_hv.set_title("Height & Velocity")
        ax_hv.grid(True, alpha=0.3)
        ax_hv.tick_params(axis="y", labelcolor=colors["height"])

        ax_v = ax_hv.twinx()
        ax_v.set_ylabel("Velocity (m/s)", color=colors["velocity"])
        ax_v.tick_params(axis="y", labelcolor=colors["velocity"])

        line_h, = ax_hv.plot([], [], color=colors["height"], linewidth=2, label="Height")
        line_v, = ax_v.plot([], [], color=colors["velocity"], linewidth=2, label="Velocity")
        point_h, = ax_hv.plot([], [], "o", color=colors["height"], markersize=8)
        point_v, = ax_v.plot([], [], "o", color=colors["velocity"], markersize=8)

        # ===== 左下：推力 =====
        ax_thrust = axes[1, 0]
        ax_thrust.set_xlabel("Time (s)")
        ax_thrust.set_ylabel("Thrust (N)", color=colors["thrust"])
        ax_thrust.set_title("Thrust")
        ax_thrust.grid(True, alpha=0.3)
        ax_thrust.set_ylim(0, 350)

        line_thrust, = ax_thrust.plot([], [], color=colors["thrust"], linewidth=2, label="Thrust")
        point_thrust, = ax_thrust.plot([], [], "o", color=colors["thrust"], markersize=8)

        # ===== 右下：燃料 =====
        ax_fuel = axes[1, 1]
        ax_fuel.set_xlabel("Time (s)")
        ax_fuel.set_ylabel("Fuel (kg)", color=colors["fuel"])
        ax_fuel.set_title("Fuel")
        ax_fuel.grid(True, alpha=0.3)
        ax_fuel.set_ylim(0, 6)

        line_fuel, = ax_fuel.plot([], [], color=colors["fuel"], linewidth=2, label="Fuel")
        point_fuel, = ax_fuel.plot([], [], "o", color=colors["fuel"], markersize=8)

        # 设置坐标轴范围
        t_max = max(traj["time"]) * 1.1
        for ax in [ax_hv, ax_thrust, ax_fuel]:
            ax.set_xlim(0, t_max)
        ax_hv.set_ylim(0, 55)
        ax_v.set_ylim(min(traj["velocity"]) * 1.2 - 1, max(traj["velocity"]) * 1.2 + 1)

        def init():
            """初始化动画"""
            rocket_body.set_xy((-0.3, 0))
            flame.set_data([], [])
            trail.set_data([], [])
            info_text.set_text("")
            line_h.set_data([], [])
            line_v.set_data([], [])
            point_h.set_data([], [])
            point_v.set_data([], [])
            line_thrust.set_data([], [])
            point_thrust.set_data([], [])
            line_fuel.set_data([], [])
            point_fuel.set_data([], [])
            return [rocket_body, flame, trail, info_text, line_h, line_v, point_h, point_v,
                    line_thrust, point_thrust, line_fuel, point_fuel]

        def animate(frame):
            """更新动画帧"""
            t = traj["time"][frame]
            h = traj["height"][frame]
            v = traj["velocity"][frame]
            thrust = traj["thrust"][frame]
            fuel = traj["fuel"][frame]

            # 更新火箭位置
            rocket_body.set_xy((-0.3, h))

            # 更新火焰（根据推力大小）
            flame_length = thrust / 300.0 * 2.0
            flame.set_data([0, 0], [h, h - flame_length])

            # 更新轨迹
            trail.set_data(traj["height"][:frame+1], traj["height"][:frame+1])

            # 更新信息文本
            status = "SUCCESS" if success and frame >= n_steps else "RUNNING"
            if frame >= n_steps:
                status = "SUCCESS" if success else "CRASH"
            info_text.set_text(f"Time: {t:.1f}s\nHeight: {h:.1f}m\nVelocity: {v:.1f}m/s\n"
                              f"Thrust: {thrust:.0f}N\nFuel: {fuel:.2f}kg\nStatus: {status}")

            # 更新曲线
            t_data = traj["time"][:frame+1]
            line_h.set_data(t_data, traj["height"][:frame+1])
            line_v.set_data(t_data, traj["velocity"][:frame+1])
            point_h.set_data([t], [h])
            point_v.set_data([t], [v])

            line_thrust.set_data(t_data, traj["thrust"][:frame+1])
            point_thrust.set_data([t], [thrust])

            line_fuel.set_data(t_data, traj["fuel"][:frame+1])
            point_fuel.set_data([t], [fuel])

            return [rocket_body, flame, trail, info_text, line_h, line_v, point_h, point_v,
                    line_thrust, point_thrust, line_fuel, point_fuel]

        # 创建动画
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                        frames=n_steps + 1, interval=1000//fps, blit=True)

        # 保存当前episode为GIF
        gif_output = output_path.replace(".mp4", f"_ep{idx+1}.gif")
        print(f"Saving episode {idx+1} to {gif_output}...")
        anim.save(gif_output, writer='pillow', fps=fps//2)

        print(f"  Episode {idx+1}: {n_steps} steps, "
              f"{'SUCCESS' if success else 'CRASH'}, "
              f"Time={traj['time'][-1]:.1f}s, Fuel={traj['fuel'][-1]:.2f}kg")

    plt.close(fig)
    return True


def create_summary_plot(trajectories, output_path):
    """创建汇总图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Rocket Landing Episodes Summary", fontsize=14, fontweight="bold")

    colors = plt.cm.Set2(np.linspace(0, 1, len(trajectories)))

    for idx, (traj, color) in enumerate(zip(trajectories, colors)):
        label = f"Ep{idx+1} ({'✓' if traj['success'] else '✗'})"

        axes[0, 0].plot(traj["time"], traj["height"], color=color, linewidth=2, label=label)
        axes[0, 1].plot(traj["time"], traj["velocity"], color=color, linewidth=2, label=label)
        axes[1, 0].plot(traj["time"], traj["thrust"], color=color, linewidth=2, label=label)
        axes[1, 1].plot(traj["time"], traj["fuel"], color=color, linewidth=2, label=label)

    axes[0, 0].set_title("Height vs Time")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Height (m)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Velocity vs Time")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Velocity (m/s)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Thrust vs Time")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Thrust (N)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Fuel vs Time")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Fuel (kg)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate rocket landing demo animation")
    parser.add_argument("--model", type=str, default="results/sweeps/v2_reward/models/final_model.zip",
                        help="Path to trained model")
    parser.add_argument("--stats", type=str, default="results/sweeps/v2_reward/models/vec_normalize.pkl",
                        help="Path to normalization stats")
    parser.add_argument("--output", type=str, default="results/demo.mp4",
                        help="Output path for demo video")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=20,
                        help="Frames per second")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model, device="cpu")
    print("Model loaded successfully!")

    # 加载归一化统计
    obs_rms = load_normalization_stats(args.stats)
    if obs_rms is not None:
        print(f"Loaded normalization stats from {args.stats}")

    # 运行多个episode
    trajectories = []
    for ep in range(args.episodes):
        print(f"\nRunning episode {ep+1}/{args.episodes}...")
        env = create_env()
        env.reset(seed=args.seed + ep)
        traj = run_episode(model, env, obs_rms)
        trajectories.append(traj)
        env.close()

    # 生成动画
    print(f"\nGenerating animations...")
    create_animation(trajectories, str(output_path), fps=args.fps)

    # 生成汇总图
    summary_path = str(output_path).replace(".mp4", "_summary.png")
    create_summary_plot(trajectories, summary_path)

    # 保存轨迹数据
    data_path = str(output_path).replace(".mp4", "_data.json")
    save_data = []
    for traj in trajectories:
        # 转换numpy类型为Python原生类型
        save_data.append({
            "time": [float(t) for t in traj["time"]],
            "height": [float(h) for h in traj["height"]],
            "velocity": [float(v) for v in traj["velocity"]],
            "thrust": [float(t) for t in traj["thrust"]],
            "fuel": [float(f) for f in traj["fuel"]],
            "action": [float(a) for a in traj["action"]],
            "success": bool(traj["success"]),
            "terminal_reason": str(traj["terminal_reason"]),
            "total_reward": float(traj["total_reward"]),
        })
    with open(data_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Trajectory data saved to {data_path}")

    # 打印统计
    print("\n" + "="*50)
    print("Demo Statistics:")
    print("="*50)
    for idx, traj in enumerate(trajectories):
        print(f"Episode {idx+1}: {'SUCCESS' if traj['success'] else 'CRASH'}, "
              f"Time={traj['time'][-1]:.1f}s, Fuel={traj['fuel'][-1]:.2f}kg")
    print("="*50)


if __name__ == "__main__":
    main()
