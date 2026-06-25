"""Train, evaluate, and compare a pure energy-guided PPO strategy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.rocket_env import RocketLandingEnv
from envs.rocket_env_energy import RocketLandingEnergyEnv
from experiment_utils import (
    STANDARD_EVAL_OPTIONS,
    evaluate_model_rollouts,
    load_obs_rms,
    summarize_rollouts,
    write_evaluation_outputs,
)
from generate_experiment_trajectory_comparisons import normalize_trajectory, plot_comparison


BASELINE_MODEL = "results/models/baseline_ppo/final_model.zip"
BASELINE_STATS = "results/models/baseline_ppo/vec_normalize.pkl"


def make_energy_env(randomize=True):
    return Monitor(RocketLandingEnergyEnv(randomize=randomize))


def train_energy_ppo(output_dir, train_steps, seed, device, init_model=None, init_stats=None):
    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_env = DummyVecEnv([lambda: make_energy_env(randomize=True)])
    if init_stats:
        env = VecNormalize.load(str(init_stats), raw_env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if init_model:
        model = PPO.load(str(init_model), env=env, device=device)
        model.tensorboard_log = str(output_dir / "tb_logs")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [128, 128], "log_std_init": -0.7},
            seed=seed,
            device=device,
            verbose=1,
            tensorboard_log=str(output_dir / "tb_logs"),
        )
        # PPO's default continuous-action mean is 0, which maps to 50% throttle
        # in this environment. Energy-optimal landing starts near zero thrust
        # and only burns near the braking switch surface.
        with torch.no_grad():
            model.policy.action_net.bias.fill_(-0.9)
    model.learn(total_timesteps=train_steps, progress_bar=False)
    model.save(str(model_dir / "pure_energy_ppo_model"))
    env.save(str(model_dir / "pure_energy_vec_normalize.pkl"))
    return model_dir / "pure_energy_ppo_model.zip", model_dir / "pure_energy_vec_normalize.pkl"


def evaluate_energy_model(model_path, stats_path, output_dir, n_episodes, seed, save_trajectories=True):
    model = PPO.load(str(model_path), device="cpu")
    obs_rms = load_obs_rms(stats_path)
    rollouts = evaluate_model_rollouts(
        model=model,
        env_factory=RocketLandingEnergyEnv,
        obs_rms=obs_rms,
        n_episodes=n_episodes,
        seed=seed,
        options=STANDARD_EVAL_OPTIONS,
        save_trajectories=save_trajectories,
    )
    summary = summarize_rollouts(rollouts, model_path=str(model_path), seed=seed)
    summary["strategy"] = "Pure Energy-Guided PPO"
    summary["stats_path"] = str(stats_path)
    write_evaluation_outputs(rollouts, summary, output_dir, save_trajectories)
    return summary, rollouts


def evaluate_baseline_for_comparison(n_episodes, seed):
    model = PPO.load(BASELINE_MODEL, device="cpu")
    obs_rms = load_obs_rms(BASELINE_STATS)
    rollouts = evaluate_model_rollouts(
        model=model,
        env_factory=RocketLandingEnv,
        obs_rms=obs_rms,
        n_episodes=n_episodes,
        seed=seed,
        options=STANDARD_EVAL_OPTIONS,
        save_trajectories=True,
    )
    return summarize_rollouts(rollouts, model_path=BASELINE_MODEL, seed=seed), rollouts


def plot_energy_accounting(rollout, output_path):
    trajectory = rollout.get("trajectory", [])
    if not trajectory:
        return
    t = np.array([p["time"] for p in trajectory])
    potential = []
    kinetic = []
    fuel_energy = []
    for p in trajectory:
        mass = p.get("mass", 10.0 + p.get("fuel", 0.0))
        h = max(p.get("height", 0.0), 0.0)
        v = p.get("velocity", 0.0)
        fuel = max(p.get("fuel", 0.0), 0.0)
        potential.append(mass * 9.81 * h)
        kinetic.append(0.5 * mass * v**2)
        fuel_energy.append(0.5 * fuel * 200.0**2)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t, potential, label="Potential energy mgh", linewidth=2)
    ax1.plot(t, kinetic, label="Kinetic energy 1/2mv^2", linewidth=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Mechanical energy (J)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(t, fuel_energy, color="tab:green", label="Fuel available energy proxy", linewidth=2, alpha=0.75)
    ax2.set_ylabel("Fuel energy proxy (J)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.suptitle("Pure Energy-Guided PPO Energy Accounting")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_reward_breakdown(rollout, output_path):
    trajectory = rollout.get("trajectory", [])
    if not trajectory:
        return
    component_names = sorted(
        {
            key
            for point in trajectory
            for key in point.get("reward_breakdown", {}).keys()
        }
    )
    if not component_names:
        return

    t = np.array([p["time"] for p in trajectory])
    fig, ax = plt.subplots(figsize=(12, 7))
    for name in component_names:
        values = np.array([p.get("reward_breakdown", {}).get(name, 0.0) for p in trajectory], dtype=float)
        ax.plot(t, values, linewidth=1.8, label=name)
    ax.set_title("Pure Energy Reward Breakdown")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward contribution")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_metric_comparison(results, output_path):
    labels = list(results.keys())
    metrics = [
        ("success_rate", "Success rate"),
        ("mean_final_velocity_error", "Mean |v_final| (m/s)"),
        ("mean_fuel_used", "Fuel used (kg)"),
        ("mean_landing_time", "Landing time (s)"),
        ("mean_abs_throttle_delta", "Throttle delta"),
    ]
    x = np.arange(len(labels))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (key, label) in enumerate(metrics):
        vals = [results[name].get(key, 0.0) for name in labels]
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Baseline PPO vs Pure Energy-Guided PPO")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pure energy-guided PPO experiment.")
    parser.add_argument("--output-dir", type=str, default="results/experiments/07_energy_ppo")
    parser.add_argument("--model", type=str, default=None, help="Existing energy PPO model path.")
    parser.add_argument("--stats", type=str, default=None, help="Existing energy PPO VecNormalize path.")
    parser.add_argument("--train-steps", type=int, default=300_000)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-train", action="store_true", help="Train from scratch even if output model files exist.")
    parser.add_argument("--init-model", type=str, default=None, help="Optional model to fine-tune.")
    parser.add_argument("--init-stats", type=str, default=None, help="Optional VecNormalize stats to continue from.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model) if args.model else output_dir / "models" / "pure_energy_ppo_model.zip"
    stats_path = Path(args.stats) if args.stats else output_dir / "models" / "pure_energy_vec_normalize.pkl"
    if args.force_train and (args.init_model or args.init_stats):
        raise ValueError("--force-train is for from-scratch runs and cannot be combined with --init-model/--init-stats")
    if not args.skip_train and (args.force_train or not (model_path.exists() and stats_path.exists())):
        model_path, stats_path = train_energy_ppo(
            output_dir,
            args.train_steps,
            args.seed,
            args.device,
            init_model=args.init_model,
            init_stats=args.init_stats,
        )
    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError(f"Energy PPO model/stats not found: {model_path}, {stats_path}")

    energy_summary, energy_rollouts = evaluate_energy_model(
        model_path,
        stats_path,
        output_dir / "eval",
        args.n_episodes,
        args.seed,
        save_trajectories=True,
    )
    baseline_summary, baseline_rollouts = evaluate_baseline_for_comparison(args.n_episodes, args.seed)
    comparison = {
        "Baseline PPO": baseline_summary,
        "Pure Energy-Guided PPO": energy_summary,
    }
    with open(output_dir / "baseline_vs_pure_energy_ppo.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    representative = {
        "Baseline PPO": normalize_trajectory(next((r for r in baseline_rollouts if r["success"]), baseline_rollouts[0])),
        "Pure Energy-Guided PPO": normalize_trajectory(next((r for r in energy_rollouts if r["success"]), energy_rollouts[0])),
    }
    representative_energy_rollout = next((r for r in energy_rollouts if r["success"]), energy_rollouts[0])
    plot_comparison(
        representative,
        "Baseline PPO vs Pure Energy-Guided PPO Trajectory",
        output_dir / "baseline_vs_pure_energy_trajectory.png",
    )
    plot_metric_comparison(comparison, output_dir / "baseline_vs_pure_energy_metrics.png")
    plot_energy_accounting(representative_energy_rollout, output_dir / "pure_energy_accounting.png")
    plot_reward_breakdown(representative_energy_rollout, output_dir / "pure_energy_reward_breakdown.png")

    print("Pure Energy-Guided PPO evaluation:")
    print(json.dumps(energy_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
