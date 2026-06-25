"""Generate 2x2 trajectory comparison figures for all major experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from envs.rocket_env import RocketLandingEnv
from envs.rocket_env_safe import RocketLandingEnvSafe
from experiment_utils import ROBUSTNESS_SCENARIOS, auto_find_stats, load_obs_rms, rollout_model


PPO_MODEL = "results/sweeps/v2_reward/models/final_model.zip"
PPO_STATS = "results/sweeps/v2_reward/models/vec_normalize.pkl"


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pick_representative(entry):
    if isinstance(entry, dict) and "trajectory" in entry:
        return entry
    if isinstance(entry, dict) and ("success" in entry or "failure" in entry):
        return entry.get("success") or entry.get("failure")
    return entry


def normalize_trajectory(raw):
    raw = pick_representative(raw)
    if raw is None:
        return None
    if "trajectory" in raw:
        points = raw["trajectory"]
        terminal = raw.get("terminal_reason", "unknown")
        initial_conditions = raw.get("initial_conditions", {})
    else:
        points = raw
        terminal = "unknown"
        initial_conditions = {}
    if not points:
        return None

    times = [float(p.get("time", i * 0.05)) for i, p in enumerate(points)]
    heights = [float(p.get("height", p.get("h", 0.0))) for p in points]
    velocities = [float(p.get("velocity", p.get("v", 0.0))) for p in points]
    fuels = [float(p.get("fuel", p.get("fuel_remaining", np.nan))) for p in points]
    throttles = []
    for p in points:
        if "throttle" in p:
            throttles.append(float(p["throttle"]))
        elif "action" in p:
            throttles.append(float((np.clip(p["action"], -1.0, 1.0) + 1.0) / 2.0))
        elif "thrust" in p:
            throttles.append(float(np.clip(p["thrust"] / 300.0, 0.0, 1.0)))
        else:
            throttles.append(np.nan)
    return {
        "time": np.asarray(times),
        "height": np.asarray(heights),
        "velocity": np.asarray(velocities),
        "throttle": np.asarray(throttles),
        "fuel": np.asarray(fuels),
        "terminal_reason": terminal,
        "initial_conditions": initial_conditions,
    }


def plot_comparison(traces, title, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    panels = [
        ("height", "Height vs Time", "Height (m)", axes[0, 0]),
        ("velocity", "Velocity vs Time", "Velocity (m/s)", axes[0, 1]),
        ("throttle", "Throttle vs Time", "Throttle", axes[1, 0]),
        ("fuel", "Fuel vs Time", "Fuel (kg)", axes[1, 1]),
    ]
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
    cmap = plt.get_cmap("tab20")

    for idx, (label, trace) in enumerate(traces.items()):
        if trace is None:
            continue
        line_style = styles[idx % len(styles)]
        color = cmap(idx % 20)
        status = trace.get("terminal_reason", "unknown")
        legend = f"{label} ({status})"
        for key, panel_title, ylabel, ax in panels:
            ax.plot(trace["time"], trace[key], linestyle=line_style, color=color, linewidth=2.0, label=legend)

    for key, panel_title, ylabel, ax in panels:
        ax.set_title(panel_title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if key == "height":
            ax.axhline(0, color="red", linewidth=1.0, alpha=0.5)
        if key == "velocity":
            ax.axhline(0, color="red", linewidth=1.0, alpha=0.45)
            ax.axhline(2, color="orange", linestyle="--", linewidth=1.0, alpha=0.45)
            ax.axhline(-2, color="orange", linestyle="--", linewidth=1.0, alpha=0.45)
        if key == "throttle":
            ax.set_ylim(-0.05, 1.05)
        if key == "fuel":
            ax.axhline(0, color="red", linestyle="--", linewidth=1.0, alpha=0.45)
        ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def load_ppo():
    model = PPO.load(PPO_MODEL, device="cpu")
    obs_rms = load_obs_rms(PPO_STATS)
    return model, obs_rms


def run_ppo_trace(options, seed=42, env_factory=RocketLandingEnv):
    model, obs_rms = load_ppo()
    env = env_factory()
    rollout = rollout_model(model, env, obs_rms=obs_rms, seed=seed, reset_options=options, save_trajectory=True)
    env.close()
    return normalize_trajectory(rollout)


def generate_safety(output_dir):
    model, obs_rms = load_ppo()
    traces = {}
    for label, env_factory in [("without_safety", RocketLandingEnv), ("with_safety", RocketLandingEnvSafe)]:
        env = env_factory()
        rollout = rollout_model(
            model,
            env,
            obs_rms=obs_rms,
            seed=42,
            reset_options=ROBUSTNESS_SCENARIOS["standard"],
            save_trajectory=True,
        )
        env.close()
        traces[label] = normalize_trajectory(rollout)
    plot_comparison(traces, "Safety Mechanism Trajectory Comparison", output_dir / "safety_trajectory_comparison.png")


def generate_robustness_and_generalization(output_dir):
    scenario_traces = {
        name: run_ppo_trace(options, seed=42 + idx * 100)
        for idx, (name, options) in enumerate(ROBUSTNESS_SCENARIOS.items())
    }
    metadata = {
        name: {
            "terminal_reason": trace["terminal_reason"] if trace else None,
            "initial_conditions": trace["initial_conditions"] if trace else None,
        }
        for name, trace in scenario_traces.items()
    }
    with open(output_dir / "robustness_trajectory_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    plot_comparison(
        scenario_traces,
        "Robustness Trajectory Comparison: All Scenarios",
        output_dir / "robustness_all_scenarios_trajectory_comparison.png",
    )

    groups = {
        "robustness_initial_conditions.png": ["standard", "random_height", "random_velocity", "random_mass", "random_fuel"],
        "robustness_physics_disturbance.png": ["standard", "gravity_bias", "thrust_bias"],
        "robustness_noise_delay.png": ["standard", "sensor_noise", "action_delay_1", "action_delay_2"],
        "robustness_combined.png": ["standard", "combined"],
        "generalization_trajectory_comparison.png": ["standard", "random_height", "random_velocity", "random_mass", "random_fuel"],
    }
    for filename, labels in groups.items():
        title = "Generalization Trajectory Comparison" if filename.startswith("generalization") else filename.replace("_", " ").replace(".png", "").title()
        plot_comparison(
            {label: scenario_traces[label] for label in labels},
            title,
            output_dir / filename,
        )


def generate_from_existing_json(json_path, title, output_path):
    data = load_json(json_path)
    traces = {label: normalize_trajectory(value) for label, value in data.items()}
    plot_comparison(traces, title, output_path)


def generate_nested_trajectory_comparisons(nested_json_path, output_dir, prefix, suite_title):
    data_path = Path(nested_json_path)
    if not data_path.exists():
        return
    data = load_json(data_path)
    output_dir = Path(output_dir)

    # For each ablation mode, compare scenarios.
    for mode, scenarios in data.items():
        traces = {scenario: normalize_trajectory(value) for scenario, value in scenarios.items()}
        plot_comparison(
            traces,
            f"{suite_title}: {mode} Across Scenarios",
            output_dir / f"{prefix}_{mode}_scenario_trajectory_comparison.png",
        )

    # For each scenario, compare ablation modes.
    scenario_names = list(next(iter(data.values())).keys())
    for scenario in scenario_names:
        traces = {mode: normalize_trajectory(data[mode][scenario]) for mode in data}
        plot_comparison(
            traces,
            f"{suite_title}: {scenario} Across Ablations",
            output_dir / f"{prefix}_{scenario}_ablation_trajectory_comparison.png",
        )


def generate_rl(output_dir):
    data_path = Path("results/reproducible/rl_comparison/rl_comparison_trajectories.json")
    if data_path.exists():
        generate_from_existing_json(data_path, "RL Algorithm Trajectory Comparison", output_dir / "rl_algorithm_trajectory_comparison.png")
        return

    algorithms = {
        "PPO": (PPO, "experiments/rl_comparison/models/PPO/model.zip", "experiments/rl_comparison/models/PPO/vec_normalize.pkl"),
        "SAC": (SAC, "experiments/rl_comparison/models/SAC/model.zip", "experiments/rl_comparison/models/SAC/vec_normalize.pkl"),
        "TD3": (TD3, "experiments/rl_comparison/models/TD3/model.zip", "experiments/rl_comparison/models/TD3/vec_normalize.pkl"),
    }
    traces = {}
    for label, (cls, model_path, stats_path) in algorithms.items():
        model = cls.load(model_path, device="cpu")
        obs_rms = load_obs_rms(stats_path)
        env = RocketLandingEnv()
        rollout = rollout_model(model, env, obs_rms=obs_rms, seed=42, reset_options=ROBUSTNESS_SCENARIOS["standard"], save_trajectory=True)
        env.close()
        traces[label] = normalize_trajectory(rollout)
    plot_comparison(traces, "RL Algorithm Trajectory Comparison", output_dir / "rl_algorithm_trajectory_comparison.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate trajectory comparison figures for all experiments.")
    parser.add_argument("--output-dir", type=str, default="results/reproducible/trajectory_comparisons")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_safety(output_dir)
    generate_robustness_and_generalization(output_dir)
    generate_from_existing_json(
        "results/reproducible/controller_comparison_trajectories.json",
        "Controller Trajectory Comparison",
        output_dir / "controller_trajectory_comparison.png",
    )
    generate_from_existing_json(
        "results/reproducible/ablation/reward_ablation_trajectories.json",
        "Reward Ablation Trajectory Comparison",
        output_dir / "ablation_trajectory_comparison.png",
    )
    generate_rl(output_dir)
    generate_nested_trajectory_comparisons(
        "results/reproducible/ablation_scenarios/ablation_generalization_trajectories.json",
        output_dir / "ablation_scenarios",
        "generalization",
        "Ablation Generalization",
    )
    generate_nested_trajectory_comparisons(
        "results/reproducible/ablation_scenarios/ablation_robustness_trajectories.json",
        output_dir / "ablation_scenarios",
        "robustness",
        "Ablation Robustness",
    )


if __name__ == "__main__":
    main()
