"""Unified generalization/robustness benchmark for learned and classical policies.

The script evaluates every requested policy on the same scenario definitions
from ``experiment_utils.ROBUSTNESS_SCENARIOS``. It supports Stable-Baselines3
PPO/SAC/TD3 policies, the pure-energy PPO environment, and classical
PID/MPC/ET-MPC controllers through one output schema.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from controller_comparison_full import EventTriggeredMPCController, MPCController, PIDController, rollout_controller
from envs.rocket_env import RocketLandingEnv
from envs.rocket_env_energy import RocketLandingEnergyBaseObsEnv, RocketLandingEnergyEnv
from envs.rocket_env_safe import RocketLandingEnvSafe
from experiment_utils import (
    ROBUSTNESS_SCENARIOS,
    auto_find_stats,
    evaluate_model_rollouts,
    load_obs_rms,
    summarize_rollouts,
)
from generate_experiment_trajectory_comparisons import normalize_trajectory, plot_comparison


SB3_ALGORITHMS = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
ENV_FACTORIES = {
    "base": RocketLandingEnv,
    "energy": RocketLandingEnergyEnv,
    "energy_base_obs": RocketLandingEnergyBaseObsEnv,
    "safe": RocketLandingEnvSafe,
}
CONTROLLERS = {
    "PID": PIDController,
    "MPC": MPCController,
    "ET-MPC": EventTriggeredMPCController,
}
GENERALIZATION_SCENARIOS = {"standard", "random_height", "random_velocity", "random_mass", "random_fuel"}
ROBUSTNESS_ONLY_SCENARIOS = {
    "standard",
    "gravity_bias",
    "thrust_bias",
    "sensor_noise",
    "action_delay_1",
    "action_delay_2",
    "combined",
}


@dataclass
class PolicySpec:
    label: str
    kind: str
    env: str = "base"
    algo: str | None = None
    model: str | None = None
    stats: str | None = None
    controller: str | None = None


def parse_policy_spec(raw: str) -> PolicySpec:
    fields: dict[str, str] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --policy item {item!r}; expected key=value")
        key, value = item.split("=", 1)
        fields[key.strip().lower()] = value.strip()

    kind = fields.get("type", fields.get("kind", "sb3")).lower()
    label = fields.get("label")
    if kind == "controller":
        controller = fields.get("controller", fields.get("name", "PID")).upper()
        if controller not in CONTROLLERS:
            raise ValueError(f"Unsupported controller {controller!r}; choose from {sorted(CONTROLLERS)}")
        return PolicySpec(label=label or controller, kind="controller", env=fields.get("env", "base"), controller=controller)

    algo = fields.get("algo", "PPO").upper()
    if algo not in SB3_ALGORITHMS:
        raise ValueError(f"Unsupported SB3 algorithm {algo!r}; choose from {sorted(SB3_ALGORITHMS)}")
    model = fields.get("model")
    if not model:
        raise ValueError("SB3 policies require model=<path>")
    return PolicySpec(
        label=label or algo,
        kind="sb3",
        env=fields.get("env", "base"),
        algo=algo,
        model=model,
        stats=fields.get("stats"),
    )


def legacy_policy_from_args(args) -> PolicySpec:
    return PolicySpec(
        label=args.label or args.algo,
        kind="sb3",
        env=args.env,
        algo=args.algo,
        model=args.model,
        stats=args.stats,
    )


def scenario_subset(name: str) -> dict[str, dict[str, Any]]:
    if name == "all":
        return ROBUSTNESS_SCENARIOS
    if name == "generalization":
        return {k: v for k, v in ROBUSTNESS_SCENARIOS.items() if k in GENERALIZATION_SCENARIOS}
    if name == "robustness":
        return {k: v for k, v in ROBUSTNESS_SCENARIOS.items() if k in ROBUSTNESS_ONLY_SCENARIOS}
    raise ValueError(f"Unknown scenario set: {name}")


def load_sb3_policy(spec: PolicySpec):
    if spec.env not in ENV_FACTORIES:
        raise ValueError(f"Unsupported env {spec.env!r}; choose from {sorted(ENV_FACTORIES)}")
    cls = SB3_ALGORITHMS[spec.algo or "PPO"]
    model = cls.load(str(spec.model), device="cpu")
    stats_path = auto_find_stats(str(spec.model), spec.stats)
    obs_rms = load_obs_rms(stats_path)
    return model, obs_rms, ENV_FACTORIES[spec.env], stats_path


def evaluate_sb3_policy(spec: PolicySpec, scenarios, n_episodes, seed, save_trajectories):
    model, obs_rms, env_factory, stats_path = load_sb3_policy(spec)
    results = {}
    representatives = {}
    for idx, (scenario_name, options) in enumerate(scenarios.items()):
        scenario_seed = seed + idx * 10_000
        print(f"{spec.label} / {scenario_name}")
        rollouts = evaluate_model_rollouts(
            model=model,
            env_factory=env_factory,
            obs_rms=obs_rms,
            n_episodes=n_episodes,
            seed=scenario_seed,
            options=options,
            save_trajectories=save_trajectories,
        )
        summary = summarize_rollouts(rollouts, model_path=str(spec.model), seed=scenario_seed)
        summary.update(
            {
                "label": spec.label,
                "policy_type": "sb3",
                "algorithm": spec.algo,
                "env": spec.env,
                "stats_path": str(stats_path) if stats_path else None,
                "scenario": scenario_name,
                "reset_options": options,
            }
        )
        results[scenario_name] = summary
        if save_trajectories:
            # Fixed episode index keeps every policy on exactly the same seed
            # and initial condition in trajectory comparison figures.
            representatives[scenario_name] = rollouts[0] if rollouts else None
        print_summary(summary)
    return results, representatives


def evaluate_controller_policy(spec: PolicySpec, scenarios, n_episodes, seed, save_trajectories):
    controller = CONTROLLERS[spec.controller or "PID"]()
    results = {}
    representatives = {}
    for idx, (scenario_name, options) in enumerate(scenarios.items()):
        scenario_seed = seed + idx * 10_000
        print(f"{spec.label} / {scenario_name}")
        rollouts = [
            rollout_controller(controller, scenario_seed + episode, options, save_trajectory=save_trajectories)
            for episode in range(n_episodes)
        ]
        summary = summarize_rollouts(rollouts, seed=scenario_seed)
        summary.update(
            {
                "label": spec.label,
                "policy_type": "controller",
                "controller": spec.controller,
                "env": "base",
                "scenario": scenario_name,
                "reset_options": options,
            }
        )
        if spec.controller == "ET-MPC":
            summary["mean_trigger_rate"] = float(np.mean([r.get("trigger_rate", 0.0) for r in rollouts]))
        results[scenario_name] = summary
        if save_trajectories:
            representatives[scenario_name] = rollouts[0] if rollouts else None
        print_summary(summary)
    return results, representatives


def evaluate_policy(spec: PolicySpec, scenarios, n_episodes, seed, save_trajectories):
    if spec.kind == "controller":
        return evaluate_controller_policy(spec, scenarios, n_episodes, seed, save_trajectories)
    return evaluate_sb3_policy(spec, scenarios, n_episodes, seed, save_trajectories)


def print_summary(summary):
    print(
        f"  success={summary['success_rate']:.1%}, "
        f"crash={summary['crash_rate']:.1%}, "
        f"time={summary['mean_landing_time']:.2f}s, "
        f"fuel={summary['mean_fuel_used']:.2f}kg, "
        f"|v_f|={summary['mean_final_velocity_error']:.2f}m/s"
    )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(comparison, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario_group",
                "scenario",
                "label",
                "policy_type",
                "algorithm_or_controller",
                "env",
                "n_episodes",
                "success_rate",
                "crash_rate",
                "timeout_rate",
                "mean_landing_time",
                "mean_fuel_used",
                "mean_final_velocity_error",
                "mean_abs_throttle_delta",
                "mean_max_velocity",
                "mean_max_acceleration",
                "terminal_reason_counts",
            ]
        )
        for scenario in comparison["scenarios"]:
            group = "generalization" if scenario in GENERALIZATION_SCENARIOS else "robustness"
            for label in comparison["labels"]:
                summary = comparison["results"][scenario][label]
                writer.writerow(
                    [
                        group,
                        scenario,
                        label,
                        summary.get("policy_type"),
                        summary.get("algorithm") or summary.get("controller"),
                        summary.get("env"),
                        summary["n_episodes"],
                        summary["success_rate"],
                        summary["crash_rate"],
                        summary["timeout_rate"],
                        summary["mean_landing_time"],
                        summary["mean_fuel_used"],
                        summary["mean_final_velocity_error"],
                        summary["mean_abs_throttle_delta"],
                        summary["mean_max_velocity"],
                        summary["mean_max_acceleration"],
                        json.dumps(summary["terminal_reason_counts"], ensure_ascii=False),
                    ]
                )
    print(f"Saved {output_path}")


def build_comparison(policy_results, scenarios):
    labels = list(policy_results.keys())
    scenario_names = list(scenarios.keys())
    return {
        "labels": labels,
        "scenarios": scenario_names,
        "generalization_scenarios": [s for s in scenario_names if s in GENERALIZATION_SCENARIOS],
        "robustness_scenarios": [s for s in scenario_names if s in ROBUSTNESS_ONLY_SCENARIOS],
        "results_by_policy": policy_results,
        "results": {
            scenario: {label: policy_results[label][scenario] for label in labels}
            for scenario in scenario_names
        },
    }


def plot_metric_grid(comparison, output_path):
    scenarios = comparison["scenarios"]
    labels = comparison["labels"]
    results = comparison["results"]
    metrics = [
        ("success_rate", "Success Rate", "Success rate"),
        ("mean_landing_time", "Landing Time", "Time (s)"),
        ("mean_fuel_used", "Fuel Used", "Fuel (kg)"),
        ("mean_final_velocity_error", "Terminal Speed Error", "|v_final| (m/s)"),
    ]
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(labels), 1)
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("Unified Generalization and Robustness Benchmark", fontsize=16, fontweight="bold")
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        for i, label in enumerate(labels):
            values = [results[scenario][label].get(metric, 0.0) for scenario in scenarios]
            ax.bar(x + (i - (len(labels) - 1) / 2) * width, values, width, label=label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        if metric == "success_rate":
            ax.set_ylim(0.0, 1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_success_heatmap(comparison, output_path):
    scenarios = comparison["scenarios"]
    labels = comparison["labels"]
    matrix = np.array(
        [[comparison["results"][scenario][label]["success_rate"] for scenario in scenarios] for label in labels]
    )
    fig, ax = plt.subplots(
        figsize=(max(12, len(scenarios) * 1.15), max(3.5, len(labels) * 0.75 + 1.5))
    )
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Success Rate Across All Scenarios")
    for row in range(len(labels)):
        for col in range(len(scenarios)):
            value = matrix[row, col]
            ax.text(col, row, f"{100 * value:.0f}%", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Success rate")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def write_trajectory_comparisons(representatives, scenarios, output_dir):
    output_dir = Path(output_dir)
    for scenario in scenarios:
        traces = {
            label: normalize_trajectory(policy_reps.get(scenario))
            for label, policy_reps in representatives.items()
            if policy_reps.get(scenario) is not None
        }
        if traces:
            plot_comparison(
                traces,
                f"Unified Scenario Trajectory Comparison: {scenario}",
                output_dir / f"{scenario}_trajectory_comparison.png",
                require_shared_start=True,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Unified scenario benchmark for PPO, energy PPO, SAC/TD3, and controllers.")
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        help=(
            "Policy spec as comma-separated key=value pairs. Examples: "
            "'label=Baseline PPO,type=sb3,algo=PPO,env=base,model=...,stats=...' or "
            "'label=PID,type=controller,controller=PID'. Repeat for multiple policies."
        ),
    )
    parser.add_argument("--scenario-set", choices=["all", "generalization", "robustness"], default="all")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="results/experiments/03_robustness/benchmark")
    parser.add_argument("--save-trajectories", action="store_true")

    # Legacy single-policy arguments are kept so older commands still work.
    parser.add_argument("--model", type=str, default="results/models/baseline_ppo/final_model.zip")
    parser.add_argument("--stats", type=str, default="results/models/baseline_ppo/vec_normalize.pkl")
    parser.add_argument("--env", choices=sorted(ENV_FACTORIES), default="base")
    parser.add_argument("--algo", choices=sorted(SB3_ALGORITHMS), default="PPO")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = scenario_subset(args.scenario_set)
    specs = [parse_policy_spec(raw) for raw in args.policy] if args.policy else [legacy_policy_from_args(args)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_results = {}
    representatives = {}
    for spec in specs:
        results, reps = evaluate_policy(spec, scenarios, args.n_episodes, args.seed, args.save_trajectories)
        policy_results[spec.label] = results
        representatives[spec.label] = reps
        if args.output and len(specs) == 1:
            write_json(args.output, results)

    comparison = build_comparison(policy_results, scenarios)
    write_json(output_dir / "scenario_comparison.json", comparison)
    write_json(output_dir / "results_by_policy.json", policy_results)
    write_csv(comparison, output_dir / "scenario_comparison.csv")
    plot_metric_grid(comparison, output_dir / "scenario_comparison_metrics.png")
    plot_success_heatmap(comparison, output_dir / "success_rate_heatmap.png")

    if args.save_trajectories:
        write_json(output_dir / "representative_trajectories.json", representatives)
        write_trajectory_comparisons(representatives, scenarios, output_dir / "trajectory_comparisons")

    print(f"Saved unified scenario benchmark to: {output_dir}")


if __name__ == "__main__":
    main()
