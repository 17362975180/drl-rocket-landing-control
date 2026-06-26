"""Evaluate every reward-ablation policy on generalization and robustness scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rocket_landing_control.core.experiment_utils import ROBUSTNESS_SCENARIOS, evaluate_model_rollouts, summarize_rollouts
from rocket_landing_control.studies.reward_ablation import ABLATION_MODES, RocketLandingEnvAblation


GENERALIZATION_SCENARIOS = {
    "standard": ROBUSTNESS_SCENARIOS["standard"],
    "random_height": ROBUSTNESS_SCENARIOS["random_height"],
    "random_velocity": ROBUSTNESS_SCENARIOS["random_velocity"],
    "random_mass": ROBUSTNESS_SCENARIOS["random_mass"],
    "random_fuel": ROBUSTNESS_SCENARIOS["random_fuel"],
}

ROBUST_ONLY_SCENARIOS = {
    "standard": ROBUSTNESS_SCENARIOS["standard"],
    "gravity_bias": ROBUSTNESS_SCENARIOS["gravity_bias"],
    "thrust_bias": ROBUSTNESS_SCENARIOS["thrust_bias"],
    "sensor_noise": ROBUSTNESS_SCENARIOS["sensor_noise"],
    "action_delay_1": ROBUSTNESS_SCENARIOS["action_delay_1"],
    "action_delay_2": ROBUSTNESS_SCENARIOS["action_delay_2"],
    "combined": ROBUSTNESS_SCENARIOS["combined"],
}


def load_model_and_obs_rms(model_path: Path, stats_path: Path):
    model = PPO.load(str(model_path), device="cpu")
    eval_env = DummyVecEnv([lambda: RocketLandingEnvAblation()])
    vec_norm = VecNormalize.load(str(stats_path), eval_env)
    obs_rms = vec_norm.obs_rms
    vec_norm.close()
    return model, obs_rms


def evaluate_mode_scenarios(mode, model_path, stats_path, scenarios, n_episodes, seed):
    model, obs_rms = load_model_and_obs_rms(model_path, stats_path)
    results = {}
    representative = {}
    for idx, (scenario_name, options) in enumerate(scenarios.items()):
        print(f"  {mode} / {scenario_name}")
        rollouts = evaluate_model_rollouts(
            model=model,
            env_factory=lambda mode=mode: RocketLandingEnvAblation(ablation_mode=mode),
            obs_rms=obs_rms,
            n_episodes=n_episodes,
            seed=seed + idx * 10_000,
            options=options,
            save_trajectories=True,
        )
        summary = summarize_rollouts(rollouts, model_path=str(model_path), seed=seed + idx * 10_000)
        summary["ablation_mode"] = mode
        summary["scenario"] = scenario_name
        summary["reset_options"] = options
        results[scenario_name] = summary
        representative[scenario_name] = {
            "success": next((r for r in rollouts if r["success"]), None),
            "failure": next((r for r in rollouts if not r["success"]), None),
        }
    return results, representative


def write_csv(output_path, generalization, robustness):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "suite",
                "ablation_mode",
                "scenario",
                "n_episodes",
                "success_rate",
                "crash_rate",
                "timeout_rate",
                "mean_final_velocity_error",
                "mean_fuel_used",
                "mean_abs_throttle_delta",
                "terminal_reason_counts",
            ]
        )
        for suite_name, suite in [("generalization", generalization), ("robustness", robustness)]:
            for mode, scenarios in suite.items():
                for scenario, summary in scenarios.items():
                    writer.writerow(
                        [
                            suite_name,
                            mode,
                            scenario,
                            summary["n_episodes"],
                            summary["success_rate"],
                            summary["crash_rate"],
                            summary["timeout_rate"],
                            summary["mean_final_velocity_error"],
                            summary["mean_fuel_used"],
                            summary["mean_abs_throttle_delta"],
                            json.dumps(summary["terminal_reason_counts"], ensure_ascii=False),
                        ]
                    )


def plot_heatmap(data, title, output_path, metric="success_rate"):
    modes = list(data.keys())
    scenarios = list(next(iter(data.values())).keys())
    matrix = np.array([[data[mode][scenario].get(metric, 0.0) for scenario in scenarios] for mode in modes])

    fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 1.2), max(5, len(modes) * 0.65)))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0 if metric.endswith("rate") else None)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(modes)))
    ax.set_xticklabels(scenarios, rotation=35, ha="right")
    ax.set_yticklabels(modes)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    for i in range(len(modes)):
        for j in range(len(scenarios)):
            value = matrix[i, j]
            label = f"{value:.0%}" if metric.endswith("rate") else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run generalization and robustness tests for every ablation model.")
    parser.add_argument("--source-dir", type=str, default="results/final_report/ablation")
    parser.add_argument("--output-dir", type=str, default="results/reproducible/ablation_scenarios")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--modes", type=str, default=",".join(ABLATION_MODES))
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    generalization_results = {}
    robustness_results = {}
    generalization_trajectories = {}
    robustness_trajectories = {}

    for mode_idx, mode in enumerate(modes):
        model_path = source_dir / mode / "model.zip"
        stats_path = source_dir / mode / "vec_normalize.pkl"
        if not model_path.exists() or not stats_path.exists():
            raise FileNotFoundError(f"Missing ablation model or stats for {mode}: {model_path}, {stats_path}")

        print(f"Ablation mode: {mode}")
        gen, gen_traj = evaluate_mode_scenarios(
            mode,
            model_path,
            stats_path,
            GENERALIZATION_SCENARIOS,
            args.n_episodes,
            args.seed + mode_idx * 100_000,
        )
        rob, rob_traj = evaluate_mode_scenarios(
            mode,
            model_path,
            stats_path,
            ROBUST_ONLY_SCENARIOS,
            args.n_episodes,
            args.seed + 500_000 + mode_idx * 100_000,
        )
        generalization_results[mode] = gen
        robustness_results[mode] = rob
        generalization_trajectories[mode] = gen_traj
        robustness_trajectories[mode] = rob_traj

    with open(output_dir / "ablation_generalization.json", "w", encoding="utf-8") as f:
        json.dump(generalization_results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "ablation_robustness.json", "w", encoding="utf-8") as f:
        json.dump(robustness_results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "ablation_generalization_trajectories.json", "w", encoding="utf-8") as f:
        json.dump(generalization_trajectories, f, indent=2, ensure_ascii=False)
    with open(output_dir / "ablation_robustness_trajectories.json", "w", encoding="utf-8") as f:
        json.dump(robustness_trajectories, f, indent=2, ensure_ascii=False)

    write_csv(output_dir / "ablation_scenario_summary.csv", generalization_results, robustness_results)
    plot_heatmap(
        generalization_results,
        "Ablation Generalization Success Rate",
        output_dir / "ablation_generalization_success_heatmap.png",
    )
    plot_heatmap(
        robustness_results,
        "Ablation Robustness Success Rate",
        output_dir / "ablation_robustness_success_heatmap.png",
    )
    print(f"Saved ablation scenario tests to {output_dir}")


if __name__ == "__main__":
    main()
