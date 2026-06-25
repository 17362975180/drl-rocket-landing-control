"""Formal evaluation for PPO rocket landing models.

Default protocol: 100 randomized in-distribution episodes:
height 45-55 m, initial velocity -1..1 m/s, nominal mass/fuel, no disturbance.
"""

from __future__ import annotations

import argparse
import json

from stable_baselines3 import PPO

from envs.rocket_env import RocketLandingEnv
from experiment_utils import (
    ROBUSTNESS_SCENARIOS,
    STANDARD_EVAL_OPTIONS,
    auto_find_stats,
    evaluate_model_rollouts,
    load_obs_rms,
    summarize_rollouts,
    write_evaluation_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO rocket landing model.")
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 model file.")
    parser.add_argument("--stats", type=str, default=None, help="Path to VecNormalize stats file.")
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--output-dir", type=str, default="results/reproducible/main_eval")
    parser.add_argument("--scenario", type=str, default="standard", choices=sorted(ROBUSTNESS_SCENARIOS))
    parser.add_argument("--options-json", type=str, default=None, help="JSON object overriding reset options.")
    parser.add_argument("--save-trajectories", action="store_true", help="Save trajectory JSON files.")
    return parser.parse_args()


def scenario_options(name: str, options_json: str | None):
    options = dict(ROBUSTNESS_SCENARIOS.get(name, STANDARD_EVAL_OPTIONS))
    if options_json:
        options.update(json.loads(options_json))
    return options


def evaluate_model(
    model_path: str,
    stats_path: str | None = None,
    n_episodes: int = 100,
    seed: int = 42,
    output_dir: str = "results/reproducible/main_eval",
    save_trajectories: bool = False,
    scenario: str = "standard",
    options_json: str | None = None,
):
    stats_path = auto_find_stats(model_path, stats_path)
    print(f"Loading model: {model_path}")
    print(f"Normalization stats: {stats_path or 'None'}")
    model = PPO.load(model_path, device="cpu")
    obs_rms = load_obs_rms(stats_path)
    options = scenario_options(scenario, options_json)

    print(f"Running {n_episodes} episodes, scenario={scenario}, seed={seed}")
    rollouts = evaluate_model_rollouts(
        model=model,
        env_factory=RocketLandingEnv,
        obs_rms=obs_rms,
        n_episodes=n_episodes,
        seed=seed,
        options=options,
        save_trajectories=save_trajectories,
    )
    summary = summarize_rollouts(rollouts, model_path=model_path, seed=seed)
    summary["scenario"] = scenario
    summary["reset_options"] = options
    summary["stats_path"] = stats_path
    write_evaluation_outputs(rollouts, summary, output_dir, save_trajectories)

    counts = summary["terminal_reason_counts"]
    print("=" * 60)
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Crash rate: {summary['crash_rate']:.1%}")
    print(f"Mean |final h|: {summary['mean_final_height_error']:.3f} m")
    print(f"Mean |final v|: {summary['mean_final_velocity_error']:.3f} m/s")
    print(f"Mean fuel used: {summary['mean_fuel_used']:.3f} kg")
    print(f"Mean throttle delta: {summary['mean_abs_throttle_delta']:.4f}")
    print(f"Terminal reasons: {counts}")
    print(f"Saved evaluation outputs to: {output_dir}")
    return summary


def main():
    args = parse_args()
    evaluate_model(
        model_path=args.model,
        stats_path=args.stats,
        n_episodes=args.n_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        save_trajectories=args.save_trajectories,
        scenario=args.scenario,
        options_json=args.options_json,
    )


if __name__ == "__main__":
    main()
