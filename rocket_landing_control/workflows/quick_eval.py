"""Quick evaluation for intermediate and final PPO checkpoints."""

from __future__ import annotations

import argparse
import json
import os

from stable_baselines3 import PPO

from rocket_landing_control.envs.rocket_env import RocketLandingEnv
from rocket_landing_control.core.experiment_utils import (
    STANDARD_EVAL_OPTIONS,
    auto_find_stats,
    evaluate_model_rollouts,
    load_obs_rms,
    summarize_rollouts,
)


def quick_evaluate(model_path, stats_path=None, n_episodes=10, seed=42, options=None):
    model = PPO.load(model_path, device="cpu")
    stats_path = auto_find_stats(model_path, stats_path)
    obs_rms = load_obs_rms(stats_path)
    rollouts = evaluate_model_rollouts(
        model=model,
        env_factory=RocketLandingEnv,
        obs_rms=obs_rms,
        n_episodes=n_episodes,
        seed=seed,
        options=options or STANDARD_EVAL_OPTIONS,
        save_trajectories=False,
    )
    return summarize_rollouts(rollouts, model_path=model_path, seed=seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Quick evaluation of a rocket landing model.")
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 model file.")
    parser.add_argument("--stats", type=str, default=None, help="Path to VecNormalize stats file.")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = quick_evaluate(args.model, args.stats, args.n_episodes, args.seed)
    print(f"Success rate: {result['success_rate']:.1%}")
    print(f"Crash rate: {result['crash_rate']:.1%}")
    print(f"Mean |final h|: {result['mean_final_height_error']:.3f} m")
    print(f"Mean |final v|: {result['mean_final_velocity_error']:.3f} m/s")
    print(f"Mean fuel used: {result['mean_fuel_used']:.3f} kg")
    print(f"Terminal reasons: {result['terminal_reason_counts']}")

    output_path = args.output or os.path.join(os.path.dirname(args.model), "quick_eval_latest.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
