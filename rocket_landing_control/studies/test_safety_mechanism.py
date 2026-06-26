"""Compare PPO with and without the safety shield."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from rocket_landing_control.envs.rocket_env import RocketLandingEnv
from rocket_landing_control.envs.rocket_env_safe import RocketLandingEnvSafe
from rocket_landing_control.core.experiment_utils import (
    STANDARD_EVAL_OPTIONS,
    auto_find_stats,
    evaluate_model_rollouts,
    load_obs_rms,
    summarize_rollouts,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Safety shield comparison.")
    parser.add_argument("--model", type=str, default="results/reproducible/main/models/final_model.zip")
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=str, default="results/reproducible/safety_comparison.json")
    return parser.parse_args()


def evaluate_env(label, env_factory, model, obs_rms, args):
    rollouts = evaluate_model_rollouts(
        model=model,
        env_factory=env_factory,
        obs_rms=obs_rms,
        n_episodes=args.n_episodes,
        seed=args.seed,
        options=STANDARD_EVAL_OPTIONS,
        save_trajectories=False,
    )
    summary = summarize_rollouts(rollouts, model_path=args.model, seed=args.seed)
    summary["label"] = label
    print(
        f"{label}: success={summary['success_rate']:.1%}, "
        f"crash={summary['crash_rate']:.1%}, "
        f"fuel={summary['mean_fuel_used']:.2f}, "
        f"throttle_delta={summary['mean_abs_throttle_delta']:.4f}, "
        f"safety_rate={summary['mean_safety_intervention_rate']:.3f}"
    )
    return summary


def main():
    args = parse_args()
    stats_path = auto_find_stats(args.model, args.stats)
    model = PPO.load(args.model, device="cpu")
    obs_rms = load_obs_rms(stats_path)

    results = {
        "without_safety": evaluate_env("without_safety", RocketLandingEnv, model, obs_rms, args),
        "with_safety": evaluate_env("with_safety", RocketLandingEnvSafe, model, obs_rms, args),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved safety comparison to: {output}")


if __name__ == "__main__":
    main()
