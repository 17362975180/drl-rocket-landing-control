"""Optional multi-algorithm comparison: PPO vs SAC vs TD3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.rocket_env import RocketLandingEnv
from experiment_utils import STANDARD_EVAL_OPTIONS, evaluate_model_rollouts, summarize_rollouts


ALGORITHMS = {"PPO": PPO, "SAC": SAC, "TD3": TD3}


def make_env():
    return Monitor(RocketLandingEnv(randomize=True))


def train_algorithm(name, output_dir, train_steps, seed, device):
    model_dir = output_dir / name
    model_path = model_dir / "model.zip"
    stats_path = model_dir / "vec_normalize.pkl"
    if model_path.exists() and stats_path.exists():
        return model_path, stats_path

    model_dir.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    cls = ALGORITHMS[name]
    kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": 3e-4,
        "seed": seed,
        "device": device,
        "verbose": 0,
        "policy_kwargs": {"net_arch": [128, 128]},
    }
    if name == "PPO":
        kwargs.update({"n_steps": 2048, "batch_size": 64, "n_epochs": 10, "ent_coef": 0.01})
    if name == "TD3":
        kwargs["action_noise"] = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))
    model = cls(**kwargs)
    model.learn(total_timesteps=train_steps, progress_bar=False)
    model.save(str(model_dir / "model"))
    env.save(str(stats_path))
    return model_path, stats_path


def evaluate_algorithm(name, model_path, stats_path, n_episodes, seed):
    cls = ALGORITHMS[name]
    model = cls.load(str(model_path), device="cpu")
    eval_env = DummyVecEnv([lambda: RocketLandingEnv()])
    vec_norm = VecNormalize.load(str(stats_path), eval_env)
    obs_rms = vec_norm.obs_rms
    vec_norm.close()
    rollouts = evaluate_model_rollouts(
        model=model,
        env_factory=RocketLandingEnv,
        obs_rms=obs_rms,
        n_episodes=n_episodes,
        seed=seed,
        options=STANDARD_EVAL_OPTIONS,
        save_trajectories=True,
    )
    return summarize_rollouts(rollouts, model_path=str(model_path), seed=seed), rollouts


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate PPO, SAC, and TD3.")
    parser.add_argument("--output-dir", type=str, default="results/reproducible/rl_comparison")
    parser.add_argument("--source-dir", type=str, default=None,
                        help="Optional directory containing ALGO/model.zip and vec_normalize.pkl.")
    parser.add_argument("--train-steps", type=int, default=150_000)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--algorithms", type=str, default="PPO,SAC,TD3")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir) if args.source_dir else None
    names = [name.strip().upper() for name in args.algorithms.split(",") if name.strip()]

    results = {}
    trajectories = {}
    for i, name in enumerate(names):
        if name not in ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {name}")
        print(f"Algorithm: {name}")
        if source_dir and (source_dir / name / "model.zip").exists():
            model_path = source_dir / name / "model.zip"
            stats_path = source_dir / name / "vec_normalize.pkl"
        else:
            model_path, stats_path = train_algorithm(name, output_dir, args.train_steps, args.seed + i, args.device)
        summary, rollouts = evaluate_algorithm(name, model_path, stats_path, args.n_episodes, args.seed + i * 1000)
        summary["algorithm"] = name
        summary["source_model"] = str(model_path)
        results[name] = summary
        trajectories[name] = {
            "success": next((r for r in rollouts if r["success"]), None),
            "failure": next((r for r in rollouts if not r["success"]), None),
        }
        print(f"  success={summary['success_rate']:.1%}, fuel={summary['mean_fuel_used']:.2f}")

    with open(output_dir / "rl_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "rl_comparison_trajectories.json", "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)
    print(f"Saved RL comparison results to: {output_dir}")


if __name__ == "__main__":
    main()
