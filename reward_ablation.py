"""Reward ablation experiment for PPO rocket landing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.rocket_env import RocketLandingEnv
from experiment_utils import STANDARD_EVAL_OPTIONS, evaluate_model_rollouts, summarize_rollouts


ABLATION_MODES = ["full", "no_fuel", "no_smooth", "no_safety", "no_success", "basic"]


class RocketLandingEnvAblation(RocketLandingEnv):
    """Same physics, modified reward components."""

    def __init__(self, ablation_mode="full", **kwargs):
        super().__init__(**kwargs)
        if ablation_mode not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation mode: {ablation_mode}")
        self.ablation_mode = ablation_mode

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        breakdown = info.get("reward_breakdown", {})
        if self.ablation_mode == "full":
            return obs, reward, terminated, truncated, info

        if self.ablation_mode == "no_fuel":
            reward -= breakdown.get("fuel", 0.0)
        elif self.ablation_mode == "no_smooth":
            reward -= breakdown.get("smooth", 0.0)
        elif self.ablation_mode == "no_safety":
            reward -= breakdown.get("safety", 0.0)
            reward -= breakdown.get("efficiency", 0.0)
        elif self.ablation_mode == "no_success":
            if info.get("terminal_reason") == "success":
                reward -= breakdown.get("terminal", 0.0)
        elif self.ablation_mode == "basic":
            reward = breakdown.get("height", 0.0) + breakdown.get("velocity", 0.0)

        return obs, float(reward), terminated, truncated, info


def make_env(mode):
    return Monitor(RocketLandingEnvAblation(ablation_mode=mode, randomize=True))


def train_variant(mode, output_dir, train_steps, seed, device):
    model_dir = output_dir / mode
    model_path = model_dir / "model.zip"
    stats_path = model_dir / "vec_normalize.pkl"
    if model_path.exists() and stats_path.exists():
        return model_path, stats_path

    model_dir.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([lambda: make_env(mode)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
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
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        device=device,
        verbose=0,
    )
    model.learn(total_timesteps=train_steps, progress_bar=False)
    model.save(str(model_dir / "model"))
    env.save(str(stats_path))
    return model_path, stats_path


def evaluate_variant(model_path, stats_path, mode, n_episodes, seed):
    model = PPO.load(str(model_path), device="cpu")
    eval_env = DummyVecEnv([lambda: RocketLandingEnvAblation(ablation_mode=mode, randomize=False)])
    vec_norm = VecNormalize.load(str(stats_path), eval_env)
    obs_rms = vec_norm.obs_rms
    vec_norm.close()
    rollouts = evaluate_model_rollouts(
        model=model,
        env_factory=lambda: RocketLandingEnvAblation(ablation_mode=mode),
        obs_rms=obs_rms,
        n_episodes=n_episodes,
        seed=seed,
        options=STANDARD_EVAL_OPTIONS,
        save_trajectories=True,
    )
    return summarize_rollouts(rollouts, model_path=str(model_path), seed=seed), rollouts


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate reward ablations.")
    parser.add_argument("--output-dir", type=str, default="results/reproducible/ablation")
    parser.add_argument("--source-dir", type=str, default=None,
                        help="Optional directory containing pre-trained mode/model.zip and vec_normalize.pkl.")
    parser.add_argument("--train-steps", type=int, default=150_000)
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--modes", type=str, default=",".join(ABLATION_MODES))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir) if args.source_dir else None
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    results = {}
    trajectories = {}
    for i, mode in enumerate(modes):
        print(f"Ablation: {mode}")
        if source_dir and (source_dir / mode / "model.zip").exists():
            model_path = source_dir / mode / "model.zip"
            stats_path = source_dir / mode / "vec_normalize.pkl"
        else:
            model_path, stats_path = train_variant(mode, output_dir, args.train_steps, args.seed + i, args.device)
        summary, rollouts = evaluate_variant(model_path, stats_path, mode, args.n_episodes, args.seed + i * 1000)
        summary["ablation_mode"] = mode
        summary["source_model"] = str(model_path)
        results[mode] = summary
        trajectories[mode] = {
            "success": next((r for r in rollouts if r["success"]), None),
            "failure": next((r for r in rollouts if not r["success"]), None),
        }
        print(f"  success={summary['success_rate']:.1%}, fuel={summary['mean_fuel_used']:.2f}")

    with open(output_dir / "reward_ablation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "reward_ablation_trajectories.json", "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)
    print(f"Saved ablation results to: {output_dir}")


if __name__ == "__main__":
    main()
