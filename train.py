"""Train PPO for the rocket vertical soft-landing task."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.rocket_env import RocketLandingEnv
from experiment_utils import STANDARD_EVAL_OPTIONS, evaluate_model_rollouts, summarize_rollouts


class QuickEvalCallback(BaseCallback):
    """Evaluate the current policy on the standard randomized protocol."""

    def __init__(
        self,
        eval_interval: int,
        eval_episodes: int,
        output_dir: str,
        seed: int,
        target_success: float,
        stop_after_successes: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.output_dir = output_dir
        self.seed = seed
        self.target_success = target_success
        self.stop_after_successes = stop_after_successes
        self.eval_results = []
        self.best_success_rate = -1.0
        self.consecutive_successes = 0

    def _on_step(self) -> bool:
        if self.n_calls <= 0 or self.n_calls % self.eval_interval != 0:
            return True

        obs_rms = getattr(self.training_env, "obs_rms", None)
        rollouts = evaluate_model_rollouts(
            model=self.model,
            env_factory=RocketLandingEnv,
            obs_rms=obs_rms,
            n_episodes=self.eval_episodes,
            seed=self.seed + self.n_calls,
            options=STANDARD_EVAL_OPTIONS,
            save_trajectories=False,
        )
        result = summarize_rollouts(rollouts, seed=self.seed + self.n_calls)
        result["step"] = self.n_calls
        self.eval_results.append(result)

        history_path = os.path.join(self.output_dir, "quick_eval_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.eval_results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "quick_eval_latest.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if result["success_rate"] > self.best_success_rate:
            self.best_success_rate = result["success_rate"]
            best_model_path = os.path.join(self.output_dir, "models", "best_model")
            self.model.save(best_model_path)
            if hasattr(self.training_env, "save"):
                self.training_env.save(os.path.join(self.output_dir, "models", "best_vec_normalize.pkl"))

        if result["success_rate"] >= self.target_success:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0

        if self.verbose:
            print(
                f"[step {self.n_calls}] success={result['success_rate']:.1%}, "
                f"crash={result['crash_rate']:.1%}, "
                f"|v_f|={result['mean_final_velocity_error']:.2f}, "
                f"fuel={result['mean_fuel_used']:.2f}"
            )

        if self.stop_after_successes > 0 and self.consecutive_successes >= self.stop_after_successes:
            print(
                f"Early stop: success rate >= {self.target_success:.0%} for "
                f"{self.consecutive_successes} consecutive evals."
            )
            return False
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="PPO rocket landing training")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--net-arch", type=str, default="128,128")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--fixed-train", action="store_true", help="Disable training randomization.")
    parser.add_argument("--target-success", type=float, default=0.70)
    parser.add_argument("--stop-after-successes", type=int, default=3)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def make_env(randomize: bool):
    return Monitor(RocketLandingEnv(randomize=randomize))


def main():
    args = parse_args()
    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    run_name = args.run_name or datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("results", "reproducible", run_name)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    config = vars(args).copy()
    config["net_arch"] = net_arch
    config["standard_eval_options"] = STANDARD_EVAL_OPTIONS
    with open(os.path.join(output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    train_env = DummyVecEnv([lambda: make_env(randomize=not args.fixed_train)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    callback = QuickEvalCallback(
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        output_dir=output_dir,
        seed=args.seed,
        target_success=args.target_success,
        stop_after_successes=args.stop_after_successes,
        verbose=args.verbose,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=args.verbose,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        policy_kwargs={"net_arch": net_arch},
        tensorboard_log=os.path.join(output_dir, "tb_logs"),
    )

    print(f"Training run: {run_name}")
    print(f"Output: {output_dir}")
    print(f"Randomized training: {not args.fixed_train}")
    model.learn(total_timesteps=args.total_steps, callback=callback, progress_bar=args.progress_bar)

    final_model_path = os.path.join(output_dir, "models", "final_model")
    model.save(final_model_path)
    train_env.save(os.path.join(output_dir, "models", "vec_normalize.pkl"))

    summary = {
        "run_name": run_name,
        "total_steps_requested": args.total_steps,
        "total_steps_completed": model.num_timesteps,
        "best_success_rate": callback.best_success_rate,
        "final_eval": callback.eval_results[-1] if callback.eval_results else None,
        "num_evals": len(callback.eval_results),
        "model_path": final_model_path + ".zip",
        "stats_path": os.path.join(output_dir, "models", "vec_normalize.pkl"),
    }
    with open(os.path.join(output_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Training complete.")
    print(f"Best quick success rate: {callback.best_success_rate:.1%}")
    print(f"Model saved to: {final_model_path}.zip")
    return model, train_env, callback.eval_results


if __name__ == "__main__":
    main()
