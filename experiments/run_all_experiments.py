"""Orchestrate reproducible rocket landing experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(name, command, timeout):
    print("=" * 70)
    print(name)
    print(" ".join(str(x) for x in command))
    print("=" * 70)
    result = subprocess.run(command, cwd=ROOT, text=True, timeout=timeout)
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the reproducible experiment suite.")
    parser.add_argument("--model", type=str, required=True, help="Trained PPO model path.")
    parser.add_argument("--stats", type=str, default=None, help="VecNormalize stats path.")
    parser.add_argument("--output-dir", type=str, default="results/reproducible/final_report")
    parser.add_argument("--main-episodes", type=int, default=100)
    parser.add_argument("--robust-episodes", type=int, default=50)
    parser.add_argument("--controller-episodes", type=int, default=100)
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-rl-comparison", action="store_true")
    parser.add_argument("--ablation-train-steps", type=int, default=150_000)
    parser.add_argument("--rl-train-steps", type=int, default=150_000)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    stats_args = ["--stats", args.stats] if args.stats else []

    run_command(
        "Experiment 1: PPO main evaluation",
        [
            sys.executable,
            "-m",
            "rocket_landing_control.workflows.evaluate",
            "--model",
            args.model,
            *stats_args,
            "--n-episodes",
            str(args.main_episodes),
            "--output-dir",
            str(out / "main_eval"),
            "--save-trajectories",
        ],
        timeout=3600,
    )

    run_command(
        "Experiment 2: safety shield comparison",
        [
            sys.executable,
            "-m",
            "rocket_landing_control.studies.test_safety_mechanism",
            "--model",
            args.model,
            *stats_args,
            "--n-episodes",
            str(args.robust_episodes),
            "--output",
            str(out / "safety_comparison.json"),
        ],
        timeout=3600,
    )

    run_command(
        "Experiment 3: robustness",
        [
            sys.executable,
            "-m",
            "rocket_landing_control.studies.robustness_full_test",
            "--model",
            args.model,
            *stats_args,
            "--n-episodes",
            str(args.robust_episodes),
            "--output",
            str(out / "robustness_full.json"),
        ],
        timeout=7200,
    )

    run_command(
        "Experiment 4: controller comparison",
        [
            sys.executable,
            "-m",
            "rocket_landing_control.studies.controller_comparison_full",
            "--model",
            args.model,
            *stats_args,
            "--n-episodes",
            str(args.controller_episodes),
            "--output",
            str(out / "controller_comparison.json"),
            "--save-trajectories",
        ],
        timeout=7200,
    )

    if args.run_ablation:
        run_command(
            "Experiment 5: reward ablation",
            [
                sys.executable,
                "-m",
                "rocket_landing_control.studies.reward_ablation",
                "--output-dir",
                str(out / "ablation"),
                "--train-steps",
                str(args.ablation_train_steps),
            ],
            timeout=24 * 3600,
        )

    if args.run_rl_comparison:
        run_command(
            "Experiment 6: PPO/SAC/TD3 comparison",
            [
                sys.executable,
                "-m",
                "rocket_landing_control.studies.rl_comparison",
                "--output-dir",
                str(out / "rl_comparison"),
                "--train-steps",
                str(args.rl_train_steps),
            ],
            timeout=24 * 3600,
        )

    run_command(
        "Generate main figures",
        [
            sys.executable,
            "-m",
            "rocket_landing_control.visualization.plot_results",
            "--result-dir",
            str(out / "figures"),
            "--eval-dir",
            str(out / "main_eval"),
        ],
        timeout=1800,
    )
    print("All requested experiments completed.")


if __name__ == "__main__":
    main()
