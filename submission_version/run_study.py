"""Run the four-group final study with paired seeds across every scenario."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = Path("results")
SEED = 2026

STANDARD = RESULTS / "01_standard_ppo"
ENERGY = RESULTS / "02_energy_ppo"
CONTROLS = RESULTS / "03_control_comparison"
OTHER_RL = RESULTS / "04_other_rl_comparison"


def policy(label, *, model=None, stats=None, env="base", algo="PPO", controller=None):
    if controller:
        return f"label={label},type=controller,controller={controller}"
    return f"label={label},type=sb3,algo={algo},env={env},model={model},stats={stats}"


BASELINE = policy(
    "Standard PPO",
    model=STANDARD / "models" / "full" / "model.zip",
    stats=STANDARD / "models" / "full" / "vec_normalize.pkl",
)
ENERGY_FULL = policy(
    "Energy PPO",
    model=ENERGY / "models" / "full" / "model.zip",
    stats=ENERGY / "models" / "full" / "vec_normalize.pkl",
    env="energy",
)


GROUPS = {
    "standard_analysis": (
        STANDARD / "analysis",
        [BASELINE],
    ),
    "standard_ablation": (
        STANDARD / "ablation",
        [
            policy(
                f"Standard-{mode}",
                model=STANDARD / "models" / "ablation" / mode / "model.zip",
                stats=STANDARD / "models" / "ablation" / mode / "vec_normalize.pkl",
            )
            for mode in ("full", "no_fuel", "no_smooth", "no_safety", "no_success", "basic")
        ],
    ),
    "energy_analysis": (
        ENERGY / "analysis",
        [ENERGY_FULL],
    ),
    "energy_ablation": (
        ENERGY / "ablation",
        [
            ENERGY_FULL,
            policy(
                "Energy-NoEnergyObservation",
                model=ENERGY / "models" / "ablation" / "no_energy_observation" / "model.zip",
                stats=ENERGY / "models" / "ablation" / "no_energy_observation" / "vec_normalize.pkl",
                env="energy_base_obs",
            ),
            policy(
                "Energy-PreTimeOptimization",
                model=ENERGY / "models" / "ablation" / "pre_time_optimization" / "model.zip",
                stats=ENERGY / "models" / "ablation" / "pre_time_optimization" / "vec_normalize.pkl",
                env="energy",
            ),
        ],
    ),
    "twin_comparison": (
        ENERGY / "vs_standard",
        [BASELINE, ENERGY_FULL],
    ),
    "control_comparison": (
        CONTROLS,
        [
            BASELINE,
            ENERGY_FULL,
            policy("PID", controller="PID"),
            policy("MPC", controller="MPC"),
            policy("ET-MPC", controller="ET-MPC"),
        ],
    ),
    "rl_comparison": (
        OTHER_RL,
        [
            BASELINE,
            ENERGY_FULL,
            policy(
                "SAC",
                model=OTHER_RL / "models" / "SAC" / "model.zip",
                stats=OTHER_RL / "models" / "SAC" / "vec_normalize.pkl",
                algo="SAC",
            ),
            policy(
                "TD3",
                model=OTHER_RL / "models" / "TD3" / "model.zip",
                stats=OTHER_RL / "models" / "TD3" / "vec_normalize.pkl",
                algo="TD3",
            ),
        ],
    ),
}


def run_command(*args, timeout=86400):
    command = [sys.executable, *map(str, args)]
    print(f"\n{'=' * 76}\n{' '.join(command)}\n{'=' * 76}")
    subprocess.run(command, cwd=ROOT, check=True, timeout=timeout)


def validate_models(policies):
    for raw in policies:
        fields = dict(item.split("=", 1) for item in raw.split(",") if "=" in item)
        if fields.get("type") != "sb3":
            continue
        for key in ("model", "stats"):
            path = Path(fields[key])
            if not path.is_file():
                raise FileNotFoundError(f"Missing {key}: {path}")


def run_group(name, episodes):
    output_dir, policies = GROUPS[name]
    validate_models(policies)
    args = [
        "robustness_full_test.py",
        "--scenario-set", "all",
        "--n-episodes", episodes,
        "--seed", SEED,
        "--save-trajectories",
        "--output-dir", output_dir,
    ]
    for item in policies:
        args.extend(["--policy", item])
    run_command(*args)


def make_demo(analysis_dir, label, output_dir):
    representatives = json.loads((analysis_dir / "representative_trajectories.json").read_text(encoding="utf-8"))
    rollout = representatives[label]["standard"]
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_file = output_dir / "standard_demo_trajectory.json"
    trajectory_file.write_text(json.dumps(rollout, indent=2, ensure_ascii=False), encoding="utf-8")
    run_command(
        "generate_demo_animation.py",
        "--trajectory", trajectory_file,
        "--output", output_dir / "standard_landing_demo.gif",
        timeout=3600,
    )


def main():
    parser = argparse.ArgumentParser(description="Run final paired-seed study groups.")
    parser.add_argument(
        "group",
        choices=[*GROUPS, "demos", "failure_analysis", "report_alignment", "all"],
        nargs="?",
        default="all",
    )
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    selected = list(GROUPS) if args.group == "all" else ([args.group] if args.group in GROUPS else [])
    for name in selected:
        run_group(name, args.episodes)

    if args.group in ("demos", "all"):
        make_demo(STANDARD / "analysis", "Standard PPO", STANDARD / "demo")
        make_demo(ENERGY / "analysis", "Energy PPO", ENERGY / "demo")

    if args.group in ("failure_analysis", "all"):
        run_command("failure_case_analysis.py", "--attempts", 300, timeout=3600)

    if args.group in ("report_alignment", "all"):
        run_command("report_alignment_analysis.py", timeout=3600)

    print("\nFinal study completed.")


if __name__ == "__main__":
    main()
