"""Verify that the reproducible experiment deliverables are present and sane."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rocket_landing_control.core.experiment_utils import ROBUSTNESS_SCENARIOS


ROOT = Path("results/reproducible")


def require(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def load_json(path):
    require(path.exists(), f"{path} exists")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    energy_root = ROOT / "energy_ppo_from_scratch_time"
    if not (energy_root / "baseline_vs_pure_energy_ppo.json").exists():
        energy_root = ROOT / "energy_ppo_time_finetune"
    if not (energy_root / "baseline_vs_pure_energy_ppo.json").exists():
        energy_root = ROOT / "energy_ppo_success"
    if not (energy_root / "baseline_vs_pure_energy_ppo.json").exists():
        energy_root = ROOT / "energy_ppo"

    main_eval = load_json(ROOT / "baseline_existing_model_eval_100" / "eval_results.json")
    require(main_eval["n_episodes"] == 100, "main PPO evaluation has 100 episodes")
    require(main_eval["success_rate"] >= 0.70, "main PPO success rate >= 70%")
    require(main_eval["terminal_reason_counts"].get("success", 0) >= 70, "main PPO has at least 70 successes")

    csv_path = ROOT / "baseline_existing_model_eval_100" / "eval_episodes.csv"
    require(csv_path.exists(), "main eval_episodes.csv exists")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 100, "eval_episodes.csv has 100 rows")
    require(len({r["initial_height"] for r in rows}) > 1, "initial heights are randomized")
    require(len({r["initial_velocity"] for r in rows}) > 1, "initial velocities are randomized")

    safety = load_json(ROOT / "safety_comparison.json")
    require({"without_safety", "with_safety"}.issubset(safety), "safety comparison has both variants")

    robustness = load_json(ROOT / "robustness_full.json")
    required_robustness = {
        "standard",
        "random_height",
        "random_velocity",
        "random_mass",
        "random_fuel",
        "gravity_bias",
        "thrust_bias",
        "sensor_noise",
        "action_delay_1",
        "action_delay_2",
        "combined",
    }
    require(required_robustness.issubset(robustness), "robustness scenarios are complete")

    controller = load_json(ROOT / "controller_comparison.json")
    require({"PPO", "PID"}.issubset(controller), "controller comparison includes PPO and PID")

    ablation = load_json(ROOT / "ablation" / "reward_ablation.json")
    require(
        {"full", "no_fuel", "no_smooth", "no_safety", "no_success", "basic"}.issubset(ablation),
        "reward ablation modes are complete",
    )
    ablation_gen = load_json(ROOT / "ablation_scenarios" / "ablation_generalization.json")
    ablation_rob = load_json(ROOT / "ablation_scenarios" / "ablation_robustness.json")
    require(set(ablation).issubset(ablation_gen), "ablation generalization includes every ablation mode")
    require(set(ablation).issubset(ablation_rob), "ablation robustness includes every ablation mode")
    require(
        {"standard", "random_height", "random_velocity", "random_mass", "random_fuel"}.issubset(
            next(iter(ablation_gen.values()))
        ),
        "ablation generalization scenarios are complete",
    )
    require(
        {"standard", "gravity_bias", "thrust_bias", "sensor_noise", "action_delay_1", "action_delay_2", "combined"}.issubset(
            next(iter(ablation_rob.values()))
        ),
        "ablation robustness scenarios are complete",
    )

    rl = load_json(ROOT / "rl_comparison" / "rl_comparison.json")
    require({"PPO", "SAC", "TD3"}.issubset(rl), "optional RL comparison includes PPO/SAC/TD3")

    energy = load_json(energy_root / "baseline_vs_pure_energy_ppo.json")
    require({"Baseline PPO", "Pure Energy-Guided PPO"}.issubset(energy), "pure energy PPO comparison has both strategies")
    require(energy["Pure Energy-Guided PPO"]["n_episodes"] == 100, "pure energy PPO evaluation has 100 episodes")
    require(energy["Pure Energy-Guided PPO"]["success_rate"] >= 0.70, "pure energy PPO success rate >= 70%")
    require(
        "energy_ppo_from_scratch_time" in energy["Pure Energy-Guided PPO"].get("model_path", ""),
        "official pure energy PPO result is trained in the from-scratch time directory",
    )
    require(
        energy["Pure Energy-Guided PPO"]["mean_landing_time"] <= energy["Baseline PPO"]["mean_landing_time"],
        "time-efficient energy PPO lands no slower than baseline PPO",
    )
    require(
        "terminal_reason_counts" in energy["Pure Energy-Guided PPO"],
        "pure energy PPO records terminal reason counts",
    )
    scenario_comparison = load_json(energy_root / "scenarios" / "standard_comparison" / "scenario_comparison.json")
    unified_results = load_json(energy_root / "scenarios" / "standard_comparison" / "results_by_policy.json")
    require(
        {"Baseline PPO", "Pure Energy PPO"}.issubset(scenario_comparison["labels"]),
        "unified scenario comparison includes baseline and pure energy PPO",
    )
    require(
        set(ROBUSTNESS_SCENARIOS.keys()).issubset(scenario_comparison["scenarios"]),
        "unified scenario comparison includes all standard scenarios",
    )
    require(
        {"standard", "random_height", "random_velocity", "random_mass", "random_fuel"}.issubset(
            scenario_comparison["generalization_scenarios"]
        ),
        "unified scenario comparison includes generalization scenarios",
    )
    require(
        {"standard", "gravity_bias", "thrust_bias", "sensor_noise", "action_delay_1", "action_delay_2", "combined"}.issubset(
            scenario_comparison["robustness_scenarios"]
        ),
        "unified scenario comparison includes robustness scenarios",
    )
    require(
        "Pure Energy PPO" in unified_results,
        "unified scenario results include pure energy PPO",
    )
    require(
        min(s["success_rate"] for s in unified_results["Pure Energy PPO"].values()) >= 0.70,
        "unified pure energy PPO scenario success rates >= 70%",
    )

    for path in [
        ROOT / "baseline_existing_model_figures" / "success_trajectory.png",
        ROOT / "failure_case_trajectory.json",
        ROOT / "failure_case_figures" / "trajectory.png",
        ROOT / "failure_case_figures" / "hv_phase.png",
        ROOT / "baseline_existing_model_figures" / "eval_summary.png",
        ROOT / "figures" / "training_curves.png",
        ROOT / "figures" / "safety_comparison.png",
        ROOT / "figures" / "robustness_results.png",
        ROOT / "figures" / "controller_comparison.png",
        ROOT / "figures" / "reward_ablation.png",
        ROOT / "figures" / "rl_comparison.png",
        ROOT / "trajectory_comparisons" / "controller_trajectory_comparison.png",
        ROOT / "trajectory_comparisons" / "safety_trajectory_comparison.png",
        ROOT / "trajectory_comparisons" / "robustness_all_scenarios_trajectory_comparison.png",
        ROOT / "trajectory_comparisons" / "robustness_initial_conditions.png",
        ROOT / "trajectory_comparisons" / "robustness_physics_disturbance.png",
        ROOT / "trajectory_comparisons" / "robustness_noise_delay.png",
        ROOT / "trajectory_comparisons" / "robustness_combined.png",
        ROOT / "trajectory_comparisons" / "generalization_trajectory_comparison.png",
        ROOT / "trajectory_comparisons" / "ablation_trajectory_comparison.png",
        ROOT / "ablation_scenarios" / "ablation_generalization_success_heatmap.png",
        ROOT / "ablation_scenarios" / "ablation_robustness_success_heatmap.png",
        ROOT / "ablation_scenarios" / "ablation_scenario_summary.csv",
        ROOT / "ablation_scenarios" / "ablation_generalization_trajectories.json",
        ROOT / "ablation_scenarios" / "ablation_robustness_trajectories.json",
        ROOT / "trajectory_comparisons" / "ablation_scenarios" / "generalization_full_scenario_trajectory_comparison.png",
        ROOT / "trajectory_comparisons" / "ablation_scenarios" / "generalization_random_fuel_ablation_trajectory_comparison.png",
        ROOT / "trajectory_comparisons" / "ablation_scenarios" / "robustness_full_scenario_trajectory_comparison.png",
        ROOT / "trajectory_comparisons" / "ablation_scenarios" / "robustness_combined_ablation_trajectory_comparison.png",
        ROOT / "trajectory_comparisons" / "rl_algorithm_trajectory_comparison.png",
        ROOT / "trajectory_comparisons" / "robustness_trajectory_metadata.json",
        energy_root / "models" / "pure_energy_ppo_model.zip",
        energy_root / "models" / "pure_energy_vec_normalize.pkl",
        energy_root / "eval" / "eval_results.json",
        energy_root / "eval" / "eval_episodes.csv",
        energy_root / "eval" / "all_trajectories.json",
        energy_root / "baseline_vs_pure_energy_ppo.json",
        energy_root / "pure_energy_accounting.png",
        energy_root / "pure_energy_reward_breakdown.png",
        energy_root / "baseline_vs_pure_energy_trajectory.png",
        energy_root / "baseline_vs_pure_energy_metrics.png",
        energy_root / "scenarios" / "standard_comparison" / "scenario_comparison.json",
        energy_root / "scenarios" / "standard_comparison" / "scenario_comparison.csv",
        energy_root / "scenarios" / "standard_comparison" / "results_by_policy.json",
        energy_root / "scenarios" / "standard_comparison" / "representative_trajectories.json",
        energy_root / "scenarios" / "standard_comparison" / "scenario_comparison_metrics.png",
        energy_root / "scenarios" / "standard_comparison" / "trajectory_comparisons" / "random_height_trajectory_comparison.png",
        energy_root / "scenarios" / "standard_comparison" / "trajectory_comparisons" / "random_fuel_trajectory_comparison.png",
        energy_root / "scenarios" / "standard_comparison" / "trajectory_comparisons" / "combined_trajectory_comparison.png",
        ROOT / "landing_demo.gif",
        ROOT / "VERIFIED_RESULTS.md",
        ROOT / "verified_summary.json",
        Path("docs/reports/REPORT_REPRODUCIBLE.md"),
        Path("docs/AI_USAGE.md"),
    ]:
        require(path.exists() and path.stat().st_size > 0, f"{path} is present and non-empty")

    print("All reproducible deliverables verified.")


if __name__ == "__main__":
    main()
