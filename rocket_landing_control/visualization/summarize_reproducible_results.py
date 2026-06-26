"""Create a concise verified summary from reproducible experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/reproducible")
ENERGY_ROOT = ROOT / "energy_ppo_from_scratch_time"
if not (ENERGY_ROOT / "baseline_vs_pure_energy_ppo.json").exists():
    ENERGY_ROOT = ROOT / "energy_ppo_time_finetune"
if not (ENERGY_ROOT / "baseline_vs_pure_energy_ppo.json").exists():
    ENERGY_ROOT = ROOT / "energy_ppo_success"
if not (ENERGY_ROOT / "baseline_vs_pure_energy_ppo.json").exists():
    ENERGY_ROOT = ROOT / "energy_ppo"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    main_eval = load(ROOT / "baseline_existing_model_eval_100" / "eval_results.json")
    safety = load(ROOT / "safety_comparison.json")
    robustness = load(ROOT / "robustness_full.json")
    controller = load(ROOT / "controller_comparison.json")
    ablation = load(ROOT / "ablation" / "reward_ablation.json")
    rl_path = ROOT / "rl_comparison" / "rl_comparison.json"
    rl = load(rl_path) if rl_path.exists() else {}
    energy_path = ENERGY_ROOT / "baseline_vs_pure_energy_ppo.json"
    energy = load(energy_path) if energy_path.exists() else {}
    scenario_comparison_path = ENERGY_ROOT / "scenarios" / "standard_comparison" / "scenario_comparison.json"
    scenario_comparison = load(scenario_comparison_path) if scenario_comparison_path.exists() else {}
    ablation_gen_path = ROOT / "ablation_scenarios" / "ablation_generalization.json"
    ablation_rob_path = ROOT / "ablation_scenarios" / "ablation_robustness.json"
    ablation_gen = load(ablation_gen_path) if ablation_gen_path.exists() else {}
    ablation_rob = load(ablation_rob_path) if ablation_rob_path.exists() else {}

    summary = {
        "main_ppo_standard_100": {
            "success_rate": main_eval["success_rate"],
            "n_episodes": main_eval["n_episodes"],
            "mean_final_velocity_error": main_eval["mean_final_velocity_error"],
            "mean_fuel_used": main_eval["mean_fuel_used"],
            "mean_abs_throttle_delta": main_eval["mean_abs_throttle_delta"],
            "terminal_reason_counts": main_eval["terminal_reason_counts"],
        },
        "safety_comparison": {
            k: {
                "success_rate": v["success_rate"],
                "crash_rate": v["crash_rate"],
                "mean_fuel_used": v["mean_fuel_used"],
                "mean_abs_throttle_delta": v["mean_abs_throttle_delta"],
                "mean_safety_intervention_rate": v["mean_safety_intervention_rate"],
            }
            for k, v in safety.items()
        },
        "robustness_success_rates": {k: v["success_rate"] for k, v in robustness.items()},
        "controller_comparison": {
            k: {
                "success_rate": v["success_rate"],
                "mean_fuel_used": v["mean_fuel_used"],
                "mean_landing_time": v["mean_landing_time"],
                "mean_abs_throttle_delta": v["mean_abs_throttle_delta"],
                "mean_final_velocity_error": v["mean_final_velocity_error"],
            }
            for k, v in controller.items()
        },
        "reward_ablation": {
            k: {
                "success_rate": v["success_rate"],
                "crash_rate": v["crash_rate"],
                "mean_fuel_used": v["mean_fuel_used"],
                "mean_final_velocity_error": v["mean_final_velocity_error"],
                "source_model": v.get("source_model"),
            }
            for k, v in ablation.items()
        },
        "rl_comparison": {
            k: {
                "success_rate": v["success_rate"],
                "mean_fuel_used": v["mean_fuel_used"],
                "mean_abs_throttle_delta": v["mean_abs_throttle_delta"],
                "mean_final_velocity_error": v["mean_final_velocity_error"],
                "source_model": v.get("source_model"),
            }
            for k, v in rl.items()
        },
        "pure_energy_ppo": {
            k: {
                "success_rate": v["success_rate"],
                "crash_rate": v["crash_rate"],
                "mean_fuel_used": v["mean_fuel_used"],
                "mean_abs_throttle_delta": v["mean_abs_throttle_delta"],
                "mean_final_velocity_error": v["mean_final_velocity_error"],
                "terminal_reason_counts": v["terminal_reason_counts"],
                "model_path": v.get("model_path"),
                "stats_path": v.get("stats_path"),
            }
            for k, v in energy.items()
        },
        "same_seed_baseline_vs_energy_success_rates": {
            scenario: {
                label: summary["success_rate"]
                for label, summary in labels.items()
            }
            for scenario, labels in scenario_comparison.get("results", {}).items()
        },
        "ablation_generalization_success_rates": {
            mode: {scenario: summary["success_rate"] for scenario, summary in scenarios.items()}
            for mode, scenarios in ablation_gen.items()
        },
        "ablation_robustness_success_rates": {
            mode: {scenario: summary["success_rate"] for scenario, summary in scenarios.items()}
            for mode, scenarios in ablation_rob.items()
        },
        "figures": {
            "training_curves": str(ROOT / "figures" / "training_curves.png"),
            "main_trajectory": str(ROOT / "baseline_existing_model_figures" / "success_trajectory.png"),
            "failure_trajectory": str(ROOT / "failure_case_figures" / "trajectory.png"),
            "main_eval_summary": str(ROOT / "baseline_existing_model_figures" / "eval_summary.png"),
            "safety": str(ROOT / "figures" / "safety_comparison.png"),
            "safety_trajectory_comparison": str(ROOT / "trajectory_comparisons" / "safety_trajectory_comparison.png"),
            "robustness": str(ROOT / "figures" / "robustness_results.png"),
            "robustness_all_trajectory_comparison": str(ROOT / "trajectory_comparisons" / "robustness_all_scenarios_trajectory_comparison.png"),
            "robustness_initial_conditions": str(ROOT / "trajectory_comparisons" / "robustness_initial_conditions.png"),
            "robustness_physics_disturbance": str(ROOT / "trajectory_comparisons" / "robustness_physics_disturbance.png"),
            "robustness_noise_delay": str(ROOT / "trajectory_comparisons" / "robustness_noise_delay.png"),
            "robustness_combined": str(ROOT / "trajectory_comparisons" / "robustness_combined.png"),
            "generalization_trajectory_comparison": str(ROOT / "trajectory_comparisons" / "generalization_trajectory_comparison.png"),
            "controller": str(ROOT / "figures" / "controller_comparison.png"),
            "controller_trajectory_comparison": str(ROOT / "trajectory_comparisons" / "controller_trajectory_comparison.png"),
            "ablation": str(ROOT / "figures" / "reward_ablation.png"),
            "ablation_trajectory_comparison": str(ROOT / "trajectory_comparisons" / "ablation_trajectory_comparison.png"),
            "ablation_generalization_heatmap": str(ROOT / "ablation_scenarios" / "ablation_generalization_success_heatmap.png"),
            "ablation_robustness_heatmap": str(ROOT / "ablation_scenarios" / "ablation_robustness_success_heatmap.png"),
            "ablation_generalization_full_trajectory": str(ROOT / "trajectory_comparisons" / "ablation_scenarios" / "generalization_full_scenario_trajectory_comparison.png"),
            "ablation_robustness_full_trajectory": str(ROOT / "trajectory_comparisons" / "ablation_scenarios" / "robustness_full_scenario_trajectory_comparison.png"),
            "ablation_generalization_random_fuel_by_mode": str(ROOT / "trajectory_comparisons" / "ablation_scenarios" / "generalization_random_fuel_ablation_trajectory_comparison.png"),
            "ablation_robustness_combined_by_mode": str(ROOT / "trajectory_comparisons" / "ablation_scenarios" / "robustness_combined_ablation_trajectory_comparison.png"),
            "rl_comparison": str(ROOT / "figures" / "rl_comparison.png"),
            "rl_algorithm_trajectory_comparison": str(ROOT / "trajectory_comparisons" / "rl_algorithm_trajectory_comparison.png"),
            "pure_energy_accounting": str(ENERGY_ROOT / "pure_energy_accounting.png"),
            "pure_energy_reward_breakdown": str(ENERGY_ROOT / "pure_energy_reward_breakdown.png"),
            "baseline_vs_pure_energy_trajectory": str(ENERGY_ROOT / "baseline_vs_pure_energy_trajectory.png"),
            "baseline_vs_pure_energy_metrics": str(ENERGY_ROOT / "baseline_vs_pure_energy_metrics.png"),
            "same_seed_scenario_comparison": str(
                ENERGY_ROOT / "scenarios" / "standard_comparison" / "scenario_comparison_metrics.png"
            ),
            "same_seed_random_height_trajectory": str(
                ENERGY_ROOT
                / "scenarios"
                / "standard_comparison"
                / "trajectory_comparisons"
                / "random_height_trajectory_comparison.png"
            ),
            "demo_animation": str(ROOT / "landing_demo.gif"),
        },
    }

    out_json = ROOT / "verified_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md = [
        "# Verified Reproducible Results",
        "",
        "All values below are read from JSON/CSV outputs under `results/reproducible`.",
        "",
        "## Experiment 1: PPO Main Line",
        f"- Standard randomized test: {main_eval['n_episodes']} episodes",
        f"- Success rate: {pct(main_eval['success_rate'])}",
        f"- Mean final velocity error: {main_eval['mean_final_velocity_error']:.3f} m/s",
        f"- Mean fuel used: {main_eval['mean_fuel_used']:.3f} kg",
        f"- Mean action smoothness metric: {main_eval['mean_abs_throttle_delta']:.4f}",
        f"- Terminal reasons: `{main_eval['terminal_reason_counts']}`",
        "",
        "## Experiment 2: Safety Shield",
        "| Variant | Success | Crash | Fuel (kg) | Throttle Delta | Intervention Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, v in safety.items():
        md.append(
            f"| {name} | {pct(v['success_rate'])} | {pct(v['crash_rate'])} | "
            f"{v['mean_fuel_used']:.3f} | {v['mean_abs_throttle_delta']:.4f} | "
            f"{v['mean_safety_intervention_rate']:.3f} |"
        )

    md += [
        "",
        "## Experiment 3: Robustness",
        "| Scenario | Success | Crash | Mean abs(v_final) (m/s) | Fuel (kg) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, v in robustness.items():
        md.append(
            f"| {name} | {pct(v['success_rate'])} | {pct(v['crash_rate'])} | "
            f"{v['mean_final_velocity_error']:.3f} | {v['mean_fuel_used']:.3f} |"
        )

    md += [
        "",
        "## Experiment 4: Controller Comparison",
        "| Controller | Success | Fuel (kg) | Throttle Delta | Mean abs(v_final) (m/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, v in controller.items():
        md.append(
            f"| {name} | {pct(v['success_rate'])} | {v['mean_fuel_used']:.3f} | "
            f"{v['mean_abs_throttle_delta']:.4f} | {v['mean_final_velocity_error']:.3f} |"
        )

    md += [
        "",
        "## Experiment 5: Reward Ablation",
        "| Mode | Success | Crash | Fuel (kg) | Mean abs(v_final) (m/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, v in ablation.items():
        md.append(
            f"| {name} | {pct(v['success_rate'])} | {pct(v['crash_rate'])} | "
            f"{v['mean_fuel_used']:.3f} | {v['mean_final_velocity_error']:.3f} |"
        )

    if rl:
        md += [
            "",
            "## Experiment 6: RL Algorithm Comparison",
            "| Algorithm | Success | Fuel (kg) | Throttle Delta | Mean abs(v_final) (m/s) |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, v in rl.items():
            md.append(
                f"| {name} | {pct(v['success_rate'])} | {v['mean_fuel_used']:.3f} | "
                f"{v['mean_abs_throttle_delta']:.4f} | {v['mean_final_velocity_error']:.3f} |"
            )

    if energy:
        md += [
            "",
            "## Experiment 7: Pure Energy-Guided PPO",
            "| Strategy | Success | Crash | Fuel (kg) | Throttle Delta | Mean abs(v_final) (m/s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, v in energy.items():
            md.append(
                f"| {name} | {pct(v['success_rate'])} | {pct(v['crash_rate'])} | "
                f"{v['mean_fuel_used']:.3f} | {v['mean_abs_throttle_delta']:.4f} | "
                f"{v['mean_final_velocity_error']:.3f} |"
            )
        pure = energy.get("Pure Energy-Guided PPO")
        if pure:
            md += [
                "",
                "This strategy changes only the reward signal. Physics, observations, actions, fuel consumption, mass variation, and termination conditions remain inherited from the base environment.",
                f"Terminal reasons for the pure energy policy: `{pure['terminal_reason_counts']}`.",
            ]

    if ablation_gen:
        md += [
            "",
            "## Experiment 5b: Ablation Generalization Matrix",
            "| Mode | " + " | ".join(next(iter(ablation_gen.values())).keys()) + " |",
            "|---" + "|---:" * len(next(iter(ablation_gen.values())).keys()) + "|",
        ]
        for mode, scenarios in ablation_gen.items():
            md.append("| " + mode + " | " + " | ".join(pct(v["success_rate"]) for v in scenarios.values()) + " |")

    if ablation_rob:
        md += [
            "",
            "## Experiment 5c: Ablation Robustness Matrix",
            "| Mode | " + " | ".join(next(iter(ablation_rob.values())).keys()) + " |",
            "|---" + "|---:" * len(next(iter(ablation_rob.values())).keys()) + "|",
        ]
        for mode, scenarios in ablation_rob.items():
            md.append("| " + mode + " | " + " | ".join(pct(v["success_rate"]) for v in scenarios.values()) + " |")

    md += [
        "",
        "## Figures",
    ]
    for label, path in summary["figures"].items():
        md.append(f"- {label}: `{path}`")

    out_md = ROOT / "VERIFIED_RESULTS.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
