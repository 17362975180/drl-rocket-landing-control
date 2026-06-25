"""Verify the final four-group study and paired trajectory comparisons."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


RESULTS = Path("results")
SCENARIOS = [
    "standard", "random_height", "random_velocity", "random_mass", "random_fuel",
    "gravity_bias", "thrust_bias", "sensor_noise", "action_delay_1", "action_delay_2", "combined",
]
GROUPS = [
    RESULTS / "01_standard_ppo" / "analysis",
    RESULTS / "01_standard_ppo" / "ablation",
    RESULTS / "02_energy_ppo" / "analysis",
    RESULTS / "02_energy_ppo" / "ablation",
    RESULTS / "02_energy_ppo" / "vs_standard",
    RESULTS / "03_control_comparison",
    RESULTS / "04_other_rl_comparison",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def load(path):
    require(path.is_file() and path.stat().st_size > 0, f"{path} exists")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_initial_conditions(rollout):
    values = rollout.get("initial_conditions", {})
    keys = ("initial_height", "initial_velocity", "dry_mass", "initial_fuel", "gravity_scale", "thrust_scale", "sensor_noise", "action_delay_steps")
    return np.array([float(values[key]) for key in keys])


def verify_group(group):
    comparison = load(group / "scenario_comparison.json")
    require(comparison["scenarios"] == SCENARIOS, f"{group} covers all 11 scenarios")
    for scenario in SCENARIOS:
        for label in comparison["labels"]:
            summary = comparison["results"][scenario][label]
            require(summary["n_episodes"] == 30, f"{group.name}/{scenario}/{label} has 30 episodes")
            require(summary["seed"] == 2026 + SCENARIOS.index(scenario) * 10_000, f"{group.name}/{scenario}/{label} uses paired seed")

    representatives = load(group / "representative_trajectories.json")
    for scenario in SCENARIOS:
        starts = [canonical_initial_conditions(representatives[label][scenario]) for label in comparison["labels"]]
        require(all(np.allclose(starts[0], item, rtol=0.0, atol=1e-10) for item in starts[1:]), f"{group.name}/{scenario} starts are identical")
        figure = group / "trajectory_comparisons" / f"{scenario}_trajectory_comparison.png"
        require(figure.is_file() and figure.stat().st_size > 0, f"{figure} exists")
    heatmap = group / "success_rate_heatmap.png"
    require(heatmap.is_file() and heatmap.stat().st_size > 0, f"{heatmap} exists")


def verify_failure_analysis():
    directory = RESULTS / "05_failure_analysis"
    report = directory / "FAILURE_ANALYSIS.md"
    require(report.is_file() and report.stat().st_size > 0, f"{report} exists")
    primary = load(directory / "report_primary_case.json")
    require(primary["terminal_reason"] == "crash", "report primary case is a real crash")
    require(abs(primary["final_v"] + 28.026412565503282) < 1e-9, "report primary case matches the cited terminal velocity")
    require(primary["evidence_classification"] == "reward_ablation_basic_failure", "report primary case has truthful provenance")
    for name in ("report_primary_case_trajectory.png", "report_primary_case_hv_phase.png", "FAILURE_CASE_PROVENANCE.json"):
        path = directory / name
        require(path.is_file() and path.stat().st_size > 0, f"{path} exists")
    for stem in ("standard_ppo_case", "energy_ppo_case"):
        case = load(directory / f"{stem}.json")
        require(case["scenario"] in SCENARIOS, f"{stem} uses a declared scenario")
        require(case["case_type"] in {"failure", "worst_success_no_failure_found"}, f"{stem} has a valid case type")
        if case["case_type"] == "failure":
            require(not case["success"], f"{stem} is a real failed rollout")
        figure = directory / f"{stem}.png"
        require(figure.is_file() and figure.stat().st_size > 0, f"{figure} exists")


def verify_report_alignment():
    directory = RESULTS / "06_report_alignment"
    evaluation = load(directory / "standard_eval_100" / "standard_eval_100.json")
    require(evaluation["seed"] == 2026 and evaluation["n_episodes"] == 100, "paired standard evaluation uses 100 episodes and seed 2026")
    for label in ("Standard PPO", "Energy PPO"):
        result = evaluation["results"][label]
        require(result["n_episodes"] == 100 and result["seed"] == 2026, f"{label} has paired 100-episode evidence")
    unified = load(directory / "unified_comparison" / "unified_success_rates.json")
    require(unified["scenarios"] == SCENARIOS and len(unified["labels"]) == 15, "unified heatmap covers 15 methods and all scenarios")
    required = [
        directory / "REPORT_ALIGNMENT.md",
        directory / "standard_eval_100" / "standard_metrics.png",
        directory / "unified_comparison" / "unified_success_heatmap.png",
        directory / "energy_interpretability" / "energy_accounting.png",
        directory / "energy_interpretability" / "reward_breakdown.png",
        directory / "failure_phase_plots" / "standard_ppo_case_height_velocity_phase.png",
        directory / "failure_phase_plots" / "energy_ppo_case_height_velocity_phase.png",
        directory / "training_evidence" / "training_curves.png",
        directory / "training_evidence" / "standard_ppo_training_curves.png",
        directory / "training_evidence" / "PROVENANCE.json",
    ]
    for path in required:
        require(path.is_file() and path.stat().st_size > 0, f"{path} exists")


def verify_final_report():
    report_pdf = Path("report") / "深度强化学习报告.pdf"
    report_docx = Path("report") / "深度强化学习报告.docx"
    require(report_pdf.is_file() and report_pdf.stat().st_size > 0, f"{report_pdf} exists")
    require(report_docx.is_file() and report_docx.stat().st_size > 0, f"{report_docx} exists")


def main():
    for group in GROUPS:
        verify_group(group)
    for demo in (
        RESULTS / "01_standard_ppo" / "demo" / "standard_landing_demo.gif",
        RESULTS / "02_energy_ppo" / "demo" / "standard_landing_demo.gif",
    ):
        require(demo.is_file() and demo.stat().st_size > 0, f"{demo} exists")
    verify_failure_analysis()
    verify_report_alignment()
    verify_final_report()
    print("All final study outputs verified.")


if __name__ == "__main__":
    main()
