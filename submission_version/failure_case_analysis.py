"""Find and explain real failure or worst-case trajectories for the twin PPO models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs.rocket_env import RocketLandingEnv
from envs.rocket_env_energy import RocketLandingEnergyEnv
from experiment_utils import ROBUSTNESS_SCENARIOS, load_obs_rms, make_episode_options, rollout_model
from generate_experiment_trajectory_comparisons import normalize_trajectory, plot_comparison


ROOT = Path("results/05_failure_analysis")
REFERENCE_CASE = Path("reference_cases/baseline_fuel_exhaustion.json")
SCENARIO_ORDER = [
    "action_delay_2",
    "combined",
    "random_fuel",
    "sensor_noise",
    "random_height",
    "random_mass",
    "gravity_bias",
    "thrust_bias",
    "action_delay_1",
    "random_velocity",
    "standard",
]


def find_case(label, model_path, stats_path, env_factory, attempts, base_seed):
    model = PPO.load(str(model_path), device="cpu")
    obs_rms = load_obs_rms(stats_path)
    worst = None
    worst_score = float("-inf")

    for scenario_index, scenario in enumerate(SCENARIO_ORDER):
        scenario_seed = base_seed + scenario_index * 100_000
        options_template = ROBUSTNESS_SCENARIOS[scenario]
        for episode in range(attempts):
            seed = scenario_seed + episode
            options = make_episode_options(options_template, episode, scenario_seed)
            env = env_factory()
            rollout = rollout_model(
                model,
                env,
                obs_rms=obs_rms,
                seed=seed,
                reset_options=options,
                save_trajectory=True,
            )
            env.close()
            rollout["model_label"] = label
            rollout["scenario"] = scenario
            rollout["seed"] = seed
            rollout["search_episode"] = episode
            if not rollout["success"]:
                rollout["case_type"] = "failure"
                return rollout

            score = abs(rollout["final_v"]) + 0.05 * rollout["fuel_used"] + 0.01 * rollout["max_acceleration"]
            if score > worst_score:
                worst_score = score
                worst = rollout

    if worst is not None:
        worst["case_type"] = "worst_success_no_failure_found"
    return worst


def diagnosis(case):
    if case["case_type"] != "failure":
        return "在设定搜索范围内未发现失败；这里保存的是按终端速度、燃料和加速度评分得到的最差成功轨迹"
    reasons = []
    scenario = case["scenario"]
    if scenario.startswith("action_delay"):
        reasons.append("动作延迟降低了制动时机的准确性")
    if scenario == "random_fuel":
        reasons.append("初始燃料减少后，剩余制动能力不足")
    if scenario == "random_height":
        reasons.append("初始高度变化使原有制动切换点发生偏移")
    if scenario == "sensor_noise":
        reasons.append("观测噪声干扰了对制动状态的判断")
    if scenario == "combined":
        reasons.append("多种扰动叠加了模型偏差和控制时序误差")
    if abs(case["final_v"]) > 2.0:
        reasons.append("触地速度超过 2 m/s 的软着陆阈值")
    if case["fuel_used"] >= case["initial_conditions"].get("initial_fuel", 5.0) - 1e-3:
        reasons.append("燃料已基本耗尽")
    return "；".join(reasons) or f"终止原因为 {case['terminal_reason']}"


def save_case(case, slug):
    ROOT.mkdir(parents=True, exist_ok=True)
    json_path = ROOT / f"{slug}.json"
    json_path.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_comparison(
        {case["model_label"]: normalize_trajectory(case)},
        f"{case['model_label']} {case['case_type']} — {case['scenario']} (seed={case['seed']})",
        ROOT / f"{slug}.png",
    )
    return json_path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_report_primary_case():
    """Render the verified basic-reward fuel-exhaustion case."""
    case = json.loads(REFERENCE_CASE.read_text(encoding="utf-8"))
    case.update(
        {
            "model_label": "Basic-reward PPO",
            "case_type": "report_primary_failure",
            "evidence_classification": "reward_ablation_basic_failure",
        }
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "report_primary_case.json").write_text(
        json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_comparison(
        {"Basic-reward PPO": normalize_trajectory(case)},
        "Report primary failure: high-altitude fuel exhaustion",
        ROOT / "report_primary_case_trajectory.png",
    )

    trajectory = case["trajectory"]
    height = [step["height"] for step in trajectory]
    velocity = [step["velocity"] for step in trajectory]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(height, velocity, color="#C00000", linewidth=2)
    ax.scatter([height[0]], [velocity[0]], color="#4472C4", label="start", zorder=3)
    ax.scatter([height[-1]], [velocity[-1]], color="#C00000", label="crash", zorder=3)
    ax.axhspan(-2, 2, color="#70AD47", alpha=0.12, label="soft-landing speed band")
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Vertical velocity (m/s)")
    ax.set_title("Height-velocity phase: fuel-exhaustion failure")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "report_primary_case_hv_phase.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    exhaustion = next((step for step in trajectory if step["fuel"] <= 1e-9), None)
    provenance = {
        "reference_file": str(REFERENCE_CASE),
        "sha256": file_sha256(REFERENCE_CASE),
        "exact_original_match": "results/reproducible/failure_case_trajectory.json",
        "exact_dataset_match": "results/reproducible/ablation/reward_ablation_trajectories.json -> basic.failure",
        "supported_label": "basic reward ablation failure",
        "unsupported_without_checkpoint": "early-training Baseline PPO checkpoint failure",
    }
    (ROOT / "FAILURE_CASE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return case, exhaustion


def main():
    parser = argparse.ArgumentParser(description="Search real twin-PPO failure cases.")
    parser.add_argument("--attempts", type=int, default=300, help="Maximum seeds per scenario and model.")
    parser.add_argument("--seed", type=int, default=50_000)
    args = parser.parse_args()

    primary_case, exhaustion = save_report_primary_case()
    cases = [
        find_case(
            "Standard PPO",
            "results/01_standard_ppo/models/full/model.zip",
            "results/01_standard_ppo/models/full/vec_normalize.pkl",
            RocketLandingEnv,
            args.attempts,
            args.seed,
        ),
        find_case(
            "Energy PPO",
            "results/02_energy_ppo/models/full/model.zip",
            "results/02_energy_ppo/models/full/vec_normalize.pkl",
            RocketLandingEnergyEnv,
            args.attempts,
            args.seed,
        ),
    ]

    for case, slug in zip(cases, ("standard_ppo_case", "energy_ppo_case")):
        save_case(case, slug)

    lines = [
        "# 失败案例分析",
        "",
        "## 1. 基础奖励消融：高空燃料耗尽",
        "",
        "该轨迹来自基础奖励消融模型，是报告中初步探索阶段使用的主要失败案例。",
        "",
        f"- 案例分类：`{primary_case['evidence_classification']}`",
        f"- 初始高度：`{primary_case['initial_conditions']['initial_height']:.3f} m`",
        f"- 初始速度：`{primary_case['initial_conditions']['initial_velocity']:.3f} m/s`",
        f"- 初始燃料：`{primary_case['initial_conditions']['initial_fuel']:.3f} kg`",
        f"- 终止原因：`{primary_case['terminal_reason']}`",
        f"- 最终速度：`{primary_case['final_v']:.3f} m/s`",
        f"- 燃料消耗：`{primary_case['fuel_used']:.3f} kg`",
        f"- 终止时间：`{primary_case['landing_time']:.3f} s`",
        f"- 燃料耗尽状态：`t = {exhaustion['time']:.2f} s`，`h = {exhaustion['height']:.2f} m`" if exhaustion else "- 未找到燃料耗尽时刻",
        "",
        "策略在火箭仍处于高空时便耗尽全部燃料。燃料耗尽后，油门指令无法继续产生实际推力，火箭进入无动力下落状态，最终高速撞地。该案例说明，燃料不能只作为较小的即时惩罚项，还应表示为完成后续制动所需的有限物理资源。",
        "",
        "原始轨迹与奖励消融数据中的 `basic.failure` 条目一致，因此该案例应表述为“基础奖励消融失败”，不应写成 Standard PPO 正式模型或某个缺少记录的中间训练模型失败。",
        "",
        "## 2. 正式模型扩展压力失败",
        "",
        f"在统一 11 场景范围内，对两个主要 PPO 模型按每场景最多 {args.attempts} 个种子进行扩展搜索。以下案例用于说明正式 30 回合评价之外仍可能存在的控制边界。",
        "",
    ]
    for index, case in enumerate(cases, start=1):
        lines += [
            f"### 2.{index} {case['model_label']}",
            "",
            f"- 案例类型：`{case['case_type']}`",
            f"- 场景：`{case['scenario']}`",
            f"- 随机种子：`{case['seed']}`",
            f"- 终止原因：`{case['terminal_reason']}`",
            f"- 最终速度：`{case['final_v']:.3f} m/s`",
            f"- 燃料消耗：`{case['fuel_used']:.3f} kg`",
            f"- 终止时间：`{case['landing_time']:.3f} s`",
            f"- 原因分析：{diagnosis(case)}",
            "",
        ]
    report = ROOT / "FAILURE_ANALYSIS.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {report}")


if __name__ == "__main__":
    main()
