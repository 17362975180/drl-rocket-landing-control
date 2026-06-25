"""Generate task-brief-aligned analysis from the current submission models/results."""

from __future__ import annotations

import csv
import json
import math
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from envs.rocket_env import RocketLandingEnv
from envs.rocket_env_energy import RocketLandingEnergyEnv
from experiment_utils import evaluate_model_rollouts, load_obs_rms, summarize_rollouts


ROOT = Path("results/06_report_alignment")
SCENARIOS = [
    "standard", "random_height", "random_velocity", "random_mass", "random_fuel",
    "gravity_bias", "thrust_bias", "sensor_noise", "action_delay_1", "action_delay_2", "combined",
]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def standard_eval_100():
    output = ROOT / "standard_eval_100"
    specs = [
        ("Standard PPO", Path("results/01_standard_ppo/models/full"), RocketLandingEnv),
        ("Energy PPO", Path("results/02_energy_ppo/models/full"), RocketLandingEnergyEnv),
    ]
    summaries = {}
    representatives = {}
    for label, model_dir, env_factory in specs:
        model = PPO.load(str(model_dir / "model.zip"), device="cpu")
        obs_rms = load_obs_rms(model_dir / "vec_normalize.pkl")
        rollouts = evaluate_model_rollouts(
            model=model,
            env_factory=env_factory,
            obs_rms=obs_rms,
            n_episodes=100,
            seed=2026,
            options={},
            save_trajectories=True,
        )
        summaries[label] = summarize_rollouts(rollouts, model_path=str(model_dir / "model.zip"), seed=2026)
        representatives[label] = rollouts[0]
    save_json(output / "standard_eval_100.json", {"seed": 2026, "n_episodes": 100, "results": summaries})
    save_json(output / "paired_representative_trajectories.json", representatives)

    metrics = ["success_rate", "mean_final_velocity_error", "mean_fuel_used", "mean_landing_time", "mean_abs_throttle_delta"]
    titles = ["Success rate", "Final speed error (m/s)", "Fuel used (kg)", "Landing time (s)", "Throttle variation"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    labels = list(summaries)
    colors = ["#4472C4", "#ED7D31"]
    for ax, metric, title in zip(axes, metrics, titles):
        values = [summaries[label][metric] for label in labels]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for i, value in enumerate(values):
            shown = 100 * value if metric == "success_rate" else value
            suffix = "%" if metric == "success_rate" else ""
            ax.text(i, value, f"{shown:.2f}{suffix}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Twin PPO paired-seed standard evaluation (100 episodes, seed=2026)")
    fig.tight_layout()
    fig.savefig(output / "standard_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summaries


def unified_heatmap():
    sources = [
        Path("results/01_standard_ppo/ablation/scenario_comparison.json"),
        Path("results/02_energy_ppo/ablation/scenario_comparison.json"),
        Path("results/03_control_comparison/scenario_comparison.json"),
        Path("results/04_other_rl_comparison/scenario_comparison.json"),
    ]
    by_label = {}
    for source in sources:
        data = load(source)
        for label in data["labels"]:
            if label in by_label:
                continue
            by_label[label] = [data["results"][scenario][label]["success_rate"] for scenario in SCENARIOS]
    preferred = [
        "Energy PPO", "Standard PPO", "Energy-NoEnergyObservation", "Energy-PreTimeOptimization",
        "Standard-full", "Standard-no_fuel", "Standard-no_smooth", "Standard-no_safety",
        "Standard-no_success", "Standard-basic", "SAC", "TD3", "PID", "MPC", "ET-MPC",
    ]
    labels = [label for label in preferred if label in by_label]
    matrix = np.array([by_label[label] for label in labels])
    output = ROOT / "unified_comparison"
    output.mkdir(parents=True, exist_ok=True)

    with (output / "unified_success_rates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *SCENARIOS, "mean"])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist(), float(np.mean(row))])
    save_json(output / "unified_success_rates.json", {"labels": labels, "scenarios": SCENARIOS, "success_rates": matrix.tolist()})

    fig, ax = plt.subplots(figsize=(15, 8))
    image = ax.imshow(matrix * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(SCENARIOS)), SCENARIOS, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(SCENARIOS)):
            value = matrix[i, j] * 100
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7, color="white" if value < 35 else "black")
    fig.colorbar(image, ax=ax, label="Success rate (%)")
    ax.set_title("Unified success-rate benchmark: 15 methods × 11 scenarios (30 episodes each)")
    fig.tight_layout()
    fig.savefig(output / "unified_success_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def energy_interpretability():
    rollout = load(Path("results/02_energy_ppo/demo/standard_demo_trajectory.json"))
    trajectory = rollout["trajectory"]
    time = np.array([step["time"] for step in trajectory])
    height = np.array([step["height"] for step in trajectory])
    velocity = np.array([step["velocity"] for step in trajectory])
    mass = np.array([step["mass"] for step in trajectory])
    fuel = np.array([step["fuel"] for step in trajectory])
    potential = mass * 9.81 * np.maximum(height, 0)
    kinetic = 0.5 * mass * velocity ** 2
    dry_mass = np.maximum(mass - fuel, 1e-6)
    delta_v = 200.0 * np.log(np.maximum(mass / dry_mass, 1.0))
    fuel_capacity = 0.5 * mass * delta_v ** 2
    output = ROOT / "energy_interpretability"
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(time, potential, label="Potential energy")
    ax.plot(time, kinetic, label="Kinetic energy")
    ax.plot(time, fuel_capacity, label="Fuel braking-capacity proxy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy / proxy (J)")
    ax.set_title("Energy PPO landing energy accounting")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "energy_accounting.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    keys = sorted({key for step in trajectory for key in step.get("reward_breakdown", {})})
    fig, ax = plt.subplots(figsize=(12, 6))
    for key in keys:
        values = [step.get("reward_breakdown", {}).get(key, 0.0) for step in trajectory]
        ax.plot(time, values, label=key, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward contribution")
    ax.set_title("Energy PPO reward decomposition")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "reward_breakdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def failure_phase_plots():
    output = ROOT / "failure_phase_plots"
    output.mkdir(parents=True, exist_ok=True)
    for stem, title in (("standard_ppo_case", "Standard PPO failure"), ("energy_ppo_case", "Energy PPO failure")):
        case = load(Path("results/05_failure_analysis") / f"{stem}.json")
        h = [step["height"] for step in case["trajectory"]]
        v = [step["velocity"] for step in case["trajectory"]]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.plot(h, v, color="#C00000", linewidth=2)
        ax.scatter([h[0]], [v[0]], color="#4472C4", label="start", zorder=3)
        ax.scatter([h[-1]], [v[-1]], color="#C00000", label="terminal", zorder=3)
        ax.axhspan(-2, 2, color="#70AD47", alpha=0.12, label="soft-landing speed band")
        ax.set_xlabel("Height (m)")
        ax.set_ylabel("Vertical velocity (m/s)")
        ax.set_title(f"{title}: {case['scenario']} (seed={case['seed']})")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{stem}_height_velocity_phase.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_evidence():
    output = ROOT / "training_evidence"
    raw = output / "raw"
    standard_data = load(raw / "standard_ppo_training_curves_data.json")
    accumulator = EventAccumulator(str(raw / "energy_ppo_tensorboard.tfevents"))
    accumulator.Reload()

    fig_standard, axes_standard = plt.subplots(2, 2, figsize=(13, 8))
    standard_panels = (
        (axes_standard[0, 0], "train/loss", "Training loss"),
        (axes_standard[0, 1], "train/value_loss", "Value loss"),
        (axes_standard[1, 0], "train/policy_gradient_loss", "Policy-gradient loss"),
        (axes_standard[1, 1], "train/entropy_loss", "Entropy loss"),
    )
    for ax, tag, title in standard_panels:
        series = standard_data[tag]
        steps = np.asarray(series["steps"], dtype=float)
        values = np.asarray(series["values"], dtype=float)
        ax.plot(steps, values, color="#4C78A8", linewidth=0.8, alpha=0.28, label="Raw")
        window = min(15, len(values))
        if window > 1:
            kernel = np.ones(window) / window
            smooth = np.convolve(values, kernel, mode="valid")
            ax.plot(steps[window - 1 :], smooth, color="#1F4E79", linewidth=2.0, label=f"Moving average ({window})")
        ax.set_title(title)
        ax.set_xlabel("Training steps")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig_standard.suptitle("Standard PPO training diagnostics")
    fig_standard.tight_layout()
    fig_standard.savefig(output / "standard_ppo_training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig_standard)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, tag, title in (
        (axes[0, 0], "train/loss", "Standard PPO training loss"),
        (axes[1, 0], "train/value_loss", "Standard PPO value loss"),
    ):
        series = standard_data[tag]
        ax.plot(series["steps"], series["values"], linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Training steps")
        ax.grid(alpha=0.25)
    for ax, tag, title in (
        (axes[0, 1], "rollout/ep_rew_mean", "Energy PPO mean episode reward"),
        (axes[1, 1], "rollout/ep_len_mean", "Energy PPO mean episode length"),
    ):
        events = accumulator.Scalars(tag)
        ax.plot([event.step for event in events], [event.value for event in events], linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Training steps")
        ax.grid(alpha=0.25)
    fig.suptitle("Submitted-checkpoint training evidence (different logged metrics; not a direct score comparison)")
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "note": "Raw training evidence copied into submission_version after checkpoint identity was verified by SHA-256.",
        "standard_checkpoint": {
            "path": "results/01_standard_ppo/models/full/model.zip",
            "sha256": sha256(Path("results/01_standard_ppo/models/full/model.zip")),
            "matched_original": "results/sweeps/v2_reward/models/final_model.zip",
        },
        "energy_checkpoint": {
            "path": "results/02_energy_ppo/models/full/model.zip",
            "sha256": sha256(Path("results/02_energy_ppo/models/full/model.zip")),
            "matched_original": "results/reproducible/energy_ppo_from_scratch_time/models/pure_energy_ppo_model.zip",
        },
        "raw_files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(raw.iterdir()) if path.is_file()
        },
    }
    save_json(output / "PROVENANCE.json", provenance)


def write_report(summaries):
    lines = [
        "# 报告补充实验与证据说明", "",
        "本目录汇总最终报告使用的标准评价、统一方法对比、能量可解释性、失败轨迹和训练过程证据。所有数值均由 `submission_version` 中保存的模型和实验结果生成。", "",
        "## 1. 配对 100 回合标准评价", "",
        "| 模型 | 成功率 | 最终速度误差 | 燃料消耗 | 着陆时间 | 动作变化量 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for label, result in summaries.items():
        lines.append(
            f"| {label} | {100 * result['success_rate']:.1f}% | {result['mean_final_velocity_error']:.3f} m/s | "
            f"{result['mean_fuel_used']:.3f} kg | {result['mean_landing_time']:.3f} s | {result['mean_abs_throttle_delta']:.4f} |"
        )
    lines += [
        "", "两个模型均使用随机种子 2026—2125，因此配对评价中的第 k 个回合具有相同的环境重置种子。", "",
        "## 2. 目录内容", "",
        "- `standard_eval_100/`：Standard PPO 与 Energy PPO 的配对 100 回合指标、原始 JSON 和对比图；",
        "- `unified_comparison/`：全部最终方法在 11 个统一场景中的成功率热图；",
        "- `energy_interpretability/`：Energy PPO 的能量变化与奖励分量图；",
        "- `failure_phase_plots/`：真实失败案例的高度—速度相图；",
        "- `training_evidence/`：训练曲线、原始日志及模型文件 SHA-256 来源记录。", "",
        "Standard PPO 和 Energy PPO 的训练日志记录指标不同，因此训练图主要用于说明各自的收敛过程，不宜直接比较两种奖励数值的绝对大小。", "",
    ]
    (ROOT / "REPORT_ALIGNMENT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    summaries = standard_eval_100()
    unified_heatmap()
    energy_interpretability()
    failure_phase_plots()
    training_evidence()
    write_report(summaries)
    print(f"Saved report-aligned analysis to {ROOT}")


if __name__ == "__main__":
    main()
