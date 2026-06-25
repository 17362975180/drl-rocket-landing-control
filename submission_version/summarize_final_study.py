"""Create a compact Markdown summary of the final four-group study."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def average_success(path):
    data = load(path / "scenario_comparison.json")
    return {
        label: sum(data["results"][scenario][label]["success_rate"] for scenario in data["scenarios"]) / len(data["scenarios"])
        for label in data["labels"]
    }


def main():
    sections = [
        ("Standard PPO 奖励消融", ROOT / "01_standard_ppo" / "ablation"),
        ("Energy PPO 结构消融", ROOT / "02_energy_ppo" / "ablation"),
        ("Standard PPO 与 Energy PPO 对比", ROOT / "02_energy_ppo" / "vs_standard"),
        ("传统控制算法对比", ROOT / "03_control_comparison"),
        ("其他强化学习算法对比", ROOT / "04_other_rl_comparison"),
    ]
    lines = [
        "# 正式实验结果汇总",
        "",
        "所有统一场景比较均采用每场景 30 回合、配对随机种子 2026；轨迹图固定使用 episode 0，保证不同方法具有相同初始条件。",
        "",
    ]
    for title, path in sections:
        lines += [f"## {title}", "", "| 方法 | 11 场景平均成功率 |", "|---|---:|"]
        for label, value in average_success(path).items():
            lines.append(f"| {label} | {100 * value:.1f}% |")
        heatmap = (path / "success_rate_heatmap.png").as_posix()
        trajectories = (path / "trajectory_comparisons").as_posix()
        lines += ["", f"- 成功率热图：`{heatmap}`", f"- 各场景轨迹：`{trajectories}/`", ""]
    lines += [
        "## 失败案例分析",
        "",
        "失败案例包括基础奖励导致的高空燃料耗尽，以及两个主要模型在扩展压力搜索中出现的临界失败。",
        "",
        f"- 分析说明：`{(ROOT / '05_failure_analysis' / 'FAILURE_ANALYSIS.md').as_posix()}`",
        f"- 原始轨迹与图像：`{(ROOT / '05_failure_analysis').as_posix()}/`",
        "",
    ]
    output = ROOT / "FINAL_STUDY_SUMMARY.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
