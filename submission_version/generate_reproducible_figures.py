"""Generate report-ready figures from reproducible JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import (
    ABLATION,
    CONTROLLERS,
    MAIN_EVAL,
    OVERVIEW_FIGURES,
    RL_ALGORITHMS,
    ROBUSTNESS,
    SAFETY,
)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_grouped_metrics(data, title, output_path, metrics):
    labels = list(data.keys())
    x = np.arange(len(labels))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
    for i, (metric, ylabel) in enumerate(metrics):
        values = [data[label].get(metric, 0.0) for label in labels]
        offset = (i - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_training_curves(data, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPO Training Curves")
    for ax, (name, series) in zip(axes.flat, data.items()):
        ax.plot(series.get("steps", []), series.get("values", []), linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel("Training steps")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate reproducible report figures.")
    parser.add_argument("--output-dir", type=str, default=str(OVERVIEW_FIGURES))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    files = {
        "training": MAIN_EVAL / "training_curves_data.json",
        "safety": SAFETY / "safety_comparison.json",
        "robustness": ROBUSTNESS / "robustness_full.json",
        "controller": CONTROLLERS / "controller_comparison.json",
        "ablation": ABLATION / "reward_ablation.json",
        "rl": RL_ALGORITHMS / "rl_comparison.json",
    }

    if files["training"].exists():
        plot_training_curves(load_json(files["training"]), output_dir / "training_curves.png")

    if files["safety"].exists():
        plot_grouped_metrics(
            load_json(files["safety"]),
            "Safety Shield Comparison",
            output_dir / "safety_comparison.png",
            [
                ("success_rate", "Success rate"),
                ("crash_rate", "Crash rate"),
                ("mean_fuel_used", "Fuel used (kg)"),
                ("mean_abs_throttle_delta", "Throttle delta"),
            ],
        )

    if files["robustness"].exists():
        plot_grouped_metrics(
            load_json(files["robustness"]),
            "Robustness Test Results",
            output_dir / "robustness_results.png",
            [
                ("success_rate", "Success rate"),
                ("crash_rate", "Crash rate"),
                ("mean_final_velocity_error", "|final velocity| (m/s)"),
                ("mean_fuel_used", "Fuel used (kg)"),
            ],
        )

    if files["controller"].exists():
        plot_grouped_metrics(
            load_json(files["controller"]),
            "Controller Comparison",
            output_dir / "controller_comparison.png",
            [
                ("success_rate", "Success rate"),
                ("mean_fuel_used", "Fuel used (kg)"),
                ("mean_abs_throttle_delta", "Throttle delta"),
                ("mean_final_velocity_error", "|final velocity| (m/s)"),
            ],
        )

    if files["ablation"].exists():
        plot_grouped_metrics(
            load_json(files["ablation"]),
            "Reward Ablation",
            output_dir / "reward_ablation.png",
            [
                ("success_rate", "Success rate"),
                ("mean_fuel_used", "Fuel used (kg)"),
                ("mean_abs_throttle_delta", "Throttle delta"),
            ],
        )

    if files["rl"].exists():
        plot_grouped_metrics(
            load_json(files["rl"]),
            "RL Algorithm Comparison",
            output_dir / "rl_comparison.png",
            [
                ("success_rate", "Success rate"),
                ("mean_fuel_used", "Fuel used (kg)"),
                ("mean_abs_throttle_delta", "Throttle delta"),
            ],
        )


if __name__ == "__main__":
    main()
