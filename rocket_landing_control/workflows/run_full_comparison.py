"""Generate grouped comparison charts + unified heatmap.

Groups:
  1. Pure Energy PPO vs PPO vs SAC vs TD3
  2. PPO (full) vs PPO ablation variants
  3. Pure Energy PPO variants (v1/v2/v3/from_scratch_time)
  4. Pure Energy PPO vs PPO vs PID vs MPC(tuned) vs ET-MPC(tuned)
  + Unified heatmap: all strategies in one view
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rocket_landing_control.core.experiment_utils import ROBUSTNESS_SCENARIOS
from rocket_landing_control.visualization.generate_experiment_trajectory_comparisons import normalize_trajectory, plot_comparison

OUTPUT_DIR = Path("results/reproducible/final_comparison")
SCENARIOS = list(ROBUSTNESS_SCENARIOS.keys())


# ── helpers ───────────────────────────────────────────────────────────

def load_json(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_strategy_data() -> dict[str, dict[str, dict]]:
    """Load all strategy data: {label: {scenario: result_dict}}."""
    all_data: dict[str, dict[str, dict]] = {}

    # ── Baseline PPO + Pure Energy PPO v4 ──
    cmp = load_json("results/reproducible/energy_ppo_from_scratch_time/scenarios/standard_comparison/results_by_policy.json")
    if cmp:
        for label in ["Baseline PPO", "Pure Energy PPO"]:
            if label in cmp:
                all_data[label] = {}
                for sc in SCENARIOS:
                    entry = cmp[label].get(sc, {})
                    if isinstance(entry, dict):
                        all_data[label][sc] = entry

    # ── PPO ablation ──
    abl_rob = load_json("results/reproducible/ablation_scenarios/ablation_robustness.json")
    abl_gen = load_json("results/reproducible/ablation_scenarios/ablation_generalization.json")
    for mode in ["full", "no_fuel", "no_smooth", "no_safety", "no_success", "basic"]:
        label = f"PPO_{mode}"
        all_data[label] = {}
        for sc in SCENARIOS:
            entry = None
            if abl_gen and mode in abl_gen and sc in abl_gen[mode]:
                entry = abl_gen[mode][sc]
            if abl_rob and mode in abl_rob and sc in abl_rob[mode]:
                entry = abl_rob[mode][sc]
            if isinstance(entry, dict):
                all_data[label][sc] = entry

    # ── New experiments (Energy PPO v1/v2/v3, SAC, TD3, PID) ──
    new_dirs = {
        "EnergyPPO_v1": OUTPUT_DIR / "energy_v1",
        "EnergyPPO_v2": OUTPUT_DIR / "energy_v2",
        "EnergyPPO_v3": OUTPUT_DIR / "energy_v3",
        "PID": OUTPUT_DIR / "controllers",
        "SAC": OUTPUT_DIR / "rl_algo",
        "TD3": OUTPUT_DIR / "rl_algo",
    }
    loaded_results: dict[Path, dict] = {}
    for label, dirpath in new_dirs.items():
        if dirpath not in loaded_results:
            loaded_results[dirpath] = load_json(dirpath / "results_by_policy.json") or {}
    for label, dirpath in new_dirs.items():
        dir_data = loaded_results[dirpath]
        if label in dir_data:
            all_data[label] = {}
            for sc in SCENARIOS:
                entry = dir_data[label].get(sc, {})
                if isinstance(entry, dict):
                    all_data[label][sc] = entry

    # ── Tuned MPC/ET-MPC (from mpc_tuning results) ──
    mpc_tuned = load_json(OUTPUT_DIR / "mpc_tuning" / "mpc_tuning_results.json")
    if mpc_tuned:
        for label in ["MPC_tuned", "ET-MPC_tuned"]:
            if label in mpc_tuned:
                all_data[label] = {}
                for sc in SCENARIOS:
                    entry = mpc_tuned[label].get(sc, {})
                    if isinstance(entry, dict):
                        all_data[label][sc] = entry

    return all_data


def load_trajectories() -> dict[str, dict[str, dict]]:
    """Load all trajectory data: {label: {scenario: trajectory_dict}}."""
    all_trajs: dict[str, dict[str, dict]] = {}

    # ── Baseline PPO + Pure Energy PPO v4 ──
    cmp_trajs = load_json("results/reproducible/energy_ppo_from_scratch_time/scenarios/standard_comparison/representative_trajectories.json")
    if cmp_trajs:
        for label in ["Baseline PPO", "Pure Energy PPO"]:
            if label in cmp_trajs:
                all_trajs[label] = {}
                for sc in SCENARIOS:
                    traj = cmp_trajs[label].get(sc)
                    if traj is not None:
                        all_trajs[label][sc] = traj

    # ── PPO ablation ──
    abl_rob_t = load_json("results/reproducible/ablation_scenarios/ablation_robustness_trajectories.json")
    abl_gen_t = load_json("results/reproducible/ablation_scenarios/ablation_generalization_trajectories.json")
    for mode in ["full", "no_fuel", "no_smooth", "no_safety", "no_success", "basic"]:
        label = f"PPO_{mode}"
        all_trajs[label] = {}
        for src in [abl_rob_t, abl_gen_t]:
            if src and mode in src:
                for sc in SCENARIOS:
                    if sc in src[mode] and sc not in all_trajs[label]:
                        all_trajs[label][sc] = src[mode][sc]

    # ── New experiments ──
    new_dirs = {
        "EnergyPPO_v1": OUTPUT_DIR / "energy_v1",
        "EnergyPPO_v2": OUTPUT_DIR / "energy_v2",
        "EnergyPPO_v3": OUTPUT_DIR / "energy_v3",
        "PID": OUTPUT_DIR / "controllers",
        "SAC": OUTPUT_DIR / "rl_algo",
        "TD3": OUTPUT_DIR / "rl_algo",
    }
    loaded_trajs: dict[Path, dict] = {}
    for label, dirpath in new_dirs.items():
        if dirpath not in loaded_trajs:
            loaded_trajs[dirpath] = load_json(dirpath / "representative_trajectories.json") or {}
    for label, dirpath in new_dirs.items():
        dir_trajs = loaded_trajs[dirpath]
        if label in dir_trajs:
            all_trajs[label] = {}
            for sc in SCENARIOS:
                traj = dir_trajs[label].get(sc)
                if traj is not None:
                    all_trajs[label][sc] = traj

    return all_trajs


# ── heatmap ───────────────────────────────────────────────────────────

def plot_heatmap(
    matrix: dict[str, dict[str, float]],
    scenarios: list[str],
    title: str,
    output_path: Path,
    figsize: tuple[float, float] | None = None,
):
    strategies = list(matrix.keys())
    n_strat = len(strategies)
    n_scen = len(scenarios)

    data = np.full((n_strat, n_scen), np.nan)
    for i, strat in enumerate(strategies):
        for j, scen in enumerate(scenarios):
            val = matrix[strat].get(scen)
            if val is not None:
                data[i, j] = val

    if figsize is None:
        figsize = (max(10, n_scen * 1.2), max(3, n_strat * 0.55))
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(n_scen))
    ax.set_xticklabels(scenarios, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_strat))
    ax.set_yticklabels(strategies, fontsize=10)

    for i in range(n_strat):
        for j in range(n_scen):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.3 or val > 0.8 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Success Rate", shrink=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  heatmap → {output_path}")


# ── CSV ───────────────────────────────────────────────────────────────

def write_csv(matrix: dict[str, dict[str, float]], scenarios: list[str], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strategies = list(matrix.keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy"] + scenarios)
        for strat in strategies:
            row = [strat]
            for sc in scenarios:
                val = matrix[strat].get(sc)
                row.append(f"{val:.1%}" if val is not None else "")
            writer.writerow(row)
    print(f"  csv → {output_path}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("Loading all data...")
    all_data = load_strategy_data()
    all_trajs = load_trajectories()
    print(f"  loaded {len(all_data)} strategies: {list(all_data.keys())}")

    scenarios = SCENARIOS

    # ── Group 1: RL Algorithm Comparison ──
    g1_labels = ["Pure Energy PPO", "Baseline PPO", "SAC", "TD3"]
    g1_name = "RL Algorithm Comparison"
    g1_dir = OUTPUT_DIR / "group1_rl_algorithms"
    print(f"\n{'='*60}\nGroup 1: {g1_name}")
    g1_matrix = {l: {sc: all_data[l][sc].get("success_rate", 0.0) for sc in scenarios if sc in all_data.get(l, {})} for l in g1_labels if l in all_data}
    plot_heatmap(g1_matrix, scenarios, f"Group 1: {g1_name}", g1_dir / "heatmap.png")
    write_csv(g1_matrix, scenarios, g1_dir / "results.csv")

    # ── Group 2: PPO Ablation ──
    g2_labels = ["PPO_full", "PPO_no_fuel", "PPO_no_smooth", "PPO_no_safety", "PPO_no_success", "PPO_basic"]
    g2_name = "PPO Reward Ablation"
    g2_dir = OUTPUT_DIR / "group2_ppo_ablation"
    print(f"\n{'='*60}\nGroup 2: {g2_name}")
    g2_matrix = {l: {sc: all_data[l][sc].get("success_rate", 0.0) for sc in scenarios if sc in all_data.get(l, {})} for l in g2_labels if l in all_data}
    plot_heatmap(g2_matrix, scenarios, f"Group 2: {g2_name}", g2_dir / "heatmap.png")
    write_csv(g2_matrix, scenarios, g2_dir / "results.csv")

    # ── Group 3: Energy PPO Variants ──
    g3_labels = ["Pure Energy PPO", "EnergyPPO_v1", "EnergyPPO_v2", "EnergyPPO_v3"]
    g3_name = "Energy PPO Variants"
    g3_dir = OUTPUT_DIR / "group3_energy_ppo_variants"
    print(f"\n{'='*60}\nGroup 3: {g3_name}")
    g3_matrix = {l: {sc: all_data[l][sc].get("success_rate", 0.0) for sc in scenarios if sc in all_data.get(l, {})} for l in g3_labels if l in all_data}
    plot_heatmap(g3_matrix, scenarios, f"Group 3: {g3_name}", g3_dir / "heatmap.png")
    write_csv(g3_matrix, scenarios, g3_dir / "results.csv")

    # ── Group 4: Controller Comparison (with tuned MPC/ET-MPC) ──
    g4_labels = ["Pure Energy PPO", "Baseline PPO", "PID", "MPC_tuned", "ET-MPC_tuned"]
    g4_name = "Controller Comparison (MPC/ET-MPC tuned)"
    g4_dir = OUTPUT_DIR / "group4_controllers"
    print(f"\n{'='*60}\nGroup 4: {g4_name}")
    g4_matrix = {l: {sc: all_data[l][sc].get("success_rate", 0.0) for sc in scenarios if sc in all_data.get(l, {})} for l in g4_labels if l in all_data}
    plot_heatmap(g4_matrix, scenarios, f"Group 4: {g4_name}", g4_dir / "heatmap.png")
    write_csv(g4_matrix, scenarios, g4_dir / "results.csv")

    # ── Unified Heatmap: ALL strategies ──
    unified_dir = OUTPUT_DIR / "unified_heatmap"
    print(f"\n{'='*60}\nUnified Heatmap")
    # Order: Energy PPO variants, PPO + ablation, RL algos, Controllers
    unified_labels = [
        "Pure Energy PPO", "EnergyPPO_v1", "EnergyPPO_v2", "EnergyPPO_v3",
        "Baseline PPO", "PPO_full", "PPO_no_fuel", "PPO_no_smooth", "PPO_no_safety", "PPO_no_success", "PPO_basic",
        "SAC", "TD3",
        "PID", "MPC_tuned", "ET-MPC_tuned",
    ]
    unified_matrix = {}
    for l in unified_labels:
        if l in all_data:
            unified_matrix[l] = {sc: all_data[l][sc].get("success_rate", 0.0) for sc in scenarios if sc in all_data[l]}
    plot_heatmap(unified_matrix, scenarios, "Unified Robustness: All Strategies × All Scenarios", unified_dir / "robustness_heatmap.png", figsize=(16, 10))
    write_csv(unified_matrix, scenarios, unified_dir / "full_comparison.csv")

    # Save JSON
    with open(unified_dir / "full_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"strategies": list(unified_matrix.keys()), "scenarios": scenarios, "matrix": unified_matrix}, f, indent=2, ensure_ascii=False)
    print(f"  json → {unified_dir / 'full_comparison.json'}")

    print(f"\nDone! All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
