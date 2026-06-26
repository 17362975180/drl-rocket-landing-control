"""Tune MPC/ET-MPC controllers and compare with original + RL strategies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rocket_landing_control.studies.controller_comparison_full import (
    EventTriggeredMPCController,
    MPCController,
    PIDController,
    rollout_controller,
)
from rocket_landing_control.envs.rocket_env import RocketLandingEnv
from rocket_landing_control.core.experiment_utils import ROBUSTNESS_SCENARIOS, load_obs_rms, normalize_obs
from rocket_landing_control.visualization.generate_experiment_trajectory_comparisons import normalize_trajectory, plot_comparison
from stable_baselines3 import PPO

OUTPUT_DIR = Path("results/reproducible/final_comparison/mpc_tuning")
SEED = 1000


# ── tuned MPC variants ────────────────────────────────────────────────

class TunedMPCController(MPCController):
    """MPC with better cost tuning (same speed as original)."""

    def __init__(self, horizon=60, grid_size=21):
        super().__init__(horizon=horizon, grid_size=grid_size)

    def target_velocity(self, height):
        if height <= 0:
            return 0.0
        # More aggressive braking profile
        braking_profile = -max(2.0, min(18.0, np.sqrt(max(0.0, 2.0 * height))))
        if height > 30:
            return min(-14.0, braking_profile)
        if height > 15:
            return min(-10.0, braking_profile)
        if height > 5:
            return min(-5.0, braking_profile)
        return max(-1.5, braking_profile)

    def _rollout_cost(self, env, throttle):
        h = float(env.height)
        v = float(env.velocity)
        fuel = float(env.fuel_remaining)
        thrust = float(env.current_thrust)
        cost = 0.01 * throttle * throttle + 0.03 * abs(throttle - self.last_throttle)
        for step in range(self.horizon):
            alpha = min(1.0, env.dt / max(env.thrust_delay, 1e-6))
            thrust += alpha * (throttle * env.T_max - thrust)
            fuel = max(0.0, fuel - (thrust / env.exhaust_v) * env.dt)
            mass = env.dry_mass + fuel
            drag = -env.drag_coeff * v * abs(v)
            acc = (thrust + drag - mass * env.g) / mass
            v += acc * env.dt
            h += v * env.dt
            target_v = self.target_velocity(max(h, 0.0))
            # Stronger velocity tracking + height urgency
            height_urgency = max(0.0, 1.0 - h / 50.0)  # more urgent near ground
            cost += (1.0 + 0.5 * height_urgency) * abs(v - target_v)
            cost += 0.03 * max(h, 0.0)
            if h <= 0:
                # Much stronger crash penalty
                cost += 2000.0 * max(abs(v) - 2.0, 0.0)
                if abs(v) <= 2.0:
                    cost -= 100.0  # bonus for soft landing
                break
        cost += 8.0 * abs(max(h, 0.0)) + 15.0 * abs(v if h <= 5 else v - self.target_velocity(h))
        return cost


class TunedETMPCController(TunedMPCController):
    """Event-triggered version of TunedMPC."""

    def __init__(self, horizon=60, grid_size=21, threshold=0.15):
        super().__init__(horizon=horizon, grid_size=grid_size)
        self.threshold = threshold
        self.last_state = None
        self.trigger_count = 0
        self.total_count = 0

    def reset(self):
        super().reset()
        self.last_state = None
        self.trigger_count = 0
        self.total_count = 0

    def predict_action(self, env, obs):
        self.total_count += 1
        state = np.array([env.height / 50.0, env.velocity / 10.0, env.fuel_remaining / 5.0, env.current_thrust / 300.0])
        if self.last_state is None or np.linalg.norm(state - self.last_state) > self.threshold:
            self.trigger_count += 1
            action = super().predict_action(env, obs)
        else:
            action = np.array([2.0 * self.last_throttle - 1.0], dtype=np.float32)
        self.last_state = state
        return action


# ── evaluate ──────────────────────────────────────────────────────────

def evaluate_controller(controller_cls, label, scenarios, n_episodes=30, seed=SEED):
    results = {}
    representatives = {}
    for idx, (scenario_name, options) in enumerate(scenarios.items()):
        scenario_seed = seed + idx * 10_000
        rollouts = []
        for ep in range(n_episodes):
            ctrl = controller_cls()
            r = rollout_controller(ctrl, scenario_seed + ep, options, save_trajectory=(ep == 0))
            rollouts.append(r)
        successes = sum(1 for r in rollouts if r["success"])
        sr = successes / n_episodes
        # pick representative
        rep = next((r for r in rollouts if r["success"]), rollouts[0])
        results[scenario_name] = {
            "success_rate": sr,
            "crash_rate": sum(1 for r in rollouts if r["terminal_reason"] == "crash") / n_episodes,
            "n_episodes": n_episodes,
            "mean_fuel_used": float(np.mean([r["fuel_used"] for r in rollouts])),
            "mean_landing_time": float(np.mean([r["landing_time"] for r in rollouts])),
            "mean_final_velocity_error": float(np.mean([abs(r["final_v"]) for r in rollouts])),
        }
        representatives[scenario_name] = rep
        print(f"  {label}/{scenario_name}: {sr:.0%}")
    return results, representatives


# ── main ──────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = ROBUSTNESS_SCENARIOS
    scenario_list = list(scenarios.keys())

    # Evaluate tuned controllers
    configs = [
        (MPCController, "MPC_original"),
        (TunedMPCController, "MPC_tuned"),
        (EventTriggeredMPCController, "ET-MPC_original"),
        (TunedETMPCController, "ET-MPC_tuned"),
    ]

    all_results = {}
    all_reps = {}
    for cls, label in configs:
        print(f"\nEvaluating {label}...")
        results, reps = evaluate_controller(cls, label, scenarios)
        all_results[label] = results
        all_reps[label] = reps

    # Also load PPO and Energy PPO for comparison
    ppo_model = PPO.load("results/sweeps/v2_reward/models/final_model.zip", device="cpu")
    ppo_rms = load_obs_rms("results/sweeps/v2_reward/models/vec_normalize.pkl")
    energy_model = PPO.load("results/reproducible/energy_ppo_from_scratch_time/models/pure_energy_ppo_model.zip", device="cpu")
    energy_rms = load_obs_rms("results/reproducible/energy_ppo_from_scratch_time/models/pure_energy_vec_normalize.pkl")

    from rocket_landing_control.envs.rocket_env_energy import RocketLandingEnergyEnv
    from rocket_landing_control.core.experiment_utils import evaluate_model_rollouts, summarize_rollouts

    for label, model, rms, env_factory in [
        ("Baseline PPO", ppo_model, ppo_rms, RocketLandingEnv),
        ("Pure Energy PPO", energy_model, energy_rms, RocketLandingEnergyEnv),
    ]:
        print(f"\nEvaluating {label}...")
        all_results[label] = {}
        all_reps[label] = {}
        for idx, (scenario_name, options) in enumerate(scenarios.items()):
            scenario_seed = SEED + idx * 10_000
            rollouts = evaluate_model_rollouts(model, env_factory, rms, 30, scenario_seed, options, save_trajectories=True)
            summary = summarize_rollouts(rollouts)
            all_results[label][scenario_name] = summary
            rep = next((r for r in rollouts if r["success"]), rollouts[0])
            all_reps[label][scenario_name] = rep
            print(f"  {label}/{scenario_name}: {summary['success_rate']:.0%}")

    # ── heatmap ──
    strategies = list(all_results.keys())
    data = np.full((len(strategies), len(scenario_list)), np.nan)
    for i, strat in enumerate(strategies):
        for j, sc in enumerate(scenario_list):
            val = all_results[strat].get(sc, {}).get("success_rate")
            if val is not None:
                data[i, j] = val

    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(scenario_list)))
    ax.set_xticklabels(scenario_list, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=10)
    for i in range(len(strategies)):
        for j in range(len(scenario_list)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.3 or val > 0.8 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")
    ax.set_title("MPC Tuning Comparison: Original vs Tuned (+ RL baselines)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Success Rate", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mpc_tuning_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUTPUT_DIR / 'mpc_tuning_heatmap.png'}")

    # ── per-scenario trajectory comparison ──
    for scenario_name in scenario_list:
        traces = {}
        for label in strategies:
            rep = all_reps.get(label, {}).get(scenario_name)
            if rep:
                norm = normalize_trajectory(rep)
                if norm:
                    traces[label] = norm
        if traces:
            plot_comparison(
                traces,
                f"MPC Tuning: {scenario_name} (unified seed={SEED})",
                OUTPUT_DIR / f"{scenario_name}_trajectory.png",
            )

    # ── save JSON ──
    with open(OUTPUT_DIR / "mpc_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved {OUTPUT_DIR / 'mpc_tuning_results.json'}")

    # ── CSV ──
    import csv
    with open(OUTPUT_DIR / "mpc_tuning_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy"] + scenario_list)
        for strat in strategies:
            row = [strat]
            for sc in scenario_list:
                val = all_results[strat].get(sc, {}).get("success_rate")
                row.append(f"{val:.0%}" if val is not None else "")
            writer.writerow(row)
    print(f"Saved {OUTPUT_DIR / 'mpc_tuning_results.csv'}")


if __name__ == "__main__":
    main()
