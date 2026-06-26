"""Generate trajectory comparisons with UNIFIED initial conditions.

For each scenario, ALL strategies are rolled out from the SAME initial state
(same seed + same reset options), so the trajectories are directly comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from rocket_landing_control.studies.controller_comparison_full import (
    EventTriggeredMPCController,
    MPCController,
    PIDController,
    rollout_controller,
)
from rocket_landing_control.studies.tune_mpc import TunedMPCController, TunedETMPCController
from rocket_landing_control.envs.rocket_env import RocketLandingEnv
from rocket_landing_control.envs.rocket_env_energy import RocketLandingEnergyEnv
from rocket_landing_control.core.experiment_utils import ROBUSTNESS_SCENARIOS, load_obs_rms, normalize_obs
from rocket_landing_control.visualization.generate_experiment_trajectory_comparisons import normalize_trajectory, plot_comparison

OUTPUT_DIR = Path("results/reproducible/final_comparison/unified_trajectories")
SEED = 1000  # unified seed for all strategies


# ── strategy definitions ──────────────────────────────────────────────

def make_group1_strategies():
    """Pure Energy PPO vs PPO vs SAC vs TD3"""
    return {
        "Pure Energy PPO": {
            "kind": "sb3", "algo": "PPO",
            "model": "results/reproducible/energy_ppo_from_scratch_time/models/pure_energy_ppo_model.zip",
            "stats": "results/reproducible/energy_ppo_from_scratch_time/models/pure_energy_vec_normalize.pkl",
            "env": "energy",
        },
        "Baseline PPO": {
            "kind": "sb3", "algo": "PPO",
            "model": "results/sweeps/v2_reward/models/final_model.zip",
            "stats": "results/sweeps/v2_reward/models/vec_normalize.pkl",
            "env": "base",
        },
        "SAC": {
            "kind": "sb3", "algo": "SAC",
            "model": "experiments/rl_comparison/models/SAC/model.zip",
            "stats": "experiments/rl_comparison/models/SAC/vec_normalize.pkl",
            "env": "base",
        },
        "TD3": {
            "kind": "sb3", "algo": "TD3",
            "model": "experiments/rl_comparison/models/TD3/model.zip",
            "stats": "experiments/rl_comparison/models/TD3/vec_normalize.pkl",
            "env": "base",
        },
    }


def make_group2_strategies():
    """PPO full vs ablation variants"""
    modes = ["full", "no_fuel", "no_smooth", "no_safety", "no_success", "basic"]
    strats = {}
    for mode in modes:
        strats[f"PPO_{mode}"] = {
            "kind": "sb3", "algo": "PPO",
            "model": f"results/final_report/ablation/{mode}/model.zip",
            "stats": f"results/final_report/ablation/{mode}/vec_normalize.pkl",
            "env": "base",
        }
    return strats


def make_group3_strategies():
    """Energy PPO variants"""
    variants = {
        "Pure Energy PPO (v4)": ("energy_ppo_from_scratch_time", "pure_energy"),
        "EnergyPPO_v1 (ratio_obs)": ("energy_ppo_ratio_obs_probe2", "pure_energy"),
        "EnergyPPO_v2 (success)": ("energy_ppo_success", "pure_energy"),
        "EnergyPPO_v3 (time_finetune)": ("energy_ppo_time_finetune", "pure_energy"),
    }
    strats = {}
    for label, (dir_name, prefix) in variants.items():
        strats[label] = {
            "kind": "sb3", "algo": "PPO",
            "model": f"results/reproducible/{dir_name}/models/{prefix}_ppo_model.zip",
            "stats": f"results/reproducible/{dir_name}/models/{prefix}_vec_normalize.pkl",
            "env": "energy",
        }
    return strats


def make_group4_strategies():
    """Energy PPO vs PPO vs PID/MPC/ET-MPC"""
    return {
        "Pure Energy PPO": {
            "kind": "sb3", "algo": "PPO",
            "model": "results/reproducible/energy_ppo_from_scratch_time/models/pure_energy_ppo_model.zip",
            "stats": "results/reproducible/energy_ppo_from_scratch_time/models/pure_energy_vec_normalize.pkl",
            "env": "energy",
        },
        "Baseline PPO": {
            "kind": "sb3", "algo": "PPO",
            "model": "results/sweeps/v2_reward/models/final_model.zip",
            "stats": "results/sweeps/v2_reward/models/vec_normalize.pkl",
            "env": "base",
        },
        "PID": {"kind": "controller", "controller": "PID"},
        "MPC": {"kind": "controller", "controller": "MPC"},
        "ET-MPC": {"kind": "controller", "controller": "ET-MPC"},
    }


# ── rollout helpers ───────────────────────────────────────────────────

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
ENV_MAP = {"base": RocketLandingEnv, "energy": RocketLandingEnergyEnv}
CTRL_MAP = {"PID": PIDController, "MPC": TunedMPCController, "ET-MPC": TunedETMPCController}

# cache loaded models
_model_cache: dict[str, tuple] = {}


def _load_sb3(spec: dict):
    key = (spec["model"], spec.get("stats"))
    if key not in _model_cache:
        cls = ALGO_MAP[spec["algo"]]
        model = cls.load(spec["model"], device="cpu")
        obs_rms = load_obs_rms(spec.get("stats"))
        _model_cache[key] = (model, obs_rms)
    return _model_cache[key]


def rollout_sb3(spec: dict, scenario_options: dict, seed: int) -> dict | None:
    model, obs_rms = _load_sb3(spec)
    env_cls = ENV_MAP[spec["env"]]
    env = env_cls()
    obs, info = env.reset(seed=seed, options=scenario_options)
    trajectory = []
    done = truncated = False
    while not (done or truncated):
        obs_norm = normalize_obs(obs, obs_rms)
        action, _ = model.predict(obs_norm, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        trajectory.append({
            "time": float(env.time),
            "height": float(env.height),
            "velocity": float(env.velocity),
            "thrust": float(env.current_thrust),
            "fuel": float(env.fuel_remaining),
            "mass": float(env.dry_mass + env.fuel_remaining),
            "action": float(np.asarray(action).reshape(-1)[0]),
        })
    env.close()
    return {
        "trajectory": trajectory,
        "terminal_reason": info.get("terminal_reason", "unknown"),
        "initial_conditions": info.get("initial_conditions", {}),
    }


def rollout_ctrl(spec: dict, scenario_options: dict, seed: int) -> dict | None:
    controller = CTRL_MAP[spec["controller"]]()
    env = RocketLandingEnv()
    controller.reset()
    obs, info = env.reset(seed=seed, options=scenario_options)
    trajectory = []
    done = truncated = False
    while not (done or truncated):
        action = controller.predict_action(env, obs)
        obs, reward, done, truncated, info = env.step(action)
        trajectory.append({
            "time": float(env.time),
            "height": float(env.height),
            "velocity": float(env.velocity),
            "thrust": float(env.current_thrust),
            "fuel": float(env.fuel_remaining),
            "mass": float(env.dry_mass + env.fuel_remaining),
            "action": float(action[0]),
        })
    env.close()
    return {
        "trajectory": trajectory,
        "terminal_reason": info.get("terminal_reason", "unknown"),
        "initial_conditions": info.get("initial_conditions", {}),
    }


def rollout_strategy(spec: dict, scenario_options: dict, seed: int) -> dict | None:
    if spec["kind"] == "sb3":
        return rollout_sb3(spec, scenario_options, seed)
    return rollout_ctrl(spec, scenario_options, seed)


# ── main ──────────────────────────────────────────────────────────────

def run_group(group_name: str, strategies: dict[str, dict]):
    print(f"\n{'='*60}")
    print(f"  {group_name}")
    print(f"{'='*60}")

    group_dir = OUTPUT_DIR / group_name.replace(" ", "_").replace("/", "_")
    group_dir.mkdir(parents=True, exist_ok=True)

    all_trajs: dict[str, dict[str, dict]] = {}

    for scenario_name, scenario_options in ROBUSTNESS_SCENARIOS.items():
        print(f"\n  Scenario: {scenario_name}")
        scenario_traces = {}
        for label, spec in strategies.items():
            try:
                result = rollout_strategy(spec, scenario_options, seed=SEED)
                norm = normalize_trajectory(result)
                if norm is not None:
                    scenario_traces[label] = norm
                    status = norm["terminal_reason"]
                    print(f"    {label:30s} → {status}")
                else:
                    print(f"    {label:30s} → [no trajectory data]")
            except Exception as e:
                print(f"    {label:30s} → ERROR: {e}")

        if scenario_traces:
            plot_comparison(
                scenario_traces,
                f"{group_name}: {scenario_name} (unified seed={SEED})",
                group_dir / f"{scenario_name}_trajectory.png",
            )

    print(f"\n  Saved to: {group_dir}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = [
        ("group1_rl_algorithms", make_group1_strategies()),
        ("group2_ppo_ablation", make_group2_strategies()),
        ("group3_energy_ppo_variants", make_group3_strategies()),
        ("group4_controllers", make_group4_strategies()),
    ]
    for group_name, strategies in groups:
        run_group(group_name, strategies)
    print(f"\nAll done! Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
