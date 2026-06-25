"""Shared evaluation helpers for reproducible rocket-landing experiments."""

from __future__ import annotations

import csv
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np


STANDARD_EVAL_OPTIONS = {
    "randomize": True,
    "initial_height_range": (45.0, 55.0),
    "initial_velocity_range": (-1.0, 1.0),
    "dry_mass": 10.0,
    "initial_fuel": 5.0,
    "gravity_scale": 1.0,
    "thrust_scale": 1.0,
    "sensor_noise": 0.0,
    "action_delay_steps": 0,
}


ROBUSTNESS_SCENARIOS = {
    "standard": STANDARD_EVAL_OPTIONS,
    "random_height": {**STANDARD_EVAL_OPTIONS, "initial_height_range": (30.0, 70.0)},
    "random_velocity": {**STANDARD_EVAL_OPTIONS, "initial_velocity_range": (-5.0, 5.0)},
    "random_mass": {**STANDARD_EVAL_OPTIONS, "dry_mass_range": (8.0, 12.0)},
    "random_fuel": {**STANDARD_EVAL_OPTIONS, "initial_fuel_range": (3.0, 7.0)},
    "gravity_bias": {**STANDARD_EVAL_OPTIONS, "gravity_scale_range": (0.9, 1.1)},
    "thrust_bias": {**STANDARD_EVAL_OPTIONS, "thrust_scale_range": (0.9, 1.1)},
    "sensor_noise": {**STANDARD_EVAL_OPTIONS, "sensor_noise": 0.05},
    "action_delay_1": {**STANDARD_EVAL_OPTIONS, "action_delay_steps": 1},
    "action_delay_2": {**STANDARD_EVAL_OPTIONS, "action_delay_steps": 2},
    "combined": {
        **STANDARD_EVAL_OPTIONS,
        "initial_height_range": (30.0, 70.0),
        "initial_velocity_range": (-3.0, 3.0),
        "dry_mass_range": (9.0, 11.0),
        "initial_fuel_range": (4.0, 6.0),
        "gravity_scale_range": (0.95, 1.05),
        "thrust_scale_range": (0.95, 1.05),
        "sensor_noise": 0.03,
        "action_delay_steps": 1,
    },
}


def normalize_obs(obs: np.ndarray, obs_rms: Any | None) -> np.ndarray:
    if obs_rms is None:
        return obs
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)


def load_obs_rms(stats_path: str | os.PathLike[str] | None):
    if not stats_path:
        return None
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return getattr(stats, "obs_rms", stats)


def auto_find_stats(model_path: str, explicit_stats: str | None = None) -> str | None:
    if explicit_stats:
        return explicit_stats
    model_dir = Path(model_path).resolve().parent
    candidates = [
        model_dir / "vec_normalize.pkl",
        model_dir / "vec_normalize_stats.pkl",
        model_dir / "vec_normalize_stats_v7.pkl",
        model_dir.parent / "vec_normalize.pkl",
        model_dir.parent / "vec_normalize_stats.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def make_episode_options(base_options: dict[str, Any] | None, episode: int, seed: int | None):
    options = dict(base_options or STANDARD_EVAL_OPTIONS)
    options.setdefault("randomize", True)
    return options


def rollout_model(
    model: Any,
    env: Any,
    obs_rms: Any | None = None,
    seed: int | None = None,
    reset_options: dict[str, Any] | None = None,
    deterministic: bool = True,
    save_trajectory: bool = False,
):
    obs, info = env.reset(seed=seed, options=reset_options)
    obs_for_policy = normalize_obs(obs, obs_rms)
    done = False
    truncated = False
    episode_reward = 0.0
    actions = []
    velocities = [float(env.velocity)]
    trajectory = []
    safety_interventions = 0

    while not (done or truncated):
        action, _ = model.predict(obs_for_policy, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32)
        raw_obs, reward, done, truncated, info = env.step(action)
        if info.get("safety_intervention", False):
            safety_interventions += 1
        episode_reward += float(reward)
        actions.append(float(action.reshape(-1)[0]))
        velocities.append(float(env.velocity))

        if save_trajectory:
            trajectory.append(
                {
                    "time": float(env.time),
                    "height": float(env.height),
                    "velocity": float(env.velocity),
                    "thrust": float(env.current_thrust),
                    "mass": float(env.dry_mass + env.fuel_remaining),
                    "fuel": float(env.fuel_remaining),
                    "action": float(action.reshape(-1)[0]),
                    "throttle": float((np.clip(action.reshape(-1)[0], -1.0, 1.0) + 1.0) / 2.0),
                    "reward": float(reward),
                    "reward_breakdown": info.get("reward_breakdown", {}),
                }
            )
        obs_for_policy = normalize_obs(raw_obs, obs_rms)

    terminal_reason = info.get("terminal_reason", "unknown")
    throttle_delta = 0.0
    if len(actions) > 1:
        throttle_delta = float(np.mean(np.abs(np.diff(actions))))
    accelerations = []
    if len(velocities) > 1:
        accelerations = [abs(velocities[i + 1] - velocities[i]) / env.dt for i in range(len(velocities) - 1)]

    result = {
        "terminal_reason": terminal_reason,
        "success": terminal_reason == "success",
        "final_h": float(env.height),
        "final_v": float(env.velocity),
        "fuel_used": float(env.fuel_used),
        "landing_time": float(env.time),
        "reward": float(episode_reward),
        "mean_abs_throttle_delta": throttle_delta,
        "max_velocity": float(max(abs(v) for v in velocities)),
        "max_acceleration": float(max(accelerations) if accelerations else 0.0),
        "safety_interventions": int(safety_interventions),
        "safety_intervention_rate": float(safety_interventions / max(len(actions), 1)),
        "initial_conditions": info.get("initial_conditions", {}).copy(),
        "trajectory": trajectory,
    }
    return result


def summarize_rollouts(rollouts: list[dict[str, Any]], model_path: str | None = None, seed: int | None = None):
    n = len(rollouts)
    counts = Counter(r["terminal_reason"] for r in rollouts)
    successes = counts.get("success", 0)
    crashes = counts.get("crash", 0)
    timeouts = counts.get("timeout", 0)
    out_of_bounds = counts.get("out_of_bounds", 0)
    velocity_exceeded = counts.get("velocity_exceeded", 0)
    acceleration_exceeded = counts.get("acceleration_exceeded", 0)

    def mean(key: str):
        return float(np.mean([r[key] for r in rollouts])) if rollouts else 0.0

    def std(key: str):
        return float(np.std([r[key] for r in rollouts])) if rollouts else 0.0

    return {
        "model_path": model_path,
        "n_episodes": n,
        "seed": seed,
        "success_rate": successes / n if n else 0.0,
        "crash_rate": crashes / n if n else 0.0,
        "timeout_rate": timeouts / n if n else 0.0,
        "out_of_bounds_rate": out_of_bounds / n if n else 0.0,
        "velocity_exceeded_rate": velocity_exceeded / n if n else 0.0,
        "acceleration_exceeded_rate": acceleration_exceeded / n if n else 0.0,
        "mean_final_height_error": mean_abs(rollouts, "final_h"),
        "std_final_height_error": std_abs(rollouts, "final_h"),
        "mean_final_velocity_error": mean_abs(rollouts, "final_v"),
        "std_final_velocity_error": std_abs(rollouts, "final_v"),
        "mean_fuel_used": mean("fuel_used"),
        "std_fuel_used": std("fuel_used"),
        "mean_landing_time": mean("landing_time"),
        "std_landing_time": std("landing_time"),
        "mean_episode_reward": mean("reward"),
        "std_episode_reward": std("reward"),
        "mean_abs_throttle_delta": mean("mean_abs_throttle_delta"),
        "mean_max_velocity": mean("max_velocity"),
        "mean_max_acceleration": mean("max_acceleration"),
        "mean_safety_interventions": mean("safety_interventions"),
        "mean_safety_intervention_rate": mean("safety_intervention_rate"),
        "terminal_reason_counts": dict(counts),
    }


def mean_abs(rollouts: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([abs(r[key]) for r in rollouts])) if rollouts else 0.0


def std_abs(rollouts: list[dict[str, Any]], key: str) -> float:
    return float(np.std([abs(r[key]) for r in rollouts])) if rollouts else 0.0


def evaluate_model_rollouts(
    model: Any,
    env_factory: Callable[[], Any],
    obs_rms: Any | None,
    n_episodes: int,
    seed: int | None,
    options: dict[str, Any] | None = None,
    save_trajectories: bool = False,
):
    rollouts = []
    for episode in range(n_episodes):
        env = env_factory()
        ep_seed = None if seed is None else seed + episode
        ep_options = make_episode_options(options, episode, seed)
        rollouts.append(
            rollout_model(
                model=model,
                env=env,
                obs_rms=obs_rms,
                seed=ep_seed,
                reset_options=ep_options,
                save_trajectory=save_trajectories,
            )
        )
        env.close()
    return rollouts


def write_evaluation_outputs(
    rollouts: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: str | os.PathLike[str],
    save_trajectories: bool,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(output / "eval_episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "terminal_reason",
                "success",
                "initial_height",
                "initial_velocity",
                "dry_mass",
                "initial_fuel",
                "gravity_scale",
                "thrust_scale",
                "sensor_noise",
                "action_delay_steps",
                "final_h",
                "final_v",
                "fuel_used",
                "landing_time",
                "reward",
                "mean_abs_throttle_delta",
                "max_velocity",
                "max_acceleration",
                "safety_interventions",
                "safety_intervention_rate",
            ]
        )
        for episode, rollout in enumerate(rollouts):
            ic = rollout.get("initial_conditions", {})
            writer.writerow(
                [
                    episode,
                    rollout["terminal_reason"],
                    rollout["success"],
                    ic.get("initial_height"),
                    ic.get("initial_velocity"),
                    ic.get("dry_mass"),
                    ic.get("initial_fuel"),
                    ic.get("gravity_scale"),
                    ic.get("thrust_scale"),
                    ic.get("sensor_noise"),
                    ic.get("action_delay_steps"),
                    rollout["final_h"],
                    rollout["final_v"],
                    rollout["fuel_used"],
                    rollout["landing_time"],
                    rollout["reward"],
                    rollout["mean_abs_throttle_delta"],
                    rollout["max_velocity"],
                    rollout["max_acceleration"],
                    rollout["safety_interventions"],
                    rollout["safety_intervention_rate"],
                ]
            )

    if save_trajectories:
        with open(output / "all_trajectories.json", "w", encoding="utf-8") as f:
            json.dump(rollouts, f, indent=2, ensure_ascii=False)
        successes = [r for r in rollouts if r["terminal_reason"] == "success"]
        failures = [r for r in rollouts if r["terminal_reason"] != "success"]
        if successes:
            with open(output / "success_trajectories.json", "w", encoding="utf-8") as f:
                json.dump(successes[0], f, indent=2, ensure_ascii=False)
        if failures:
            with open(output / "fail_trajectories.json", "w", encoding="utf-8") as f:
                json.dump(failures[0], f, indent=2, ensure_ascii=False)
