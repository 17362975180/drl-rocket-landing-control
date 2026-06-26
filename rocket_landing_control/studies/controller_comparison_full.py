"""Controller comparison: PPO vs PID vs lightweight MPC vs ET-MPC."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rocket_landing_control.envs.rocket_env import RocketLandingEnv
from rocket_landing_control.core.experiment_utils import STANDARD_EVAL_OPTIONS, auto_find_stats, load_obs_rms, normalize_obs, summarize_rollouts


class PPOController:
    def __init__(self, model, obs_rms):
        self.model = model
        self.obs_rms = obs_rms

    def reset(self):
        pass

    def predict_action(self, env, obs):
        action, _ = self.model.predict(normalize_obs(obs, self.obs_rms), deterministic=True)
        return np.asarray(action, dtype=np.float32)


class PIDController:
    def __init__(self):
        self.kp_v = 0.10
        self.ki_v = 0.003
        self.kd_v = 0.005
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def target_velocity(self, height):
        if height > 30:
            return -15.0
        if height > 15:
            return -10.0
        if height > 5:
            return -5.0
        return -1.5

    def predict_action(self, env, obs):
        target_v = self.target_velocity(env.height)
        error = target_v - env.velocity
        self.integral = np.clip(self.integral + error * env.dt, -20.0, 20.0)
        derivative = (error - self.prev_error) / env.dt
        self.prev_error = error
        hover = (env.dry_mass + env.fuel_remaining) * env.g / max(env.T_max, 1e-6)
        throttle = hover + self.kp_v * error + self.ki_v * self.integral + self.kd_v * derivative
        throttle = float(np.clip(throttle, 0.0, 1.0))
        return np.array([2.0 * throttle - 1.0], dtype=np.float32)


class MPCController:
    def __init__(self, horizon=60, grid_size=21):
        self.horizon = horizon
        self.grid = np.linspace(0.0, 1.0, grid_size)
        self.last_throttle = 0.0

    def reset(self):
        self.last_throttle = 0.0

    def target_velocity(self, height):
        if height <= 0:
            return 0.0
        braking_profile = -max(1.5, min(15.0, np.sqrt(max(0.0, 1.2 * height))))
        if height > 30:
            return min(-12.0, braking_profile)
        if height > 15:
            return min(-8.0, braking_profile)
        if height > 5:
            return min(-4.0, braking_profile)
        return max(-1.5, braking_profile)

    def predict_action(self, env, obs):
        best_u = self.last_throttle
        best_cost = float("inf")
        for u in self.grid:
            cost = self._rollout_cost(env, float(u))
            if cost < best_cost:
                best_cost = cost
                best_u = float(u)
        self.last_throttle = best_u
        return np.array([2.0 * best_u - 1.0], dtype=np.float32)

    def _rollout_cost(self, env, throttle):
        h = float(env.height)
        v = float(env.velocity)
        fuel = float(env.fuel_remaining)
        thrust = float(env.current_thrust)
        cost = 0.02 * throttle * throttle + 0.05 * abs(throttle - self.last_throttle)
        for _ in range(self.horizon):
            alpha = min(1.0, env.dt / max(env.thrust_delay, 1e-6))
            thrust += alpha * (throttle * env.T_max - thrust)
            fuel = max(0.0, fuel - (thrust / env.exhaust_v) * env.dt)
            mass = env.dry_mass + fuel
            drag = -env.drag_coeff * v * abs(v)
            acc = (thrust + drag - mass * env.g) / mass
            v += acc * env.dt
            h += v * env.dt
            target_v = self.target_velocity(max(h, 0.0))
            cost += 0.8 * abs(v - target_v) + 0.02 * max(h, 0.0)
            if h <= 0:
                cost += 500.0 * max(abs(v) - 2.0, 0.0)
                break
        cost += 6.0 * abs(max(h, 0.0)) + 12.0 * abs(v if h <= 5 else v - self.target_velocity(h))
        return cost


class EventTriggeredMPCController(MPCController):
    def __init__(self, horizon=60, grid_size=21, threshold=0.25):
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


def rollout_controller(controller, seed, options, save_trajectory=False):
    env = RocketLandingEnv()
    controller.reset()
    obs, info = env.reset(seed=seed, options=options)
    done = truncated = False
    reward_sum = 0.0
    actions = []
    velocities = [float(env.velocity)]
    trajectory = []
    while not (done or truncated):
        action = controller.predict_action(env, obs)
        obs, reward, done, truncated, info = env.step(action)
        reward_sum += float(reward)
        actions.append(float(action[0]))
        velocities.append(float(env.velocity))
        if save_trajectory:
            trajectory.append(
                {
                    "time": float(env.time),
                    "height": float(env.height),
                    "velocity": float(env.velocity),
                    "thrust": float(env.current_thrust),
                    "fuel": float(env.fuel_remaining),
                    "mass": float(env.dry_mass + env.fuel_remaining),
                    "action": float(action[0]),
                    "reward": float(reward),
                }
            )
    throttle_delta = float(np.mean(np.abs(np.diff(actions)))) if len(actions) > 1 else 0.0
    accelerations = [abs(velocities[i + 1] - velocities[i]) / env.dt for i in range(len(velocities) - 1)]
    result = {
        "terminal_reason": info.get("terminal_reason", "unknown"),
        "success": info.get("terminal_reason") == "success",
        "final_h": float(env.height),
        "final_v": float(env.velocity),
        "fuel_used": float(env.fuel_used),
        "landing_time": float(env.time),
        "reward": reward_sum,
        "mean_abs_throttle_delta": throttle_delta,
        "max_velocity": float(max(abs(v) for v in velocities)),
        "max_acceleration": float(max(accelerations) if accelerations else 0.0),
        "safety_interventions": 0,
        "safety_intervention_rate": 0.0,
        "initial_conditions": info.get("initial_conditions", {}).copy(),
        "trajectory": trajectory,
    }
    if isinstance(controller, EventTriggeredMPCController):
        result["trigger_rate"] = controller.trigger_count / max(controller.total_count, 1)
    env.close()
    return result


def evaluate_controller(name, controller, n_episodes, seed, save_trajectories):
    rollouts = [
        rollout_controller(controller, seed + ep, STANDARD_EVAL_OPTIONS, save_trajectory=save_trajectories)
        for ep in range(n_episodes)
    ]
    summary = summarize_rollouts(rollouts, seed=seed)
    if name == "ET-MPC":
        summary["mean_trigger_rate"] = float(np.mean([r.get("trigger_rate", 0.0) for r in rollouts]))
    print(
        f"{name}: success={summary['success_rate']:.1%}, "
        f"crash={summary['crash_rate']:.1%}, fuel={summary['mean_fuel_used']:.2f}, "
        f"smooth={summary['mean_abs_throttle_delta']:.4f}"
    )
    return summary, rollouts


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PPO with classical controllers.")
    parser.add_argument("--model", type=str, default="results/reproducible/main/models/final_model.zip")
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--output", type=str, default="results/reproducible/controller_comparison.json")
    parser.add_argument("--save-trajectories", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    stats_path = auto_find_stats(args.model, args.stats)
    ppo_model = PPO.load(args.model, device="cpu")
    obs_rms = load_obs_rms(stats_path)
    controllers = {
        "PPO": PPOController(ppo_model, obs_rms),
        "PID": PIDController(),
        "MPC": MPCController(),
        "ET-MPC": EventTriggeredMPCController(),
    }

    results = {}
    trajectories = {}
    for i, (name, controller) in enumerate(controllers.items()):
        summary, rollouts = evaluate_controller(
            name,
            controller,
            args.n_episodes,
            args.seed + i * 10_000,
            args.save_trajectories,
        )
        results[name] = summary
        if args.save_trajectories:
            trajectories[name] = {
                "success": next((r for r in rollouts if r["success"]), None),
                "failure": next((r for r in rollouts if not r["success"]), None),
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if args.save_trajectories:
        with open(output.with_name(output.stem + "_trajectories.json"), "w", encoding="utf-8") as f:
            json.dump(trajectories, f, indent=2, ensure_ascii=False)
    print(f"Saved controller comparison to: {output}")


if __name__ == "__main__":
    main()
