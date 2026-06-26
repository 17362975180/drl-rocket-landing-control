"""Rocket vertical soft-landing environment — unified version.

Supports three modes via constructor parameters:

- **standard** (default): 4-dim observation, hand-crafted reward.
- **energy**: 7-dim observation (4 base + 3 energy ratios), pure energy-guided reward.

Safety shielding can be enabled on any mode via ``safety_enabled=True``.

State: [height, velocity, fuel_remaining, current_thrust], normalized.
(+ 3 energy ratios when reward_mode="energy")
Action: one continuous value in [-1, 1], mapped to throttle [0, 1].
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RocketLandingEnv(gym.Env):
    """Physics-based 1D rocket landing environment.

    Parameters
    ----------
    reward_mode : ``"standard"`` | ``"energy"``
        Selects the reward function.  ``"energy"`` also extends the
        observation vector from 4 to 7 dimensions.
    safety_enabled : bool
        When *True*, four safety shielding rules are applied to every
        action before it reaches the physics simulation.
    render_mode : str | None
        Gymnasium render mode.
    randomize : bool
        Whether to randomize initial conditions on ``reset()``.
    max_velocity, max_acceleration, max_height, max_time : float
        Episode termination bounds.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str | None = None,
        randomize: bool = False,
        max_velocity: float = 50.0,
        max_acceleration: float = 35.0,
        max_height: float = 120.0,
        max_time: float = 100.0,
        reward_mode: str = "standard",
        safety_enabled: bool = False,
    ):
        super().__init__()

        # ── reward / safety mode ──────────────────────────────────────
        assert reward_mode in ("standard", "energy"), reward_mode
        self.reward_mode = reward_mode
        self.safety_enabled = safety_enabled

        # ── physics parameters ────────────────────────────────────────
        self.nominal_g = 9.81
        self.nominal_dry_mass = 10.0
        self.nominal_initial_fuel = 5.0
        self.nominal_T_max = 300.0

        self.g = self.nominal_g
        self.dry_mass = self.nominal_dry_mass
        self.initial_fuel = self.nominal_initial_fuel
        self.T_max = self.nominal_T_max
        self.exhaust_v = 200.0
        self.drag_coeff = 0.02
        self.thrust_delay = 0.05
        self.dt = 0.05
        self.initial_height = 50.0
        self.initial_velocity = 0.0

        self.randomize = randomize
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_height = max_height
        self.max_time = max_time

        self.default_randomization = {
            "initial_height_range": (45.0, 55.0),
            "initial_velocity_range": (-1.0, 1.0),
        }

        # ── spaces ───────────────────────────────────────────────────
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        if self.reward_mode == "energy":
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.render_mode = render_mode

        # ── mutable episode state ────────────────────────────────────
        self.height: float | None = None
        self.velocity: float | None = None
        self.fuel_remaining: float | None = None
        self.current_thrust: float | None = None
        self.time: float | None = None
        self.fuel_used: float | None = None
        self.last_action: float | None = None
        self.last_acceleration = 0.0
        self.max_abs_acceleration_seen = 0.0
        self.max_abs_velocity_seen = 0.0
        self.terminal_reason = "none"
        self._reward_breakdown: dict[str, float] = {}
        self.initial_conditions: dict[str, float] = {}
        self.sensor_noise = 0.0
        self.action_delay_steps = 0
        self._action_buffer: deque[np.ndarray] = deque()

        # ── safety-shielding state (only meaningful when safety_enabled) ──
        self.danger_height = 10.0
        self.danger_velocity = -5.0
        self.min_throttle_danger = 0.5
        self.max_throttle_rate = 0.3
        self.last_throttle: float | None = None
        self.last_velocity_for_accel: float | None = None
        self.safety_max_acceleration = 20.0
        self.fuel_threshold = 1.0
        self.max_throttle_low_fuel = 0.5
        self.safety_intervention = False

        # ── energy-mode bookkeeping (set in reset) ───────────────────
        self.initial_mechanical_energy = 1.0
        self.initial_fuel_energy = 1.0
        self.prev_energy_state: dict[str, float] = {}

    # -----------------------------------------------------------------
    # reset
    # -----------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = dict(options or {})
        episode_config = dict(self.default_randomization) if options.pop("randomize", self.randomize) else {}
        episode_config.update(options)

        height = self._resolve_value("initial_height", self.initial_height, episode_config)
        velocity = self._resolve_value("initial_velocity", self.initial_velocity, episode_config)
        dry_mass = self._resolve_value("dry_mass", self.dry_mass, episode_config)
        initial_fuel = self._resolve_value("initial_fuel", self.initial_fuel, episode_config)
        gravity_scale = self._resolve_value("gravity_scale", 1.0, episode_config)
        thrust_scale = self._resolve_value("thrust_scale", 1.0, episode_config)

        self.g = self.nominal_g * gravity_scale
        self.dry_mass = max(0.1, dry_mass)
        self.initial_fuel = max(0.0, initial_fuel)
        self.T_max = max(0.0, self.nominal_T_max * thrust_scale)
        self.sensor_noise = max(0.0, self._resolve_value("sensor_noise", 0.0, episode_config))
        self.action_delay_steps = int(max(0, self._resolve_value("action_delay_steps", 0, episode_config)))
        self._action_buffer.clear()

        self.height = max(0.0, height)
        self.velocity = velocity
        self.fuel_remaining = self.initial_fuel
        self.current_thrust = 0.0
        self.time = 0.0
        self.fuel_used = 0.0
        self.last_action = None
        self.last_acceleration = 0.0
        self.max_abs_acceleration_seen = 0.0
        self.max_abs_velocity_seen = abs(self.velocity)
        self.terminal_reason = "none"
        self.initial_conditions = {
            "initial_height": float(self.height),
            "initial_velocity": float(self.velocity),
            "dry_mass": float(self.dry_mass),
            "initial_fuel": float(self.initial_fuel),
            "gravity_scale": float(gravity_scale),
            "thrust_scale": float(thrust_scale),
            "sensor_noise": float(self.sensor_noise),
            "action_delay_steps": float(self.action_delay_steps),
        }

        # safety state
        self.last_throttle = None
        self.last_velocity_for_accel = None
        self.safety_intervention = False

        # energy state
        if self.reward_mode == "energy":
            self.initial_mechanical_energy = max(self._mechanical_energy(), 1.0)
            self.initial_fuel_energy = max(self._fuel_available_energy(), 1.0)
            self.prev_energy_state = self._energy_state()
            self._reward_breakdown = self._empty_energy_breakdown()
        else:
            self._reward_breakdown = self._empty_reward_breakdown()

        return self._get_obs(), self._get_info()

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------
    def _resolve_value(self, name: str, default: float, config: dict[str, Any]) -> float:
        range_key = f"{name}_range"
        if range_key in config:
            low, high = config[range_key]
            return float(self.np_random.uniform(low, high))
        if name in config:
            return float(config[name])
        return float(default)

    # -----------------------------------------------------------------
    # observation
    # -----------------------------------------------------------------
    def _get_obs(self):
        obs = np.array(
            [
                self.height / 50.0,
                self.velocity / 10.0,
                self.fuel_remaining / max(self.nominal_initial_fuel, 1e-6),
                self.current_thrust / max(self.nominal_T_max, 1e-6),
            ],
            dtype=np.float32,
        )
        if self.sensor_noise > 0:
            obs = obs + self.np_random.normal(0.0, self.sensor_noise, size=obs.shape).astype(np.float32)

        if self.reward_mode == "energy":
            energy = self._energy_state()
            energy_obs = np.array(
                [
                    np.clip(energy["rho_brake"], 0.0, 3.0) / 3.0,
                    np.clip(energy["rho_fuel"], 0.0, 3.0) / 3.0,
                    np.clip(energy["rho_impact"], 0.0, 30.0) / 30.0,
                ],
                dtype=np.float32,
            )
            return np.concatenate([obs, energy_obs]).astype(np.float32)

        return obs

    # -----------------------------------------------------------------
    # info
    # -----------------------------------------------------------------
    def _get_info(self):
        info = {
            "success": self.terminal_reason == "success",
            "crash": self.terminal_reason == "crash",
            "fuel_used": float(self.fuel_used),
            "final_h": float(self.height),
            "final_v": float(self.velocity),
            "terminal_reason": self.terminal_reason,
            "reward_breakdown": self._reward_breakdown.copy(),
            "initial_conditions": self.initial_conditions.copy(),
            "height": float(self.height),
            "velocity": float(self.velocity),
            "fuel_remaining": float(self.fuel_remaining),
            "current_thrust": float(self.current_thrust),
            "mass": float(self.dry_mass + self.fuel_remaining),
            "time": float(self.time),
            "last_acceleration": float(self.last_acceleration),
            "max_abs_acceleration": float(self.max_abs_acceleration_seen),
            "max_abs_velocity": float(self.max_abs_velocity_seen),
        }
        if self.reward_mode == "energy":
            info["energy"] = self._energy_state()
        if self.safety_enabled:
            info["raw_throttle"] = getattr(self, "_raw_throttle", None)
            info["safe_throttle"] = getattr(self, "_safe_throttle", None)
            info["safety_intervention"] = self.safety_intervention
        return info

    # -----------------------------------------------------------------
    # action delay
    # -----------------------------------------------------------------
    def _delayed_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_delay_steps <= 0:
            return action
        self._action_buffer.append(np.array(action, dtype=np.float32))
        if len(self._action_buffer) > self.action_delay_steps:
            return self._action_buffer.popleft()
        return np.zeros_like(action, dtype=np.float32)

    # -----------------------------------------------------------------
    # safety shielding
    # -----------------------------------------------------------------
    def _apply_safety_mechanisms(self, throttle: float) -> float:
        """Apply four safety rules to a throttle in [0, 1]."""
        safe = throttle

        # 1. low-altitude / high-speed rule
        if self.height < self.danger_height and self.velocity < self.danger_velocity:
            safe = max(safe, self.min_throttle_danger)

        # 2. throttle rate limit
        if self.last_throttle is not None:
            delta = safe - self.last_throttle
            if abs(delta) > self.max_throttle_rate:
                safe = self.last_throttle + np.sign(delta) * self.max_throttle_rate
            safe = np.clip(safe, 0.0, 1.0)

        # 3. acceleration limit
        if self.last_velocity_for_accel is not None:
            accel = abs(self.velocity - self.last_velocity_for_accel) / self.dt
            if accel > self.safety_max_acceleration:
                safe *= self.safety_max_acceleration / accel

        # 4. fuel conservation
        if self.fuel_remaining < self.fuel_threshold:
            safe = min(safe, self.max_throttle_low_fuel)

        # bookkeeping
        self.last_throttle = np.clip(safe, 0.0, 1.0)
        self.last_velocity_for_accel = self.velocity
        return float(np.clip(safe, 0.0, 1.0))

    # -----------------------------------------------------------------
    # step
    # -----------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = self._delayed_action(action)
        raw_action = float(action[0])
        throttle = float(np.clip((raw_action + 1.0) / 2.0, 0.0, 1.0))

        # ── safety shielding (before physics) ────────────────────────
        self._raw_throttle = throttle
        if self.safety_enabled:
            throttle = self._apply_safety_mechanisms(throttle)
            self.safety_intervention = abs(throttle - self._raw_throttle) > 0.01
            action = np.array([throttle * 2.0 - 1.0], dtype=np.float32)
        else:
            self.safety_intervention = False
        self._safe_throttle = throttle

        # ── physics (identical for all modes) ────────────────────────
        target_thrust = throttle * self.T_max
        alpha = min(1.0, self.dt / max(self.thrust_delay, 1e-6))
        self.current_thrust += alpha * (target_thrust - self.current_thrust)

        fuel_consumed = (self.current_thrust / self.exhaust_v) * self.dt
        if self.fuel_remaining <= 0.0:
            self.current_thrust = 0.0
            fuel_consumed = 0.0
        elif fuel_consumed > self.fuel_remaining:
            max_thrust = (self.fuel_remaining / self.dt) * self.exhaust_v
            self.current_thrust = min(self.current_thrust, max_thrust)
            fuel_consumed = (self.current_thrust / self.exhaust_v) * self.dt

        self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_consumed)
        self.fuel_used += fuel_consumed
        if self.fuel_remaining <= 0.0:
            self.current_thrust = 0.0

        total_mass = self.dry_mass + self.fuel_remaining
        drag = -self.drag_coeff * self.velocity * abs(self.velocity)
        net_force = self.current_thrust + drag - total_mass * self.g
        acceleration = net_force / total_mass
        self.last_acceleration = acceleration
        self.max_abs_acceleration_seen = max(self.max_abs_acceleration_seen, abs(acceleration))

        self.velocity += acceleration * self.dt
        self.height += self.velocity * self.dt
        self.time += self.dt
        self.max_abs_velocity_seen = max(self.max_abs_velocity_seen, abs(self.velocity))

        # ── reward (mode-dependent) ──────────────────────────────────
        if self.reward_mode == "energy":
            # In the original energy env, super().step() runs
            # _calculate_reward() first — this sets self.terminal_reason
            # which _calculate_energy_reward() reads.  We replicate
            # that by running the standard reward first (discarded).
            _std_reward, terminated, truncated = self._calculate_reward(throttle)
            prev_state = self.prev_energy_state
            prev_height = self.height - self.velocity * self.dt  # reconstruct
            prev_action = self.last_action
            reward = self._calculate_energy_reward(prev_state, prev_height, prev_action)
        else:
            reward, terminated, truncated = self._calculate_reward(throttle)

        self.last_action = throttle

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # termination (shared by both modes)
    # -----------------------------------------------------------------
    def _check_termination(self) -> tuple[bool, bool]:
        terminated = False
        truncated = False

        if self.height <= 0.0:
            self.height = 0.0
            terminated = True
            if abs(self.velocity) <= 2.0:
                self.terminal_reason = "success"
            else:
                self.terminal_reason = "crash"
        elif self.height > self.max_height:
            terminated = True
            self.terminal_reason = "out_of_bounds"
        elif abs(self.velocity) > self.max_velocity:
            terminated = True
            self.terminal_reason = "velocity_exceeded"
        elif abs(self.last_acceleration) > self.max_acceleration:
            terminated = True
            self.terminal_reason = "acceleration_exceeded"
        elif self.time > self.max_time:
            truncated = True
            self.terminal_reason = "timeout"

        return terminated, truncated

    # -----------------------------------------------------------------
    # standard reward (original from base env)
    # -----------------------------------------------------------------
    def _empty_reward_breakdown(self) -> dict[str, float]:
        return {
            "height": 0.0,
            "velocity": 0.0,
            "fuel": 0.0,
            "smooth": 0.0,
            "approach": 0.0,
            "efficiency": 0.0,
            "time": 0.0,
            "safety": 0.0,
            "terminal": 0.0,
        }

    def _calculate_reward(self, throttle: float):
        breakdown = self._empty_reward_breakdown()
        terminated = False
        truncated = False
        self.terminal_reason = "none"

        if self.height <= 0.0:
            self.height = 0.0
            terminated = True
            if abs(self.velocity) <= 2.0:
                self.terminal_reason = "success"
                breakdown["terminal"] = 1000.0
                breakdown["terminal"] += max(0.0, 300.0 * (1.0 - self.time / 30.0))
                if self.initial_fuel > 0:
                    breakdown["terminal"] += 200.0 * (self.fuel_remaining / self.initial_fuel)
            else:
                self.terminal_reason = "crash"
                breakdown["terminal"] = -1000.0
        elif self.height > self.max_height:
            terminated = True
            self.terminal_reason = "out_of_bounds"
            breakdown["terminal"] = -1000.0
        elif abs(self.velocity) > self.max_velocity:
            terminated = True
            self.terminal_reason = "velocity_exceeded"
            breakdown["terminal"] = -1000.0
        elif abs(self.last_acceleration) > self.max_acceleration:
            terminated = True
            self.terminal_reason = "acceleration_exceeded"
            breakdown["terminal"] = -1000.0
        elif self.time > self.max_time:
            truncated = True
            self.terminal_reason = "timeout"
            breakdown["terminal"] = -500.0

        if terminated or truncated:
            self._reward_breakdown = breakdown
            return sum(breakdown.values()), terminated, truncated

        breakdown["height"] = -1.0 * (self.height / 50.0)

        if self.height > 30:
            target_v = -15.0
        elif self.height > 15:
            target_v = -10.0
        elif self.height > 5:
            target_v = -5.0
        else:
            target_v = -2.0
        breakdown["velocity"] = -0.3 * abs(self.velocity - target_v)

        if self.height > 20:
            breakdown["efficiency"] = 1.0 if throttle < 0.1 else -2.0
        elif self.height > 10:
            if self.velocity < -15:
                breakdown["efficiency"] = 1.0 if throttle > 0.5 else -1.0
            else:
                breakdown["efficiency"] = 0.5 if throttle < 0.1 else -0.5
        else:
            if self.velocity < -5:
                breakdown["efficiency"] = 2.0 if throttle > 0.5 else -3.0
            elif self.velocity < -2:
                breakdown["efficiency"] = 1.0 if 0.3 < throttle < 0.7 else -0.5
            else:
                breakdown["efficiency"] = 0.5 if throttle < 0.3 else -1.0

        fuel_ratio = self.fuel_remaining / max(self.initial_fuel, 1e-6)
        breakdown["fuel"] = fuel_ratio * (2.0 if self.height > 20 else 1.0)
        if self.last_action is not None:
            breakdown["smooth"] = -0.05 * abs(throttle - self.last_action)
        if self.height < 10.0 and abs(self.velocity) < 3.0:
            breakdown["approach"] = (10.0 - self.height) / 10.0
        if abs(self.velocity) > 35.0:
            breakdown["safety"] -= 5.0 * ((abs(self.velocity) - 35.0) / 15.0)
        if abs(self.last_acceleration) > 25.0:
            breakdown["safety"] -= 2.0 * ((abs(self.last_acceleration) - 25.0) / 10.0)
        breakdown["time"] = -0.05

        self._reward_breakdown = breakdown
        return sum(breakdown.values()), terminated, truncated

    # -----------------------------------------------------------------
    # energy reward (from rocket_env_energy.py)
    # -----------------------------------------------------------------
    def _empty_energy_breakdown(self):
        return {
            "coast_efficiency": 0.0,
            "switch_surface": 0.0,
            "braking_reserve": 0.0,
            "fuel_energy_efficiency": 0.0,
            "energy_power": 0.0,
            "impact_energy": 0.0,
            "energy_smoothness": 0.0,
            "terminal_energy_time": 0.0,
        }

    def _mass(self):
        return float(self.dry_mass + self.fuel_remaining)

    def _potential_energy(self):
        return self._mass() * self.g * max(float(self.height), 0.0)

    def _kinetic_energy(self):
        return 0.5 * self._mass() * float(self.velocity) ** 2

    def _mechanical_energy(self):
        return self._potential_energy() + self._kinetic_energy()

    def _fuel_available_energy(self):
        return 0.5 * max(float(self.fuel_remaining), 0.0) * self.exhaust_v**2

    def _safe_kinetic_energy(self, mass=None):
        m = float(self._mass() if mass is None else mass)
        return 0.5 * m * 2.0**2

    def _braking_capacity(self, height=None, mass=None):
        h = max(float(self.height if height is None else height), 0.0)
        m = float(self._mass() if mass is None else mass)
        max_net_braking_force = max(self.T_max - m * self.g, 0.0)
        return max_net_braking_force * h

    def _fuel_delta_v_capacity(self, mass=None):
        m = float(self._mass() if mass is None else mass)
        dry_mass = max(float(self.dry_mass), 1e-6)
        if m <= dry_mass:
            return 0.0
        return float(self.exhaust_v * np.log(m / dry_mass))

    def _fuel_braking_energy_capacity(self, mass=None):
        m = float(self._mass() if mass is None else mass)
        delta_v = self._fuel_delta_v_capacity(m)
        return 0.5 * m * delta_v**2

    def _energy_state(self):
        mass = self._mass()
        potential = self._potential_energy()
        kinetic = self._kinetic_energy()
        fuel_available = self._fuel_available_energy()
        braking_capacity = self._braking_capacity(float(self.height), mass)
        fuel_braking_capacity = self._fuel_braking_energy_capacity(mass)
        safe_kinetic = self._safe_kinetic_energy(mass)
        return {
            "potential": potential,
            "kinetic": kinetic,
            "mechanical": potential + kinetic,
            "fuel_available": fuel_available,
            "fuel_braking_capacity": fuel_braking_capacity,
            "fuel_remaining": float(self.fuel_remaining),
            "height": float(self.height),
            "velocity": float(self.velocity),
            "mass": mass,
            "thrust": float(self.current_thrust),
            "safe_kinetic": safe_kinetic,
            "braking_capacity": braking_capacity,
            "rho_brake": kinetic / max(braking_capacity, 1e-6),
            "rho_fuel": kinetic / max(fuel_braking_capacity, 1e-6),
            "rho_impact": kinetic / max(safe_kinetic, 1e-6),
        }

    def _calculate_energy_reward(self, prev_state, prev_height, prev_action):
        breakdown = self._empty_energy_breakdown()
        current = self._energy_state()

        initial_mechanical = max(self.initial_mechanical_energy, 1.0)
        initial_fuel_energy = max(self.initial_fuel_energy, 1.0)

        braking_capacity = max(current["braking_capacity"], 1e-6)
        fuel_capacity = max(current["fuel_braking_capacity"], 1e-6)
        safe_kinetic = max(current["safe_kinetic"], 1e-6)
        rho_brake = current["kinetic"] / braking_capacity
        rho_fuel = current["kinetic"] / fuel_capacity
        rho_impact = current["kinetic"] / safe_kinetic

        fuel_energy_used = max(prev_state["fuel_available"] - current["fuel_available"], 0.0)

        displacement = abs(float(self.height) - prev_height)
        thrust_work = float(self.current_thrust) * displacement
        useful_braking_energy = max(prev_state["kinetic"] - current["kinetic"], 0.0)
        work_efficiency = useful_braking_energy / max(fuel_energy_used, 1e-6) if fuel_energy_used > 0 else 0.0
        thrust_work_ratio = thrust_work / initial_mechanical
        fuel_used_ratio = fuel_energy_used / initial_fuel_energy
        potential_increase = max(current["potential"] - prev_state["potential"], 0.0) / initial_mechanical
        ascent_kinetic = current["kinetic"] / initial_mechanical if current["velocity"] > 0.0 else 0.0
        descent_gate = 1.0 if prev_state["velocity"] < 0.0 else 0.0

        throttle = self.current_thrust / max(self.T_max, 1e-6)
        over_brake = max(rho_brake - 1.0, 0.0)
        coast_surface = np.exp(-((rho_brake - 0.32) / 0.32) ** 2)
        switch_surface = np.exp(-((rho_brake - 0.88) / 0.12) ** 2)
        burn_gate = np.clip((rho_brake - 0.65) / 0.25, 0.0, 1.0)
        early_burn_penalty = max(0.65 - rho_brake, 0.0) * fuel_used_ratio

        mechanical_removed = max(prev_state["mechanical"] - current["mechanical"], 0.0)
        p_remove = mechanical_removed / max(self.dt, 1e-6)
        p_waste = fuel_energy_used / max(self.dt, 1e-6)
        p_ref = max(initial_mechanical / 2.5, 1.0)
        useful_brake_ratio = np.clip(work_efficiency / 0.25, 0.0, 1.0)

        breakdown["coast_efficiency"] = 0.55 * (1.0 - throttle) * coast_surface
        breakdown["switch_surface"] = (
            8.0 * switch_surface * (0.25 + 0.75 * throttle)
            - 14.0 * np.clip(max(0.65 - rho_brake, 0.0), 0.0, 1.0) * throttle
        )
        breakdown["braking_reserve"] = -170.0 * np.clip(over_brake, 0.0, 3.0) ** 2
        breakdown["fuel_energy_efficiency"] = (
            -35.0 * np.clip(rho_fuel - 1.0, 0.0, 3.0) ** 2
            - 2200.0 * early_burn_penalty
        )
        breakdown["energy_power"] = (
            6.0 * burn_gate * descent_gate * np.clip(p_remove / p_ref, 0.0, 1.0)
            - 1.4 * np.clip(p_waste / p_ref, 0.0, 1.0) * (1.0 - useful_brake_ratio)
            - 0.20 * np.clip(thrust_work_ratio, 0.0, 1.0)
            - 450.0 * np.clip(potential_increase, 0.0, 0.5)
        )

        breakdown["impact_energy"] = -300.0 * np.clip(ascent_kinetic, 0.0, 2.0)

        if prev_action is not None:
            breakdown["energy_smoothness"] = -0.1 * abs(throttle - prev_action)

        if abs(self.velocity) > 35.0:
            excess_energy = 0.5 * current["mass"] * (abs(self.velocity) - 35.0) ** 2
            breakdown["impact_energy"] -= 20.0 * np.clip(excess_energy / initial_mechanical, 0.0, 5.0)
        if abs(self.last_acceleration) > 25.0:
            accel_energy_proxy = 0.5 * current["mass"] * (abs(self.last_acceleration) - 25.0) ** 2 * self.dt**2
            breakdown["impact_energy"] -= 5.0 * np.clip(accel_energy_proxy / initial_mechanical, 0.0, 5.0)

        # ── terminal ────────────────────────────────────────────────
        terminated, truncated = self._check_termination()
        if terminated or truncated:
            residual_mechanical = current["mechanical"] / initial_mechanical
            fuel_remaining_ratio = current["fuel_available"] / initial_fuel_energy
            if self.terminal_reason == "success":
                safe_impact_bonus = 1850.0 * np.exp(-rho_impact)
                episode_power = self.initial_mechanical_energy / max(self.time, 1e-6)
                time_energy_bonus = 360.0 * np.clip(episode_power / 1200.0, 0.0, 1.0)
                breakdown["terminal_energy_time"] = safe_impact_bonus + time_energy_bonus + 45.0 * fuel_remaining_ratio
            elif self.terminal_reason == "crash":
                reserve_penalty = np.clip(over_brake + max(rho_fuel - 1.0, 0.0), 0.0, 4.0)
                breakdown["terminal_energy_time"] = -2200.0 - 300.0 * np.clip(rho_impact, 0.0, 40.0) - 380.0 * reserve_penalty
            elif self.terminal_reason == "timeout":
                breakdown["terminal_energy_time"] = -300.0 - 300.0 * np.clip(residual_mechanical, 0.0, 5.0)
            elif self.terminal_reason == "out_of_bounds":
                breakdown["terminal_energy_time"] = -600000.0 - 2000.0 * np.clip(residual_mechanical, 0.0, 5.0)
            else:
                breakdown["terminal_energy_time"] = -600.0 - 300.0 * np.clip(residual_mechanical, 0.0, 5.0)

        self._reward_breakdown = {k: float(v) for k, v in breakdown.items()}
        self.prev_energy_state = current
        return sum(self._reward_breakdown.values())

    # -----------------------------------------------------------------
    # render / close
    # -----------------------------------------------------------------
    def render(self, mode="human"):
        return None

    def close(self):
        return None
