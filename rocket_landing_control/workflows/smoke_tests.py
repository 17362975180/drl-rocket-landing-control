"""Lightweight reproducibility and environment checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rocket_landing_control.envs.rocket_env import RocketLandingEnv
from rocket_landing_control.envs.rocket_env_energy import RocketLandingEnergyEnv
from rocket_landing_control.core.experiment_utils import STANDARD_EVAL_OPTIONS


def check_reset_reproducibility():
    env = RocketLandingEnv()
    obs1, info1 = env.reset(seed=123, options=STANDARD_EVAL_OPTIONS)
    obs2, info2 = env.reset(seed=123, options=STANDARD_EVAL_OPTIONS)
    assert np.allclose(obs1, obs2), "same seed/options should reproduce observation"
    assert info1["initial_conditions"] == info2["initial_conditions"], "same seed should reproduce initial conditions"
    obs3, info3 = env.reset(seed=124, options=STANDARD_EVAL_OPTIONS)
    assert info1["initial_conditions"] != info3["initial_conditions"], "different seeds should sample different initial conditions"


def check_action_and_fuel_constraints():
    env = RocketLandingEnv()
    env.reset(seed=1, options={"initial_fuel": 0.01})
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.array([10.0], dtype=np.float32))
        assert 0.0 <= env.fuel_remaining <= env.initial_fuel + 1e-9
        if env.fuel_remaining <= 0:
            assert env.current_thrust == 0.0
            break


def check_terminal_and_reward_breakdown():
    env = RocketLandingEnv()
    env.reset(seed=1, options={"initial_height": 0.1, "initial_velocity": -20.0})
    done = truncated = False
    info = {}
    while not (done or truncated):
        obs, reward, done, truncated, info = env.step(np.array([-1.0], dtype=np.float32))
    assert info["terminal_reason"] in {"crash", "velocity_exceeded", "acceleration_exceeded"}
    expected_keys = {"height", "velocity", "fuel", "smooth", "approach", "efficiency", "time", "safety", "terminal"}
    assert expected_keys.issubset(info["reward_breakdown"].keys())


def check_energy_env_preserves_physics_and_reports_energy():
    action = np.array([0.25], dtype=np.float32)
    options = {"initial_height": 50.0, "initial_velocity": -1.0, "initial_fuel": 5.0, "dry_mass": 10.0}
    base_env = RocketLandingEnv()
    energy_env = RocketLandingEnergyEnv()
    base_env.reset(seed=7, options=options)
    _, energy_info = energy_env.reset(seed=7, options=options)
    assert {"potential", "kinetic", "mechanical", "fuel_available"}.issubset(energy_info["energy"].keys())

    base_env.step(action)
    _, _, _, _, energy_info = energy_env.step(action)
    assert np.isclose(base_env.height, energy_env.height), "energy env must preserve base physics height"
    assert np.isclose(base_env.velocity, energy_env.velocity), "energy env must preserve base physics velocity"
    assert np.isclose(base_env.fuel_remaining, energy_env.fuel_remaining), "energy env must preserve base fuel physics"
    forbidden = {"height", "velocity", "fuel", "approach", "efficiency", "time"}
    assert forbidden.isdisjoint(energy_info["reward_breakdown"].keys()), "energy reward must not expose base reward terms"
    expected = {
        "coast_efficiency",
        "switch_surface",
        "braking_reserve",
        "fuel_energy_efficiency",
        "energy_power",
        "impact_energy",
        "terminal_energy_time",
        "energy_smoothness",
    }
    assert expected.issubset(energy_info["reward_breakdown"].keys())


def check_energy_reward_prefers_early_coast_to_early_burn():
    options = {"initial_height": 50.0, "initial_velocity": 0.0, "initial_fuel": 5.0, "dry_mass": 10.0}
    coast_env = RocketLandingEnergyEnv()
    burn_env = RocketLandingEnergyEnv()
    coast_env.reset(seed=11, options=options)
    burn_env.reset(seed=11, options=options)

    _, coast_reward, _, _, coast_info = coast_env.step(np.array([-1.0], dtype=np.float32))
    _, burn_reward, _, _, burn_info = burn_env.step(np.array([1.0], dtype=np.float32))
    assert coast_reward > burn_reward, "early engine-off coasting should be better than early full burn"
    assert coast_info["reward_breakdown"]["coast_efficiency"] >= 0.0
    assert burn_info["reward_breakdown"]["fuel_energy_efficiency"] < coast_info["reward_breakdown"]["fuel_energy_efficiency"]


def main():
    checks = [
        check_reset_reproducibility,
        check_action_and_fuel_constraints,
        check_terminal_and_reward_breakdown,
        check_energy_env_preserves_physics_and_reports_energy,
        check_energy_reward_prefers_early_coast_to_early_burn,
    ]
    passed = []
    for check in checks:
        check()
        passed.append(check.__name__)
        print(f"PASS {check.__name__}")
    output = Path("results/reproducible/smoke_tests.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"passed": passed}, f, indent=2)
    print(f"Saved smoke test report to: {output}")


if __name__ == "__main__":
    main()
