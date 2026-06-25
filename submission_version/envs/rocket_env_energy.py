"""Energy-guided reward variant — backward-compatible wrapper.

This file is kept so that existing ``from envs.rocket_env_energy import
RocketLandingEnergyEnv`` imports continue to work without changes.
The actual implementation lives in :class:`envs.rocket_env.RocketLandingEnv`.
"""

from envs.rocket_env import RocketLandingEnv


class RocketLandingEnergyEnv(RocketLandingEnv):
    """RocketLandingEnv with energy reward mode."""

    def __init__(self, **kwargs):
        kwargs.setdefault("reward_mode", "energy")
        super().__init__(**kwargs)


class RocketLandingEnergyBaseObsEnv(RocketLandingEnv):
    """Energy reward with the four-dimensional base observation (ablation)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("reward_mode", "energy")
        kwargs.setdefault("energy_observation", False)
        super().__init__(**kwargs)
