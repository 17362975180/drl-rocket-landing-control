"""Energy-guided reward variant — backward-compatible wrapper.

This file is kept so that existing ``from rocket_landing_control.envs.rocket_env_energy import
RocketLandingEnergyEnv`` imports continue to work without changes.
The actual implementation lives in :class:`envs.rocket_env.RocketLandingEnv`.
"""

from rocket_landing_control.envs.rocket_env import RocketLandingEnv


class RocketLandingEnergyEnv(RocketLandingEnv):
    """RocketLandingEnv with energy reward mode."""

    def __init__(self, **kwargs):
        kwargs.setdefault("reward_mode", "energy")
        super().__init__(**kwargs)
