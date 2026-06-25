"""Safety-shielded variant — backward-compatible wrapper.

This file is kept so that existing ``from envs.rocket_env_safe import
RocketLandingEnvSafe`` imports continue to work without changes.
The actual implementation lives in :class:`envs.rocket_env.RocketLandingEnv`.
"""

from envs.rocket_env import RocketLandingEnv


class RocketLandingEnvSafe(RocketLandingEnv):
    """RocketLandingEnv with safety shielding enabled by default."""

    def __init__(self, **kwargs):
        kwargs.setdefault("safety_enabled", True)
        super().__init__(**kwargs)

    def enable_safety(self):
        self.safety_enabled = True

    def disable_safety(self):
        self.safety_enabled = False
