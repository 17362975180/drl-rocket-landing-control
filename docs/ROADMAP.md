# Roadmap

This project is already usable as a compact rocket-landing DRL artifact. The
next upgrades should make it easier to discover, run, cite, and extend.

## High-Impact Next Steps

- Package the environment as an installable module with a Gymnasium registration
  entry point, so users can run `gymnasium.make(...)`.
- Add a notebook tutorial that trains a tiny PPO policy, evaluates the included
  checkpoint, and visualizes one trajectory.
- Add a benchmark table generated from JSON artifacts, with success rate,
  landing velocity, fuel use, and runtime for each controller.
- Add a small web or Streamlit demo where visitors can replay trajectories and
  compare PPO, PID, and MPC behavior.
- Publish model cards for the included PPO checkpoint and any future trained
  policies.
- Create pinned GitHub issues for good first contributions: new disturbances,
  new controllers, plotting improvements, and documentation polish.

## Research Extensions

- Extend the environment from 1D vertical landing to 2D landing with lateral
  drift and attitude control.
- Add curriculum learning and domain randomization experiments.
- Compare PPO against newer off-policy and model-based RL baselines.
- Evaluate safety shielding as a formal intervention policy, not only as an
  environment option.
- Track energy-aware reward variants with fixed seeds and generated summary
  tables.

## Documentation Improvements

- Add a diagram of the environment state, action, reward, and termination flow.
- Add a "How to reproduce the paper-style figures" guide.
- Add expected runtime and hardware notes for each experiment tier.
- Add an FAQ for common Windows, PyTorch, and Stable-Baselines3 setup issues.

## Community Improvements

- Add GitHub topics for discoverability.
- Keep issues small and labeled by difficulty.
- Add release tags when the public API or benchmark artifacts change.
- Consider a short project page with the landing GIF, benchmark table, and links
  to the report and code.
