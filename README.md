# DRL Rocket Landing Control

[![CI](https://github.com/17362975180/drl-rocket-landing-control/actions/workflows/ci.yml/badge.svg)](https://github.com/17362975180/drl-rocket-landing-control/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](requirements.txt)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Rocket%20Landing-green.svg)](rocket_landing_control/envs/rocket_env.py)

Deep reinforcement learning project for one-dimensional vertical rocket soft
landing. The project trains and evaluates PPO-based controllers, compares them
with classical control baselines, and includes reproducible evaluation scripts
for robustness, reward ablation, and controller comparison.

![Rocket landing demo](results/reproducible/landing_demo.gif)

## Why This Project Is Interesting

- A compact rocket soft-landing benchmark that is easy to inspect, modify, and
  run locally.
- Physics constraints are explicit: fuel depletion, thrust lag, drag, mass
  changes, action delay, sensor noise, and safety shielding.
- The repository compares learned controllers with PID, MPC, event-triggered
  MPC, SAC, and TD3 baselines instead of showing only one successful run.
- Reproducibility artifacts are kept small enough for GitHub while still giving
  readers concrete metrics and trajectories to inspect.

## Project Status

This repository is prepared for public release as a compact research/code
artifact. It includes runnable source code, a small reference PPO checkpoint,
selected verified summaries, the final course-submission snapshot, and a
reproducibility report. Large regenerated experiment folders and local scratch
files are intentionally excluded from version control.

## Highlights

- Gymnasium environments for rocket dynamics, fuel limits, thrust inertia, drag,
  and safety constraints.
- Standard PPO and energy-aware PPO training workflows.
- Evaluation scripts for 100-episode standard tests and 11-scenario robustness
  / generalization tests.
- Baseline comparisons against PID, MPC, event-triggered MPC, SAC, and TD3.
- Lightweight verified summaries are kept in the repository; large generated
  experiment artifacts are intentionally excluded from Git.

## Key Results

Representative verified results from the public artifacts:

| Evaluation | Result |
| --- | --- |
| Standard PPO quick reference model | Included in `saved_models/ppo_rocket_v7.zip` |
| Fast smoke checks | Seed reproducibility, fuel constraints, terminal rewards, energy reward behavior |
| Robustness protocol | 11 standard/generalization/disturbance scenarios |
| Public demo | `results/reproducible/landing_demo.gif` |

For the full experimental narrative, see
`docs/reports/REPORT_REPRODUCIBLE.md` and
`results/reproducible/VERIFIED_RESULTS.md`.

## Repository Layout

```text
.
|-- configs/                      # PPO configuration files
|-- docs/                         # Roadmap, release notes, structure guide
|   `-- reports/                  # Long-form experiment reports
|-- experiments/                  # Experiment entry points
|-- rocket_landing_control/       # Source package and runnable modules
|   |-- core/                     # Shared evaluation and reproducibility helpers
|   |-- envs/                     # Rocket landing environments
|   |-- studies/                  # Ablations, robustness, and controller comparisons
|   |-- visualization/            # Plotting, animation, and figure generation
|   `-- workflows/                # Training, evaluation, smoke tests, verification
|-- saved_models/                 # Small reference PPO model and normalization stats
|-- results/reproducible/         # Lightweight verified summaries
|-- scripts/                      # Repository maintenance helpers
|-- submission_version/           # Final course-submission snapshot and report
`-- requirements.txt
```

The local workspace may contain `tmp/`, `.venv/`, full `results/`, TensorBoard
logs, rendered documents, and large JSON trajectory dumps. These are ignored so
the GitHub repository stays clean and reproducible.

## Included Artifacts

The public repository keeps a lightweight set of artifacts:

- `saved_models/ppo_rocket_v7.zip`
- `saved_models/vec_normalize_stats_v7.pkl`
- `results/reproducible/*.json`
- `results/reproducible/VERIFIED_RESULTS.md`
- `results/reproducible/landing_demo.gif`
- `submission_version/report/深度强化学习报告.pdf`
- `submission_version/report/深度强化学习报告.docx`

Full regenerated experiment folders are not tracked. Re-run the relevant
scripts to recreate them locally.

## Installation

Python 3.10 or 3.11 is recommended. PyTorch and Stable-Baselines3 compatibility
with Python 3.13 is not assumed.

Windows PowerShell:

```powershell
.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

Cross-platform manual setup:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Check

Run the fast smoke tests:

```bash
python -m rocket_landing_control.workflows.smoke_tests
```

Run a quick evaluation with the included reference model:

```bash
python -m rocket_landing_control.workflows.quick_eval --model saved_models/ppo_rocket_v7.zip --n-episodes 10
```

The full reproducibility verifier checks generated artifacts under `results/`.
It is meant for the complete local research workspace; a fresh GitHub clone only
contains lightweight summaries and will not include every generated figure,
training run, and trajectory file. After regenerating or restoring the full
result artifacts, run:

```bash
python -m rocket_landing_control.workflows.verify_reproducible_outputs
```

## Training

Train a PPO landing controller:

```bash
python -m rocket_landing_control.workflows.train --run-name main --total-steps 500000 --eval-interval 10000 --eval-episodes 20
```

Run the formal 100-episode evaluation:

```bash
python -m rocket_landing_control.workflows.evaluate \
  --model saved_models/ppo_rocket_v7.zip \
  --stats saved_models/vec_normalize_stats_v7.pkl \
  --n-episodes 100 \
  --output-dir results/local_eval \
  --save-trajectories
```

## Reproducibility Notes

Verified summary artifacts:

- `results/reproducible/VERIFIED_RESULTS.md`
- `results/reproducible/verified_summary.json`
- `results/reproducible/landing_demo.gif`
- `docs/reports/REPORT_REPRODUCIBLE.md`

Large generated outputs are excluded from version control. Re-run the scripts
above to regenerate local artifacts under `results/`.

## Report

The final course report is stored in:

- `submission_version/report/深度强化学习报告.pdf`
- `submission_version/report/深度强化学习报告.docx`

## AI Usage

AI-assisted development and review notes are documented in
`docs/AI_USAGE.md`.

## Structure

The repository structure is documented in `docs/STRUCTURE.md`. The short
version: public-facing documentation lives in `docs/`, source code lives in
`rocket_landing_control/`, experiment orchestration lives in `experiments/`, and
the root stays reserved for project metadata and high-level folders.

## Roadmap

Planned improvements are tracked in `docs/ROADMAP.md`. The most useful next
upgrades are packaging the environment as an installable Gymnasium environment,
adding a notebook tutorial, and publishing richer benchmark tables.

## Release Notes

The release preparation checklist and GitHub publishing commands are documented
in `docs/OPEN_SOURCE_RELEASE.md`.

## License

This project is released under the MIT License. See `LICENSE` for details.
