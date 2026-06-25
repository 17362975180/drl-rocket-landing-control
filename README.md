# DRL Rocket Landing Control

Deep reinforcement learning project for one-dimensional vertical rocket soft
landing. The project trains and evaluates PPO-based controllers, compares them
with classical control baselines, and includes reproducible evaluation scripts
for robustness, reward ablation, and controller comparison.

## Highlights

- Gymnasium environments for rocket dynamics, fuel limits, thrust inertia, drag,
  and safety constraints.
- Standard PPO and energy-aware PPO training workflows.
- Evaluation scripts for 100-episode standard tests and 11-scenario robustness
  / generalization tests.
- Baseline comparisons against PID, MPC, event-triggered MPC, SAC, and TD3.
- Lightweight verified summaries are kept in the repository; large generated
  experiment artifacts are intentionally excluded from Git.

## Repository Layout

```text
.
|-- envs/                         # Rocket landing environments
|-- configs/                      # PPO configuration files
|-- experiments/                  # Experiment entry points
|-- saved_models/                 # Small reference PPO model and normalization stats
|-- results/reproducible/         # Lightweight verified summaries
|-- submission_version/           # Final course-submission snapshot and report
|-- train.py                      # PPO training entry point
|-- evaluate.py                   # Single-model evaluation
|-- robustness_full_test.py       # Multi-scenario robustness evaluation
|-- run_full_comparison.py        # Full comparison workflow
|-- smoke_tests.py                # Fast environment and physics checks
`-- verify_reproducible_outputs.py
```

The local workspace may contain `tmp/`, `.venv/`, full `results/`, TensorBoard
logs, rendered documents, and large JSON trajectory dumps. These are ignored so
the GitHub repository stays clean and reproducible.

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
python smoke_tests.py
```

The full reproducibility verifier checks generated artifacts under `results/`.
It passes in the complete local research workspace, but a fresh GitHub clone only
contains lightweight summaries. After regenerating or restoring the full result
artifacts, run:

```bash
python verify_reproducible_outputs.py
```

## Training

Train a PPO landing controller:

```bash
python train.py --run-name main --total-steps 500000 --eval-interval 10000 --eval-episodes 20
```

Run a quick evaluation with the included reference model:

```bash
python quick_eval.py --model saved_models/ppo_rocket_v7.zip --n-episodes 10
```

Run the formal 100-episode evaluation:

```bash
python evaluate.py \
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
- `REPORT_REPRODUCIBLE.md`

Large generated outputs are excluded from version control. Re-run the scripts
above to regenerate local artifacts under `results/`.

## Report

The final course report is stored in:

- `submission_version/report/深度强化学习报告.pdf`
- `submission_version/report/深度强化学习报告.docx`

## AI Usage

AI-assisted development and review notes are documented in `AI_USAGE.md`.

## License

This project is released under the MIT License. See `LICENSE` for details.
