# Repository Structure

The repository is organized to keep the GitHub landing page readable while
preserving the original research scripts and reproducibility paths.

## Top-Level Shape

```text
.
|-- README.md                     # Project entry point
|-- LICENSE / CITATION.cff         # Open-source and citation metadata
|-- requirements.txt               # Python dependency list
|-- setup_env.ps1                  # Windows setup helper
|-- configs/                       # PPO and experiment configuration
|-- experiments/                   # Grouped experiment launchers
|-- docs/                          # Roadmap, release and structure notes
|   `-- reports/                   # Long-form experiment reports
|-- rocket_landing_control/        # Source package and runnable modules
|   |-- core/                      # Shared evaluation and reproducibility helpers
|   |-- envs/                      # Gymnasium rocket landing environments
|   |-- studies/                   # Ablations, robustness, and controller comparisons
|   |-- visualization/             # Plotting, animation, and figure generation
|   `-- workflows/                 # Training, evaluation, smoke tests, verification
|-- scripts/                       # Repository maintenance helpers
|-- saved_models/                  # Small reference model artifacts
|-- results/reproducible/          # Lightweight verified public artifacts
`-- submission_version/            # Course-submission snapshot and final report
```

## Design Intent

- Keep the root directory calm. Project metadata, documentation, reproducible
  artifacts, and one source package are enough for the first screen.
- Move long-form project narrative into `docs/reports/`, where reports can
  grow without overwhelming the landing page.
- Keep `rocket_landing_control/envs/` small and focused, because the
  environments are the reusable core of the project.
- Separate runnable workflows, experiment studies, and visualization helpers so
  the package reads like an intentional project instead of an experiment dump.
- Keep generated heavy artifacts out of Git, while preserving small public
  summaries under `results/reproducible/`.
- Preserve the original `submission_version/` as a stable archival snapshot.

## Future Refinement

The next structural upgrade should turn this package layout into an installable
distribution with command entry points, for example:

```text
pip install -e .
rocket-landing-smoke
rocket-landing-evaluate --model saved_models/ppo_rocket_v7.zip
```

That refactor would also allow a Gymnasium registration entry point.
