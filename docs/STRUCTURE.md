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
|-- envs/                          # Gymnasium rocket landing environments
|-- configs/                       # PPO and experiment configuration
|-- experiments/                   # Grouped experiment launchers
|-- docs/                          # Roadmap, release and structure notes
|   `-- reports/                   # Long-form experiment reports
|-- scripts/                       # Repository maintenance helpers
|-- saved_models/                  # Small reference model artifacts
|-- results/reproducible/          # Lightweight verified public artifacts
|-- submission_version/            # Course-submission snapshot and final report
|-- train.py                       # Main PPO training entry point
|-- evaluate.py                    # Model evaluation entry point
|-- quick_eval.py                  # Fast reference-model check
|-- smoke_tests.py                 # Lightweight CI checks
`-- verify_reproducible_outputs.py # Full local artifact verifier
```

## Design Intent

- Keep the root directory useful, not empty. The commands a new visitor is most
  likely to run stay visible.
- Move long-form project narrative into `docs/reports/`, where reports can
  grow without overwhelming the landing page.
- Keep `envs/` small and focused, because the environments are the reusable core
  of the project.
- Keep generated heavy artifacts out of Git, while preserving small public
  summaries under `results/reproducible/`.
- Preserve the original `submission_version/` as a stable archival snapshot.

## Future Refinement

The next structural upgrade should be a package-style layout, for example:

```text
src/rocket_landing_control/
|-- envs/
|-- controllers/
|-- evaluation/
`-- plotting/
```

That refactor would make imports cleaner and allow an installable Gymnasium
registration entry point. It should be done as a dedicated compatibility pass,
because several current research scripts import one another by root-level module
name.
