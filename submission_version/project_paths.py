"""Canonical paths for the clean submission result layout."""

from pathlib import Path


RESULTS = Path("results")
MODELS = RESULTS / "models"
BASELINE_MODELS = MODELS / "baseline_ppo"

EXPERIMENTS = RESULTS / "experiments"
MAIN_EVAL = EXPERIMENTS / "01_main_evaluation"
SAFETY = EXPERIMENTS / "02_safety"
ROBUSTNESS = EXPERIMENTS / "03_robustness"
CONTROLLERS = EXPERIMENTS / "04_controllers"
ABLATION = EXPERIMENTS / "05_ablation"
RL_ALGORITHMS = EXPERIMENTS / "06_rl_algorithms"
ENERGY_PPO = EXPERIMENTS / "07_energy_ppo"

FIGURES = RESULTS / "figures"
OVERVIEW_FIGURES = FIGURES / "overview"
TRAJECTORY_FIGURES = FIGURES / "trajectories"

REPORTS = RESULTS / "reports"
DEMO = RESULTS / "demo"

BASELINE_MODEL = BASELINE_MODELS / "final_model.zip"
BASELINE_STATS = BASELINE_MODELS / "vec_normalize.pkl"


def create_result_directories() -> None:
    for path in (
        BASELINE_MODELS,
        MAIN_EVAL,
        SAFETY,
        ROBUSTNESS,
        CONTROLLERS,
        ABLATION,
        RL_ALGORITHMS,
        ENERGY_PPO,
        OVERVIEW_FIGURES,
        TRAJECTORY_FIGURES,
        REPORTS,
        DEMO,
    ):
        path.mkdir(parents=True, exist_ok=True)
