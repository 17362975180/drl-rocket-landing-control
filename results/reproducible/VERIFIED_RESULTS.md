# Verified Reproducible Results

All values below are read from JSON/CSV outputs under `results/reproducible`.

## Experiment 1: PPO Main Line
- Standard randomized test: 100 episodes
- Success rate: 100.0%
- Mean final velocity error: 1.117 m/s
- Mean fuel used: 4.456 kg
- Mean action smoothness metric: 0.0146
- Terminal reasons: `{'success': 100}`

## Experiment 2: Safety Shield
| Variant | Success | Crash | Fuel (kg) | Throttle Delta | Intervention Rate |
|---|---:|---:|---:|---:|---:|
| without_safety | 100.0% | 0.0% | 4.435 | 0.0146 | 0.000 |
| with_safety | 100.0% | 0.0% | 4.437 | 0.0146 | 0.009 |

## Experiment 3: Robustness
| Scenario | Success | Crash | Mean abs(v_final) (m/s) | Fuel (kg) |
|---|---:|---:|---:|---:|
| standard | 100.0% | 0.0% | 1.219 | 4.347 |
| random_height | 48.0% | 52.0% | 2.372 | 4.320 |
| random_velocity | 100.0% | 0.0% | 1.070 | 4.503 |
| random_mass | 98.0% | 2.0% | 1.410 | 4.224 |
| random_fuel | 20.0% | 80.0% | 4.090 | 4.000 |
| gravity_bias | 98.0% | 2.0% | 1.302 | 4.309 |
| thrust_bias | 100.0% | 0.0% | 1.124 | 4.440 |
| sensor_noise | 90.0% | 10.0% | 1.323 | 4.419 |
| action_delay_1 | 100.0% | 0.0% | 1.076 | 4.493 |
| action_delay_2 | 100.0% | 0.0% | 1.022 | 4.543 |
| combined | 40.0% | 60.0% | 3.498 | 4.265 |

## Experiment 4: Controller Comparison
| Controller | Success | Fuel (kg) | Throttle Delta | Mean abs(v_final) (m/s) |
|---|---:|---:|---:|---:|
| PPO | 100.0% | 4.391 | 0.0148 | 1.178 |
| PID | 100.0% | 4.238 | 0.0564 | 1.682 |
| MPC | 0.0% | 5.000 | 0.0144 | 4.985 |
| ET-MPC | 0.0% | 1.824 | 0.0001 | 19.923 |

## Experiment 5: Reward Ablation
| Mode | Success | Crash | Fuel (kg) | Mean abs(v_final) (m/s) |
|---|---:|---:|---:|---:|
| full | 100.0% | 0.0% | 4.416 | 1.156 |
| no_fuel | 100.0% | 0.0% | 4.416 | 1.283 |
| no_smooth | 100.0% | 0.0% | 4.409 | 1.421 |
| no_safety | 90.0% | 10.0% | 4.399 | 1.571 |
| no_success | 100.0% | 0.0% | 4.485 | 1.089 |
| basic | 0.0% | 100.0% | 5.000 | 28.142 |

## Experiment 6: RL Algorithm Comparison
| Algorithm | Success | Fuel (kg) | Throttle Delta | Mean abs(v_final) (m/s) |
|---|---:|---:|---:|---:|
| PPO | 100.0% | 4.421 | 0.0147 | 1.150 |
| SAC | 100.0% | 3.169 | 0.0498 | 1.687 |
| TD3 | 30.0% | 3.090 | 0.4529 | 2.083 |

## Experiment 7: Pure Energy-Guided PPO
| Strategy | Success | Crash | Fuel (kg) | Throttle Delta | Mean abs(v_final) (m/s) |
|---|---:|---:|---:|---:|---:|
| Baseline PPO | 100.0% | 0.0% | 4.406 | 0.0147 | 1.163 |
| Pure Energy-Guided PPO | 100.0% | 0.0% | 3.015 | 0.0408 | 0.299 |

This strategy changes only the reward signal. Physics, observations, actions, fuel consumption, mass variation, and termination conditions remain inherited from the base environment.
Terminal reasons for the pure energy policy: `{'success': 100}`.

## Experiment 5b: Ablation Generalization Matrix
| Mode | standard | random_height | random_velocity | random_mass | random_fuel |
|---|---:|---:|---:|---:|---:|
| full | 100.0% | 46.7% | 100.0% | 83.3% | 26.7% |
| no_fuel | 100.0% | 33.3% | 100.0% | 86.7% | 6.7% |
| no_smooth | 100.0% | 53.3% | 100.0% | 90.0% | 26.7% |
| no_safety | 93.3% | 40.0% | 93.3% | 40.0% | 10.0% |
| no_success | 100.0% | 56.7% | 100.0% | 80.0% | 20.0% |
| basic | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Experiment 5c: Ablation Robustness Matrix
| Mode | standard | gravity_bias | thrust_bias | sensor_noise | action_delay_1 | action_delay_2 | combined |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 100.0% | 100.0% | 100.0% | 93.3% | 100.0% | 100.0% | 46.7% |
| no_fuel | 100.0% | 83.3% | 96.7% | 83.3% | 100.0% | 100.0% | 20.0% |
| no_smooth | 100.0% | 100.0% | 100.0% | 90.0% | 100.0% | 100.0% | 43.3% |
| no_safety | 93.3% | 73.3% | 93.3% | 70.0% | 100.0% | 86.7% | 16.7% |
| no_success | 100.0% | 96.7% | 100.0% | 86.7% | 100.0% | 100.0% | 43.3% |
| basic | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Figures
- training_curves: `results\reproducible\figures\training_curves.png`
- main_trajectory: `results\reproducible\baseline_existing_model_figures\success_trajectory.png`
- failure_trajectory: `results\reproducible\failure_case_figures\trajectory.png`
- main_eval_summary: `results\reproducible\baseline_existing_model_figures\eval_summary.png`
- safety: `results\reproducible\figures\safety_comparison.png`
- safety_trajectory_comparison: `results\reproducible\trajectory_comparisons\safety_trajectory_comparison.png`
- robustness: `results\reproducible\figures\robustness_results.png`
- robustness_all_trajectory_comparison: `results\reproducible\trajectory_comparisons\robustness_all_scenarios_trajectory_comparison.png`
- robustness_initial_conditions: `results\reproducible\trajectory_comparisons\robustness_initial_conditions.png`
- robustness_physics_disturbance: `results\reproducible\trajectory_comparisons\robustness_physics_disturbance.png`
- robustness_noise_delay: `results\reproducible\trajectory_comparisons\robustness_noise_delay.png`
- robustness_combined: `results\reproducible\trajectory_comparisons\robustness_combined.png`
- generalization_trajectory_comparison: `results\reproducible\trajectory_comparisons\generalization_trajectory_comparison.png`
- controller: `results\reproducible\figures\controller_comparison.png`
- controller_trajectory_comparison: `results\reproducible\trajectory_comparisons\controller_trajectory_comparison.png`
- ablation: `results\reproducible\figures\reward_ablation.png`
- ablation_trajectory_comparison: `results\reproducible\trajectory_comparisons\ablation_trajectory_comparison.png`
- ablation_generalization_heatmap: `results\reproducible\ablation_scenarios\ablation_generalization_success_heatmap.png`
- ablation_robustness_heatmap: `results\reproducible\ablation_scenarios\ablation_robustness_success_heatmap.png`
- ablation_generalization_full_trajectory: `results\reproducible\trajectory_comparisons\ablation_scenarios\generalization_full_scenario_trajectory_comparison.png`
- ablation_robustness_full_trajectory: `results\reproducible\trajectory_comparisons\ablation_scenarios\robustness_full_scenario_trajectory_comparison.png`
- ablation_generalization_random_fuel_by_mode: `results\reproducible\trajectory_comparisons\ablation_scenarios\generalization_random_fuel_ablation_trajectory_comparison.png`
- ablation_robustness_combined_by_mode: `results\reproducible\trajectory_comparisons\ablation_scenarios\robustness_combined_ablation_trajectory_comparison.png`
- rl_comparison: `results\reproducible\figures\rl_comparison.png`
- rl_algorithm_trajectory_comparison: `results\reproducible\trajectory_comparisons\rl_algorithm_trajectory_comparison.png`
- pure_energy_accounting: `results\reproducible\energy_ppo_from_scratch_time\pure_energy_accounting.png`
- pure_energy_reward_breakdown: `results\reproducible\energy_ppo_from_scratch_time\pure_energy_reward_breakdown.png`
- baseline_vs_pure_energy_trajectory: `results\reproducible\energy_ppo_from_scratch_time\baseline_vs_pure_energy_trajectory.png`
- baseline_vs_pure_energy_metrics: `results\reproducible\energy_ppo_from_scratch_time\baseline_vs_pure_energy_metrics.png`
- same_seed_scenario_comparison: `results\reproducible\energy_ppo_from_scratch_time\scenarios\standard_comparison\scenario_comparison_metrics.png`
- same_seed_random_height_trajectory: `results\reproducible\energy_ppo_from_scratch_time\scenarios\standard_comparison\trajectory_comparisons\random_height_trajectory_comparison.png`
- demo_animation: `results\reproducible\landing_demo.gif`
