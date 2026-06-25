# 正式实验结果汇总

所有统一场景比较均采用每场景 30 回合、配对随机种子 2026；轨迹图固定使用 episode 0，保证不同方法具有相同初始条件。

## Standard PPO 奖励消融

| 方法 | 11 场景平均成功率 |
|---|---:|
| Standard-full | 80.3% |
| Standard-no_fuel | 73.0% |
| Standard-no_smooth | 82.4% |
| Standard-no_safety | 61.2% |
| Standard-no_success | 80.3% |
| Standard-basic | 0.0% |

- 成功率热图：`results/01_standard_ppo/ablation/success_rate_heatmap.png`
- 各场景轨迹：`results/01_standard_ppo/ablation/trajectory_comparisons/`

## Energy PPO 结构消融

| 方法 | 11 场景平均成功率 |
|---|---:|
| Energy PPO | 100.0% |
| Energy-NoEnergyObservation | 0.0% |
| Energy-PreTimeOptimization | 98.5% |

- 成功率热图：`results/02_energy_ppo/ablation/success_rate_heatmap.png`
- 各场景轨迹：`results/02_energy_ppo/ablation/trajectory_comparisons/`

## Standard PPO 与 Energy PPO 对比

| 方法 | 11 场景平均成功率 |
|---|---:|
| Standard PPO | 80.3% |
| Energy PPO | 100.0% |

- 成功率热图：`results/02_energy_ppo/vs_standard/success_rate_heatmap.png`
- 各场景轨迹：`results/02_energy_ppo/vs_standard/trajectory_comparisons/`

## 传统控制算法对比

| 方法 | 11 场景平均成功率 |
|---|---:|
| Standard PPO | 80.3% |
| Energy PPO | 100.0% |
| PID | 97.9% |
| MPC | 9.4% |
| ET-MPC | 0.0% |

- 成功率热图：`results/03_control_comparison/success_rate_heatmap.png`
- 各场景轨迹：`results/03_control_comparison/trajectory_comparisons/`

## 其他强化学习算法对比

| 方法 | 11 场景平均成功率 |
|---|---:|
| Standard PPO | 80.3% |
| Energy PPO | 100.0% |
| SAC | 75.8% |
| TD3 | 34.2% |

- 成功率热图：`results/04_other_rl_comparison/success_rate_heatmap.png`
- 各场景轨迹：`results/04_other_rl_comparison/trajectory_comparisons/`

## 失败案例分析

失败案例包括基础奖励导致的高空燃料耗尽，以及两个主要模型在扩展压力搜索中出现的临界失败。

- 分析说明：`results/05_failure_analysis/FAILURE_ANALYSIS.md`
- 原始轨迹与图像：`results/05_failure_analysis/`
