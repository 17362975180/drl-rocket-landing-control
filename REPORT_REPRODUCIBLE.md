# 火箭垂直软着陆控制 PPO 可复现实验报告

本报告对应项目目录 `C:/Users/ycy/Desktop/drl-control-project`。所有实验结论均来自本地真实运行生成的 JSON/CSV/PNG 文件，核心汇总见 `results/reproducible/verified_summary.json` 与 `results/reproducible/VERIFIED_RESULTS.md`。最终验证脚本 `verify_reproducible_outputs.py` 已通过全部检查。

## 摘要

本项目研究一维火箭垂直软着陆控制问题：智能体需要控制连续油门，使火箭从约 50 m 高度下降并安全触地，同时降低燃料消耗、终端速度、控制抖动和失败率。项目完成了基础 PPO、燃料与安全约束、泛化与鲁棒性测试、传统控制器对比、奖励函数消融、多算法对比，以及新增的纯能量诱导 PPO 策略。

主线 PPO 在标准 100 次随机测试中达到 100% 成功率，超过课程要求的 70%。进一步提出的 Pure Energy-Guided PPO 在不改变物理环境、不使用课程训练、不加载旧模型微调的条件下，从零训练达到 100% 成功率，并将平均着陆时间从 baseline PPO 的 7.293 s 降至 4.716 s，平均燃料消耗从 4.406 kg 降至 3.015 kg，终端速度误差从 1.163 m/s 降至 0.299 m/s。

## 1. 问题背景

火箭软着陆控制具有连续状态、连续动作、非线性动力学、终端安全约束和燃料资源约束。控制目标包括：

- 最终高度接近 0 m；
- 触地速度满足 `|v| <= 2 m/s`；
- 尽量节省燃料；
- 控制输入尽量平滑；
- 避免坠毁、越界、速度或加速度超限；
- 在随机初始条件和扰动下保持稳定。

该问题适合使用 PPO 等连续控制强化学习算法，也适合与 PID、MPC、SAC、TD3 等方法进行横向比较。

## 2. 动力学模型

环境实现位于 `envs/rocket_env.py`。状态变量包括高度 `h`、速度 `v`、剩余燃料 `fuel` 和当前推力 `T`。动作 `a in [-1, 1]` 映射为油门：

```text
throttle = (a + 1) / 2
T_target = throttle * T_max
```

主要物理参数如下：

| 参数 | 数值 |
|---|---:|
| 干质量 | 10 kg |
| 初始燃料 | 5 kg |
| 最大推力 | 300 N |
| 排气速度 | 200 m/s |
| 重力加速度 | 9.81 m/s^2 |
| 时间步长 | 0.05 s |
| 推力惯性时间常数 | 0.05 s |
| 阻力系数 | 0.02 |

离散更新公式为：

```text
fuel_consumed = T / exhaust_v * dt
mass = dry_mass + fuel_remaining
drag = -drag_coeff * v * |v|
acceleration = (T + drag - mass * g) / mass
v_next = v + acceleration * dt
h_next = h + v_next * dt
```

燃料耗尽后推力强制为 0。环境记录终止原因、最大速度、最大加速度、燃料消耗、轨迹和奖励分解。

## 3. 强化学习建模

基础 PPO 的观测空间为 4 维归一化向量：

| 维度 | 含义 | 归一化 |
|---|---|---|
| 0 | 高度 | `h / 50` |
| 1 | 速度 | `v / 10` |
| 2 | 剩余燃料 | `fuel / 5` |
| 3 | 当前推力 | `T / 300` |

动作空间为连续一维 `[-1, 1]`，映射到油门 `[0, 1]`。标准测试协议为 100 个随机 episode：

```text
initial_height in [45, 55] m
initial_velocity in [-1, 1] m/s
dry_mass = 10 kg
initial_fuel = 5 kg
gravity_scale = 1
thrust_scale = 1
sensor_noise = 0
action_delay_steps = 0
```

终止条件如下：

| 条件 | 终止原因 |
|---|---|
| `h <= 0` 且 `|v| <= 2` | success |
| `h <= 0` 且 `|v| > 2` | crash |
| `h > 120` | out_of_bounds |
| `|v| > 50` | velocity_exceeded |
| `|acceleration| > 35` | acceleration_exceeded |
| `time > 100 s` | timeout |

## 4. PPO 算法与训练配置

PPO 使用裁剪策略梯度目标限制策略更新幅度：

```text
L_clip(theta) = E[min(r_t(theta) A_t, clip(r_t(theta), 1-epsilon, 1+epsilon) A_t)]
```

其中 `r_t(theta)` 是新旧策略概率比，`A_t` 是优势函数。本项目使用 Stable-Baselines3 PPO，并保存模型与 `VecNormalize` 统计，保证训练和评估时观测归一化一致。

主线训练入口：

```powershell
python train.py --run-name main --total-steps 500000 --eval-interval 10000 --eval-episodes 20
```

主线复评入口：

```powershell
python evaluate.py --model results/sweeps/v2_reward/models/final_model.zip --stats results/sweeps/v2_reward/models/vec_normalize.pkl --n-episodes 100 --output-dir results/reproducible/baseline_existing_model_eval_100 --save-trajectories
```

## 5. 实验 1：PPO 主线训练与标准测试

结果文件：`results/reproducible/baseline_existing_model_eval_100/eval_results.json`

| 指标 | 结果 |
|---|---:|
| 测试次数 | 100 |
| 成功率 | 100.0% |
| 坠毁率 | 0.0% |
| 平均终端速度误差 | 1.117 m/s |
| 平均燃料消耗 | 4.456 kg |
| 平均动作变化量 | 0.0146 |
| 终止原因 | `{'success': 100}` |

结论：基础 PPO 已稳定达到课程要求。`eval_episodes.csv` 中初始高度和初始速度均为随机采样，不是重复同一初始条件。

主要图表：

- 训练曲线：`results/reproducible/figures/training_curves.png`
- 成功轨迹：`results/reproducible/baseline_existing_model_figures/success_trajectory.png`
- 标准评估统计：`results/reproducible/baseline_existing_model_figures/eval_summary.png`
- 演示动画：`results/reproducible/landing_demo.gif`

## 6. 实验 2：燃料模型与安全约束增强

安全机制实现位于 `envs/rocket_env_safe.py`，测试入口为 `test_safety_mechanism.py`。该实验比较无安全屏蔽和有安全屏蔽 PPO。

结果文件：`results/reproducible/safety_comparison.json`

| 变体 | 成功率 | 坠毁率 | 平均燃料 | 动作变化量 | 安全介入率 |
|---|---:|---:|---:|---:|---:|
| without_safety | 100.0% | 0.0% | 4.435 kg | 0.0146 | 0.000 |
| with_safety | 100.0% | 0.0% | 4.437 kg | 0.0146 | 0.0086 |

结论：标准场景下 PPO 本身已较稳定，安全屏蔽只少量触发，因此没有显著改变成功率和燃料消耗。其价值主要体现在扰动、低空高速或异常动作场景下的保护。

轨迹图：`results/reproducible/trajectory_comparisons/safety_trajectory_comparison.png`

## 7. 实验 3：泛化与鲁棒性测试

统一场景定义位于 `experiment_utils.ROBUSTNESS_SCENARIOS`。测试入口统一为：

```powershell
python robustness_full_test.py ...
```

场景分为两类：

- 泛化场景：`standard`、`random_height`、`random_velocity`、`random_mass`、`random_fuel`
- 鲁棒场景：`gravity_bias`、`thrust_bias`、`sensor_noise`、`action_delay_1`、`action_delay_2`、`combined`

基础 PPO 结果文件：`results/reproducible/robustness_full.json`

| 场景 | 成功率 | 坠毁率 | 终端速度误差 | 燃料消耗 |
|---|---:|---:|---:|---:|
| standard | 100.0% | 0.0% | 1.219 m/s | 4.347 kg |
| random_height | 48.0% | 52.0% | 2.372 m/s | 4.320 kg |
| random_velocity | 100.0% | 0.0% | 1.070 m/s | 4.503 kg |
| random_mass | 98.0% | 2.0% | 1.410 m/s | 4.224 kg |
| random_fuel | 20.0% | 80.0% | 4.090 m/s | 4.000 kg |
| gravity_bias | 98.0% | 2.0% | 1.302 m/s | 4.309 kg |
| thrust_bias | 100.0% | 0.0% | 1.124 m/s | 4.440 kg |
| sensor_noise | 90.0% | 10.0% | 1.323 m/s | 4.419 kg |
| action_delay_1 | 100.0% | 0.0% | 1.076 m/s | 4.493 kg |
| action_delay_2 | 100.0% | 0.0% | 1.022 m/s | 4.543 kg |
| combined | 40.0% | 60.0% | 3.498 m/s | 4.265 kg |

结论：基础 PPO 对速度扰动、推力偏差和动作延迟较稳健，但对大范围高度、燃料变化和综合扰动较敏感。失败主要表现为触地速度超过安全阈值。

图表：

- 鲁棒统计：`results/reproducible/figures/robustness_results.png`
- 泛化轨迹：`results/reproducible/trajectory_comparisons/generalization_trajectory_comparison.png`
- 鲁棒总览轨迹：`results/reproducible/trajectory_comparisons/robustness_all_scenarios_trajectory_comparison.png`

## 8. 实验 4：传统控制器对比

入口：`controller_comparison_full.py`

结果文件：`results/reproducible/controller_comparison.json`

| 控制器 | 成功率 | 平均燃料 | 动作变化量 | 终端速度误差 |
|---|---:|---:|---:|---:|
| PPO | 100.0% | 4.391 kg | 0.0148 | 1.178 m/s |
| PID | 100.0% | 4.238 kg | 0.0564 | 1.682 m/s |
| MPC | 0.0% | 5.000 kg | 0.0144 | 4.985 m/s |
| ET-MPC | 0.0% | 1.824 kg | 0.0001 | 19.923 m/s |

结论：PPO 与 PID 都能完成标准随机测试。PID 燃料略低，但动作变化量约为 PPO 的 3.8 倍，控制更抖。当前轻量 MPC/ET-MPC 未充分优化，作为传统控制失败基线说明 MPC 对代价函数、预测窗口和优化器质量更敏感。

轨迹图：`results/reproducible/trajectory_comparisons/controller_trajectory_comparison.png`

## 9. 实验 5：奖励函数消融

入口：`reward_ablation.py`

结果文件：`results/reproducible/ablation/reward_ablation.json`

| 奖励变体 | 成功率 | 坠毁率 | 平均燃料 | 终端速度误差 |
|---|---:|---:|---:|---:|
| full | 100.0% | 0.0% | 4.416 kg | 1.156 m/s |
| no_fuel | 100.0% | 0.0% | 4.416 kg | 1.283 m/s |
| no_smooth | 100.0% | 0.0% | 4.409 kg | 1.421 m/s |
| no_safety | 90.0% | 10.0% | 4.399 kg | 1.571 m/s |
| no_success | 100.0% | 0.0% | 4.485 kg | 1.089 m/s |
| basic | 0.0% | 100.0% | 5.000 kg | 28.142 m/s |

结论：只保留基础高度/速度误差的 `basic` 奖励完全失败；去掉安全项会明显降低成功率；去掉燃料或平滑项仍可落地，但终端速度误差变大。完整奖励函数更均衡。

进一步对所有消融策略进行泛化/鲁棒测试，结果位于：

- `results/reproducible/ablation_scenarios/ablation_generalization.json`
- `results/reproducible/ablation_scenarios/ablation_robustness.json`
- `results/reproducible/ablation_scenarios/ablation_scenario_summary.csv`
- `results/reproducible/trajectory_comparisons/ablation_scenarios/`

关键结论：`no_fuel` 在随机燃料场景明显退化；`no_safety` 在质量变化、噪声和综合扰动下退化明显；`basic` 在所有场景均失败。

## 10. 实验 6：多算法对比

入口：`rl_comparison.py`

结果文件：`results/reproducible/rl_comparison/rl_comparison.json`

| 算法 | 成功率 | 平均燃料 | 动作变化量 | 终端速度误差 |
|---|---:|---:|---:|---:|
| PPO | 100.0% | 4.421 kg | 0.0147 | 1.150 m/s |
| SAC | 100.0% | 3.169 kg | 0.0498 | 1.687 m/s |
| TD3 | 30.0% | 3.090 kg | 0.4529 | 2.083 m/s |

结论：SAC 在本评估中成功率与 PPO 相同且更省燃料，但终端速度误差更大、动作更不平滑。TD3 燃料较低但成功率不足，说明其策略更激进或训练稳定性不足。

轨迹图：`results/reproducible/trajectory_comparisons/rl_algorithm_trajectory_comparison.png`

## 11. 纯能量诱导 PPO 策略

纯能量策略实现位于 `envs/rocket_env_energy.py`，实验入口为 `energy_ppo_experiment.py`。该策略保持 PPO 算法、物理仿真、动作空间、燃料消耗、质量变化和终止条件不变，只替换奖励函数，并额外将能量比值加入观测。

能量建模如下：

```text
E_p = mgh
E_k = 0.5 m v^2
E_m = E_p + E_k
E_f = 0.5 fuel exhaust_v^2
W_brake = max(T_max - mg, 0) h
rho_brake = E_k / W_brake
rho_fuel = E_k / E_fuel_capacity
rho_impact = E_k / E_safe
```

核心思想：

- `rho_brake` 较小时，制动储备充足，应关火滑落，避免过早点火；
- `rho_brake` 接近切换面时，开始集中制动；
- `rho_brake > 1` 时，剩余高度不足以安全制动，给予强惩罚；
- 使用能量处理功率鼓励更快完成安全制动：

```text
P_remove = max(E_mech_prev - E_mech_now, 0) / dt
terminal_efficiency = E_initial / T_landing
```

奖励分解只包含能量项：

```text
coast_efficiency
switch_surface
braking_reserve
fuel_energy_efficiency
energy_power
impact_energy
energy_smoothness
terminal_energy_time
```

正式训练命令：

```powershell
python energy_ppo_experiment.py --output-dir results/reproducible/energy_ppo_from_scratch_time --train-steps 300000 --n-episodes 100 --seed 17000 --force-train
```

注意：该结果是从零训练，不使用课程训练，不加载 `energy_ppo_success` 或 `energy_ppo_time_finetune` 等旧模型。

### 11.1 标准 100 次测试对比

结果文件：`results/reproducible/energy_ppo_from_scratch_time/baseline_vs_pure_energy_ppo.json`

| 策略 | 成功率 | 平均着陆时间 | 平均燃料 | 终端速度误差 |
|---|---:|---:|---:|---:|
| Baseline PPO | 100.0% | 7.293 s | 4.406 kg | 1.163 m/s |
| Pure Energy-Guided PPO | 100.0% | 4.716 s | 3.015 kg | 0.299 m/s |

结论：纯能量 PPO 不仅保持 100% 成功率，而且明显更快、更省燃料、终端速度更小，学习到了接近 suicide-burn 的晚点火制动策略。

图表：

- 轨迹对比：`results/reproducible/energy_ppo_from_scratch_time/baseline_vs_pure_energy_trajectory.png`
- 指标对比：`results/reproducible/energy_ppo_from_scratch_time/baseline_vs_pure_energy_metrics.png`
- 能量会计图：`results/reproducible/energy_ppo_from_scratch_time/pure_energy_accounting.png`
- 奖励分解图：`results/reproducible/energy_ppo_from_scratch_time/pure_energy_reward_breakdown.png`

### 11.2 统一场景泛化与鲁棒对比

统一测试入口：

```powershell
python robustness_full_test.py --n-episodes 50 --seed 18000 --save-trajectories --output-dir results/reproducible/energy_ppo_from_scratch_time/scenarios/standard_comparison --policy "label=Baseline PPO,type=sb3,algo=PPO,env=base,model=results/sweeps/v2_reward/models/final_model.zip,stats=results/sweeps/v2_reward/models/vec_normalize.pkl" --policy "label=Pure Energy PPO,type=sb3,algo=PPO,env=energy,model=results/reproducible/energy_ppo_from_scratch_time/models/pure_energy_ppo_model.zip,stats=results/reproducible/energy_ppo_from_scratch_time/models/pure_energy_vec_normalize.pkl"
```

结果文件：

- `results/reproducible/energy_ppo_from_scratch_time/scenarios/standard_comparison/scenario_comparison.csv`
- `results/reproducible/energy_ppo_from_scratch_time/scenarios/standard_comparison/scenario_comparison.json`
- `results/reproducible/energy_ppo_from_scratch_time/scenarios/standard_comparison/results_by_policy.json`
- `results/reproducible/energy_ppo_from_scratch_time/scenarios/standard_comparison/trajectory_comparisons/`

同 seed、同 50 次/场景成功率如下：

| 场景 | Baseline PPO | Pure Energy PPO |
|---|---:|---:|
| standard | 100.0% | 100.0% |
| random_height | 48.0% | 100.0% |
| random_velocity | 100.0% | 100.0% |
| random_mass | 94.0% | 100.0% |
| random_fuel | 26.0% | 100.0% |
| gravity_bias | 100.0% | 100.0% |
| thrust_bias | 100.0% | 100.0% |
| sensor_noise | 84.0% | 100.0% |
| action_delay_1 | 100.0% | 100.0% |
| action_delay_2 | 100.0% | 98.0% |
| combined | 40.0% | 100.0% |

结论：纯能量 PPO 在相同场景、相同 seed、相同 episode 数下显著提升泛化和鲁棒性。主要弱点是两步动作延迟会增大末端速度误差并带来少量坠毁。

## 12. PPO 策略演化总结

本项目中的 PPO 不是一次性完成，而是逐步演化：

1. 基础 PPO：先保证标准随机测试中稳定软着陆。
2. 安全 PPO：加入燃料、质量、安全边界、动作约束和安全屏蔽，降低危险动作风险。
3. 鲁棒 PPO：引入随机初始条件、物理扰动、噪声和延迟，发现策略薄弱场景。
4. 对比 PPO：与 PID、MPC、SAC、TD3 比较，明确 PPO 的稳定性、燃料和控制平滑性特点。
5. 消融 PPO：逐项删除奖励组件，证明成功奖励、安全项、燃料项和平滑项的作用。
6. 纯能量 PPO：从经验奖励进一步演化为物理先验奖励，用能量比值和能量处理功率诱导晚点火、省时、省燃料策略。

核心变化可以概括为：

```text
经验任务奖励 -> 安全约束奖励 -> 鲁棒评估闭环 -> 奖励消融验证 -> 纯能量物理诱导
```

## 13. 失败案例分析

基础 PPO 的主要失败集中在：

- 大范围高度变化：`random_height` 成功率 48%，说明制动时机对高度分布敏感；
- 随机燃料：`random_fuel` 成功率 20%，说明低燃料或高燃料下推力预算不匹配；
- 综合扰动：`combined` 成功率 40%，说明多扰动叠加会破坏原策略的安全裕度；
- 基础奖励消融：`basic` 成功率 0%，说明只依赖高度/速度误差无法形成稳定制动策略。

失败轨迹示例：

- `results/reproducible/failure_case_trajectory.json`
- `results/reproducible/failure_case_figures/trajectory.png`
- `results/reproducible/failure_case_figures/hv_phase.png`

改进方向包括扩大域随机化、加入更强安全屏蔽、改进 MPC 代价函数、引入在线自适应，或使用纯能量诱导奖励提升物理可解释泛化能力。

## 14. 统一测试接口

为避免不同脚本使用不同场景和指标，本项目将泛化/鲁棒测试统一到：

```text
robustness_full_test.py
```

该脚本支持：

- 基础 PPO：`type=sb3,algo=PPO,env=base`
- 纯能量 PPO：`type=sb3,algo=PPO,env=energy`
- SAC / TD3：`type=sb3,algo=SAC/TD3`
- PID / MPC / ET-MPC：`type=controller,controller=PID/MPC/ET-MPC`

所有策略共享同一套 `ROBUSTNESS_SCENARIOS`、同一指标、同一 CSV/JSON 输出和同一 2x2 轨迹对比图格式。

## 15. 可复现性与验证

关键验证命令：

```powershell
python smoke_tests.py
python summarize_reproducible_results.py
python verify_reproducible_outputs.py
```

最终验证结果：

```text
All reproducible deliverables verified.
```

验证内容包括：

- 主线 PPO 100 次标准测试成功率不少于 70%；
- `eval_episodes.csv` 中初始高度和速度确实随机；
- 安全、鲁棒、控制器、消融、多算法、纯能量 PPO 输出文件存在；
- 纯能量 PPO 来自 `energy_ppo_from_scratch_time`，不是旧的 fine-tune 结果；
- 统一场景对比包含 Baseline PPO 和 Pure Energy PPO；
- 纯能量 PPO 在统一场景测试中各场景成功率不少于 70%；
- 所有关键图表非空。

## 16. AI 使用说明

本项目使用 AI 工具辅助完成代码检查、实验脚本整理、报告结构设计、统一接口重构和结果一致性核对。人工确认内容包括：

- Python 3.10 虚拟环境与依赖安装；
- PPO、纯能量 PPO 和各实验脚本真实运行；
- JSON/CSV/PNG/GIF 输出文件存在且可追溯；
- 标准测试、泛化测试、鲁棒测试、消融测试和对比实验的数值来自本地结果文件；
- 报告结论与 `verified_summary.json`、`scenario_comparison.csv` 等结果一致。

不能表述为“本项目完全由 AI 生成”。合理表述为：AI 参与了工程实现、脚本整理和报告初稿生成，实验配置、真实运行结果、图表和最终结论均基于本地运行输出进行人工核对。

## 17. 总结

本项目完成了从基础 PPO 到纯能量诱导 PPO 的完整实验闭环。基础 PPO 已在标准 100 次随机测试中达到 100% 成功率。安全机制、鲁棒性测试、传统控制器对比、奖励消融和多算法对比共同验证了策略的能力边界。

最重要的改进是 Pure Energy-Guided PPO：它不依赖原始高度误差、速度误差或经验成功奖励，而是通过 `rho_brake = E_k / W_brake`、能量处理功率和终端能量-时间效率，引导智能体从零学习晚点火制动策略。最终它在标准测试中达到 100% 成功率，并显著缩短着陆时间、降低燃料消耗和终端速度误差；在统一泛化/鲁棒场景中也明显优于 baseline PPO。

本系统目前已经具备可运行、可训练、可评估、可画图、可复现和可报告的完整能力。
