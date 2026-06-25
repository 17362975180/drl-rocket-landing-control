# 深度强化学习火箭垂直软着陆控制

本工程是课程大作业的最终提交版本，研究一维火箭垂直软着陆控制。项目以 **Standard PPO** 和 **Energy PPO** 为主要模型，包含动力学环境、训练模型、标准评价、鲁棒性与泛化测试、奖励/结构消融、传统控制器对比、其他强化学习算法对比及失败案例分析。

## 1. 建议查看顺序

1. 实验报告：`report/深度强化学习报告.pdf`
2. 本说明文件：`README.md`
3. 主要实验结果：`results/`
4. 核心环境代码：`envs/rocket_env.py`、`envs/rocket_env_energy.py`
5. 统一实验入口：`run_study.py`

报告对应的主要结论如下：

- Standard PPO 在标准 100 回合评价中成功率为 100.0%；
- Standard PPO 在统一 11 场景中的平均成功率为 80.3%；
- Energy PPO 在标准 100 回合评价和统一 11 场景正式测试中成功率均为 100.0%；
- 与 Standard PPO 相比，Energy PPO 的平均最终速度误差由 1.174 m/s 降至 0.287 m/s，平均燃料消耗由 4.396 kg 降至 3.005 kg，平均着陆时间由 7.283 s 缩短至 4.707 s。

## 2. 工程结构

```text
submission_version/
├── report/
│   ├── 深度强化学习报告.pdf          # 最终实验报告，建议首先查看
│   └── 深度强化学习报告.docx         # Word 版本
├── envs/
│   ├── rocket_env.py                 # 基础动力学与 Standard PPO 环境
│   └── rocket_env_energy.py          # Energy PPO 能量观测与奖励环境
├── configs/                          # 训练配置
├── experiments/                      # 实验配置说明
├── results/
│   ├── 01_standard_ppo/              # Standard PPO 模型、demo、分析与奖励消融
│   ├── 02_energy_ppo/                # Energy PPO 模型、demo、分析、消融及双模型对比
│   ├── 03_control_comparison/        # PPO 与 PID、MPC、ET-MPC 对比
│   ├── 04_other_rl_comparison/       # PPO 与 SAC、TD3 对比
│   ├── 05_failure_analysis/          # 基础奖励失败与压力失败案例
│   ├── 06_report_alignment/          # 100 回合评价、统一热图及可解释性结果
│   └── FINAL_STUDY_SUMMARY.md        # 实验结果汇总
├── run_study.py                      # 全部正式实验的统一入口
├── AGENTS.md                         # AI Agent 每次任务开始前读取的项目规则
├── robustness_full_test.py           # 统一 11 场景评价程序
├── failure_case_analysis.py          # 失败案例分析
├── report_alignment_analysis.py      # 报告所用补充图表与数据
├── smoke_tests.py                    # 环境与物理约束快速检查
├── verify_reproducible_outputs.py    # 已有正式结果完整性检查
├── setup_env.ps1                     # Windows 环境安装脚本
└── requirements.txt                  # Python 依赖
```

`__pycache__/`、临时日志和本地虚拟环境均不是实验内容，运行时即使重新生成也不影响结果。

## 3. 运行环境

- Windows 10/11
- Python 3.10 或 3.11
- PyTorch 2.0+
- Gymnasium 0.29+
- Stable-Baselines3 2.3+

在 PowerShell 中进入本目录后执行：

```powershell
.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

也可以手动安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 4. 快速检查（无需重新训练）

首先检查环境动力学、随机种子复现、燃料约束和能量奖励：

```powershell
python smoke_tests.py
```

然后检查报告引用的正式结果是否完整，包括 11 个场景、30 回合评价、配对随机种子、模型、热图、轨迹图、失败案例和 100 回合标准评价：

```powershell
python verify_reproducible_outputs.py
```

上述两个命令不会重新训练模型。正式模型和报告所用实验结果已经保存在 `results/` 中。

已有着陆动画可直接查看：

- `results/01_standard_ppo/demo/standard_landing_demo.gif`
- `results/02_energy_ppo/demo/standard_landing_demo.gif`

## 5. 复现实验

### 5.1 分组复现

以下命令按统一协议重新运行指定实验组。每个场景默认测试 30 回合，比较方法共享配对随机种子 `2026`，轨迹图使用相同初始条件。

```powershell
python run_study.py standard_analysis --episodes 30
python run_study.py standard_ablation --episodes 30
python run_study.py energy_analysis --episodes 30
python run_study.py energy_ablation --episodes 30
python run_study.py twin_comparison --episodes 30
python run_study.py control_comparison --episodes 30
python run_study.py rl_comparison --episodes 30
```

补充实验：

```powershell
python run_study.py demos
python run_study.py failure_analysis
python run_study.py report_alignment
```

### 5.2 完整复现

```powershell
python run_study.py all --episodes 30
```

完整复现会依次运行所有正式场景和绘图任务，耗时明显长于快速检查。重新运行会更新 `results/` 中对应的结果文件。

## 6. 统一评价协议

- 共 11 个标准、泛化和鲁棒性场景；
- 每个正式场景测试 30 回合；
- 标准指标对比额外使用 100 回合配对评价；
- 同一场景内所有方法使用相同 episode seed；
- 代表轨迹固定使用 episode 0；
- 成功条件为火箭触地且触地速度绝对值不超过 2 m/s；
- 燃料耗尽后实际推力严格为 0。

11 个场景包括：标准随机、随机高度、随机速度、随机质量、随机燃料、重力偏差、推力偏差、传感器噪声、1 步动作延迟、2 步动作延迟和综合扰动。

## 7. 主要程序说明

| 文件 | 作用 |
|---|---|
| `envs/rocket_env.py` | 动力学、燃料消耗、推力惯性、终止条件和 Standard PPO 奖励 |
| `envs/rocket_env_energy.py` | Energy PPO 的能量观测和能量奖励 |
| `train.py` | PPO 训练入口 |
| `evaluate.py` | 单模型评价入口 |
| `robustness_full_test.py` | 多策略统一 11 场景评价与轨迹绘制 |
| `run_study.py` | 正式实验统一调度入口 |
| `reward_ablation.py` | Standard PPO 奖励消融 |
| `energy_ppo_experiment.py` | Energy PPO 训练与实验 |
| `controller_comparison_full.py` | PID、MPC、ET-MPC 对比 |
| `rl_comparison.py` | SAC、TD3 对比 |
| `failure_case_analysis.py` | 失败样本搜索、分类和绘图 |

## 8. 说明

- 报告中的表格和图片均来自 `results/` 中保存的正式运行结果；
- 所有正式比较采用统一场景、统一回合数和配对随机种子；
- `results/` 中同时保留 PNG、GIF、CSV 和 JSON，便于核对图表与原始统计数据；
- AI 工具、使用方式及人工复核过程见 `AI_USAGE.md`；项目级 Agent 提示词见 `AGENTS.md`。
