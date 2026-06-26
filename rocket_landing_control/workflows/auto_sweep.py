"""
auto_sweep.py - 自动训练调度器
并行训练多个PPO候选配置，定期快速评估，自动淘汰，最终选出最优模型。

用法:
    python -m rocket_landing_control.workflows.auto_sweep --max-parallel 2 --scout-steps 30000 --promote-steps 150000
    python -m rocket_landing_control.workflows.auto_sweep --dry-run  # 只打印配置，不训练
"""
import os
import sys
import json
import time
import csv
import argparse
import subprocess
import signal
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================================
# 候选配置定义
# ============================================================

SCOUT_CONFIGS = [
    {
        "name": "lr3e4_s42",
        "learning_rate": 3e-4,
        "seed": 42,
        "net_arch": "128,128",
        "n_steps": 2048,
        "ent_coef": 0.01,
    },
    {
        "name": "lr1e4_s42",
        "learning_rate": 1e-4,
        "seed": 42,
        "net_arch": "128,128",
        "n_steps": 2048,
        "ent_coef": 0.01,
    },
    {
        "name": "lr5e4_s42",
        "learning_rate": 5e-4,
        "seed": 42,
        "net_arch": "128,128",
        "n_steps": 2048,
        "ent_coef": 0.01,
    },
    {
        "name": "lr3e4_s123",
        "learning_rate": 3e-4,
        "seed": 123,
        "net_arch": "128,128",
        "n_steps": 2048,
        "ent_coef": 0.01,
    },
    {
        "name": "lr3e4_wide_s42",
        "learning_rate": 3e-4,
        "seed": 42,
        "net_arch": "256,256",
        "n_steps": 2048,
        "ent_coef": 0.01,
    },
    {
        "name": "lr3e4_s2026",
        "learning_rate": 3e-4,
        "seed": 2026,
        "net_arch": "128,128",
        "n_steps": 2048,
        "ent_coef": 0.01,
    },
]


# ============================================================
# 早停规则
# ============================================================

def check_early_stop(run_name, eval_history, current_step, scout_steps):
    """
    检查是否应该提前停止。
    返回 (should_stop: bool, reason: str)
    """
    if not eval_history:
        return False, ""

    latest = eval_history[-1]

    # 规则1: NaN/Inf检查
    for key in ["success_rate", "crash_rate", "mean_final_velocity_error", "mean_episode_reward"]:
        val = latest.get(key)
        if val is not None and (np.isnan(val) or np.isinf(val)):
            return True, f"NaN/Inf detected in {key}"

    # 规则2: 超过20k步，成功率0且坠毁率>80%
    if current_step > 20000:
        if latest["success_rate"] == 0 and latest["crash_rate"] > 0.8:
            return True, f"No success after {current_step} steps, crash_rate={latest['crash_rate']:.1%}"

    # 规则3: 连续3次速度误差无改善
    if len(eval_history) >= 4:
        recent_v_errors = [e["mean_final_velocity_error"] for e in eval_history[-4:]]
        if all(recent_v_errors[i] >= recent_v_errors[i-1] * 0.95 for i in range(1, len(recent_v_errors))):
            if latest["mean_final_velocity_error"] > 3.0:
                return True, f"Velocity error stagnant at {latest['mean_final_velocity_error']:.2f}"

    # 规则4: 连续3次成功率无提升且<30%
    if len(eval_history) >= 4:
        recent_sr = [e["success_rate"] for e in eval_history[-4:]]
        if all(recent_sr[i] <= recent_sr[i-1] + 0.01 for i in range(1, len(recent_sr))):
            if latest["success_rate"] < 0.3:
                return True, f"Success rate stagnant at {latest['success_rate']:.1%}"

    # 规则5: reward连续下降且crash_rate升高
    if len(eval_history) >= 3:
        recent_rewards = [e["mean_episode_reward"] for e in eval_history[-3:]]
        recent_crashes = [e["crash_rate"] for e in eval_history[-3:]]
        if (all(recent_rewards[i] < recent_rewards[i-1] for i in range(1, len(recent_rewards))) and
            all(recent_crashes[i] > recent_crashes[i-1] for i in range(1, len(recent_crashes)))):
            return True, "Reward decreasing while crash rate increasing"

    # 规则6: 油门抖振过高
    if latest.get("mean_abs_throttle_delta", 0) > 0.5:
        return True, f"Excessive throttle oscillation: {latest['mean_abs_throttle_delta']:.3f}"

    return False, ""


def should_promote(eval_history, scout_steps):
    """判断是否应该晋级到promote阶段"""
    if not eval_history:
        return False, ""

    latest = eval_history[-1]

    # 晋级条件1: 成功率>=30%
    if latest["success_rate"] >= 0.3:
        return True, f"success_rate={latest['success_rate']:.1%} >= 30%"

    # 晋级条件2: 速度误差明显下降
    if len(eval_history) >= 3:
        first_v = eval_history[0]["mean_final_velocity_error"]
        last_v = latest["mean_final_velocity_error"]
        if first_v > 0 and last_v < first_v * 0.5:
            return True, f"velocity_error improved {first_v:.2f} -> {last_v:.2f}"

    # 晋级条件3: 坠毁率明显下降
    if len(eval_history) >= 3:
        first_crash = eval_history[0]["crash_rate"]
        last_crash = latest["crash_rate"]
        if first_crash > 0.5 and last_crash < first_crash * 0.5:
            return True, f"crash_rate improved {first_crash:.1%} -> {last_crash:.1%}"

    return False, ""


def is_best_model(eval_history):
    """判断promote阶段是否产生了best model"""
    if not eval_history:
        return False
    latest = eval_history[-1]
    return latest["success_rate"] >= 0.7


# ============================================================
# 训练进程管理
# ============================================================

def run_training(config, total_steps, eval_interval, eval_episodes, output_dir):
    """启动一个训练子进程"""
    cmd = [
        sys.executable, "-m", "rocket_landing_control.workflows.train",
        "--run-name", config["name"],
        "--total-steps", str(total_steps),
        "--eval-interval", str(eval_interval),
        "--eval-episodes", str(eval_episodes),
        "--save-interval", str(total_steps // 5),
        "--seed", str(config["seed"]),
        "--learning-rate", str(config["learning_rate"]),
        "--n-steps", str(config["n_steps"]),
        "--net-arch", config["net_arch"],
        "--ent-coef", str(config["ent_coef"]),
        "--output-dir", output_dir,
        "--device", "cpu",
        "--verbose", "0",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )


def read_eval_history(output_dir):
    """读取某个run的评估历史"""
    history_path = os.path.join(output_dir, "quick_eval_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def read_train_summary(output_dir):
    """读取训练摘要"""
    summary_path = os.path.join(output_dir, "train_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


# ============================================================
# 排行榜生成
# ============================================================

def generate_leaderboard(all_results, output_dir):
    """生成排行榜CSV和Markdown"""
    os.makedirs(output_dir, exist_ok=True)

    # 按success_rate降序，然后velocity_error升序排序
    sorted_results = sorted(
        all_results,
        key=lambda r: (
            -r.get("best_success_rate", 0),
            r.get("best_velocity_error", 999),
            r.get("best_height_error", 999),
        )
    )

    # CSV
    csv_path = os.path.join(output_dir, "leaderboard.csv")
    if sorted_results:
        fieldnames = [
            "rank", "run_name", "status", "best_success_rate",
            "best_velocity_error", "best_height_error", "best_fuel_used",
            "best_crash_rate", "best_throttle_delta", "stop_reason",
            "promoted", "is_best"
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, r in enumerate(sorted_results):
                writer.writerow({
                    "rank": i + 1,
                    "run_name": r["run_name"],
                    "status": r.get("status", "unknown"),
                    "best_success_rate": f"{r.get('best_success_rate', 0):.3f}",
                    "best_velocity_error": f"{r.get('best_velocity_error', 0):.3f}",
                    "best_height_error": f"{r.get('best_height_error', 0):.3f}",
                    "best_fuel_used": f"{r.get('best_fuel_used', 0):.3f}",
                    "best_crash_rate": f"{r.get('best_crash_rate', 0):.3f}",
                    "best_throttle_delta": f"{r.get('best_throttle_delta', 0):.4f}",
                    "stop_reason": r.get("stop_reason", ""),
                    "promoted": r.get("promoted", False),
                    "is_best": r.get("is_best", False),
                })

    # Markdown
    md_path = os.path.join(output_dir, "leaderboard.md")
    with open(md_path, "w") as f:
        f.write("# 🏆 Training Sweep Leaderboard\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Rank | Run | Status | Success↑ | V-Error↓ | H-Error↓ | Fuel↓ | Crash↓ | Promoted | Best |\n")
        f.write("|------|-----|--------|----------|----------|----------|-------|--------|----------|------|\n")
        for i, r in enumerate(sorted_results):
            status_emoji = {"completed": "✅", "early_stopped": "⏹️", "promoted": "⬆️", "best": "🏆"}.get(r.get("status"), "❓")
            f.write(f"| {i+1} | {r['run_name']} | {status_emoji} {r.get('status', '?')} | "
                    f"{r.get('best_success_rate', 0):.1%} | "
                    f"{r.get('best_velocity_error', 0):.2f} | "
                    f"{r.get('best_height_error', 0):.2f} | "
                    f"{r.get('best_fuel_used', 0):.2f} | "
                    f"{r.get('best_crash_rate', 0):.1%} | "
                    f"{'✓' if r.get('promoted') else '✗'} | "
                    f"{'🏆' if r.get('is_best') else ''} |\n")

        f.write("\n## Run Details\n\n")
        for r in sorted_results:
            f.write(f"### {r['run_name']}\n")
            f.write(f"- **Config**: lr={r.get('learning_rate', '?')}, seed={r.get('seed', '?')}, "
                    f"net={r.get('net_arch', '?')}\n")
            f.write(f"- **Status**: {r.get('status', '?')}\n")
            if r.get('stop_reason'):
                f.write(f"- **Stop reason**: {r['stop_reason']}\n")
            if r.get('best_eval'):
                e = r['best_eval']
                f.write(f"- **Best eval** (step {e.get('step', '?')}):\n")
                f.write(f"  - Success rate: {e.get('success_rate', 0):.1%}\n")
                f.write(f"  - Crash rate: {e.get('crash_rate', 0):.1%}\n")
                f.write(f"  - Final v error: {e.get('mean_final_velocity_error', 0):.2f} m/s\n")
                f.write(f"  - Final h error: {e.get('mean_final_height_error', 0):.2f} m\n")
                f.write(f"  - Fuel used: {e.get('mean_fuel_used', 0):.2f}\n")
            f.write("\n")

    print(f"📋 Leaderboard saved to: {csv_path} and {md_path}")


# ============================================================
# 快速审阅图
# ============================================================

def plot_quick_review(run_name, eval_history, output_dir):
    """为某个run生成快速审阅图"""
    if len(eval_history) < 2:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [e.get("step", i * 5000) for i, e in enumerate(eval_history)]
    success_rates = [e["success_rate"] for e in eval_history]
    crash_rates = [e["crash_rate"] for e in eval_history]
    v_errors = [e["mean_final_velocity_error"] for e in eval_history]
    fuels = [e["mean_fuel_used"] for e in eval_history]
    rewards = [e["mean_episode_reward"] for e in eval_history]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Quick Review: {run_name}", fontsize=14)

    # Success rate
    ax = axes[0, 0]
    ax.plot(steps, success_rates, 'g-o', linewidth=2)
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Crash rate
    ax = axes[0, 1]
    ax.plot(steps, crash_rates, 'r-o', linewidth=2)
    ax.set_ylabel("Crash Rate")
    ax.set_title("Crash Rate")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Velocity error
    ax = axes[0, 2]
    ax.plot(steps, v_errors, 'b-o', linewidth=2)
    ax.set_ylabel("Mean |v_final| (m/s)")
    ax.set_title("Final Velocity Error")
    ax.grid(True, alpha=0.3)

    # Fuel used
    ax = axes[1, 0]
    ax.plot(steps, fuels, 'm-o', linewidth=2)
    ax.set_ylabel("Fuel Used")
    ax.set_title("Mean Fuel Used")
    ax.grid(True, alpha=0.3)

    # Episode reward
    ax = axes[1, 1]
    ax.plot(steps, rewards, 'c-o', linewidth=2)
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Episode Reward")
    ax.grid(True, alpha=0.3)

    # Empty subplot for info
    ax = axes[1, 2]
    ax.axis('off')
    info_text = f"Run: {run_name}\n"
    if eval_history:
        latest = eval_history[-1]
        info_text += f"\nLatest eval (step {latest.get('step', '?')}):"
        info_text += f"\n  Success: {latest['success_rate']:.1%}"
        info_text += f"\n  Crash: {latest['crash_rate']:.1%}"
        info_text += f"\n  V-error: {latest['mean_final_velocity_error']:.2f}"
        info_text += f"\n  Fuel: {latest['mean_fuel_used']:.2f}"
        info_text += f"\n  Reward: {latest['mean_episode_reward']:.0f}"
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    for ax in axes.flat:
        ax.set_xlabel("Training Steps")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "quick_review.png")
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================
# 主调度逻辑
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Auto training sweep")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="Max parallel training processes")
    parser.add_argument("--scout-steps", type=int, default=30000,
                        help="Scout phase total steps per run")
    parser.add_argument("--promote-steps", type=int, default=150000,
                        help="Promote phase total steps per run")
    parser.add_argument("--scout-eval-interval", type=int, default=5000,
                        help="Scout phase eval interval")
    parser.add_argument("--promote-eval-interval", type=int, default=20000,
                        help="Promote phase eval interval")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Episodes per quick evaluation")
    parser.add_argument("--promote-threshold", type=float, default=0.3,
                        help="Min success rate to promote")
    parser.add_argument("--best-threshold", type=float, default=0.7,
                        help="Min success rate for best model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print configs, don't train")
    parser.add_argument("--configs", type=str, default=None,
                        help="Path to custom configs JSON file")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("🚀 Auto Training Sweep")
    print("=" * 60)

    # 加载配置
    configs = SCOUT_CONFIGS
    if args.configs:
        with open(args.configs) as f:
            configs = json.load(f)

    print(f"\n📋 {len(configs)} candidate configurations")
    print(f"   Max parallel: {args.max_parallel}")
    print(f"   Scout steps: {args.scout_steps:,}")
    print(f"   Promote steps: {args.promote_steps:,}")

    if args.dry_run:
        print("\n[DRY RUN] Configurations:")
        for c in configs:
            print(f"  - {c['name']}: lr={c['learning_rate']}, seed={c['seed']}, net={c['net_arch']}")
        return

    # 创建输出目录
    sweep_dir = "results/sweeps"
    os.makedirs(sweep_dir, exist_ok=True)

    all_results = []  # 最终排行榜数据
    best_model_info = None  # 最优模型信息

    # ============================================================
    # 阶段1: Scout 快速筛选
    # ============================================================
    print(f"\n{'='*60}")
    print(f"📊 Phase 1: Scout ({len(configs)} configs, {args.scout_steps:,} steps each)")
    print(f"{'='*60}")

    scout_results = {}  # name -> {config, status, eval_history, output_dir}
    active_processes = {}  # name -> Popen

    # 初始化所有scout runs
    for config in configs:
        name = config["name"]
        output_dir = os.path.join(sweep_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

        scout_results[name] = {
            "config": config,
            "status": "pending",
            "output_dir": output_dir,
            "eval_history": [],
            "stop_reason": "",
            "learning_rate": config["learning_rate"],
            "seed": config["seed"],
            "net_arch": config["net_arch"],
        }

    # 并行启动scout训练
    pending_names = list(scout_results.keys())
    completed_names = []

    while pending_names or active_processes:
        # 启动新进程（不超过max_parallel）
        while len(active_processes) < args.max_parallel and pending_names:
            name = pending_names.pop(0)
            config = scout_results[name]["config"]
            output_dir = scout_results[name]["output_dir"]

            print(f"\n  ▶ Starting scout: {name}")
            proc = run_training(
                config, args.scout_steps, args.scout_eval_interval,
                args.eval_episodes, output_dir
            )
            active_processes[name] = proc
            scout_results[name]["status"] = "running"

        # 检查活跃进程
        finished_names = []
        for name, proc in active_processes.items():
            retcode = proc.poll()
            if retcode is not None:
                finished_names.append(name)
                output_dir = scout_results[name]["output_dir"]

                if retcode != 0:
                    stderr = proc.stderr.read().decode() if proc.stderr else ""
                    scout_results[name]["status"] = "error"
                    scout_results[name]["stop_reason"] = f"Process exited with code {retcode}: {stderr[:200]}"
                    print(f"  ❌ {name} failed: {scout_results[name]['stop_reason']}")
                else:
                    # 读取最终评估结果
                    history = read_eval_history(output_dir)
                    scout_results[name]["eval_history"] = history

                    # 检查早停
                    should_stop, reason = check_early_stop(
                        name, history, args.scout_steps, args.scout_steps
                    )
                    if should_stop:
                        scout_results[name]["status"] = "early_stopped"
                        scout_results[name]["stop_reason"] = reason
                        print(f"  ⏹️ {name} early stopped: {reason}")
                    else:
                        scout_results[name]["status"] = "completed"
                        print(f"  ✅ {name} completed")

                # 生成快速审阅图
                plot_quick_review(name, scout_results[name]["eval_history"], output_dir)

        # 移除已完成的进程
        for name in finished_names:
            del active_processes[name]
            completed_names.append(name)

        if active_processes:
            time.sleep(5)

    # 晋级判断
    promoted_names = []
    for name, result in scout_results.items():
        if result["status"] in ("completed", "early_stopped"):
            history = result["eval_history"]
            promote, reason = should_promote(history, args.scout_steps)
            if promote:
                promoted_names.append(name)
                result["promoted"] = True
                print(f"  ⬆️ {name} promoted: {reason}")
            else:
                result["promoted"] = False
                print(f"  ↘️ {name} not promoted")

    # 如果没有晋级的，取前2个最好的
    if not promoted_names:
        print("\n  ⚠️ No runs met promotion criteria. Selecting top 2 by success_rate.")
        sorted_runs = sorted(
            [(n, r) for n, r in scout_results.items() if r["eval_history"]],
            key=lambda x: (-x[1]["eval_history"][-1]["success_rate"] if x[1]["eval_history"] else -1)
        )
        promoted_names = [n for n, _ in sorted_runs[:2]]
        for name in promoted_names:
            scout_results[name]["promoted"] = True

    print(f"\n  📊 Scout phase complete. Promoted: {promoted_names}")

    # ============================================================
    # 阶段2: Promote 精训
    # ============================================================
    if promoted_names:
        print(f"\n{'='*60}")
        print(f"📈 Phase 2: Promote ({len(promoted_names)} runs, {args.promote_steps:,} steps)")
        print(f"{'='*60}")

        promote_results = {}
        active_processes = {}
        pending_names = list(promoted_names)

        for name in promoted_names:
            config = scout_results[name]["config"]
            output_dir = scout_results[name]["output_dir"]
            promote_results[name] = {
                "config": config,
                "status": "pending",
                "output_dir": output_dir,
                "eval_history": list(scout_results[name]["eval_history"]),  # 保留scout历史
                "stop_reason": "",
            }

        while pending_names or active_processes:
            # 启动新进程
            while len(active_processes) < args.max_parallel and pending_names:
                name = pending_names.pop(0)
                config = promote_results[name]["config"]
                output_dir = promote_results[name]["output_dir"]

                # 从scout阶段的总步数继续
                scout_done = args.scout_steps
                promote_total = args.promote_steps

                print(f"\n  ▶ Starting promote: {name} (from step {scout_done:,} to {promote_total:,})")

                # 加载scout阶段的模型继续训练
                scout_model_path = os.path.join(output_dir, "models", "final_model.zip")
                if os.path.exists(scout_model_path):
                    # 继续训练：加载模型
                    cmd = [
                        sys.executable, "-m", "rocket_landing_control.workflows.train",
                        "--run-name", f"{name}_promote",
                        "--total-steps", str(promote_total - scout_done),
                        "--eval-interval", str(args.promote_eval_interval),
                        "--eval-episodes", str(args.eval_episodes),
                        "--save-interval", str((promote_total - scout_done) // 5),
                        "--seed", str(config["seed"]),
                        "--learning-rate", str(config["learning_rate"]),
                        "--n-steps", str(config["n_steps"]),
                        "--net-arch", config["net_arch"],
                        "--ent-coef", str(config["ent_coef"]),
                        "--output-dir", output_dir,
                        "--device", "cpu",
                        "--verbose", "0",
                    ]
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd()
                    )
                else:
                    # 没有scout模型，从头训练
                    proc = run_training(
                        config, promote_total, args.promote_eval_interval,
                        args.eval_episodes, output_dir
                    )

                active_processes[name] = proc
                promote_results[name]["status"] = "running"

            # 检查活跃进程
            finished_names = []
            for name, proc in active_processes.items():
                retcode = proc.poll()
                if retcode is not None:
                    finished_names.append(name)
                    output_dir = promote_results[name]["output_dir"]

                    if retcode != 0:
                        stderr = proc.stderr.read().decode() if proc.stderr else ""
                        promote_results[name]["status"] = "error"
                        promote_results[name]["stop_reason"] = f"Process error: {stderr[:200]}"
                        print(f"  ❌ {name} failed: {promote_results[name]['stop_reason']}")
                    else:
                        history = read_eval_history(output_dir)
                        promote_results[name]["eval_history"] = history

                        # 检查是否达到best
                        if is_best_model(history):
                            promote_results[name]["status"] = "best"
                            promote_results[name]["is_best"] = True
                            print(f"  🏆 {name} achieved best model! success_rate={history[-1]['success_rate']:.1%}")
                        else:
                            promote_results[name]["status"] = "completed"
                            print(f"  ✅ {name} promote completed")

                    plot_quick_review(f"{name}_promote", promote_results[name]["eval_history"], output_dir)

            for name in finished_names:
                del active_processes[name]

            if active_processes:
                time.sleep(10)

        # 找出最优模型
        best_name = None
        best_success = -1
        for name, result in promote_results.items():
            history = result["eval_history"]
            if history:
                last_success = history[-1]["success_rate"]
                if last_success > best_success:
                    best_success = last_success
                    best_name = name

        if best_name:
            best_model_info = {
                "run_name": best_name,
                "output_dir": promote_results[best_name]["output_dir"],
                "success_rate": best_success,
                "eval_history": promote_results[best_name]["eval_history"],
            }
            print(f"\n  🏆 Best model: {best_name} (success_rate={best_success:.1%})")

        # 收集结果到all_results
        for name, result in scout_results.items():
            history = result["eval_history"]
            best_eval = max(history, key=lambda e: e["success_rate"]) if history else {}
            all_results.append({
                "run_name": name,
                "status": result.get("status", "unknown"),
                "stop_reason": result.get("stop_reason", ""),
                "promoted": result.get("promoted", False),
                "is_best": (name == best_name),
                "learning_rate": result.get("learning_rate"),
                "seed": result.get("seed"),
                "net_arch": result.get("net_arch"),
                "best_success_rate": best_eval.get("success_rate", 0),
                "best_velocity_error": best_eval.get("mean_final_velocity_error", 0),
                "best_height_error": best_eval.get("mean_final_height_error", 0),
                "best_fuel_used": best_eval.get("mean_fuel_used", 0),
                "best_crash_rate": best_eval.get("crash_rate", 0),
                "best_throttle_delta": best_eval.get("mean_abs_throttle_delta", 0),
                "best_eval": best_eval,
            })

    # ============================================================
    # 阶段3: 生成排行榜
    # ============================================================
    print(f"\n{'='*60}")
    print(f"📋 Phase 3: Leaderboard & Final Steps")
    print(f"{'='*60}")

    generate_leaderboard(all_results, sweep_dir)

    # 保存最优模型路径
    if best_model_info:
        best_link = os.path.join(sweep_dir, "best_model_info.json")
        with open(best_link, "w") as f:
            json.dump({
                "run_name": best_model_info["run_name"],
                "model_dir": best_model_info["output_dir"],
                "success_rate": best_model_info["success_rate"],
            }, f, indent=2)
        print(f"\n  🏆 Best model info saved to: {best_link}")
        print(f"     Run: {best_model_info['run_name']}")
        print(f"     Success rate: {best_model_info['success_rate']:.1%}")
        print(f"\n  Next steps:")
        print(
            f"     python -m rocket_landing_control.workflows.evaluate "
            f"--model {best_model_info['output_dir']}/models/final_model.zip --n-episodes 100"
        )
        print(
            f"     python -m rocket_landing_control.studies.robustness_full_test "
            f"--model {best_model_info['output_dir']}/models/final_model.zip"
        )
        print(f"     python -m rocket_landing_control.visualization.plot_results --result-dir {best_model_info['output_dir']}")
    else:
        print("\n  ⚠️ No best model found. Consider running more training.")

    # 保存sweep摘要
    sweep_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_configs": len(configs),
        "promoted": len(promoted_names) if promoted_names else 0,
        "best_model": best_model_info["run_name"] if best_model_info else None,
        "best_success_rate": best_model_info["success_rate"] if best_model_info else 0,
        "all_results": all_results,
    }
    summary_path = os.path.join(sweep_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sweep_summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"✅ Sweep complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
