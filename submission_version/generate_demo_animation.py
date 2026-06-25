"""Generate a simple report/demo GIF from a saved trajectory JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_trajectory(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, data["trajectory"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate rocket landing trajectory GIF.")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="results/experiments/01_main_evaluation/success_trajectories.json",
    )
    parser.add_argument("--output", type=str, default="results/demo/landing_demo.gif")
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    meta, traj = load_trajectory(args.trajectory)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    times = [p["time"] for p in traj]
    heights = [p["height"] for p in traj]
    velocities = [p["velocity"] for p in traj]
    throttles = [p.get("throttle", 0.0) for p in traj]
    fuels = [p.get("fuel", 0.0) for p in traj]

    max_height = max(max(heights), meta.get("initial_conditions", {}).get("initial_height", 50.0))
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 2)
    ax_rocket = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1])

    ax_rocket.set_xlim(-1.0, 1.0)
    ax_rocket.set_ylim(-2.0, max_height + 5.0)
    ax_rocket.set_xlabel("Lateral position (illustrative)")
    ax_rocket.set_ylabel("Height (m)")
    ax_rocket.set_title("Rocket Vertical Soft Landing")
    ax_rocket.axhline(0, color="black", linewidth=1.5)
    rocket_marker, = ax_rocket.plot([0], [heights[0]], marker="^", markersize=18, color="tab:blue")
    flame_marker, = ax_rocket.plot([0], [heights[0] - 1.5], marker="v", markersize=10, color="tab:orange")
    text = ax_rocket.text(-0.95, max_height, "", va="top")

    ax_v.set_xlim(0, times[-1])
    ax_v.set_ylim(min(velocities) - 2, max(max(velocities), 0) + 2)
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Velocity (m/s)")
    ax_v.set_title("Velocity")
    v_line, = ax_v.plot([], [], color="tab:red")
    ax_v.grid(True, alpha=0.3)

    ax_u.set_xlim(0, times[-1])
    ax_u.set_ylim(0, 1.05)
    ax_u.set_xlabel("Time (s)")
    ax_u.set_ylabel("Throttle / Fuel ratio")
    ax_u.set_title("Control and Fuel")
    u_line, = ax_u.plot([], [], color="tab:green", label="Throttle")
    f_line, = ax_u.plot([], [], color="tab:orange", label="Fuel ratio")
    ax_u.legend()
    ax_u.grid(True, alpha=0.3)

    initial_fuel = max(meta.get("initial_conditions", {}).get("initial_fuel", max(fuels)), 1e-6)

    def update(frame):
        rocket_marker.set_data([0], [heights[frame]])
        flame_marker.set_data([0], [max(heights[frame] - 1.5, 0)])
        flame_marker.set_markersize(6 + 18 * throttles[frame])
        flame_marker.set_alpha(0.2 + 0.8 * throttles[frame])
        text.set_text(
            f"t = {times[frame]:.2f} s\n"
            f"h = {heights[frame]:.2f} m\n"
            f"v = {velocities[frame]:.2f} m/s\n"
            f"fuel = {fuels[frame]:.2f} kg"
        )
        v_line.set_data(times[: frame + 1], velocities[: frame + 1])
        u_line.set_data(times[: frame + 1], throttles[: frame + 1])
        f_line.set_data(times[: frame + 1], [f / initial_fuel for f in fuels[: frame + 1]])
        return rocket_marker, flame_marker, text, v_line, u_line, f_line

    anim = FuncAnimation(fig, update, frames=len(traj), interval=1000 / args.fps, blit=True)
    anim.save(output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved demo animation to: {output}")


if __name__ == "__main__":
    main()
