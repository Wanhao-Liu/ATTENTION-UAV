# -*- coding: utf-8 -*-
"""绘制 2D 轨迹图"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

AGENT_COLORS = ["#e75840", "#3299CC", "#115840", "#9B59B6", "#F39C12"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="baseline")
    parser.add_argument("--episode", type=int, default=0,
                        help="要绘制的 episode 编号")
    args = parser.parse_args()

    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    pkl_path = os.path.join(ROOT, "results", args.config, "test_data.pkl")
    if not os.path.exists(pkl_path):
        print(f"文件不存在: {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    trajs = data["trajectories"]
    if args.episode >= len(trajs):
        print(f"Episode {args.episode} 超出范围 (共 {len(trajs)} 个)")
        return

    ep_traj = trajs[args.episode]
    n_agents = len(ep_traj[0]["positions"])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    for i in range(n_agents):
        xs = [t["positions"][i][0] for t in ep_traj]
        ys = [t["positions"][i][1] for t in ep_traj]
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        label = "Leader" if i == 0 else f"Follower {i}"
        ax.plot(xs, ys, color=color, label=label, linewidth=1.5)
        ax.scatter(xs[0], ys[0], color=color, marker="o", s=60, zorder=5)
        ax.scatter(xs[-1], ys[-1], color=color, marker="*", s=100, zorder=5)

    ax.set_xlabel("X (normalized)")
    ax.set_ylabel("Y (normalized)")
    ax.set_title(f"{args.config} - Episode {args.episode} Trajectory")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()

    out_path = os.path.join(fig_dir,
                            f"{args.config}_traj_ep{args.episode}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"轨迹图保存至 {out_path}")


if __name__ == "__main__":
    main()
