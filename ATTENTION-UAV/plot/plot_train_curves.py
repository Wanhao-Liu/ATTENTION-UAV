# -*- coding: utf-8 -*-
"""绘制训练曲线：total/leader/follower reward + std 阴影"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CONFIGS = ["baseline", "masac_per", "masac_attn", "masac_per_attn"]
LABELS = ["MASAC", "MASAC+PER", "MASAC+Attn", "MASAC+PER+Attn"]
COLORS = ["#e75840", "#3299CC", "#115840", "#9B59B6"]


def smooth(data, weight=0.9):
    smoothed = []
    last = data[0]
    for d in data:
        last = last * weight + (1 - weight) * d
        smoothed.append(last)
    return np.array(smoothed)


def plot_single(ax, exp_name, label, color):
    pkl_path = os.path.join(ROOT, "results", exp_name, "train_data.pkl")
    if not os.path.exists(pkl_path):
        print(f"  跳过 {exp_name}: {pkl_path} 不存在")
        return
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    arr = np.array(data["all_ep_r"])
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    mean_s = smooth(mean)
    x = np.arange(len(mean_s))
    ax.plot(x, mean_s, label=label, color=color)
    ax.fill_between(x, mean_s - std * 0.95, mean_s + std * 0.95,
                    alpha=0.3, facecolor=color)


def main():
    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    for exp, label, color in zip(CONFIGS, LABELS, COLORS):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        plot_single(ax, exp, label, color)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.margins(x=0)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_train_curve.png"))
        plt.close(fig)

    print("训练曲线绘制完成")


if __name__ == "__main__":
    main()
