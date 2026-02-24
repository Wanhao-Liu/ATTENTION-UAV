# -*- coding: utf-8 -*-
"""绘制测试指标折线图：FKR, integral_V, integral_U, timesteps"""
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

METRICS = [
    ("all_ep_F", "FKR"),
    ("all_ep_V", "Integral V"),
    ("all_ep_U", "Integral U"),
    ("all_ep_T", "Timesteps"),
]


def main():
    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()

    for ax, (key, ylabel) in zip(axes, METRICS):
        for exp, label, color in zip(CONFIGS, LABELS, COLORS):
            pkl_path = os.path.join(ROOT, "results", exp, "test_data.pkl")
            if not os.path.exists(pkl_path):
                continue
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            vals = np.array(data[key])
            x = np.arange(len(vals))
            ax.plot(x, vals, label=label, color=color, alpha=0.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.margins(x=0)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "test_metrics.png"))
    plt.close(fig)
    print("测试指标图绘制完成")


if __name__ == "__main__":
    main()
