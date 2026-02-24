# -*- coding: utf-8 -*-
"""消融对比图：训练曲线叠加 + 测试指标分组柱状图"""
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


def plot_training_comparison(fig_dir):
    """Figure 1: 4 条训练曲线叠加"""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for exp, label, color in zip(CONFIGS, LABELS, COLORS):
        pkl = os.path.join(ROOT, "results", exp, "train_data.pkl")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        arr = np.array(data["all_ep_r"])
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        mean_s = smooth(mean)
        x = np.arange(len(mean_s))
        ax.plot(x, mean_s, label=label, color=color, linewidth=1.5)
        ax.fill_between(x, mean_s - std * 0.95, mean_s + std * 0.95,
                        alpha=0.2, facecolor=color)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Ablation: Training Curves")
    ax.legend()
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "ablation_training_curves.png"))
    plt.close(fig)
    print("消融训练曲线对比图完成")


def plot_test_comparison(fig_dir):
    """Figure 2: 5 个测试指标的分组柱状图"""
    metrics = [
        ("win_rate", "Win Rate"),
        ("all_ep_F", "Avg FKR"),
        ("all_ep_V", "Avg Integral V"),
        ("all_ep_U", "Avg Integral U"),
        ("all_ep_T", "Avg Timesteps"),
    ]
    values = {m[0]: [] for m in metrics}
    valid_labels = []
    valid_colors = []

    for exp, label, color in zip(CONFIGS, LABELS, COLORS):
        pkl = os.path.join(ROOT, "results", exp, "test_data.pkl")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        valid_labels.append(label)
        valid_colors.append(color)
        values["win_rate"].append(data["win_rate"])
        for key, _ in metrics[1:]:
            values[key].append(np.mean(data[key]))

    if not valid_labels:
        print("无测试数据，跳过柱状图")
        return

    n_groups = len(valid_labels)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=150)

    for ax, (key, ylabel) in zip(axes, metrics):
        x = np.arange(n_groups)
        bars = ax.bar(x, values[key], color=valid_colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_labels, rotation=30, ha="right",
                           fontsize=7)
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, values[key]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Ablation: Test Metrics Comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "ablation_test_metrics.png"))
    plt.close(fig)
    print("消融测试指标对比图完成")


def main():
    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_training_comparison(fig_dir)
    plot_test_comparison(fig_dir)


if __name__ == "__main__":
    main()
