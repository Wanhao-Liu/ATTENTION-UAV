#!/bin/bash
# ATTENTION-UAV 一键消融实验脚本
# 串行执行 4 个实验：train → test
# 用法: bash scripts/run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo " ATTENTION-UAV 消融实验"
echo "=========================================="

CONFIGS=("baseline" "masac_per" "masac_attn" "masac_per_attn")

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=========================================="
    echo " 实验: $cfg"
    echo "=========================================="

    echo "[1/2] 训练 $cfg ..."
    python scripts/train.py --config "$cfg"

    echo "[2/2] 测试 $cfg ..."
    python scripts/test.py --config "$cfg"

    echo " $cfg 完成"
done

echo ""
echo "=========================================="
echo " 全部实验完成，生成对比图 ..."
echo "=========================================="

python plot/plot_train_curves.py
python plot/plot_test_metrics.py
python plot/plot_ablation_comparison.py

echo " 所有结果保存在 results/ 和 figures/ 目录下"
