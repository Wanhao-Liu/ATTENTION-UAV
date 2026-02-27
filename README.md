# ATTENTION-UAV
python 3.8
基于 MASAC（Multi-Agent Soft Actor-Critic）的多无人机协同路径规划框架，支持 PER（优先经验回放）和 Cross-Attention 机制的消融实验。

## 项目结构

```
ATTENTION-UAV/
├── config/                # 实验配置
│   ├── baseline.py        # MASAC 基线
│   ├── masac_per.py       # MASAC + PER
│   ├── masac_attn.py      # MASAC + Attention
│   ├── masac_per_attn.py  # MASAC + PER + Attention（完整方法）
│   └── maddpg.py          # MADDPG 基线
├── env/                   # 多 UAV 路径规划环境（Pygame）
├── modules/               # 算法模块
│   ├── agent.py           # SAC Actor / Critic / Entropy
│   ├── networks.py        # MLP 和 Attention 网络
│   ├── ddpg_agent.py      # DDPG Actor / Critic
│   ├── memory.py          # 均匀经验回放
│   ├── per_memory.py      # 优先经验回放（SumTree）
│   └── noise.py           # OU 噪声
├── scripts/               # 训练与测试脚本
│   ├── train.py           # MASAC 统一训练
│   ├── train_ddpg.py      # MADDPG 训练
│   └── test.py            # 统一测试
├── plot/                  # 绘图脚本
│   ├── plot_train_curves.py
│   ├── plot_test_metrics.py
│   ├── plot_trajectory.py
│   └── plot_ablation_comparison.py
├── results/               # 实验结果（自动生成）
└── figures/               # 图表输出
```


## 使用方法

### 1. 训练

MASAC 系列（baseline / masac_per / masac_attn / masac_per_attn）：

```bash
# 训练 baseline（MASAC，无 PER 无 Attention）
python scripts/train.py --config baseline

# 训练 MASAC + PER
python scripts/train.py --config masac_per

# 训练 MASAC + Attention
python scripts/train.py --config masac_attn

# 训练 MASAC + PER + Attention（完整方法）
python scripts/train.py --config masac_per_attn
```

MADDPG 基线：

```bash
python scripts/train_ddpg.py
```

可选参数：

```bash
--ep_max 2000          # 最大训练 episode 数
--ep_len 1000          # 每个 episode 最大步数
--memory_capacity 20000 # 经验回放容量（仅 train.py）
--resume               # 从最新 checkpoint 恢复训练
--render               # 开启渲染（调试用）
```

训练结果保存在 `results/<config_name>/` 下，包含：
- `checkpoints/` — 模型权重
- `train_log.csv` — 训练日志
- `train_data.pkl` — 训练数据

### 2. 测试

```bash
# 测试 baseline（2-agent，默认 100 episodes）
python scripts/test.py --config baseline

# 测试完整方法
python scripts/test.py --config masac_per_attn

# 测试 MADDPG
python scripts/test.py --config maddpg
```

可选参数：

```bash
--n_agent 1            # leader 数量
--m_enemy 4            # follower 数量（5-agent 测试）
--test_episodes 100    # 测试 episode 数
--ep_len 1000          # 每个 episode 最大步数
--leader_ckpt path     # 指定 leader 权重路径
--follower_ckpt path   # 指定 follower 权重路径
--render               # 开启渲染
```

测试结果保存在 `results/<config_name>/` 下，包含：
- `test_metrics.csv` — 逐 episode 指标
- `test_data.pkl` — 完整测试数据

### 3. 绘图

```bash
# 训练曲线
python plot/plot_train_curves.py

# 测试指标对比
python plot/plot_test_metrics.py

# 轨迹可视化
python plot/plot_trajectory.py

# 消融实验对比
python plot/plot_ablation_comparison.py
```

### 4. 完整消融实验流程

```bash
# 1) 训练全部配置
python scripts/train.py --config baseline
python scripts/train.py --config masac_per
python scripts/train.py --config masac_attn
python scripts/train.py --config masac_per_attn
python scripts/train_ddpg.py

# 2) 测试全部配置
python scripts/test.py --config baseline
python scripts/test.py --config masac_per
python scripts/test.py --config masac_attn
python scripts/test.py --config masac_per_attn
python scripts/test.py --config maddpg

# 3) 绘制对比图
python plot/plot_ablation_comparison.py
```
