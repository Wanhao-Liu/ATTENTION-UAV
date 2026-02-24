# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class BaselineConfig:
    exp_name: str = "baseline"

    # 环境 - 训练
    n_agent_train: int = 1
    m_enemy_train: int = 1
    # 环境 - 测试
    n_agent_test: int = 1
    m_enemy_test: int = 4

    # 状态/动作维度
    state_dim: int = 7
    action_dim: int = 2
    max_action: float = 1.0
    min_action: float = -1.0

    # 训练超参数
    ep_max: int = 500
    ep_len: int = 1000
    gamma: float = 0.9
    tau: float = 0.01
    batch_size: int = 128
    memory_capacity: int = 20000
    train_num: int = 1

    # 学习率
    policy_lr: float = 1e-3
    q_lr: float = 3e-4
    value_lr: float = 3e-3

    # 网络
    actor_hidden: int = 256
    critic_hidden: int = 256

    # Attention (默认关闭)
    use_attention: bool = False
    embed_dim: int = 64
    n_heads: int = 4

    # PER (默认关闭)
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 100000
    per_eps: float = 1e-6

    # 保存/日志
    save_interval: int = 20
    save_after: int = 200
    noise_episodes: int = 20

    # OU 噪声
    ou_mu: float = 0.0
    ou_theta: float = 0.15
    ou_sigma: float = 0.2

    # 渲染
    render: bool = False

    # 设备
    device: str = "auto"
