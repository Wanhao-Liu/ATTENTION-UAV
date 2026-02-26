from dataclasses import dataclass


@dataclass
class MADDPGConfig:
    exp_name: str = "maddpg"

    # 环境 - 训练
    n_agent_train: int = 1
    m_enemy_train: int = 1
    # 环境 - 测试
    n_agent_test: int = 1
    m_enemy_test: int = 1

    # 状态/动作维度
    state_dim: int = 7
    action_dim: int = 2
    max_action: float = 1.0
    min_action: float = -1.0

    # 训练超参数（与原始 DDPG 一致）
    ep_max: int = 1000
    ep_len: int = 1000
    gamma: float = 0.95
    tau: float = 0.005
    batch_size: int = 128
    memory_capacity: int = 20000
    train_num: int = 1
    reward_scale: float = 1e-3  # 原始代码 reward/1000

    # 学习率
    policy_lr: float = 1e-3
    critic_lr: float = 1e-3

    # 网络（与原始 DDPG 一致）
    actor_hidden1: int = 50
    actor_hidden2: int = 20
    critic_hidden1: int = 40
    critic_hidden2: int = 20

    # Attention（DDPG 不使用）
    use_attention: bool = False

    # 保存/日志
    save_interval: int = 20
    save_after: int = 200
    noise_episodes: int = 50

    # OU 噪声
    ou_mu: float = 0.0
    ou_theta: float = 0.1
    ou_sigma: float = 0.1
    ou_dt: float = 1e-2

    # 随机种子
    seed: int = 42

    # 渲染
    render: bool = False

    # 设备
    device: str = "auto"
