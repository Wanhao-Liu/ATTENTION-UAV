# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    """原始 MLP Actor：obs → (mean, std)"""
    def __init__(self, inp, outp, hidden=256, max_action=1.0):
        super(ActorNet, self).__init__()
        self.max_action = max_action
        self.in_to_y1 = nn.Linear(inp, hidden)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(hidden, hidden)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden, outp)
        self.out.weight.data.normal_(0, 0.1)
        self.std_out = nn.Linear(hidden, outp)
        self.std_out.weight.data.normal_(0, 0.1)

    def forward(self, inputstate, other_obs=None):
        """other_obs 参数仅为接口兼容，MLP Actor 忽略它"""
        x = F.relu(self.in_to_y1(inputstate))
        x = F.relu(self.y1_to_y2(x))
        mean = self.max_action * torch.tanh(self.out(x))
        log_std = torch.clamp(self.std_out(x), -20, 2)
        std = log_std.exp()
        return mean, std


class CriticNet(nn.Module):
    """双 Q 网络：(global_state, action) → (q1, q2)"""
    def __init__(self, state_dim, action_dim, hidden=256):
        super(CriticNet, self).__init__()
        inp = state_dim + action_dim
        # Q1
        self.in_to_y1 = nn.Linear(inp, hidden)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(hidden, hidden)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden, 1)
        self.out.weight.data.normal_(0, 0.1)
        # Q2
        self.q2_in_to_y1 = nn.Linear(inp, hidden)
        self.q2_in_to_y1.weight.data.normal_(0, 0.1)
        self.q2_y1_to_y2 = nn.Linear(hidden, hidden)
        self.q2_y1_to_y2.weight.data.normal_(0, 0.1)
        self.q2_out = nn.Linear(hidden, 1)
        self.q2_out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        q1 = F.relu(self.in_to_y1(x))
        q1 = F.relu(self.y1_to_y2(q1))
        q1 = self.out(q1)
        q2 = F.relu(self.q2_in_to_y1(x))
        q2 = F.relu(self.q2_y1_to_y2(q2))
        q2 = self.q2_out(q2)
        return q1, q2


class AttnActorNet(nn.Module):
    """Multi-Agent Cross-Attention Actor：
    Query = 自身 obs embedding，K/V = 其他 agent obs embedding
    Attention 输出与自身 embedding 拼接后经 MLP → (mean, std)
    """
    def __init__(self, obs_dim=7, action_dim=2, embed_dim=64,
                 n_heads=4, hidden=256, max_action=1.0):
        super(AttnActorNet, self).__init__()
        self.max_action = max_action
        self.embed_dim = embed_dim
        # 观测编码器：obs(7) → embed(64)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # Multi-Head Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )
        # MLP 输出头：concat[self_embed, attn_out](128) → hidden → mean/std
        self.fc1 = nn.Linear(embed_dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean_out = nn.Linear(hidden, action_dim)
        self.std_out = nn.Linear(hidden, action_dim)

        # 仅对 MLP 输出头做 normal_(0,0.1)，保留 obs_encoder/attn 的默认初始化
        for layer in [self.fc1, self.fc2, self.mean_out, self.std_out]:
            layer.weight.data.normal_(0, 0.1)

    def forward(self, self_obs, other_obs=None):
        """
        self_obs:  (batch, obs_dim)
        other_obs: (batch, n_others, obs_dim) 或 None
        """
        self_embed = self.obs_encoder(self_obs)  # (batch, embed)
        query = self_embed.unsqueeze(1)           # (batch, 1, embed)

        if other_obs is not None and other_obs.shape[1] > 0:
            # other_obs: (batch, n_others, obs_dim)
            kv = self.obs_encoder(other_obs)      # (batch, n_others, embed)
        else:
            # 单 agent 退化：K/V 用零向量
            batch_size = self_obs.shape[0]
            kv = torch.zeros(
                batch_size, 1, self.embed_dim,
                device=self_obs.device
            )

        attn_out, _ = self.attn(query, kv, kv)   # (batch, 1, embed)
        attn_out = attn_out.squeeze(1)            # (batch, embed)

        # 拼接自身 embedding 和 attention 输出
        x = torch.cat([self_embed, attn_out], dim=-1)  # (batch, 2*embed)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.max_action * torch.tanh(self.mean_out(x))
        log_std = torch.clamp(self.std_out(x), -20, 2)
        std = log_std.exp()
        return mean, std
