# -*- coding: utf-8 -*-
"""DDPG Agent：确定性策略 + 单 Q 网络，从原始 main_DDPG.py 迁移"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def resolve_device(cfg):
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


class DDPGActorNet(nn.Module):
    """原始 DDPG Actor：obs → action（确定性）"""
    def __init__(self, inp, outp, hidden1=50, hidden2=20,
                 max_action=1.0):
        super(DDPGActorNet, self).__init__()
        self.max_action = max_action
        self.in_to_y1 = nn.Linear(inp, hidden1)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(hidden1, hidden2)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden2, outp)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.in_to_y1(x))
        x = torch.sigmoid(self.y1_to_y2(x))
        return self.max_action * torch.tanh(self.out(x))


class DDPGCriticNet(nn.Module):
    """原始 DDPG Critic：(global_state, action) → Q"""
    def __init__(self, state_dim, action_dim, hidden1=40,
                 hidden2=20):
        super(DDPGCriticNet, self).__init__()
        inp = state_dim + action_dim
        self.in_to_y1 = nn.Linear(inp, hidden1)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(hidden1, hidden2)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden2, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.in_to_y1(x))
        x = torch.sigmoid(self.y1_to_y2(x))
        return self.out(x)


class DDPGActor:
    """DDPG Actor 封装：确定性策略 + target 网络"""
    def __init__(self, cfg):
        self.device = resolve_device(cfg)
        self.action_net = DDPGActorNet(
            inp=cfg.state_dim, outp=cfg.action_dim,
            hidden1=cfg.actor_hidden1, hidden2=cfg.actor_hidden2,
            max_action=cfg.max_action
        ).to(self.device)
        self.target_net = DDPGActorNet(
            inp=cfg.state_dim, outp=cfg.action_dim,
            hidden1=cfg.actor_hidden1, hidden2=cfg.actor_hidden2,
            max_action=cfg.max_action
        ).to(self.device)
        self.target_net.load_state_dict(self.action_net.state_dict())
        self.optimizer = torch.optim.Adam(
            self.action_net.parameters(), lr=cfg.policy_lr
        )
        self.tau = cfg.tau

    def choose_action(self, self_obs, other_obs=None):
        """推理：确定性动作，接口与 SAC Actor 一致"""
        inp = torch.FloatTensor(self_obs).to(self.device)
        action = self.action_net(inp)
        return action.detach().cpu().numpy()

    def learn_a(self, s_batch):
        """训练时生成带梯度的动作"""
        return self.action_net(s_batch)

    def learn_a_target(self, s_batch):
        """target 网络生成动作（无梯度）"""
        return self.target_net(s_batch).detach()

    def learn(self, actor_loss):
        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.action_net.parameters(), max_norm=0.5)
        self.optimizer.step()

    def soft_update(self):
        for tp, p in zip(self.target_net.parameters(),
                         self.action_net.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)


class DDPGCritic:
    """DDPG Critic 封装：单 Q 网络 + target"""
    def __init__(self, cfg):
        self.device = resolve_device(cfg)
        n_total = cfg.n_agent_train + cfg.m_enemy_train
        s_dim = cfg.state_dim * n_total
        a_dim = cfg.action_dim
        self.critic = DDPGCriticNet(
            s_dim, a_dim,
            hidden1=cfg.critic_hidden1, hidden2=cfg.critic_hidden2
        ).to(self.device)
        self.target_critic = DDPGCriticNet(
            s_dim, a_dim,
            hidden1=cfg.critic_hidden1, hidden2=cfg.critic_hidden2
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr
        )
        self.lossfunc = nn.MSELoss()
        self.gamma = cfg.gamma
        self.tau = cfg.tau

    def learn_loss(self, s, a):
        """返回 -Q(s,a).mean() 作为 actor loss"""
        return -self.critic(s, a).mean()

    def learn(self, s, a, r, s_, a_):
        """Critic TD 学习"""
        Q_est = self.critic(s, a)
        Q_next = self.target_critic(s_, a_).detach()
        Q_target = r + self.gamma * Q_next
        loss = self.lossfunc(Q_est, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self):
        for tp, p in zip(self.target_critic.parameters(),
                         self.critic.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)
