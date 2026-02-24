# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from modules.networks import ActorNet, AttnActorNet, CriticNet


def resolve_device(cfg):
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


class Actor:
    """Actor 封装：根据 cfg.use_attention 选择 MLP 或 Attention Actor"""
    def __init__(self, cfg):
        self.device = resolve_device(cfg)
        if cfg.use_attention:
            self.action_net = AttnActorNet(
                obs_dim=cfg.state_dim,
                action_dim=cfg.action_dim,
                embed_dim=cfg.embed_dim,
                n_heads=cfg.n_heads,
                hidden=cfg.actor_hidden,
                max_action=cfg.max_action
            ).to(self.device)
        else:
            self.action_net = ActorNet(
                inp=cfg.state_dim,
                outp=cfg.action_dim,
                hidden=cfg.actor_hidden,
                max_action=cfg.max_action
            ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.action_net.parameters(), lr=cfg.policy_lr
        )
        self.max_action = cfg.max_action
        self.min_action = cfg.min_action

    def choose_action(self, self_obs, other_obs=None):
        """推理时选动作，返回 numpy"""
        inp = torch.FloatTensor(self_obs).to(self.device)
        if other_obs is not None:
            other_inp = torch.FloatTensor(other_obs).to(self.device).unsqueeze(0)
            inp = inp.unsqueeze(0)
            mean, std = self.action_net(inp, other_inp)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        else:
            mean, std = self.action_net(inp)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, self.min_action, self.max_action)
        return action.detach().cpu().numpy()

    def evaluate(self, self_obs_batch, other_obs_batch=None):
        """训练时评估：返回 (action, log_prob)，支持梯度"""
        mean, std = self.action_net(self_obs_batch, other_obs_batch)
        dist = torch.distributions.Normal(mean, std)
        noise = torch.distributions.Normal(0, 1)
        z = noise.sample(mean.shape).to(self.device)
        action = torch.tanh(mean + std * z)
        action = torch.clamp(action, self.min_action, self.max_action)
        log_prob = dist.log_prob(mean + std * z) - torch.log(
            1 - action.pow(2) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # [batch, 1]
        return action, log_prob

    def learn(self, actor_loss):
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()


class Critic:
    """Critic 封装：双 Q 网络 + target 网络 + soft update"""
    def __init__(self, cfg):
        self.device = resolve_device(cfg)
        n_total = cfg.n_agent_train + cfg.m_enemy_train
        s_dim = cfg.state_dim * n_total
        a_dim = cfg.action_dim
        self.critic_v = CriticNet(s_dim, a_dim, hidden=cfg.actor_hidden).to(self.device)
        self.target_critic_v = CriticNet(s_dim, a_dim, hidden=cfg.actor_hidden).to(self.device)
        self.target_critic_v.load_state_dict(self.critic_v.state_dict())
        self.optimizer = torch.optim.Adam(
            self.critic_v.parameters(), lr=cfg.value_lr, eps=1e-5
        )
        self.lossfunc = nn.MSELoss()
        self.tau = cfg.tau

    def get_v(self, s, a):
        return self.critic_v(s, a)

    def target_get_v(self, s, a):
        return self.target_critic_v(s, a)

    def soft_update(self):
        for tp, p in zip(self.target_critic_v.parameters(),
                         self.critic_v.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)

    def learn(self, current_q1, current_q2, target_q, is_weights=None):
        if is_weights is not None:
            w = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)
            loss = (w * (current_q1 - target_q).pow(2)).mean() + \
                   (w * (current_q2 - target_q).pow(2)).mean()
        else:
            loss = self.lossfunc(current_q1, target_q) + \
                   self.lossfunc(current_q2, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Entropy:
    """自动温度调节 alpha"""
    def __init__(self, cfg):
        self.device = resolve_device(cfg)
        self.target_entropy = -0.1
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.q_lr)

    def learn(self, entropy_loss):
        self.optimizer.zero_grad()
        entropy_loss.backward()
        self.optimizer.step()
        self.alpha = self.log_alpha.exp()
