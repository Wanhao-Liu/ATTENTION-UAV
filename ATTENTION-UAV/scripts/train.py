# -*- coding: utf-8 -*-
"""ATTENTION-UAV 统一训练脚本"""
import os
import sys
import csv
import time
import random
import argparse
import pickle
import numpy as np
import torch

# 将项目根目录加入 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.agent import Actor, Critic, Entropy, resolve_device
from modules.memory import Memory
from modules.per_memory import PrioritizedReplayBuffer
from modules.noise import OrnsteinUhlenbeckNoise


def parse_args():
    parser = argparse.ArgumentParser(description="MASAC Ablation Training")
    parser.add_argument("--config", type=str, default="baseline",
                        choices=["baseline", "masac_per", "masac_attn",
                                 "masac_per_attn"])
    parser.add_argument("--ep_max", type=int, default=None)
    parser.add_argument("--ep_len", type=int, default=None)
    parser.add_argument("--memory_capacity", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def load_config(name):
    if name == "baseline":
        from config.baseline import BaselineConfig
        return BaselineConfig()
    elif name == "masac_per":
        from config.masac_per import MasacPerConfig
        return MasacPerConfig()
    elif name == "masac_attn":
        from config.masac_attn import MasacAttnConfig
        return MasacAttnConfig()
    elif name == "masac_per_attn":
        from config.masac_per_attn import MasacPerAttnConfig
        return MasacPerAttnConfig()
    else:
        raise ValueError(f"Unknown config: {name}")


def train():
    args = parse_args()
    cfg = load_config(args.config)
    # 命令行覆盖
    if args.ep_max is not None:
        cfg.ep_max = args.ep_max
    if args.ep_len is not None:
        cfg.ep_len = args.ep_len
    if args.memory_capacity is not None:
        cfg.memory_capacity = args.memory_capacity
    if args.render:
        cfg.render = True

    # 设备
    device = resolve_device(cfg)

    # 固定随机种子
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # 结果目录
    result_dir = os.path.join(ROOT, "results", cfg.exp_name)
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 环境
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from env.path_env import RlGame
    env = RlGame(n=cfg.n_agent_train, m=cfg.m_enemy_train,
                 render=cfg.render).unwrapped

    n_total = cfg.n_agent_train + cfg.m_enemy_train
    s_dim = cfg.state_dim
    a_dim = cfg.action_dim
    mem_dims = 2 * s_dim * n_total + a_dim * n_total + n_total

    # 多轮训练
    all_ep_r = [[] for _ in range(cfg.train_num)]
    all_ep_r0 = [[] for _ in range(cfg.train_num)]
    all_ep_r1 = [[] for _ in range(cfg.train_num)]

    csv_path = os.path.join(result_dir, "train_log.csv")

    print(f"[{cfg.exp_name}] 训练开始 | PER={cfg.use_per} | "
          f"Attn={cfg.use_attention} | ep_max={cfg.ep_max} | device={device}")

    for k in range(cfg.train_num):
        # 实例化 agents
        actors = [Actor(cfg) for _ in range(n_total)]
        critics = [Critic(cfg) for _ in range(n_total)]
        entroys = [Entropy(cfg) for _ in range(n_total)]

        # 经验回放
        if cfg.use_per:
            buffer = PrioritizedReplayBuffer(
                capacity=cfg.memory_capacity, dims=mem_dims,
                alpha=cfg.per_alpha, beta_start=cfg.per_beta_start,
                beta_end=cfg.per_beta_end, beta_steps=cfg.per_beta_steps,
                eps=cfg.per_eps
            )
        else:
            buffer = Memory(capacity=cfg.memory_capacity, dims=mem_dims)

        # OU 噪声 — 全局一个实例，与原始一致
        ou_noise = OrnsteinUhlenbeckNoise(
            mu=np.zeros((n_total, a_dim)),
            sigma=cfg.ou_sigma, theta=cfg.ou_theta, dt=cfg.ou_dt
        )

        # 断点续训
        start_ep = 0
        if args.resume:
            ckpt_file = os.path.join(ckpt_dir, "latest.pth")
            if os.path.exists(ckpt_file):
                ckpt = torch.load(ckpt_file, map_location=device)
                for i in range(n_total):
                    actors[i].action_net.load_state_dict(ckpt[f"actor_{i}"])
                start_ep = ckpt.get("episode", 0)
                print(f"  从 episode {start_ep} 恢复")

        # CSV 日志
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "reward_total", "reward_leader",
                         "reward_follower", "timesteps", "wall_time"])

        t_start = time.time()

        for episode in range(start_ep, cfg.ep_max):
            observation = env.reset()
            reward_total = 0.0
            reward_r0 = 0.0
            reward_r1 = 0.0
            action = np.zeros((n_total, a_dim))

            for step in range(cfg.ep_len):
                for i in range(n_total):
                    obs_i = observation[i]
                    if cfg.use_attention and n_total > 1:
                        other_idx = [j for j in range(n_total) if j != i]
                        other_obs = observation[other_idx]
                        action[i] = actors[i].choose_action(obs_i, other_obs)
                    else:
                        action[i] = actors[i].choose_action(obs_i)

                if episode <= cfg.noise_episodes:
                    action = action + ou_noise()
                action = np.clip(action, -cfg.max_action, cfg.max_action)

                observation_, reward, done, win, team_counter, d = \
                    env.step(action)

                # 存储 transition
                s_flat = observation.flatten()
                a_flat = action.flatten()
                r_flat = reward.flatten()
                s_next_flat = observation_.flatten()
                buffer.store_transition(s_flat, a_flat, r_flat, s_next_flat)

                # 批量更新
                if buffer.is_ready:
                    if cfg.use_per:
                        batch, tree_idx, is_weights = buffer.sample(
                            cfg.batch_size)
                    else:
                        batch = buffer.sample(cfg.batch_size)
                        tree_idx = None
                        is_weights = None

                    # 解析 batch
                    b_s = torch.FloatTensor(batch[:, :s_dim * n_total]).to(device)
                    offset_a = s_dim * n_total
                    b_a = torch.FloatTensor(
                        batch[:, offset_a:offset_a + a_dim * n_total]).to(device)
                    offset_r = offset_a + a_dim * n_total
                    b_r = torch.FloatTensor(
                        batch[:, offset_r:offset_r + n_total]).to(device)
                    offset_s_ = offset_r + n_total
                    b_s_ = torch.FloatTensor(batch[:, offset_s_:]).to(device)

                    td_errors_all = []

                    for i in range(n_total):
                        si = s_dim * i
                        ai = a_dim * i
                        # 当前 agent 的 local obs
                        b_obs_i = b_s[:, si:si + s_dim]
                        b_obs_i_ = b_s_[:, si:si + s_dim]

                        # other agents obs (for attention)
                        other_obs = None
                        other_obs_ = None
                        if cfg.use_attention and n_total > 1:
                            others = [j for j in range(n_total) if j != i]
                            o_list = [b_s[:, s_dim*j:s_dim*(j+1)]
                                      for j in others]
                            other_obs = torch.stack(o_list, dim=1)
                            o_list_ = [b_s_[:, s_dim*j:s_dim*(j+1)]
                                       for j in others]
                            other_obs_ = torch.stack(o_list_, dim=1)

                        # target Q
                        new_a, log_p_ = actors[i].evaluate(
                            b_obs_i_, other_obs_)
                        tq1, tq2 = critics[i].target_get_v(b_s_, new_a)
                        target_q = b_r[:, i:i+1] + cfg.gamma * (
                            torch.min(tq1, tq2) - entroys[i].alpha * log_p_)

                        # current Q
                        cur_a_i = b_a[:, ai:ai + a_dim]
                        cq1, cq2 = critics[i].get_v(b_s, cur_a_i)

                        # TD error for PER
                        with torch.no_grad():
                            td_err = torch.abs(
                                cq1 - target_q.detach()).mean(dim=-1).cpu().numpy()
                            td_err = np.atleast_1d(td_err)
                        td_errors_all.append(td_err)

                        # Critic 更新
                        critics[i].learn(cq1, cq2, target_q.detach(),
                                         is_weights)

                        # Actor 更新
                        a_i, log_p = actors[i].evaluate(b_obs_i, other_obs)
                        q1, q2 = critics[i].get_v(b_s, a_i)
                        q = torch.min(q1, q2)
                        actor_loss = (entroys[i].alpha * log_p - q).mean()
                        actors[i].learn(actor_loss)

                        # Entropy 更新
                        alpha_loss = -(entroys[i].log_alpha.exp() * (
                            log_p + entroys[i].target_entropy
                        ).detach()).mean()
                        entroys[i].learn(alpha_loss)
                        entroys[i].alpha = entroys[i].log_alpha.exp()

                        # soft update
                        critics[i].soft_update()

                    # PER 优先级更新：取所有 agent TD-error 的 max
                    if cfg.use_per and tree_idx is not None:
                        max_td = np.max(np.stack(td_errors_all, axis=0),
                                        axis=0)
                        buffer.update_priorities(tree_idx, max_td)

                # 累计 reward
                reward_total += float(sum(reward))
                reward_r0 += float(reward[0])
                if n_total > 1:
                    reward_r1 += float(np.mean(reward[1:]))

                observation = observation_
                if done:
                    break

            # episode 结束
            wall = time.time() - t_start
            all_ep_r[k].append(reward_total)
            all_ep_r0[k].append(reward_r0)
            all_ep_r1[k].append(reward_r1)

            writer.writerow([episode, f"{reward_total:.2f}",
                             f"{reward_r0:.2f}", f"{reward_r1:.2f}",
                             step + 1, f"{wall:.1f}"])
            csv_file.flush()

            if episode % 10 == 0:
                print(f"  [{cfg.exp_name}] ep={episode} "
                      f"R={reward_total:.1f} "
                      f"R0={reward_r0:.1f} R1={reward_r1:.1f} "
                      f"t={wall:.0f}s")

            # Checkpoint 保存
            if episode >= cfg.save_after and \
               episode % cfg.save_interval == 0:
                ckpt_data = {"episode": episode}
                for i in range(n_total):
                    ckpt_data[f"actor_{i}"] = \
                        actors[i].action_net.state_dict()
                torch.save(ckpt_data,
                           os.path.join(ckpt_dir, "latest.pth"))
                # 保存 Leader / Follower 单独权重
                torch.save(actors[0].action_net.state_dict(),
                           os.path.join(ckpt_dir,
                                        f"actor_L_ep{episode}.pth"))
                if n_total > 1:
                    torch.save(actors[1].action_net.state_dict(),
                               os.path.join(ckpt_dir,
                                            f"actor_F_ep{episode}.pth"))

        csv_file.close()

        # 训练结束保存最终权重
        torch.save(actors[0].action_net.state_dict(),
                   os.path.join(ckpt_dir, "actor_L_final.pth"))
        if n_total > 1:
            torch.save(actors[1].action_net.state_dict(),
                       os.path.join(ckpt_dir, "actor_F_final.pth"))

    # 保存训练曲线 pickle
    data = {
        "all_ep_r": all_ep_r, "all_ep_r0": all_ep_r0,
        "all_ep_r1": all_ep_r1, "cfg": cfg.exp_name
    }
    pkl_path = os.path.join(result_dir, "train_data.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[{cfg.exp_name}] 训练完成 | 结果保存至 {result_dir}")
    env.close()


if __name__ == "__main__":
    train()
