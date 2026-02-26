# -*- coding: utf-8 -*-
"""MADDPG 训练脚本"""
import os
import sys
import csv
import time
import random
import argparse
import pickle
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.ddpg_agent import DDPGActor, DDPGCritic, resolve_device
from modules.memory import Memory
from modules.noise import OrnsteinUhlenbeckNoise


def parse_args():
    parser = argparse.ArgumentParser(description="MADDPG Training")
    parser.add_argument("--ep_max", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def train():
    args = parse_args()
    from config.maddpg import MADDPGConfig
    cfg = MADDPGConfig()
    if args.ep_max is not None:
        cfg.ep_max = args.ep_max
    if args.render:
        cfg.render = True

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

    all_ep_r = [[] for _ in range(cfg.train_num)]
    all_ep_r0 = [[] for _ in range(cfg.train_num)]
    all_ep_r1 = [[] for _ in range(cfg.train_num)]

    csv_path = os.path.join(result_dir, "train_log.csv")

    print(f"[{cfg.exp_name}] 训练开始 | DDPG | "
          f"ep_max={cfg.ep_max} | device={device}")

    for k in range(cfg.train_num):
        actors = [DDPGActor(cfg) for _ in range(n_total)]
        critics = [DDPGCritic(cfg) for _ in range(n_total)]
        buffer = Memory(capacity=cfg.memory_capacity, dims=mem_dims)

        ou_noise = OrnsteinUhlenbeckNoise(
            mu=np.zeros((n_total, a_dim)),
            sigma=cfg.ou_sigma, theta=cfg.ou_theta, dt=cfg.ou_dt
        )

        start_ep = 0
        if args.resume:
            ckpt_file = os.path.join(ckpt_dir, "latest.pth")
            if os.path.exists(ckpt_file):
                ckpt = torch.load(ckpt_file, map_location=device)
                for i in range(n_total):
                    actors[i].action_net.load_state_dict(
                        ckpt[f"actor_{i}"])
                start_ep = ckpt.get("episode", 0)
                print(f"  从 episode {start_ep} 恢复")

        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "reward_total", "reward_leader",
                         "reward_follower", "timesteps", "wall_time"])

        t_start = time.time()
        action = np.zeros((n_total, a_dim))

        for episode in range(start_ep, cfg.ep_max):
            observation = env.reset()
            reward_total = 0.0
            reward_r0 = 0.0
            reward_r1 = 0.0

            for step in range(cfg.ep_len):
                for i in range(n_total):
                    action[i] = actors[i].choose_action(
                        observation[i])

                if episode <= cfg.noise_episodes:
                    action = action + ou_noise()
                action = np.clip(action, -cfg.max_action,
                                 cfg.max_action)

                observation_, reward, done, win, team_counter, d = \
                    env.step(action)

                # 原始 DDPG: reward/1000
                scaled_r = reward.flatten() * cfg.reward_scale
                buffer.store_transition(
                    observation.flatten(), action.flatten(),
                    scaled_r, observation_.flatten())

                if buffer.is_ready:
                    b_M = buffer.sample(cfg.batch_size)
                    b_s = torch.FloatTensor(
                        b_M[:, :s_dim * n_total]).to(device)
                    off_a = s_dim * n_total
                    b_a = torch.FloatTensor(
                        b_M[:, off_a:off_a + a_dim * n_total]
                    ).to(device)
                    off_r = off_a + a_dim * n_total
                    b_r = torch.FloatTensor(
                        b_M[:, off_r:off_r + n_total]).to(device)
                    off_s_ = off_r + n_total
                    b_s_ = torch.FloatTensor(
                        b_M[:, off_s_:]).to(device)

                    for i in range(n_total):
                        si = s_dim * i
                        ai = a_dim * i
                        b_obs_i = b_s[:, si:si + s_dim]
                        b_obs_i_ = b_s_[:, si:si + s_dim]

                        # target action
                        a_target = actors[i].learn_a_target(
                            b_obs_i_)
                        # Critic 更新
                        critics[i].learn(
                            b_s,
                            b_a[:, ai:ai + a_dim],
                            b_r[:, i:i + 1],
                            b_s_, a_target)
                        # Actor 更新
                        a_eval = actors[i].learn_a(b_obs_i)
                        actor_loss = critics[i].learn_loss(
                            b_s, a_eval)
                        actors[i].learn(actor_loss)
                        # 软更新
                        actors[i].soft_update()
                        critics[i].soft_update()

                reward_total += float(sum(reward))
                reward_r0 += float(reward[0])
                if n_total > 1:
                    reward_r1 += float(np.mean(reward[1:]))

                observation = observation_
                if done:
                    break

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

            if episode >= cfg.save_after and \
               episode % cfg.save_interval == 0:
                ckpt_data = {"episode": episode}
                for i in range(n_total):
                    ckpt_data[f"actor_{i}"] = \
                        actors[i].action_net.state_dict()
                torch.save(ckpt_data,
                           os.path.join(ckpt_dir, "latest.pth"))
                torch.save(actors[0].action_net.state_dict(),
                           os.path.join(ckpt_dir,
                                        f"actor_L_ep{episode}.pth"))
                if n_total > 1:
                    torch.save(
                        actors[1].action_net.state_dict(),
                        os.path.join(ckpt_dir,
                                     f"actor_F_ep{episode}.pth"))

        csv_file.close()

        torch.save(actors[0].action_net.state_dict(),
                   os.path.join(ckpt_dir, "actor_L_final.pth"))
        if n_total > 1:
            torch.save(actors[1].action_net.state_dict(),
                       os.path.join(ckpt_dir, "actor_F_final.pth"))

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
