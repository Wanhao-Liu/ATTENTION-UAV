# -*- coding: utf-8 -*-
"""ATTENTION-UAV 统一测试脚本"""
import os
import sys
import csv
import pickle
import argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.agent import Actor, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="MASAC Ablation Test")
    parser.add_argument("--config", type=str, default="baseline",
                        choices=["baseline", "masac_per", "masac_attn",
                                 "masac_per_attn"])
    parser.add_argument("--leader_ckpt", type=str, default=None)
    parser.add_argument("--follower_ckpt", type=str, default=None)
    parser.add_argument("--n_agent", type=int, default=None)
    parser.add_argument("--m_enemy", type=int, default=None)
    parser.add_argument("--test_episodes", type=int, default=100)
    parser.add_argument("--ep_len", type=int, default=1000)
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


def test():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg)

    n_agent = args.n_agent if args.n_agent else cfg.n_agent_test
    m_enemy = args.m_enemy if args.m_enemy else cfg.m_enemy_test
    n_total = n_agent + m_enemy
    a_dim = cfg.action_dim
    ep_len = args.ep_len
    render = args.render

    result_dir = os.path.join(ROOT, "results", cfg.exp_name)
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(result_dir, exist_ok=True)

    # 环境
    if not render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from env.path_env import RlGame
    env = RlGame(n=n_agent, m=m_enemy, render=render).unwrapped

    # 加载 actors
    leader_ckpt = args.leader_ckpt or os.path.join(
        ckpt_dir, "actor_L_final.pth")
    follower_ckpt = args.follower_ckpt or os.path.join(
        ckpt_dir, "actor_F_final.pth")

    actors = []
    # Leader (agent 0)
    leader = Actor(cfg)
    ckpt_data = torch.load(leader_ckpt, map_location=device)
    if "net" in ckpt_data:
        leader.action_net.load_state_dict(ckpt_data["net"])
    else:
        leader.action_net.load_state_dict(ckpt_data)
    actors.append(leader)

    # Followers (agents 1..n_total-1)
    for i in range(1, n_total):
        follower = Actor(cfg)
        fdata = torch.load(follower_ckpt, map_location=device)
        if "net" in fdata:
            follower.action_net.load_state_dict(fdata["net"])
        else:
            follower.action_net.load_state_dict(fdata)
        actors.append(follower)

    print(f"[{cfg.exp_name}] 测试开始 | n={n_agent} m={m_enemy} "
          f"episodes={args.test_episodes}")

    # 测试指标
    win_times = 0
    all_ep_V, all_ep_U, all_ep_T, all_ep_F = [], [], [], []
    trajectories = []
    action = np.zeros((n_total, a_dim))

    for j in range(args.test_episodes):
        state = env.reset()
        integral_V = 0.0
        integral_U = 0.0
        traj_ep = []

        for timestep in range(ep_len):
            for i in range(n_total):
                if cfg.use_attention and n_total > 1:
                    other_idx = [k for k in range(n_total) if k != i]
                    other_obs = state[other_idx]
                    action[i] = actors[i].choose_action(state[i], other_obs)
                else:
                    action[i] = actors[i].choose_action(state[i])

            new_state, reward, done, win, team_counter, dis = env.step(action)

            if win:
                win_times += 1

            # 记录指标
            integral_V += float(state[0][2])  # leader speed
            integral_U += float(abs(action[0]).sum())

            # 轨迹记录
            traj_ep.append({
                "step": timestep,
                "positions": [s[:2].tolist() for s in state],
                "actions": action.copy().tolist()
            })

            state = new_state
            if render:
                env.render()
            if done:
                break

        fkr = team_counter / max(timestep + 1, 1)
        all_ep_V.append(integral_V)
        all_ep_U.append(integral_U)
        all_ep_T.append(timestep + 1)
        all_ep_F.append(fkr)
        trajectories.append(traj_ep)

        if j % 10 == 0:
            print(f"  ep={j} steps={timestep+1} win={win} FKR={fkr:.3f}")

    # 汇总
    win_rate = win_times / args.test_episodes
    avg_V = np.mean(all_ep_V)
    avg_U = np.mean(all_ep_U)
    avg_T = np.mean(all_ep_T)
    avg_F = np.mean(all_ep_F)

    print(f"\n[{cfg.exp_name}] 测试结果:")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Avg FKR:  {avg_F:.4f}")
    print(f"  Avg V:    {avg_V:.2f}")
    print(f"  Avg U:    {avg_U:.2f}")
    print(f"  Avg T:    {avg_T:.1f}")

    # 保存 CSV
    csv_path = os.path.join(result_dir, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "timesteps", "FKR", "integral_V",
                     "integral_U"])
        for j in range(args.test_episodes):
            w.writerow([j, all_ep_T[j], f"{all_ep_F[j]:.4f}",
                         f"{all_ep_V[j]:.4f}", f"{all_ep_U[j]:.4f}"])
        w.writerow(["MEAN", f"{avg_T:.1f}", f"{avg_F:.4f}",
                     f"{avg_V:.4f}", f"{avg_U:.4f}"])

    # 保存 pickle
    test_data = {
        "win_rate": win_rate, "all_ep_V": all_ep_V,
        "all_ep_U": all_ep_U, "all_ep_T": all_ep_T,
        "all_ep_F": all_ep_F, "trajectories": trajectories,
        "cfg": cfg.exp_name
    }
    pkl_path = os.path.join(result_dir, "test_data.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(test_data, f)

    print(f"  结果保存至 {result_dir}")
    env.close()


if __name__ == "__main__":
    test()
