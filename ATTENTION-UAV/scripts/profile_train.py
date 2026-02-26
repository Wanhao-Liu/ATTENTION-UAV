# -*- coding: utf-8 -*-
"""训练循环性能分析 - 测量每个操作耗时"""
import os, sys, time, argparse
import numpy as np, torch
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules.agent import Actor, Critic, Entropy, resolve_device
from modules.per_memory import PrioritizedReplayBuffer
from modules.memory import Memory
from modules.noise import OrnsteinUhlenbeckNoise

class Timer:
    def __init__(self):
        self.times = defaultdict(list)
    def start(self, n):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self._n, self._t = n, time.perf_counter()
    def stop(self):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self.times[self._n].append(time.perf_counter()-self._t)
    def report(self):
        total, rows = 0, []
        for n, v in self.times.items():
            a = np.mean(v)*1000; total += a; rows.append((n,a,len(v)))
        rows.sort(key=lambda x:-x[1])
        print("\n"+"="*60+"\n性能分析报告 (每步平均耗时)\n"+"="*60)
        for n,a,c in rows:
            print(f"  {n:<30s}{a:8.3f}ms ({a/total*100:5.1f}%) [n={c}]")
        print("-"*60+f"\n  {'TOTAL':<30s}{total:8.3f}ms\n"+"="*60)

def profile():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config",default="masac_per_attn",
        choices=["baseline","masac_per","masac_attn","masac_per_attn"])
    pa.add_argument("--profile_steps",type=int,default=200)
    args = pa.parse_args()
    cfgmap = {"baseline":"config.baseline:BaselineConfig",
              "masac_per":"config.masac_per:MasacPerConfig",
              "masac_attn":"config.masac_attn:MasacAttnConfig",
              "masac_per_attn":"config.masac_per_attn:MasacPerAttnConfig"}
    mod, cls = cfgmap[args.config].split(":")
    import importlib; cfg = getattr(importlib.import_module(mod), cls)()
    device = resolve_device(cfg)
    os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    from env.path_env import RlGame
    env = RlGame(n=cfg.n_agent_train,m=cfg.m_enemy_train,render=False).unwrapped
    nt = cfg.n_agent_train+cfg.m_enemy_train
    sd, ad = cfg.state_dim, cfg.action_dim
    md = 2*sd*nt + ad*nt + nt
    actors=[Actor(cfg) for _ in range(nt)]
    critics=[Critic(cfg) for _ in range(nt)]
    ents=[Entropy(cfg) for _ in range(nt)]
    if cfg.use_per:
        buf=PrioritizedReplayBuffer(cfg.memory_capacity,md,cfg.per_alpha,
            cfg.per_beta_start,cfg.per_beta_end,cfg.per_beta_steps,cfg.per_eps)
    else:
        buf=Memory(cfg.memory_capacity,md)
    ous=[OrnsteinUhlenbeckNoise(ad,cfg.ou_mu,cfg.ou_theta,cfg.ou_sigma)
         for _ in range(nt)]
    print(f"[Profile] {args.config} device={device} PER={cfg.use_per} "
          f"Attn={cfg.use_attention} steps={args.profile_steps}")
    print("填充buffer..."); obs=env.reset(); act=np.zeros((nt,ad))
    while not buf.is_ready:
        for i in range(nt): act[i]=ous[i].noise()
        o2,r,done,*_=env.step(act)
        buf.store_transition(obs.flatten(),act.flatten(),r.flatten(),o2.flatten())
        obs=env.reset() if done else o2
    print(f"  buffer ready, size={len(buf)}")
    run_profile_loop(args, cfg, env, actors, critics, ents, buf, ous,
                     nt, sd, ad, device)

def run_profile_loop(args,cfg,env,actors,critics,ents,buf,ous,nt,sd,ad,device):
    T=Timer(); obs=env.reset(); act=np.zeros((nt,ad)); p=0
    while p < args.profile_steps:
        T.start("1_choose_action")
        for i in range(nt):
            if cfg.use_attention and nt>1:
                oi=[j for j in range(nt) if j!=i]
                act[i]=actors[i].choose_action(obs[i],obs[oi])
            else:
                act[i]=actors[i].choose_action(obs[i])
        T.stop()
        T.start("2_env_step")
        obs_,rew,done,win,tc,d=env.step(act)
        T.stop()
        T.start("3_store_transition")
        buf.store_transition(obs.flatten(),act.flatten(),rew.flatten(),obs_.flatten())
        T.stop()
        T.start("4_buffer_sample")
        if cfg.use_per:
            batch,tidx,isw=buf.sample(cfg.batch_size)
        else:
            batch=buf.sample(cfg.batch_size); tidx=None; isw=None
        T.stop()
        T.start("5_to_tensor_GPU")
        b_s=torch.FloatTensor(batch[:,:sd*nt]).to(device)
        oa=sd*nt
        b_a=torch.FloatTensor(batch[:,oa:oa+ad*nt]).to(device)
        orr=oa+ad*nt
        b_r=torch.FloatTensor(batch[:,orr:orr+nt]).to(device)
        osn=orr+nt
        b_s_=torch.FloatTensor(batch[:,osn:]).to(device)
        T.stop()
        td_all=[]
        for i in range(nt):
            si,ai=sd*i,ad*i
            bo=b_s[:,si:si+sd]; bo_=b_s_[:,si:si+sd]
            oo=oo_=None
            if cfg.use_attention and nt>1:
                others=[j for j in range(nt) if j!=i]
                oo=torch.stack([b_s[:,sd*j:sd*(j+1)] for j in others],1)
                oo_=torch.stack([b_s_[:,sd*j:sd*(j+1)] for j in others],1)
            T.start("6_eval_targetQ")
            na,lp_=actors[i].evaluate(bo_,oo_)
            tq1,tq2=critics[i].target_get_v(b_s_,na)
            tq=b_r[:,i:i+1]+cfg.gamma*(torch.min(tq1,tq2)-ents[i].alpha*lp_)
            T.stop()
            T.start("7_currentQ")
            cq1,cq2=critics[i].get_v(b_s,b_a[:,ai:ai+ad])
            T.stop()
            T.start("8_critic_learn")
            critics[i].learn(cq1,cq2,tq.detach(),isw)
            T.stop()
            T.start("9_eval_actorQ")
            ai2,lp=actors[i].evaluate(bo,oo)
            q1,q2=critics[i].get_v(b_s,ai2)
            T.stop()
            T.start("10_actor_learn")
            aloss=(ents[i].alpha*lp-torch.min(q1,q2)).mean()
            actors[i].learn(aloss)
            T.stop()
            T.start("11_entropy_learn")
            eloss=-(ents[i].log_alpha.exp()*(lp+ents[i].target_entropy).detach()).mean()
            ents[i].learn(eloss)
            T.stop()
            T.start("12_soft_update")
            critics[i].soft_update()
            T.stop()
            with torch.no_grad():
                td=torch.abs(cq1-tq.detach()).squeeze(-1).cpu().numpy()
            td_all.append(np.atleast_1d(td))
        if cfg.use_per and tidx is not None:
            T.start("13_PER_update_pri")
            buf.update_priorities(tidx,np.max(np.stack(td_all,0),0))
            T.stop()
        obs=env.reset() if done else obs_
        p+=1
        if p%50==0: print(f"  profiled {p}/{args.profile_steps}")
    T.report()
    env.close()

if __name__=="__main__":
    profile()
