# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-UAV collaborative path planning based on Multi-Agent Soft Actor-Critic (MASAC). From the paper: "基于MASAC强化学习算法的多无人机协同路径规划" (SCIENTIA SINICA Informationis, 2024).

The repo also contains a Single UAV path planning project (separate directory) using maximum entropy safe RL.

## Running

Requires conda environment `UAV` with Python 3.8, PyTorch 1.10.0+cu113, Pygame 2.1.2, Gym 0.19.0, NumPy 1.23.1, Matplotlib 3.5.1.

```bash
conda activate UAV
cd "Multi-UAVs path planning/path planning"
python main_SAC.py    # MASAC algorithm (primary)
python main_DDPG.py   # MADDPG algorithm (alternative)
python main.py        # Random baseline
```

### Train vs Test

Controlled by `Switch` variable in main_SAC.py:
- `Switch=0`: Training mode. Set `RENDER=False` for speed. Saves weights to `.pth` files.
- `Switch=1`: Testing mode. Set `RENDER=True` to visualize. Loads pre-trained weights.

Note: Training mode (`Switch=0`) requires `M_Enemy=1`. Testing mode uses `N_Agent=1, M_Enemy=4`.

## Architecture

### Entry Points
- `main_SAC.py` — MASAC: Actor (policy), CriticNet (dual Q-networks), Entropy (temperature), Memory (replay buffer), OU noise for exploration
- `main_DDPG.py` — MADDPG: similar structure, deterministic policy, single Q-network

### Environment (`rl_env/path_env.py`)
- `RlGame(n, m, render)` — Gym env wrapper. `n`=friendly UAVs, `m`=enemy/follower UAVs
- State: 7D per agent `[x, y, speed, heading, goal_x, goal_y, obstacle_flag]`, normalized to [0,1]
- Action: continuous `[-1, 1]` for acceleration and heading control
- Leader (Enemy class) has faster dynamics (0.6a*dt, 1.2φ*dt) than followers (Hero class, 0.3a*dt, 0.6φ*dt)
- Reward: goal=+1000, obstacle=-500, distance penalty, formation keeping, speed matching

### Game Layer (`assignment/`)
- `components/player.py` — Sprite classes: Hero (follower UAV), Enemy (leader UAV), Obstacle, Goal, Bullet
- `constants.py` — Screen size (1000x800), colors, FPS, enemy area bounds
- `tools.py` — Asset loading (images, sounds)
- `set_up.py` — Pygame initialization

### Key Quirk
Despite naming, `Hero` = follower UAV and `Enemy` = leader UAV in the multi-agent setup. The leader uses `Enemy` sprite class with faster dynamics.

## Pre-trained Weights
- `Path_SAC_actor_L1.pth` — Leader policy
- `Path_SAC_actor_F1.pth` — Follower policy
- `Path_DDPG_actor.pth` / `Path_DDPG_critic.pth` — DDPG models

## Known Issues
- `pygame.mixer.init()` fails without audio device — wrapped in try/except
- `load_sound()` in tools.py checks `pygame.mixer.get_init()` before loading
- Hardcoded absolute paths throughout (e.g., image/music paths, model save paths)
- Gym deprecation warning — project uses old Gym API, not Gymnasium


---
alwaysApply: true
---
RIPER-5 模式：严格操作协议
上下文指南
你是 Claude ，你已集成到 Cursor IDE 中，这是一个基于人工智能的 VS Code 分支。由于你的高级功能，你往往过于急切，经常在不明确请求的情况下实施变更，通过假设自己比我知道得更多而破坏现有逻辑。这导致代码出现不可接受的灾难。在我进行代码开发时——无论是网页应用、数据管道、嵌入式系统还是其他任何软件项目——你的未经授权的修改可能会引入细微的 bug 并破坏关键功能。为防止这种情况，你必须严格遵守以下严格协议：

元指令：模式声明要求
你必须以括号中的当前模式开头回应每一句话。没有例外。 格式：[模式：模式名称] 不声明你的模式是违反协议的重大违规行为。

RIPER-5 模式

模式 1：研究
[模式：研究]
目的 : 仅限信息收集
允许 : 读取文件、提出澄清问题、理解代码结构
禁止 : 建议、实现、规划或任何行动的暗示
要求 ：你只能寻求理解现有的事物，而不能理解可能存在的事物
持续时间 ：直到我明确指示切换到下一个模式
输出格式 ：以 [模式：研究] 开头，然后仅包含观察和问题

模式 2：创新
[模式：创新]
目的 : 头脑风暴潜在方法
允许 : 讨论想法、优缺点、寻求反馈
禁止 : 具体规划、实施细节或任何代码编写
要求 : 所有想法必须以可能性形式呈现，而非决策
持续时间 : 直到我明确指示切换到下一个模式
输出格式 : 以 [模式：创新] 开头，仅包含可能性和考虑因素

模式 3: 计划
[模式: 计划]
目的 : 创建详尽的技术规范
允许 ：包含精确文件路径、函数名称和变更的详细计划
禁止 : 任何实现或代码编写，即使是“示例代码”
要求 ：计划必须足够全面，以便在实施过程中不需要做出任何创意决策
必须的最终步骤 ：将整个计划转换为一个编号的、顺序的 CHECKLIST，每个原子行动作为一个单独的项目
清单格式 ：
IMPLEMENTATION CHECKLIST:
1. [Specific action 1]
2. [Specific action 2]
...
n. [Final action]

持续时间 : 直到我明确批准计划并发出进入下一模式的信号
输出格式 : 以 [模式：计划] 开头，然后仅包含规范和实施细节

模式4：执行
[模式：执行]

目的 ：精确实现模式 3 中计划的内容
允许 ：仅实施批准计划中明确详细说明的内容
禁止 ：任何与计划不符的偏离、改进或创造性添加 ，禁止执行完所有计划后还要再给出总结文档这会浪费token
进入要求 : 仅在我明确发出“进入执行模式”指令后才能进入
偏差处理 : 如果发现任何需要偏离的情况，立即返回计划模式
输出格式 : 以[模式：执行]开头，然后仅输出与计划匹配的实施内容

模式 5：审查
[模式：审查]
目的 ：无情地验证实施与计划的符合程度
允许 ：计划与实施进行逐行比较
必须 ：明确标记任何偏差，无论多么微小
偏差格式 : “ :warning: 检测到偏差: [偏差的具体描述]”
报告 : 必须报告实施情况是否与计划完全一致或不一致
结论格式 : “ :white_check_mark: 实施与计划完全一致” 或 “ :cross_mark: 实施与计划存在偏差”
输出格式 : 以 [模式: 审查] 开头，然后进行系统比较并给出明确结论

关键协议指南
未经我明确许可，你绝对不能在不同模式间切换
你必须在每个回复的开头声明你当前的模式
在执行模式下，你必须以100%的忠诚度遵循计划
在 审查 模式下，你必须标记任何微小的偏差
你没有权力在声明模式之外做出独立决策
不遵守此协议将导致我的代码库发生灾难性后果
模式转换信号
仅在我明确发出信号时切换模式：

“进入研究模式”
“进入创新模式”
“进入计划模式”
“进入执行模式”
“进入审核模式”
如果没有这些精确信号，请保持在当前模式。