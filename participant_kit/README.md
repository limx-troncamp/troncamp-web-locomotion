# 选手资源包 —— TRON2（Tron2Task）与 Oli（OliTask）

**训练策略**、**构建 `solution.py`** 所需的一切，面向 TronCamp（TRON2）与 HumanoidCamp（Oli）两个
基准。本目录是一个自包含的静态资源包；`manifest.json` 是供 Web 前端列出并链接这些资源的机器可读索引。

## 内容

| 路径 | 说明 |
| --- | --- |
| [`training_terrain/`](training_terrain/) | 随机化、宽阔开放的训练地形（覆盖评测的全部地形要素；布局随机，避免过拟合单一赛道）。`TASK_C_TRAINING_TERRAIN_CFG` / `TASK_F_TRAINING_TERRAIN_CFG`。 |
| [`solution_guide/`](solution_guide/) | `solution_build_guide.md`（obs/action 接口约定、传感器索引表、速度指令）、`solution_template.py`、`obs_schema.py`。 |
| `manifest.json` | 全部资源 + 外部链接的索引（供 Web 前端使用）。 |

## TRON2 开源仓库

| 仓库 | 用途 |
| --- | --- |
| [tron2_rl_lab](https://github.com/limx-tron2/tron2_rl_lab) | TRON2 强化学习训练框架 —— 在这里训练你的策略。 |
| [TRON2_YG_LAB](https://github.com/limx-tron2/TRON2_YG_LAB) | TRON2 训练 Lab（YG 变体，含足式 `sfyg` 与轮足 `wfyg` 配置）。 |
| [robot-description](https://github.com/limx-tron2/robot-description) | TRON2 机器人描述（URDF / 网格 / 配置）。 |

## Oli 开源仓库

| 仓库 | 用途 |
| --- | --- |
| [humanoid-description](https://github.com/limxdynamics/humanoid-description) | Oli 人形（串联）机器人描述（URDF / 网格 / 配置）；评测即采用该串联模型。 |
| [humanoid-rl-deploy-python](https://github.com/limxdynamics/humanoid-rl-deploy-python) | 官方部署代码与参考 policy；可按需借鉴（不保证可直接用于本赛题）。 |

## 快速开始

1. **训练** —— 把训练地形接入你的环境（例如在 `tron2_rl_lab` 内）：
   ```python
   import sys; sys.path.insert(0, "participant_kit/training_terrain")
   from tron_camp_training_terrain import TASK_C_TRAINING_TERRAIN_CFG
   env_cfg.scene.terrain = TASK_C_TRAINING_TERRAIN_CFG   # 随机化、宽阔；但最终按固定的评测赛道评分
   ```
   Oli 人形请改用 `humanoid_camp_training_terrain.py`；机器人描述见上方「Oli 开源仓库」的
   [humanoid-description](https://github.com/limxdynamics/humanoid-description)（串联模型）。

2. **构建你的方案** —— 复制 `solution_guide/solution_template.py` → `solution.py`，实现
   `predicts(obs, current_score)`。阅读 `solution_guide/solution_build_guide.md` 里的**索引表**
   （TRON2：proprio `[6:9]` 是朝向终点的速度指令，`action_dim = (D-12)//3`；Oli：`[6:10]`，`(D-13)//3`）。

3. 用提交 CLI **提交**：你的 `--code-dir` 根目录必须有 `solution.py`，权重通过
   `--ckpt-file policy.pt` 提交（运行时会放在 `solution.py` 旁边）。`--robot` 选择赛题 ——
   `sfyg_tron2a` / `wfyg_tron2a` = TRON2 腿式 / 轮足，`oli` = 人形。详见
   `solution_guide/solution_build_guide.md` §6。

> 训练地形**刻意不是**评分赛道 —— 它随机化每一处布局，让你的策略学会泛化，而不是死记一条固定赛道。
