# 训练地形(TronCamp / HumanoidCamp)

用于**训练**你的 TRON2(Tron2ATask)或 Oli(OliTask)策略的随机化、宽阔/开放的 IsaacLab 地形。它
囊括了评测赛道的**每一种地形要素**——起始平地 → 上行斜坡 → 山顶高台 →
下行斜坡 → 坎(bumps) → 上行台阶 → 带沟壑(gullies)的桥 → 下行台阶 → 抬高高台
(踏脚石) → 终点平地——但会**在每个地形网格上以程序化方式随机化每一个参数**,
让你的策略学到的是技能本身,而不是去记忆某一个固定布局。

你是在**固定评测赛道上被评分的**,而非这个地形。在这里训练是为了泛化。

## 文件

| 文件 | 说明 |
| --- | --- |
| `mixed_terrain.py` | 随机化构建器:`make_mixed_track`、`MixedTrackCfg`(所有可调范围)、`make_training_terrain_cfg(...)`。 |
| `tron_camp_training_terrain.py` | TRON2/Tron2ATask 预设 → `TASK_C_TRAINING_TERRAIN_CFG`、`TRON2_SPAWN_Z`。 |
| `humanoid_camp_training_terrain.py` | Oli/OliTask 预设 → `TASK_F_TRAINING_TERRAIN_CFG`、`OLI_SPAWN_Z`。 |
| `terrain_base.py` | `BetterTerrainGenerator` / `BetterTerrainImporter`。增加了一个永久性保护:每个基本体都必须具有正体积(法线朝外),否则就会抛出异常——PhysX 三角碰撞体是单面的,因此一个翻转的 mesh 会让机器人直接穿模掉落。 |

## 依赖要求

- IsaacLab(v2.3.x)+ Isaac Sim、`trimesh`、`numpy`。地形本身不需要其他任何依赖。
- 把此文件夹放到 `PYTHONPATH` 上(各模块之间以顶层名相互 import,例如
  `from terrain_base import ...`)。例如:`sys.path.insert(0, ".../participant_kit/training_terrain")`。

## 在你的训练环境中使用

```python
import sys
sys.path.insert(0, "/path/to/participant_kit/training_terrain")
from tron_camp_training_terrain import TASK_C_TRAINING_TERRAIN_CFG, TRON2_SPAWN_Z

# 在你的 ManagerBasedRLEnvCfg 中:
env_cfg.scene.terrain = TASK_C_TRAINING_TERRAIN_CFG
# 在网格原点生成机器人 base,z = TRON2_SPAWN_Z(Oli:OLI_SPAWN_Z = 1.0)。
```

一个 `num_rows × num_cols` 的网格(默认 10×10 = 100 个网格)一次性给出 100 个互不相同的随机化赛道;
`difficulty` 随行数升高,形成一条课程式难度曲线。它可以直接插入 IsaacLab 的 velocity-locomotion
训练脚手架或 `tron2_rl_lab`。

## 调参

编辑 `MixedTrackCfg` 上的各范围(全为 `(low, high)`),或向
`make_training_terrain_cfg(...)` 传入你自己的 `track_cfg=`:

```python
from mixed_terrain import MixedTrackCfg, make_training_terrain_cfg
t = MixedTrackCfg()
t.step_height_range = (0.08, 0.14)     # 更平缓的台阶
t.gap_width_range   = (0.10, 0.25)     # 更窄的沟壑
cfg = make_training_terrain_cfg(width=8.0, num_rows=20, num_cols=20, track_cfg=t)
```

- `width` —— mesh 走廊宽度(默认 8 m TRON2 / 7 m Oli;评测走廊为 3 m / 5 m)。
- `n_*_range` —— 地形要素数量(台阶、坎、沟壑、高台)。地形要素**顺序固定**;只有
  参数会变化,因此每个赛道仍然会演练到每一项技能。
- `cell_length` 必须超过你的范围所能产生的最长赛道(超出部分是一段平地尾巴)。

## 与评测赛道的关系

此地形被刻意设计成**并非**评测赛道。评测赛道是单一确定性的
布局(Tron2ATask:107 m × 3 m 走廊;OliTask:68 m × 5 m 走廊),包含相同的地形要素。这个
训练地形会随机化每一种布局,从而让你的策略泛化,而不是去拟合某一个赛道。
