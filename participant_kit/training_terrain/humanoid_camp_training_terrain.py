# HumanoidCamp (Oli) 训练地形 —— 随机化、宽阔开放，覆盖评测的全部地形要素。
#
# 用它来训练 Oli 人形。你最终按固定的评测赛道评分，而不是这条训练地形。
# 接入方式见 README.md。
#
#   from humanoid_camp_training_terrain import HUMANOID_CAMP_TRAINING_TERRAIN_CFG, OLI_SPAWN_Z
#   env_cfg.scene.terrain = HUMANOID_CAMP_TRAINING_TERRAIN_CFG

from __future__ import annotations

from mixed_terrain import MixedTrackCfg, make_training_terrain_cfg

# Oli 站立高度（与评测环境中 OLI_EDU_CFG 的初始 z 一致）。
OLI_SPAWN_Z = 1.0

# Oli 的评测走廊本就是 5 m；训练时加宽到 7 m 留出余量。
TRACK_WIDTH = 7.0

# 人形：把台阶 / 高台调得比轮足 TRON2 的默认值略缓一些。
_track = MixedTrackCfg()
_track.step_height_range = (0.10, 0.16)
_track.platform_height_range = (0.15, 0.45)
_track.bump_height_range = (0.08, 0.28)

# 10 x 10 = 100 条各自独立随机化的赛道（难度随行递增）。按你的 GPU 调整。
HUMANOID_CAMP_TRAINING_TERRAIN_CFG = make_training_terrain_cfg(
    width=TRACK_WIDTH,
    cell_length=80.0,
    num_rows=10,
    num_cols=10,
    base_seed=0,
    track_cfg=_track,
)
