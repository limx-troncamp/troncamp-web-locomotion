"""obs 的（反）序列化 —— obs↔multipart 传输格式的唯一定义，按任务区分。

评测沙箱的两端都用它，以保证不漂移：
  * 打包端    → `pack_obs_to_files(obs, schema)`   把环境 obs dict 打包成 multipart 文件。
  * 重建端    → `rebuild_obs(raw, schema)`          从收到的字节重建 obs dict。

纯 numpy：不依赖 torch、不依赖 Isaac、不依赖 HTTP。任务按 key 选取自己的 schema（见
`schema_key_for_task`）；发送端把该 key 放进 `task` multipart 字段，接收端据此查到同一份 schema。

每个字段的传输格式：`np.<dtype>` 的原始小端、C 连续字节块，单样本形状为 `shape`；打包/重建时带一个
为 1 的前置 batch 维（即数组形状 == (1, *shape)）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    """传输中的一个 obs 字段。

    group == "top"   → 位于 obs[name]            （proprio / extero / goal / …）
    group == "image" → 位于 obs["image"][name]   （相机张量）
    """

    name: str
    dtype: str           # numpy dtype 字符串，如 "float32" | "uint8"
    shape: tuple         # 不含 batch 维的单样本形状，如 (480, 640, 3)
    group: str           # "top" | "image"
    required: bool = True


# --- 各任务的 schema -----------------------------------------------------------------------------
# key 即发送端发来的 `task` 值。proprio 对 TRON2 腿式与轮足都是 66（18 个关节：12 + 3*18）。`extero`
# 是完整展平的激光雷达高度扫描（channels*360 条射线），不是通道数：REAL Fairy96 = 96*360 = 34560；
# ORIG 16 通道 = 16*360 = 5760。REAL 与 ORIG 都发 `head_*`/`ee_*`；REAL 额外发 `down_*`。（旧契约的
# `video_*` 分支是死代码 —— 真实环境从不产出它 —— 且漏了 `down_*`；这里都已修正。）
# `humanoid`（OliTask SERIAL，ATEC-TaskF-OliEdu）：106 维原始 proprio + 头部 D435i 深度图（唯一外感）。
# proprio 顺序 = base_lin_vel(3) | base_ang_vel(3) | velocity_commands(4) | projected_gravity(3) |
# joint_pos_rel(31) | joint_vel_rel(31) | last_action(31)，关节顺序见 OLI_EDU_JOINT_NAMES；数值为原始
# 真值（不做预缩放，选手自行归一化）。无 extero/rgb/goal；特权高度扫描不进 obs。
OBS_SCHEMAS: dict[str, list[FieldSpec]] = {
    # TRON2 REAL 传感器套件：Fairy96 激光雷达高度扫描 + 前/下 D435i RGB-D + 末端手眼相机（EE）。
    "tron-real": [
        FieldSpec("proprio", "float32", (66,), "top"),
        FieldSpec("extero", "float32", (34560,), "top", required=False),
        FieldSpec("head_rgb", "uint8", (480, 640, 3), "image", required=False),
        FieldSpec("head_depth", "float32", (480, 640, 1), "image", required=False),
        FieldSpec("ee_rgb", "uint8", (480, 640, 3), "image", required=False),
        FieldSpec("ee_depth", "float32", (480, 640, 1), "image", required=False),
        FieldSpec("down_rgb", "uint8", (480, 640, 3), "image", required=False),
        FieldSpec("down_depth", "float32", (480, 640, 1), "image", required=False),
    ],
    # TRON2 原装套件（-Orig 变体）：16 通道激光雷达高度扫描 + 前向 D435i RGB-D + EE。图像 key 与
    # REAL 相同，仅少了下视相机。
    "tron-orig": [
        FieldSpec("proprio", "float32", (66,), "top"),
        FieldSpec("extero", "float32", (5760,), "top", required=False),
        FieldSpec("head_rgb", "uint8", (480, 640, 3), "image", required=False),
        FieldSpec("head_depth", "float32", (480, 640, 1), "image", required=False),
        FieldSpec("ee_rgb", "uint8", (480, 640, 3), "image", required=False),
        FieldSpec("ee_depth", "float32", (480, 640, 1), "image", required=False),
    ],
    # Oli 人形（OliTask SERIAL）：106 维原始 proprio + 头部深度图（唯一外感）；无 extero/rgb/goal。
    "humanoid": [
        FieldSpec("proprio", "float32", (106,), "top"),
        FieldSpec("head_depth", "float32", (60, 106, 1), "image", required=False),
    ],
}


def schema_key_for_task(task_id: str) -> str:
    """把 gym task id 映射到 schema key。未知任务抛 KeyError。"""
    t = task_id.lower()
    if "taskc" in t or "tron" in t:
        return "tron-orig" if t.endswith("-orig") else "tron-real"
    if "taskf" in t or "oli" in t:
        return "humanoid"
    raise KeyError(f"no obs schema registered for task {task_id!r}")


def get_schema(key_or_task: str) -> list[FieldSpec]:
    """既接受 schema key（'tron-real'），也接受 gym task id（如 'ATEC-…-Tron2ALegged'）。"""
    if key_or_task in OBS_SCHEMAS:
        return OBS_SCHEMAS[key_or_task]
    return OBS_SCHEMAS[schema_key_for_task(key_or_task)]


# --- （反）序列化 --------------------------------------------------------------------------------
def _to_numpy(t):
    """torch tensor / ndarray / list → ndarray，不导入 torch。"""
    if hasattr(t, "detach"):
        t = t.detach().to("cpu").numpy()
    return np.asarray(t)


def _field_value(obs, spec: FieldSpec):
    if spec.group == "image":
        return (obs.get("image") or {}).get(spec.name)
    return obs.get(spec.name)


def pack_obs_to_files(obs, schema: list[FieldSpec]) -> dict:
    """env obs dict → {field_name: (filename, bytes)}，用于 multipart POST。

    字段当且仅当存在于 `obs` 时才发出；缺失的 `required` 字段会抛异常。
    """
    files: dict = {}
    for spec in schema:
        val = _field_value(obs, spec)
        if val is None:
            if spec.required:
                raise KeyError(f"required obs field missing on pack: {spec.name}")
            continue
        arr = _to_numpy(val).astype(np.dtype(spec.dtype), copy=False)
        files[spec.name] = (f"{spec.name}.bin", np.ascontiguousarray(arr).tobytes())
    return files


def rebuild_obs(raw: dict, schema: list[FieldSpec]) -> dict:
    """{field_name: bytes} → obs dict {top 字段…, 'image': {…}}。是 pack_obs_to_files 的逆操作。

    返回形状为 (1, *spec.shape) 的 numpy 数组；调用方按需用 torch 包装。缺失的 `required` 字段会
    抛异常；缺失的可选字段被跳过（若没有任何 image 字段到达，则整个 'image' 组省略）。
    """
    obs: dict = {}
    image: dict = {}
    for spec in schema:
        b = raw.get(spec.name)
        if b is None:
            if spec.required:
                raise KeyError(f"required obs field missing on rebuild: {spec.name}")
            continue
        arr = np.frombuffer(b, dtype=np.dtype(spec.dtype)).reshape(1, *spec.shape)
        if spec.group == "image":
            image[spec.name] = arr
        else:
            obs[spec.name] = arr
    if image:
        obs["image"] = image
    return obs
