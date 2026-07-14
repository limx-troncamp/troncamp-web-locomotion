# `solution.py` 构建指南 —— TronCamp & HumanoidCamp

你提交一份 **`solution.py`**（即你的 `AlgSolution` 类）以及训练好的权重（例如 `policy.pt`）。
评测器构建环境，构造一次你的 `AlgSolution`，在每个 episode 开始时调用 `reset()`，然后
每个控制步调用 `predicts(obs, current_score)` 并把动作施加到机器人上。你的任务是：
把 `obs` 转换成一个动作。

- `solution_template.py` —— 可复制粘贴的起点（以零动作运行；若存在 `policy.pt` 则加载它）。
- `obs_schema.py` —— 观测 ↔ 线缆（wire）的（反）序列化，供参考 / 本地测试使用。平台在评测时也会
  在你的运行目录中提供它，因此你无需提交自己的副本（见 §6）。

---

## 1. `AlgSolution` 接口

```python
class AlgSolution:
    def __init__(self): ...                       # 在此加载你的 policy（只运行一次）
    def reset(self, **kwargs): ...                # 可选 —— 每个 episode 清除循环/历史状态
    def get_action_spec(self) -> dict | None: ... # 可选 —— 动作如何被解释（见 §4）
    def predicts(self, obs, current_score) -> dict:
        # 必需 —— 每个控制步被调用
        return {"action": [...], "giveup": False}
```

- `predicts` 必须返回 `{"action": <flat list of `action_dim` floats>, "giveup": <bool>}`。
- 对于 **TRON2**，`action_dim = (obs["proprio"].shape[-1] - 12) // 3`（固定头部 = 12）；对于 **Oli**，
  固定头部 = 13，因此 `action_dim = (D - 13) // 3`（见 §3）。动作向量按机器人固定的关节顺序施加（你不选择
  顺序；你通过 `get_action_spec` 选择各分组的 *模式/缩放*）。
- `giveup: True` 提前结束 episode，保留当前已得的分数。
- `current_score` 是运行中的当前分数（float），供你参考。

> **控制 / 仿真频率（TronCamp 与 HumanoidCamp 一致）** —— 评测每个**控制步**调用一次 `predicts`：
> 控制 / 推理频率 = **50 Hz**（每 20 ms 一步）；底层物理仿真 = **200 Hz**（`sim.dt = 0.005 s`），每个控制步
> 推进 `decimation = 4` 个物理子步（4 × 5 ms = 20 ms）。两个赛题一致。传感器刷新是另一档：TRON2 相机 /
> Fairy96 约 **10 Hz**（`update_period = 0.1 s`）、Oli 头部深度相机 **50 Hz**。**请按 50 Hz 训练 / 适配你的
> policy** —— 若按别的控制频率训练（如官方 deploy walk 控制器 100 Hz），评测仍以 50 Hz 驱动、步态可能失稳
> （另见 §4）。

---

## 2. `obs` 字典 —— 你每步收到的内容

`obs` 是一个字典，每个值都带有前导的批维度 1。**评测时这些值是 GPU（CUDA）上的 torch tensor** —— 沙箱容器
在 GPU 上驱动你的 `predicts`。

> **⚠️ 用 numpy 处理前先搬到 CPU**：对 CUDA tensor 直接 `np.asarray(obs["proprio"])` 会抛
> `TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() ...`。先转 CPU：
> `obs["proprio"].detach().cpu().numpy()`；或沿用 torch（模板即用 `torch.as_tensor(obs[...], device=self.device)`，
> 直接接受 CUDA tensor），则无需转换。若 `predicts` 抛异常，评测会回报「提交的代码运行时出错」并附上你的异常，便于自查。

| Key | 分组 | 出现于 | Shape | 含义 |
| --- | --- | --- | --- | --- |
| `proprio` | top | 所有任务 | `(1, 66)` TRON2 · `(1, 106)` Oli | 内部状态（见 §3） |
| `extero` | top | 仅 TRON2 | `(1, 34560)` | Fairy96 LiDAR 高度扫描，展平为 `96 × 360`（Oli 无此项） |
| `image[<cam>]` | image | 带相机的任务 | TRON2 `(1, 480, 640, 3)` rgb / `(1, 480, 640, 1)` depth · Oli `head_depth` `(1, 60, 106, 1)` | 每相机的 RGB-D |

各任务权威的字段列表（名称、dtype、shape、必需/可选）见 `obs_schema.py`
→ `OBS_SCHEMAS`。`schema_key_for_task(task_id)` 把任务 id 映射到其 schema。

### 图像 key
- **TRON2**（足式 / 轮足）：`head_rgb/head_depth`（前视）、`down_rgb/down_depth`（下视）、
  `ee_rgb/ee_depth`（末端手眼）。相机内外参见 README「传感器参考」。
- **Oli**：仅头部**深度**（`head_depth`，`(1, 60, 106, 1)`）；无胸部相机、无 RGB、无 LiDAR `extero`。

深度为度量值（**米**，`float32`），RGB 为 uint8 `HWC`。**两类机器人的深度语义不同，切勿套用同一套预处理**：

| | TRON2 RGB-D | Oli `head_depth` |
| --- | --- | --- |
| **像素值的含义** | 见 README | **沿光轴的 z-深度**，**不是**到相机的直线距离（换算见下） |
| 超出量程 / 无回波 | `inf → 0` | 返回**量程上限 `5.0`**，**不会**是 `0`/`inf`/`nan` |
| `0` 的含义 | 无效 / 无回波 | **真实的极近距回波**（如摔倒后相机贴地）——不可当作「远」 |
| 是否含机器人自身 | — | **否**，机器人不遮挡自己的视野，图像中不会出现躯干、手臂 |
| 光轴 | 见 README | **前下俯视 30°**，相机高约 1.50 m → 图像内最近像素约 **1.53 m** 深 |

> ⚠️ **z-深度 ≠ 径向距离。** `head_depth[v, u]` 是该点在相机系下的 **z 分量**（到成像平面的垂直距离）。
> 到光心的直线距离为 `z / cos θ`（θ = 该像素光线与光轴夹角）。本相机视场很宽，图像**四角**的
> θ 达 48.1°，径向距离是 z-深度的 **1.50 倍**——误当作径向距离会在角落低估约 50%。
> 5.0 m 的量程上限同样是对 z-深度而言。

> ⚠️ Oli 最常见的踩坑：把可用量程按「近处障碍」的直觉裁到 1.5 m 上下。由于俯视 30° + 头高 1.50 m，
> **图像里最近的像素也有约 1.53 m**（对应脚前约 0.90 m 的地面），这样裁会把几乎整幅图判成
> 「远 / 无信息」，视觉输入退化成常量。建议按 **0–5 m** 全量程归一化。完整外参/内参见 README「传感器参考」。

### 读取示例（在 `predicts(obs, ...)` 内）

```python
# 激光雷达高度扫描（exteroception）：仅 TRON2 有，Oli 没有 extero。
# 是展平的一维向量 channels*360，reshape 成 (channels, 360) 即逐环高度。
extero = obs.get("extero")                 # (1, channels*360) 或 None
if extero is not None:
    rings = extero.reshape(1, -1, 360)     # (1, channels, 360)

# 深度图（exteroception）：obs["image"][<cam>]，形状 (1, H, W, 1)，单位米。
# 语义按机器人不同：TRON2 为 inf→0；Oli 的 head_depth 未命中/超距为 5.0（见上表）。
images = obs.get("image") or {}            # 无相机的任务为 {}
head_depth = images.get("head_depth")      # Oli (1,60,106,1) · TRON2 (1,480,640,1) 或 None
ee_depth   = images.get("ee_depth")        # 仅 TRON2
# RGB（仅 TRON2）：images.get("head_rgb") → (1,480,640,3) uint8
```

> 一律用 `obs.get(...)` 取（缺失返回 `None`/`{}`），**不要假设某项一定存在** —— 不同 `--robot` 提供的
> 传感器不同（Oli 只有 `proprio` + `head_depth`，无 `extero`/RGB）。

---

## 3. `proprio` —— 传感器索引图（内感 / proprioception）

`proprio` 是一个有序的 float 向量。前若干项是固定的（TRON2 为 12，Oli 为 13）；其余是 `3 × action_dim` 个关节通道。

**TRON2 —— 66 维（`action_dim = 18`）：**

| Index | 字段 | 说明 |
| --- | --- | --- |
| `[0:3]` | `base_lin_vel` | base 坐标系下的线速度（仿真真值） |
| `[3:6]` | `base_ang_vel` | base 坐标系下的角速度（IMU 等价量） |
| `[6:9]` | `velocity_commands` | **目标导向的指令**，指向终点（见 §5） |
| `[9:12]` | `projected_gravity` | base 坐标系下的重力方向（朝向线索） |
| `[12:30]` | `joint_pos` | 18 个关节角（相对于默认值） |
| `[30:48]` | `joint_vel` | 18 个关节速度 |
| `[48:66]` | `last_action` | 18 个先前动作 |

通用规则：`joint_pos = proprio[12 : 12+N]`、`joint_vel = proprio[12+N : 12+2N]`、
`last_action = proprio[12+2N : 12+3N]`，其中 `N = action_dim = (D-12)//3`。

18 个关节通道的顺序 = 机器人固定的 `joint_names`（`joint_pos` / `joint_vel` / `last_action` 同序，
**也是你从 `predicts` 返回的 18 维动作的顺序**）。**sfyg 与 wfyg 不同**——wfyg 把双踝换成双轮、且轮夹在腿与臂之间：

```python
# sfyg（18 = 腿 10 + 臂 8）
TRON2A_SFYG_JOINT_NAMES = [
    # 腿 0-9
    "proximal_pitch_L_Joint", "proximal_roll_L_Joint", "proximal_yaw_L_Joint", "knee_L_Joint", "ankle_pitch_L_Joint",
    "proximal_pitch_R_Joint", "proximal_roll_R_Joint", "proximal_yaw_R_Joint", "knee_R_Joint", "ankle_pitch_R_Joint",
    # 臂 10-17
    "arm1_Joint", "arm2_Joint", "arm3_Joint", "arm4_Joint", "arm5_Joint", "arm6_Joint", "gripper1_Joint", "gripper2_Joint",
]
# wfyg（18 = 腿 8 + 轮 2 + 臂 8；腿部无踝）
TRON2A_WFYG_JOINT_NAMES = [
    # 腿 0-7
    "proximal_pitch_L_Joint", "proximal_roll_L_Joint", "proximal_yaw_L_Joint", "knee_L_Joint",
    "proximal_pitch_R_Joint", "proximal_roll_R_Joint", "proximal_yaw_R_Joint", "knee_R_Joint",
    # 轮 8-9
    "wheel_L_Joint", "wheel_R_Joint",
    # 臂 10-17
    "arm1_Joint", "arm2_Joint", "arm3_Joint", "arm4_Joint", "arm5_Joint", "arm6_Joint", "gripper1_Joint", "gripper2_Joint",
]
```

**Oli —— 106 维（`action_dim = 31`）：** 与 TRON2 同构 —— **带 `base_lin_vel`**（作为速度传感器
通道），但速度指令是 **4 维**的。关节顺序由机器人固定（serial `OLI_EDU_JOINT_NAMES`）：

| Index | 字段 | 说明 |
| --- | --- | --- |
| `[0:3]` | `base_lin_vel` | base 系线速度（速度传感器通道） |
| `[3:6]` | `base_ang_vel` | base 系角速度 |
| `[6:10]` | `velocity_commands` (`vx, vy, wz, stand_flag`) | 目标导向指令，指向终点（见 §5） |
| `[10:13]` | `projected_gravity` | base 系重力方向 |
| `[13:44]` | `joint_pos` (31) | 关节角（相对默认站姿） |
| `[44:75]` | `joint_vel` (31) | 关节速度 |
| `[75:106]` | `last_action` (31) | 上一步施加的动作 |

因此 `joint_pos = proprio[13 : 13+N]`、`joint_vel = proprio[13+N : 13+2N]`、
`last_action = proprio[13+2N : 13+3N]`，其中 `N = action_dim = (D - 13) // 3 = 31`。

`OLI_EDU_JOINT_NAMES`（31，机器人固定顺序）—— `joint_pos` / `joint_vel` / `last_action` 的通道顺序、
以及 `get_action_spec` 里 `stiffness` / `damping` 列表的顺序，**都**按此序：

```python
OLI_EDU_JOINT_NAMES = [
    # 左腿 0-5
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # 右腿 6-11
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # 腰 12-14
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # 头 15-16
    "head_yaw_joint", "head_pitch_joint",
    # 左臂 17-23
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    # 右臂 24-30
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
]
```

> **proprio 为原始真值（未做预缩放）**：线速度 / 角速度单位 m·s⁻¹ / rad·s⁻¹，`joint_pos` 为相对默认站姿的
> 弧度差，`joint_vel` 为 rad·s⁻¹。是否归一化、用什么尺度，由你自行决定（评测不预设缩放系数）。

### `extero` —— LiDAR（外感 / exteroception）
展平的高度扫描：`96 × 360` 条射线（Fairy96 = 96×360 = 34560）。若想要逐环（per-ring）结构，可 reshape
为 `(96, 360)`。LiDAR / 相机的内外参（挂载位姿、FOV、分辨率）见 README「传感器参考」。

### `image` —— 相机（外感）
`obs["image"]["head_depth"]` 等。各任务的 key 见 §2 的表格。

---

## 4. 动作规格 —— `get_action_spec()`

你从 `predicts` 返回的 `action` 是长度 `action_dim` 的扁平向量，**按 §3 的固定关节顺序**排列，并顺次切成各
动作分组——TRON2 为 `leg (+ wheel) + arm`（sfyg：腿 10 + 臂 8；wfyg：腿 8 + 轮 2 + 臂 8），Oli 为单一
`leg`（全 31）。`get_action_spec()` 只决定每个分组*如何*被解释（`mode`/`scale`/`clip`/PD），**不改变这个顺序**；
若你的 policy 内部关节序与此不同，在 `predicts` 里把网络输出重映射到 §3 的固定顺序后再返回。

返回 `None`/`{}` 使用官方默认值，或按分组覆盖：

```python
DEFAULT = {
    "leg":   {"mode": "position", "scale": 0.5, "clip": None},
    "arm":   {"mode": "position", "scale": 0.5, "clip": None},
    "wheel": {"mode": "velocity", "scale": 5.0, "clip": None},
}
```

- 分组：`leg`、`arm`、`wheel`（仅使用机器人实际拥有的那些）。**Oli 只有 `leg` 这一组，且它涵盖全部 31
  个关节**（按 §3 的 `OLI_EDU_JOINT_NAMES` 顺序）；TRON2 才分 `leg`/`arm`/`wheel`。
- `mode`：`position` | `velocity` | `effort`。对于 `position`：`joint_target = default_pos + scale * action`。
- `scale`：`(1e-6, 100]` 范围内的 float。`clip`：`None` 或 `[min, max]`（在缩放前钳制动作）。
- **`stiffness` / `damping`（可选，逐关节 PD）**：每个是一个**长度 = 该组关节数**的 float 列表（按机器人固定
  关节顺序），分别覆盖该组各关节的 `kp` / `kd`。**Oli 即长度 31**。不填则用下方"推荐/默认 PD"。安全范围
  `kp∈[0,1000]`、`kd∈[0,100]`。**力矩上限不可改**（真机规格，见下）。位置控制策略与训练时的 PD 强相关——
  想复现你的步态，请用这两个键把训练 PD 一并声明。
- 你设定动作向量*如何*被解释；而**关节顺序由机器人固定**。如果你的 policy 是按不同的内部关节顺序
  训练的，请在 `predicts` 内部把输出重映射到 §3 的固定顺序。

#### PD 覆盖示例（`stiffness` / `damping`）

列表**必须按机器人固定的关节顺序**逐关节给值（长度 = 该组关节数）。下面这些顺序就是评测施加动作的顺序：

| 机器人·分组 | 关节数 | 固定顺序（列表第 0…N-1 项） |
| --- | --- | --- |
| TRON2 `leg`（sfyg / 有踝） | 10 | `proximal_pitch_L, proximal_roll_L, proximal_yaw_L, knee_L, ankle_pitch_L,` `proximal_pitch_R, proximal_roll_R, proximal_yaw_R, knee_R, ankle_pitch_R` |
| TRON2 `leg`（wfyg / 双轮，无踝） | 8 | `proximal_pitch_L, proximal_roll_L, proximal_yaw_L, knee_L,` `proximal_pitch_R, proximal_roll_R, proximal_yaw_R, knee_R` |
| TRON2 `arm`（sfyg & wfyg） | 8 | `arm1, arm2, arm3, arm4, arm5, arm6, gripper1, gripper2` |
| TRON2 `wheel`（**仅 wfyg**） | 2 | `wheel_L, wheel_R` |
| Oli `leg`（**唯一分组，全 31 关节**） | 31 | `OLI_EDU_JOINT_NAMES`（见 §3） |

> 上表 TRON2 关节名省略了 `_Joint` 后缀（如 `proximal_pitch_L` 即 `proximal_pitch_L_Joint`、`wheel_L` 即
> `wheel_L_Joint`）；完整名与 sfyg/wfyg 的 18 维整体顺序见 §3。PD 列表是**按分组**给值（`leg` / `arm` / `wheel`
> 各自一段），与 proprio 的 18 维扁平顺序对应关系也在 §3。

只需给你想改的分组填 `stiffness`/`damping`；没填的分组、以及没填的键，都保持"默认 PD"。**手写扁平列表极易错序**——
推荐用"关节名 → (kp, kd)"字典、再按固定顺序展开，从根上杜绝错位：

```python
# 例：TRON2（sfyg）把腿部 PD 换成你自己训练用的那套；臂/轮不动 -> 用默认。
# 下面的数值即评测默认（腿·大 159.67/10.16、腿·小 53.22/3.39），把它们替换成你的训练 PD 即可。
LEG_ORDER = [  # 必须与上表 TRON2 leg(10) 一致
    "proximal_pitch_L", "proximal_roll_L", "proximal_yaw_L", "knee_L", "ankle_pitch_L",
    "proximal_pitch_R", "proximal_roll_R", "proximal_yaw_R", "knee_R", "ankle_pitch_R",
]
LEG_PD = {  # 关节名 -> (kp, kd)；左右同构只写一次
    "proximal_pitch": (159.67, 10.16), "proximal_roll": (159.67, 10.16), "knee": (159.67, 10.16),
    "proximal_yaw":   (53.22,  3.39),  "ankle_pitch":   (53.22,  3.39),
}
def _pd(joint):                       # "proximal_pitch_L" -> LEG_PD["proximal_pitch"]
    return LEG_PD[joint.rsplit("_", 1)[0]]

def get_action_spec(self):
    stiffness = [_pd(j)[0] for j in LEG_ORDER]   # 长度 10，按固定顺序
    damping   = [_pd(j)[1] for j in LEG_ORDER]
    return {"leg": {"mode": "position", "scale": 0.5,
                    "stiffness": stiffness, "damping": damping}}
    # 等价于（可直接照抄再改数）：
    # "stiffness": [159.67,159.67,53.22,159.67,53.22, 159.67,159.67,53.22,159.67,53.22]
    # "damping":   [ 10.16, 10.16, 3.39, 10.16, 3.39,  10.16, 10.16, 3.39, 10.16, 3.39]
```

```python
# 例：Oli（唯一分组 "leg" 涵盖全部 31 关节）声明你训练用的逐关节 PD。
# 关节名请直接照抄 §3 列出的 31 个，按该顺序写死在你自己的代码里（评测容器只安装
# requirements.txt 中声明的依赖，不要试图 import 评测侧的包）。
OLI_EDU_JOINT_NAMES = [ ... 见 §3 的 31 个关节，按该顺序 ... ]
MY_KP = { ... }   # 关节名 -> kp，覆盖全部 31 个（可参照下表"推荐 PD"起步）
MY_KD = { ... }   # 关节名 -> kd
def get_action_spec(self):
    return {"leg": {
        "stiffness": [MY_KP[j] for j in OLI_EDU_JOINT_NAMES],   # 长度 31，按固定顺序
        "damping":   [MY_KD[j] for j in OLI_EDU_JOINT_NAMES],
    }}
```

约束回顾：`kp∈[0,1000]`、`kd∈[0,100]`；长度必须**恰好等于**该组关节数（多/少一个都会被判 failed）；
`stiffness`/`damping` 缺省即用下表默认；**力矩上限不可改**。**所有开放构型的所有分组都支持逐关节覆盖**：
TRON2 **sfyg**（`leg` 10 / `arm` 8）、**wfyg**（`leg` 8 / `arm` 8 / `wheel` 2）与 **Oli Edu**（`leg` 31），放心声明。

### Oli 推荐 / 默认 PD、默认姿态、力矩上限

来源：`limxdynamics/humanoid-rl-deploy-python` →
`controllers/HU_D04_01/walk_controller/walk_param.yaml`（`kp`/`kd`/`default_angle`/`user_torque_limit`，逐关节
31 维）。**评测内置的默认 PD 就是这套**——不声明 `stiffness`/`damping` 时即用它；若你用别的 PD 训练，务必
按上面方式声明，否则步态会失稳。

> 同仓库另有 `controllers/HU_D04_01/mimic_controller/mimic_param.yaml` —— **mimic 控制器**的一套**不同**
> 的 kp/kd（如腿 280 / kd 5、踝 20 / kd 2…，与上表 walk 的不同）。若你的策略是 mimic 风格、按那套 PD
> 训练，请用 `get_action_spec` 的 `stiffness`/`damping` 把它声明出来，而不要沿用评测默认的 walk PD。

| 关节组 | 推荐 `kp` | 推荐 `kd` | 力矩上限(固定) |
| --- | --- | --- | --- |
| hip(p/r/y) · knee | 139.41 | 17.75 | 140 |
| ankle(pitch/roll) | 93.65 | 11.92 | 80 |
| waist yaw | 93.65 | 11.92 | 42 |
| waist roll/pitch | 93.65 | 11.92 | 80 |
| head(yaw/pitch) | 15.12 | 1.93 | 19 |
| shoulder(p/r/y) · elbow | 87.51 | 11.14 | 42 |
| wrist(y/p/r) | 15.12 | 1.93 | 19 |

默认站姿 `default_angle`（按 `OLI_EDU_JOINT_NAMES` 顺序，proprio 的 `joint_pos` 即相对它的差值）：
腿 L/R `[-0.15, 0, ∓0.05, 0.30, -0.16, 0]`、腰 `[0,0,0]`、头 `[0,0]`、
臂 L `[0.1, 0.1, -0.2, -0.2, 0,0,0]` / R `[0.1, -0.1, 0.2, -0.2, 0,0,0]`。
力矩上限是真机电机物理上限，**不开放修改**；`kp/kd` 是控制器参数，可按上方 `stiffness`/`damping` 自定义。

> **控制频率**:评测以 **50 Hz** 调用 `predicts`(`decimation 4 × sim.dt 0.005`)。若你的 policy 按别的
> 控制频率训练(例如官方 deploy walk 控制器是 100 Hz),它会被以评测的 50 Hz 驱动、步态可能不稳——
> 请按 50 Hz 训练或适配。

### TRON2（sfyg / wfyg）推荐 / 默认 PD、默认姿态

来源：腿部 PD 来自 `limx-tron2/TRON2_YG_LAB` →
`exts/bipedal_locomotion/.../assets/config/{solefoot,wheelfoot}_yg_tron2a_cfg.py`（训练框架另见
`limx-tron2/tron2_rl_lab`）。**评测内置的默认 PD 即下表**（不声明 `stiffness`/`damping` 时生效；腿部 kp/kd
与 TRON2_YG_LAB 一致）：

| 关节组 | `kp` | `kd` | 力矩上限(固定) | 关节 |
| --- | --- | --- | --- | --- |
| 腿·大（pitch/roll/knee，双腿 6） | 159.67 | 10.16 | 150 | `proximal_pitch/roll_[RL]`, `knee_[RL]` |
| 腿·小（yaw/ankle，双腿 4） | 53.22 | 3.39 | 60 | `proximal_yaw_[RL]`, `ankle_pitch_[RL]` |
| 臂（arm1–6，6） | 80 | 4 | 100 | `arm[1-6]_Joint` |
| 夹爪（gripper，2） | 80 | 4 | 10 | `gripper[12]_Joint` |
| 轮（**仅 wfyg**，2） | 0（速度控制） | 0.6 | 20 | `wheel_[RL]_Joint` |

默认姿态：腿关节 0，但 `proximal_yaw` = ∓π（左 -π / 右 +π）；臂 0；`gripper1=0.05, gripper2=-0.05`。
18 个 action 关节 = 腿·大 6 + 腿·小 4 + 臂 6 + 夹爪 2（wfyg 把双 ankle 换成双轮）。`kp/kd` 可按上方
`stiffness`/`damping` 自定义；力矩上限固定不可改。

---

## 5. 速度指令分发（`proprio[6:9]`）

环境向 proprio 槽位注入一个**目标导向的速度指令** —— 它把机器人指向终点线（一个远目标朝向伺服：
恒定的前进速度 + 一个让你转向目标的偏航角速度）。你可以：

- **跟随它**（对速度跟踪型 policy 推荐）：把该指令（TRON2 为 `proprio[6:9]`；Oli 为 `proprio[6:10]`）
  作为要跟踪的指令喂给你的 policy。良好的朝向跟踪让你保持居中；糟糕的跟踪会让你漂向边界（被惩罚）。
- **忽略它**，用你自己的逻辑来转向（例如硬编码的前进指令）。

这就是为什么该指令放在 proprio 里而不是单独的通道 —— 速度跟踪型 policy 无需在 solution 侧做任何
改动。（Oli 使用一个带 `stand_flag` 的 4 维指令。）

> **指令速度上限**：Oli 的目标导向指令前进速度可达约 **0.8 m/s**。若你的 policy 按更低的最大速度
> 训练（例如官方 deploy walk 控制器 `max_vx=0.5`），请在喂给 policy 前先把指令钳到你的训练范围。

---

## 6. 打包与提交

你用比赛提交 CLI（`submit.py`，从官网下载，纯 Python 3 标准库、无需 pip install、无需 GitHub 账号）
提交两样东西：一个**代码目录**和你的**权重文件**。提交身份只靠主办方私发的**队伍令牌**。

```bash
python3 submit.py --server https://submit.troncamp-loco.limxdynamics.com --token=<队伍令牌> \
                  --competition <tron|humanoid> \
                  --robot <oli|sfyg_tron2a|wfyg_tron2a> \
                  --ckpt-file policy.pt --code-dir ./my_solution
```

- **`--server`** 是评测服务器地址（官方 `https://submit.troncamp-loco.limxdynamics.com`）。
- **`--token`** 是主办方私发的队伍令牌。**令牌可能以 `-` 开头，务必用等号形式 `--token=<队伍令牌>`**，
  否则会被解析成选项。
- `--robot` 选择任务以及 obs/action 配置：
  - `sfyg_tron2a` → TRON2 **足式** · `wfyg_tron2a` → TRON2 **轮足式** ·
    `oli` → **Oli 人形**。
- **`--code-dir`** 必须在其根目录包含 **`solution.py`**（外加它导入的任何辅助模块）。它会被
  解包到运行目录中，因此 `solution.py` 作为顶层模块被导入。
- **`--ckpt-file`** 单独上传，并在运行时放在 **`solution.py` 旁边**。**⚠️ 无论你上传的文件原名/格式是
  什么（`policy.onnx`、`model.pt`…），平台一律以固定名 `policy.pt` 落到运行目录（并同时提供一份同内容的
  `policy.onnx`）。** 请按 `policy.pt`（或 `policy.onnx`）这个名字加载——`os.path.join(os.path.dirname(__file__),
  "policy.pt")`，onnxruntime/torch 都按文件内容读、不看扩展名；**不要假设保留你上传时的原始文件名**。
  见 `solution_template.py`。
- **在代码目录中添加一个 `requirements.txt`** 声明你额外用到的 Python 包 —— 容器构建时会 `pip install -r`。
  评测镜像**已预装** `torch` + `numpy`（基础镜像自带）以及 HTTP 服务依赖（`fastapi`/`uvicorn`/`python-multipart`）；
  **其它你在 solution.py 里 `import` 的包（例如 `onnxruntime`）必须自己在 `requirements.txt` 里声明**，否则
  容器启动即 `ModuleNotFoundError`、这份提交跑不起来（评测会把该异常回报给你，便于自查）。所以
  `requirements.txt` 是增量的 —— 只列这些额外的包（完全无额外依赖的提交可以不带它）。
- **不要放入** `server.py`、`obs_schema.py` 或 `run.sh` 到你的代码目录 —— 平台会把这些作为固定
  基础设施注入；自带会冲突。（`obs_schema.py` 在评测时*确实*存在于你的运行目录中，因此
  `import obs_schema` 可用；本资源包里的副本仅供参考 / 本地测试。）

### 代码目录布局

放进 `--code-dir` 的内容（也就是被打包进 `code.tar.gz` 的内容）：

```
my_solution/                 # 即 --code-dir
├── solution.py              # 必需 —— 你的 AlgSolution（就用这个文件名）
├── requirements.txt         # 可选 —— 额外 pip 依赖（见本 kit 中的模板；增量式）
└── <your_helper_modules>.py # 可选 —— solution.py 导入的任何模块
```

`policy.pt` **不**在这里面 —— 通过 `--ckpt-file` 单独传递；运行时它位于 `solution.py` 旁边。
一个无依赖的提交就只是 `solution.py`。

评测服务器加载你的 `solution.py`，构造 `AlgSolution`，并按上文所述通过
`reset()` / `get_action_spec()` / `predicts()` 来驱动它。用
`python3 submit.py --server https://submit.troncamp-loco.limxdynamics.com --token=<队伍令牌> --status` 查询本队提交状态与分数。
