# 选手资源包 —— TronCamp（TRON2）与 HumanoidCamp（Oli）

**训练策略**、**构建 `solution.py`** 所需的一切，面向 TronCamp（TRON2）与 HumanoidCamp（Oli）两个
基准。本目录是一个自包含的静态资源包；`manifest.json` 是供 Web 前端列出并链接这些资源的机器可读索引。

## 内容

| 路径 | 说明 |
| --- | --- |
| [`training_terrain/`](training_terrain/) | 随机化、宽阔开放的训练地形（覆盖评测的全部地形要素；布局随机，避免过拟合单一赛道）。`TRON_CAMP_TRAINING_TERRAIN_CFG` / `HUMANOID_CAMP_TRAINING_TERRAIN_CFG`。 |
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
   from tron_camp_training_terrain import TRON_CAMP_TRAINING_TERRAIN_CFG
   env_cfg.scene.terrain = TRON_CAMP_TRAINING_TERRAIN_CFG   # 随机化、宽阔；但最终按固定的评测赛道评分
   ```
   Oli 人形请改用 `humanoid_camp_training_terrain.py`；机器人描述见上方「Oli 开源仓库」的
   [humanoid-description](https://github.com/limxdynamics/humanoid-description)（串联模型）。训练时可
   选用的相机 / 雷达内外参见下方 **「传感器参考（训练可选）」** 一节。

2. **构建你的方案** —— 复制 `solution_guide/solution_template.py` → `solution.py`，实现
   `predicts(obs, current_score)`。阅读 `solution_guide/solution_build_guide.md` 里的**索引表**
   （TRON2：proprio `[6:9]` 是朝向终点的速度指令，`action_dim = (D-12)//3`；Oli：`[6:10]`，`(D-13)//3`）。

3. 用提交 CLI **提交**：你的 `--code-dir` 根目录必须有 `solution.py`，权重通过
   `--ckpt-file policy.pt` 提交（运行时会放在 `solution.py` 旁边）。`--robot` 选择赛题 ——
   `sfyg_tron2a` / `wfyg_tron2a` = TRON2 腿式 / 轮足，`oli` = 人形。详见
   `solution_guide/solution_build_guide.md` §6。

> 训练地形**刻意不是**评分赛道 —— 它随机化每一处布局，让你的策略学会泛化，而不是死记一条固定赛道。

## 传感器参考（训练可选）

评测按各 `--robot` 的 obs 契约提供传感器（TRON2：`proprio` + `extero` + `image`；Oli：`proprio` + `head_depth`），
**是否使用、如何使用由你决定**（obs/图像 key、索引表见 `solution_guide/solution_build_guide.md`
[§2](solution_guide/solution_build_guide.md)/§3）。下表给出评测同款仿真中各传感器的**内外参**，便于你在训练侧复现
相同的传感器布局。数值取自评测仿真 cfg 与机器人 USD（外参对应
[robot-description](https://github.com/limx-tron2/robot-description) /
[humanoid-description](https://github.com/limxdynamics/humanoid-description) 里的传感器 link）。

> 约定：`base_link` 与 base IMU 同一原点，坐标轴 **x 前 / y 左 / z 上**；相机采用 ROS 光学约定（**+Z 为视线方向、+Y 向下**）；
> 四元数按 `(w, x, y, z)` 给出。

### TRON2（TronCamp，评测默认套件）

**外参**（相对 `base_link` ≡ base IMU）：

| 传感器 | obs key | 挂载 link | pos (x, y, z) m | 姿态 |
| --- | --- | --- | --- | --- |
| 前视 D435i RGB-D | `head_rgb` / `head_depth` | `base_imu` | (0.22033, 0.01750, 0.18592) | rpy = (0, 2.2689, 0) rad（绕 y 俯仰） |
| 下视 D435i RGB-D | `down_rgb` / `down_depth` | `d435_Link` | (0.09680, 0.01759, −0.00409) | quat = (0.2079, 0, 0.9781, 0)（底盘下视，光轴朝下、前倾约 24°） |
| Fairy96 LiDAR | `extero` | `base_imu` | (0.18058, 0, 0.23876) | 单位姿态 (0, 0, 0) |

**内参**：

- **前视 / 下视 D435i（RGB-D）**：针孔相机，`focal_length` **11.04 mm**、`horizontal_aperture` **20.955 mm**、
  分辨率 **640×480**（W×H）、裁剪范围 **(0.1, 10.0) m**、HFOV ≈ 87°。`*_rgb` 为 uint8 `HWC`，`*_depth` 为 float32（米，`inf→0`）。
- **Fairy96 LiDAR**：**96** 通道 × **360**（`horizontal_res` 1.0°），垂直 FOV **±16°**（32°）、水平 FOV **±180°**、
  `max_distance` **30 m**、刷新 0.1 s。`extero` 是展平的高度扫描，长度 96×360 = **34560**，可 `reshape(96, 360)` 得逐环高度。

> 仿真是真机的近似：Fairy96 水平分辨率下采样到 1.0°（真机 0.25°）、量程截到 30 m（真机 150 m）；D435i 保持 640×480
> （VFOV ≈ 71° vs 真机 58°，为 4:3 与 16:9 之差）。前视外参在评测仿真中以 Isaac `convention="world"` 加载（含 −π/2 光学修正）；
> 下视相机以 `d435_Link` 位姿 + ROS 约定挂载（0 偏移）。此外还保留末端手眼相机 `ee_rgb`/`ee_depth`（`gripper_base_Link`，通用针孔）。

### Oli（HumanoidCamp）

**外参**：

| 传感器 | obs key | 挂载 link | offset pos (x, y, z) m | offset 姿态 quat | 相对 base_link（默认站姿） |
| --- | --- | --- | --- | --- | --- |
| 头部 D435i 深度 | `head_depth` | `head_pitch_link` | (0.07453, 0.01750, 0.11500) | (0.3536, −0.6124, 0.6124, −0.3536)（ROS） | pos (0.0615, 0.0175, 0.702)，光轴前下俯视约 30° |

**内参**：

- **头部 D435i 深度**：针孔深度相机，焦距 **10.6780 mm**、分辨率 **106×60**（W×H）、
  量程上限 **5.0 m**、仅深度（单位**米**，`float32`）→ `head_depth` 形状 `(1, 60, 106, 1)`。
  视场 **HFOV ≈ 88.9° × VFOV ≈ 58.1°**，每控制步（50 Hz）刷新一次。
- Oli **只暴露 `proprio` + `head_depth`**（无 RGB、无 LiDAR `extero`、无胸部相机）；IMU 位于 `base_link`。

**`head_depth` 的四条关键语义**（直接影响预处理写法，请务必按此实现）：

1. **像素值是「沿光轴的 z-深度」，不是「到相机的直线距离」。**
   即该点在相机坐标系下的 **z 分量**（到成像平面的垂直距离），而非到光心的径向距离。
   两者的换算是 `径向距离 = z深度 / cos θ`（θ = 该像素光线与光轴的夹角）。本相机视场很宽，
   两者在画面边缘差别显著：

   | 像素位置 | θ | 径向距离 / z-深度 |
   | --- | --- | --- |
   | 图像中心 | ≈ 0° | 1.00 |
   | 上下边缘 | 28.6° | 1.14 |
   | 左右边缘 | 44.2° | 1.40 |
   | **四角** | **48.1°** | **1.50** |

   > 若误当作径向距离使用，画面角落会把实际距离**低估约 50%**。做点云反投影、障碍测距时尤其注意。

2. **超出量程 / 无回波的像素一律返回量程上限 `5.0`**，**不是 `0`，也不是 `inf` / `nan`**。
   该 5.0 m 上限同样是对 **z-深度**设的——因此沿画面角落方向，真实径向距离要到约 **7.5 m** 才被截断。
   另外**不要把 `0` 当作「无效值」并置为「远」**：本流中接近 `0` 的值是**真实的极近距回波**
   （如摔倒后相机贴地），把它翻成「远」会把糊在镜头上的障碍误判为空旷。
   注意这与 TRON2 的 RGB-D 流（`inf → 0`）语义**相反**，两者不可套用同一套预处理。
3. **深度图中不包含机器人自身**。机器人对自己的相机透明；躯干、手臂**不会**遮挡视野，
   图像中不会出现自身结构。
4. **光轴前下俯视 30°**，整幅图都在地平线以下。相机高约 1.50 m，因此
   **图像内最近的像素也有约 1.53 m 的 z-深度**（对应脚前约 0.90 m 处的地面）。

> ⚠️ 第 1 条与第 4 条合起来最容易踩坑。**同一束光线上有三个不同的距离，别混用**——以图像**正下方**
> 那行（最近的地面）为例，相机高 1.502 m、该光线俯角 59.1°：
>
> | 量 | 公式 | 值 |
> | --- | --- | --- |
> | **z-深度**（`head_depth` 给的就是它） | `h / sin α × cos 29.05°` | **1.53 m** |
> | 径向距离（到光心的直线距离） | `h / sin α` | 1.75 m |
> | 水平距离（脚前多远的地面） | `h / tan α` | 0.90 m |
>
> 也就是说：**脚前仅 0.90 m 的地面，在 `head_depth` 里的读数是 1.53 m。**
> 若按「近处障碍应该 < 1.5 m」的直觉去裁剪量程，会把几乎整幅图判成「远 / 无信息」，
> 视觉输入退化为常量。默认站姿平地下，读数小于 1.8 m 的像素仅约 **15%**，等于 5.0 m 上限的约 **38%**。
> 建议按 **0–5 m** 全量程归一化，或先用上表的外参/内参在本地复现同款相机再决定裁剪策略。
