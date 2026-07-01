"""TronCamp / HumanoidCamp 基准的参考 `solution.py` 模板。

提交一个暴露 `AlgSolution` 类、且实现下述接口的 `solution.py`。评测器会构造它一次，在回合开始时调用
`reset()`，随后每个控制步调用 `predicts(obs, current_score)` 并把返回的动作施加到机器人。完整契约见
`solution_build_guide.md`。

本模板开箱即可运行（零动作）。把 `predicts` 的函数体替换成你的策略 —— 在 `__init__` 里加载训练好的
权重，并参照 `solution_build_guide.md` 里的 obs 索引表。
"""
from __future__ import annotations

import os
from typing import Any

import torch

# --- proprio 布局（TRON2，66 维）----------------------------------------------------------------
# obs["proprio"] 是单条有序浮点向量。前 12 项是固定的；其余是 3 * action_dim 个关节通道
# （joint_pos | joint_vel | last_action），因此 action_dim = (D - 12) // 3。
BASE_LIN_VEL = slice(0, 3)      # 机体系线速度（仿真真值）
BASE_ANG_VEL = slice(3, 6)      # 机体系角速度
VELOCITY_COMMAND = slice(6, 9)  # 环境注入的、朝向终点的 (vx, vy, wz) 速度指令
PROJECTED_GRAVITY = slice(9, 12)  # 机体系下的重力方向（朝向线索）
# 关节从索引 12 开始：
#   joint_pos     = proprio[12 : 12 + N]
#   joint_vel     = proprio[12 + N : 12 + 2N]
#   last_action   = proprio[12 + 2N : 12 + 3N]      其中 N = action_dim
# （Oli：proprio 106 维、固定头部 13（含 base_lin_vel + 4 维速度指令 vx,vy,wz,stand_flag），
#   故 action_dim = (D - 13) // 3，关节从索引 13 开始。详见 solution_build_guide.md §3。）

# 评测时你上传的权重（--ckpt-file）会以**固定名 `policy.pt`** 放在本文件旁边（无论你上传时的原始名/
# 格式；平台同时还放一份同内容的 `policy.onnx`）。请相对 __file__ 按此名加载——torch / onnxruntime 都按
# 文件**内容**读、不看扩展名（见 solution_build_guide.md §6）。没有权重文件时本模板仍可运行（零动作）。
POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")


class AlgSolution:
    def __init__(self):
        self.device = torch.device("cpu")
        self.policy = None          # TorchScript 模块
        self.ort = None             # onnxruntime 会话（若用 ONNX）

        if not os.path.exists(POLICY_PATH):
            print("[solution] 未找到权重，将以零动作运行（占位）")   # 模板可独立运行；正式提交务必附带权重
            return

        # --- 加载权重：按你的格式二选一 ----------------------------------------------------
        # 注意：请勿用 try/except 掩盖加载失败再退化为零动作——这会导致机器人全程静止、得 0 分，
        #       且日志中难以定位原因（属常见错误）。建议让加载失败直接抛出，以便及时暴露问题。

        # 方式一：TorchScript（policy.pt = torch.jit.save 导出）
        self.policy = torch.jit.load(POLICY_PATH, map_location=self.device).eval()

        # 方式二：ONNX —— 删掉上面方式一、取消下面注释；并在 requirements.txt 里加 onnxruntime。
        #   onnxruntime 按内容读，文件名是 policy.pt 也能正常加载 ONNX 字节。CPU 推理用 CPUExecutionProvider。
        # import onnxruntime as ort
        # so = ort.SessionOptions(); so.intra_op_num_threads = 1
        # self.ort = ort.InferenceSession(POLICY_PATH, sess_options=so, providers=["CPUExecutionProvider"])
        # self.ort_in = self.ort.get_inputs()[0].name

    def reset(self, **kwargs):
        """每个回合开始时调用一次。在这里清空任何循环 / 历史状态。"""
        pass

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        """可选。返回 None（或 {}）以使用官方默认：
            leg   -> position, scale 0.5
            arm   -> position, scale 0.5
            wheel -> velocity, scale 5.0
        也可按组覆盖。允许键：mode ∈ {position, velocity, effort}，scale ∈ (1e-6, 100]，
        clip = None 或 [min, max]，以及可选的逐关节 PD：stiffness / damping（长度=该组关节数的
        float 列表，kp∈[0,1000]、kd∈[0,100]）。关节顺序由机器人固定 —— 你只设定动作如何被解释。
        力矩上限不可改（真机规格）。示例：
            return {"leg": {"mode": "position", "scale": 0.25},
                    "wheel": {"mode": "velocity", "scale": 1.0}}
        Oli 只有一组 "leg" 且涵盖全部 31 个关节，想用自己训练的 PD 就声明 31 维列表（推荐值见指南 §4）：
            return {"leg": {"stiffness": [...31...], "damping": [...31...]}}
        """
        return None

    def predicts(self, obs, current_score) -> dict:
        """每步返回 {"action": <action_dim 个浮点数的扁平列表>, "giveup": bool}。

        obs 是一个 dict：obs["proprio"]（总有），以及可选的 obs["extero"]（激光雷达高度扫描）和
        obs["image"][<cam>]（RGB-D）。形状均为 (1, ...)（batch 为 1）。评测时其值是 **GPU（CUDA）上的
        torch tensor**：下面的 `torch.as_tensor(...)` 直接接受它；若改用 numpy，先 `.detach().cpu().numpy()`。
        """
        proprio = torch.as_tensor(obs["proprio"], dtype=torch.float32, device=self.device)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)

        # --- 外感传感器（可选；用 obs.get 取，缺失时为 None / {}）-----------------------------
        # 激光雷达高度扫描（exteroception）：**仅 TRON2** 有；**Oli 没有 extero**。
        #   是展平的一维向量 96*360，reshape 成 (96, 360) 得逐环结构（Fairy96；内外参见 README「传感器参考」）。
        extero = obs.get("extero")                    # (1, 96*360) 或 None
        # if extero is not None:
        #     extero = torch.as_tensor(extero, dtype=torch.float32, device=self.device)
        #     rings = extero.reshape(1, -1, 360)       # (1, channels, 360) 逐环高度
        #
        # 深度相机（exteroception）：obs["image"][<cam>]，形状 (1, H, W, 1)，单位米（inf 已置 0）。
        #   Oli：head_depth (1,60,106,1)。 TRON2：head_depth / ee_depth [/ down_depth] (1,480,640,1)，
        #   REAL 另有 *_rgb (1,480,640,3) uint8。
        images = obs.get("image") or {}               # {} 若该任务无相机
        head_depth = images.get("head_depth")         # (1, H, W, 1) 或 None
        # if head_depth is not None:
        #     head_depth = torch.as_tensor(head_depth, dtype=torch.float32, device=self.device)

        action_dim = (proprio.shape[-1] - 12) // 3
        # velocity_command = proprio[:, VELOCITY_COMMAND]   # 跟随它即可朝终点前进

        if self.policy is not None:                 # 方式一：TorchScript
            with torch.inference_mode():
                action = self.policy(proprio)        # 把 obs 适配成你网络期望的输入
        elif self.ort is not None:                  # 方式二：ONNX
            import numpy as np
            out = self.ort.run(None, {self.ort_in: proprio.cpu().numpy().astype(np.float32)})
            action = torch.as_tensor(out[0], device=self.device)
        else:                                        # 无权重 -> 零动作（占位）
            action = torch.zeros(proprio.shape[0], action_dim, device=self.device)

        action_list = action.squeeze(0).cpu().tolist()
        return {"action": action_list, "giveup": False}
