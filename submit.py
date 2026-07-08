#!/usr/bin/env python3
"""运控合并榜 提交 CLI（自托管云网关版，纯 Python3 标准库，无需 pip install）。

把推理代码 + 策略权重提交到组织者的评测服务器（不再走 GitHub）：
  ① 权重 policy.pt 与代码目录(打成 code.tar.gz) 经 multipart 上传到 /api/upload（令牌鉴权）；
  ② /api/submit 用 token + 赛题 + 机型 入队，服务器校验令牌 + 限流，返回 queued/rejected；
  ③ 默认就地轮询到出结果：排队中(前面几个) → 评测中(已用时) → 完成(得分)/失败(哪一环)。

  # 提交（Tron 腿式；轮式把 --robot 换成 wfyg_tron2a；人形用 --competition humanoid --robot oli）
  python3 submit.py --server https://submit.troncamp-loco.limxdynamics.com --token=<队伍令牌> \
      --competition tron --robot sfyg_tron2a --ckpt-file ./policy.pt --code-dir ./solution
  # 只提交、不等结果：加 --no-wait（之后用 --status 查）

  # 查询本队各次提交状态/分数
  python3 submit.py --server https://submit.troncamp-loco.limxdynamics.com --token=<队伍令牌> --status

鉴权：每队一个组织者私发的 **队伍令牌**（无需 GitHub 账号 / PAT）。令牌可能以 - 开头，
务必用等号形式 --token=<令牌>，否则会被解析成选项。
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

# 机型 → task id（oli→Task F 人形；sfyg/wfyg→Task C Tron）。同 competition_web/common，内联以保零依赖。
ROBOT_TASK_ID = {
    "oli":         "ATEC-TaskF-OliEdu",
    "sfyg_tron2a": "ATEC-TaskC-Tron2ALegged",
    "wfyg_tron2a": "ATEC-TaskC-Tron2AWheel",
}
ROBOT_COMPETITION = {"oli": "humanoid", "sfyg_tron2a": "tron", "wfyg_tron2a": "tron"}

_CODE_EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules",
                      ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
                      "datasets", "checkpoints", "outputs", "wandb", "logs"}
_CODE_EXCLUDE_SUFFIX = (".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".onnx",
                        ".mp4", ".avi", ".mov", ".npz", ".npy", ".parquet", ".arrow",
                        ".h5", ".hdf5", ".tar", ".tar.gz", ".tgz", ".zip", ".pyc", ".so", ".o")
_CODE_MAX_FILE_MB = 50

# 评测墙钟上限（秒），仅用于展示「已 mm:ss / 上限 mm:ss」。须与服务器 deploy.yaml 的 worker timeout_s 对齐。
WALL_LIMIT_S = 1500
POLL_INTERVAL_S = 2.0
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_TTY = sys.stderr.isatty()

# 失败环节码 → 给选手看的中文原因（**选手可见措辞的唯一来源**，全在这里好改）。
# 评测层日志（eval.log/server.log）绝不出现在这里——网关 /api/status 也不返回。
STAGE_ZH = {
    "deps":         "依赖安装未通过（请检查 requirements.txt）",
    "serve":        "提交的代码无法运行（solution.py 可能导入报错，或 policy.pt 加载失败）",
    "solution":     "提交的代码运行时出错（solution.py 在 reset/predict 执行中抛出异常）",
    "eval":         "评测运行中出错",
    "timeout":      f"评测超时（{WALL_LIMIT_S // 60} 分钟内未跑完）",
    "env":          "评测环境暂时不可用，请稍后重试或联系主办方",
    "code_missing": "提交缺少 solution.py",
}
_STAGE_FALLBACK = "评测未通过"


# ----------------------------- 代码打包 -----------------------------
def _pack_code_dir(code_dir: str) -> str:
    """把代码目录【内容】打成 .tar.gz（solution.py 在包根，无外层目录；排除 .git/缓存/大权重/数据集）。
    沙箱约定：评测机把包解到 solution/ 下，故 solution.py 必须在 tar 根。返回临时包路径。"""
    root = os.path.abspath(code_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"--code-dir 不是目录: {code_dir}")
    fd, tmp = tempfile.mkstemp(prefix="locomotion_code_", suffix=".tar.gz")
    os.close(fd)
    cap = _CODE_MAX_FILE_MB * 1024 * 1024
    skipped: list[str] = []
    with tarfile.open(tmp, "w:gz") as tf:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in _CODE_EXCLUDE_DIRS)
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                arc = os.path.relpath(full, root)  # 相对路径 = 包内路径（无外层目录）
                if os.path.islink(full):
                    continue
                if name.lower().endswith(_CODE_EXCLUDE_SUFFIX):
                    skipped.append(arc)
                    continue
                if os.path.getsize(full) > cap:
                    skipped.append(f"{arc}({os.path.getsize(full) // 1024 // 1024}MB>上限)")
                    continue
                tf.add(full, arcname=arc)
    if skipped:
        head = ", ".join(skipped[:8]) + ("…" if len(skipped) > 8 else "")
        print(f"  代码包已排除 {len(skipped)} 个大文件/产物（权重用 --ckpt-file 单独传）：{head}",
              file=sys.stderr)
    return tmp


# ----------------------------- 上传（带进度条） -----------------------------
def _human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} GB"


class _ProgressFile:
    """file-like 包装：urllib 发送请求体时按块 read()，借机回报已发字节数（纯标准库）。"""

    def __init__(self, data: bytes, callback):
        self._io = io.BytesIO(data)
        self._total = len(data)
        self._sent = 0
        self._cb = callback

    def read(self, size: int = -1) -> bytes:
        chunk = self._io.read(size)
        if chunk:
            self._sent += len(chunk)
            self._cb(self._sent, self._total)
        return chunk


def _draw_bar(label: str, sent: int, total: int, *, done: bool = False) -> None:
    if not _TTY and not done:
        return
    width = 20
    frac = 1.0 if total <= 0 else min(1.0, sent / total)
    bar = "█" * int(round(frac * width)) + "░" * (width - int(round(frac * width)))
    icon = "✓" if done else " "
    line = f"\r{icon} {label} [{bar}] {int(frac * 100):3d}%  {_human_bytes(total)}"
    sys.stderr.write(line + ("\n" if done else ""))
    sys.stderr.flush()


def _upload_file(server: str, token: str, path: str, *, label: str = "上传") -> tuple[str, str]:
    """multipart 上传文件到 /api/upload（令牌鉴权），返回 (file_url, sha256)；带进度条。"""
    with open(path, "rb") as f:
        data = f.read()
    boundary = uuid.uuid4().hex
    raw = os.path.basename(path) or "blob.bin"
    filename = "".join(c for c in raw if c.isascii() and c.isprintable() and c not in '"\\') or "blob.bin"
    pre = (f"--{boundary}\r\n"
           f'Content-Disposition: form-data; name="token"\r\n\r\n{token}\r\n'
           f"--{boundary}\r\n"
           f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
           f"Content-Type: application/octet-stream\r\n\r\n").encode()
    post = f"\r\n--{boundary}--\r\n".encode()
    body = pre + data + post
    total = len(body)

    state = {"pct": -1}

    def cb(sent: int, tot: int) -> None:
        pct = int(sent * 100 / tot) if tot else 100
        if pct != state["pct"]:          # 每个百分点才重绘，避免大权重时刷屏
            state["pct"] = pct
            _draw_bar(label, sent, tot)

    req = urllib.request.Request(
        server.rstrip("/") + "/api/upload",
        data=_ProgressFile(body, cb),
        # file-like 请求体 urllib 不会自动设 Content-Length，必须手动给，否则会报错或走 chunked。
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Content-Length": str(total)},
        method="POST")
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.loads(resp.read().decode())
    _draw_bar(label, total, total, done=True)
    return out["file_url"], out["sha256"]


# ----------------------------- 状态渲染（中文） -----------------------------
def _fmt_mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _icon_and_body(sub: dict) -> tuple[str, str]:
    """一条提交 → (图标, 中文正文)。评测阶段失败按 stage 映射，绝不显示日志。"""
    status = sub.get("status", "queued")
    if status == "queued":
        ahead = sub.get("submissions_ahead")
        running = sub.get("running_ahead") or 0
        if ahead is None:
            return "⏳", "排队中"
        parts = []
        if ahead > 0:
            parts.append(f"前方还有 {ahead} 个待评测")
        if running > 0:
            parts.append(f"另有 {running} 个评测中")
        if not parts:
            return "⏳", "排队中（即将开始评测）"
        return "⏳", "排队中（" + "，".join(parts) + "）"
    if status == "running":
        el = sub.get("running_elapsed")
        if el is None:
            return "⚙", "评测中"
        limit = sub.get("wall_limit_s") or WALL_LIMIT_S  # 优先服务器上报的真实墙钟；缺省回退本地常量
        return "⚙", f"评测中（已 {_fmt_mmss(el)} / 上限 {_fmt_mmss(limit)}）"
    if status == "done":
        total = sub.get("total")
        robot = sub.get("robot")
        tail = f"（{robot}）" if robot else ""
        return "✓", (f"完成！得分 {total}{tail}" if total is not None else f"完成{tail}")
    if status == "failed":
        base = STAGE_ZH.get(sub.get("stage"), _STAGE_FALLBACK)
        body = "失败 · " + base
        # 透传确切异常供选手自查：solution 运行时 / serve 启动导入 / eval 回合运行时（env.step 抛错，
        # 如动作维度不符）。绝对路径已在 worker 侧抹除；与 STAGE_ZH 通用兜底重复的（如「提交的代码无法
        # 运行」，或 eval 无消息时回退的「评测运行中出错」）不重复展示。timeout/env/deps 内部日志仍不外泄。
        err = sub.get("error")
        if sub.get("stage") in ("solution", "serve", "eval") and err and err not in base:
            body += f"\n         ↳ {err}"
        return "✗", body
    if status == "rejected":
        return "✗", "已拒绝 · " + (sub.get("error") or "提交被拒绝")
    return "·", str(status)


def _status_line(sub: dict) -> str:
    icon, body = _icon_and_body(sub)
    comp = sub.get("competition") or "?"
    sid = sub.get("submission_id", "?")
    return f"  {sid}  {icon} {body}   [{comp}]"


# ----------------------------- 轮询等待 -----------------------------
def _fetch_status(server: str, token: str) -> dict:
    url = server.rstrip("/") + "/api/status?" + urllib.parse.urlencode({"token": token})
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _find_sub(body: dict, sub_id: str) -> dict | None:
    return next((s for s in body.get("submissions", []) if s.get("submission_id") == sub_id), None)


def _wait_until_terminal(server: str, token: str, sub_id: str) -> dict | None:
    """轮询直到该提交进入终态(done/failed/rejected)。TTY 下单行刷新进度，非 TTY 只在状态切换时打印。
    返回最终 sub dict；Ctrl-C 中断返回 None（提交已入队，不受影响）。"""
    frame = 0
    last_len = 0
    last_status = None
    try:
        while True:
            try:
                sub = _find_sub(_fetch_status(server, token), sub_id)
            except (urllib.error.URLError, OSError):
                time.sleep(POLL_INTERVAL_S)         # 网络抖动：稍后重试，不打断等待
                continue
            if sub is None:
                time.sleep(POLL_INTERVAL_S)
                continue
            status = sub.get("status")
            if status in ("done", "failed", "rejected"):
                if _TTY and last_len:
                    sys.stderr.write("\r" + " " * last_len + "\r")
                    sys.stderr.flush()
                return sub
            _, body = _icon_and_body(sub)
            if _TTY:
                spin = _SPINNER[frame % len(_SPINNER)]
                frame += 1
                line = f"\r{spin} {body}"
                pad = " " * max(0, last_len - len(line))
                sys.stderr.write(line + pad)
                sys.stderr.flush()
                last_len = len(line)
            elif status != last_status:            # 非 TTY：只在 排队→评测 等切换时打印一行
                print(f"  …{body}", file=sys.stderr)
            last_status = status
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        if _TTY and last_len:
            sys.stderr.write("\n")
        print(f"已入队（{sub_id}），可稍后用 --status 查看结果。", file=sys.stderr)
        return None


# ----------------------------- 命令 -----------------------------
def do_submit(args) -> int:
    if args.robot not in ROBOT_TASK_ID:
        print(f"错误：--robot 非法（oli/sfyg_tron2a/wfyg_tron2a）：{args.robot}", file=sys.stderr)
        return 2
    if ROBOT_COMPETITION[args.robot] != args.competition:
        print(f"错误：--robot {args.robot} 属于赛题 {ROBOT_COMPETITION[args.robot]}，"
              f"与 --competition {args.competition} 不符", file=sys.stderr)
        return 2
    if not args.ckpt_file or not os.path.isfile(args.ckpt_file):
        print(f"错误：权重文件不存在：{args.ckpt_file}", file=sys.stderr)
        return 2
    if not args.code_dir or not os.path.isdir(args.code_dir):
        print(f"错误：代码目录不存在：{args.code_dir}", file=sys.stderr)
        return 2

    # ① 打包代码 + 上传（提交阶段：失败给清晰原因）
    try:
        tmp = _pack_code_dir(args.code_dir)
        print(f"✓ 打包代码 ({_human_bytes(os.path.getsize(tmp))})", file=sys.stderr)
        try:
            ckpt_url, ckpt_sha = _upload_file(args.server, args.token, args.ckpt_file, label="上传权重")
            code_url, code_sha = _upload_file(args.server, args.token, tmp, label="上传代码")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        print(f"✗ 上传被拒绝（HTTP {e.code}）：{detail}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"✗ 上传失败（无法连接评测服务器）：{e}", file=sys.stderr)
        return 1

    # ② 入队
    payload = {"token": args.token, "competition": args.competition, "robot": args.robot,
               "ckpt_url": ckpt_url, "ckpt_sha256": ckpt_sha,
               "code_url": code_url, "code_sha256": code_sha,
               "config_path": args.config_path}
    req = urllib.request.Request(args.server.rstrip("/") + "/api/submit",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"✗ 提交被拒绝（HTTP {e.code}）：{e.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"✗ 无法连接评测服务器：{e.reason}", file=sys.stderr)
        return 1

    sub_id = body.get("submission_id", "?")
    if body.get("status") == "rejected":
        print(f"✗ 提交被拒绝：{body.get('error', '（无原因）')}", file=sys.stderr)
        return 1
    print(f"✓ 已入队  {sub_id}", file=sys.stderr)

    # ③ 默认就地等到出结果
    if args.no_wait:
        print(f"  查看进度： submit.py --server {args.server} --token=<队伍令牌> --status",
              file=sys.stderr)
        return 0
    final = _wait_until_terminal(args.server, args.token, sub_id)
    if final is None:                       # Ctrl-C，已提示
        return 0
    icon, text = _icon_and_body(final)
    print(f"{icon} {text}", file=sys.stderr)
    return 0 if final.get("status") == "done" else 1


def do_status(args) -> int:
    try:
        body = _fetch_status(args.server, args.token)
    except urllib.error.HTTPError as e:
        print(f"查询失败（HTTP {e.code}）：{e.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"无法连接评测服务器：{e.reason}", file=sys.stderr)
        return 1
    subs = body.get("submissions", [])
    print(f"队伍 tk_{body.get('display', '?')} 的提交（{len(subs)} 条）：")
    for s in subs:
        print(_status_line(s))
    if not subs:
        print("  （暂无提交）")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="https://submit.troncamp-loco.limxdynamics.com",
                   help="评测服务器地址（默认官方服务器 %(default)s）")
    p.add_argument("--token", required=True, help="队伍令牌（以 - 开头时用 --token=<令牌>）")
    p.add_argument("--status", action="store_true", help="查询本队提交状态/分数（而非提交）")
    p.add_argument("--no-wait", action="store_true", help="只提交、不等结果（之后用 --status 查）")
    p.add_argument("--competition", choices=["tron", "humanoid"], help="赛题")
    p.add_argument("--robot", choices=["oli", "sfyg_tron2a", "wfyg_tron2a"], help="机型")
    p.add_argument("--ckpt-file", help="策略权重 policy.pt")
    p.add_argument("--code-dir", help="推理代码目录（含 solution.py）")
    p.add_argument("--config-path", default=None, help="可选 config 路径（包内相对）")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.status:
        return do_status(args)
    if not args.competition or not args.robot:
        print("错误：提交需要 --competition 和 --robot", file=sys.stderr)
        return 2
    return do_submit(args)


if __name__ == "__main__":
    sys.exit(main())
