# sim/recorded_moves.py —— 官方录制动作库 → 我们 9 维格式的【只读】转换器 + 演示查看器
#
# 背景:Reachy Mini Control 界面里那些丰富的表情/舞蹈动作,来自官方两个 HuggingFace 数据集
#   pollen-robotics/reachy-mini-emotions-library  (85 个情绪动作)
#   pollen-robotics/reachy-mini-dances-library    (19 个舞蹈动作)
# 启动 daemon 时会自动下载到本机 HF 缓存。许可 apache-2.0(README frontmatter)。
#
# 【定位】本脚本只做两件事,均为"演示素材",不碰训练数据管线:
#   1. 把官方录制动作(50Hz、head 4×4 矩阵)无损转成我们的 [T,9] @30Hz 格式;
#   2. 通过和 demo_show 同款执行层把它播给仿真,供肉眼预览这 104 个动作。
# 它【不】导入/修改 data_gen/templates.py、schema.py 或任何数据生成代码。
# 是否把这些动作纳入训练模板库 = 设计"大事",见 docs/decisions/。
#
# 用法(先起好 sim daemon):
#   $env:PYTHONUTF8=1
#   ...\python.exe -u sim\recorded_moves.py list                 # 列出全部动作名+描述
#   ...\python.exe -u sim\recorded_moves.py play amazed1         # 播放单个动作
#   ...\python.exe -u sim\recorded_moves.py play curious1 laughing1 rage1   # 依次播放多个
#
# 数据格式(实测 amazed1.json):
#   {"description": str,
#    "time": [[t0],[t1],...],                         # 秒,50Hz
#    "set_target_data": [{"head": 4×4, "antennas":[右,左], "body_yaw": φ, "check_collision": bool}, ...]}
#   head 是近中立的相对姿态(平移 ~毫米,旋转小角);录制时首帧≠中立,故查看时从中立平滑渐入。
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
import sys, json, glob
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

FPS = 30
DT = 1.0 / FPS
# 9 维顺序: 0x 1y 2z 3roll 4pitch 5yaw 6body_yaw 7ant_右 8ant_左

# HuggingFace 缓存里两个官方动作库(datasets 前缀)。跨用户目录用通配。
_HF = Path.home() / ".cache" / "huggingface" / "hub"
_LIBS = {
    "emotion": "datasets--pollen-robotics--reachy-mini-emotions-library",
    "dance":   "datasets--pollen-robotics--reachy-mini-dances-library",
}


def _snapshot_dir(lib_dir_name):
    """定位某数据集当前 snapshot 目录(HF 缓存结构:<repo>/snapshots/<hash>/)。"""
    cands = sorted((_HF / lib_dir_name / "snapshots").glob("*"))
    return cands[-1] if cands else None


def list_moves():
    """返回 {name: {"path": Path, "kind": "emotion"|"dance"}},按名字排序。"""
    out = {}
    for kind, dirname in _LIBS.items():
        snap = _snapshot_dir(dirname)
        if snap is None:
            continue
        for f in sorted(snap.glob("*.json")):
            out[f.stem] = {"path": f, "kind": kind}
    return dict(sorted(out.items()))


def load_move_9d(name_or_path, fps=FPS, ramp_s=0.3):
    """把一个官方录制动作转成 (traj [T,9] float32, description)。
    - head 4×4 → (xyz, rpy):录制 head 是相对中立的姿态,直接取其平移+欧拉角即我们的前 6 维;
      查看器 play() 再做 neutral@d 还原到仿真中立系(两处中立均近似单位阵,误差在毫米/毫弧度级)。
    - 50Hz → fps:按录制时间戳线性重采样(逐维 np.interp)。
    - ramp_s:从中立(全 0)平滑渐入首帧、末尾渐出回中立,避免查看时起步/收尾突跳
      (官方播放器由 goto 接管这段;我们用 set_target 直推,故自加渐入渐出)。
    """
    p = Path(name_or_path)
    if not p.exists():
        info = list_moves().get(str(name_or_path))
        if info is None:
            raise FileNotFoundError(f"未找到录制动作: {name_or_path}(用 list 查看可用名字)")
        p = info["path"]
    d = json.loads(p.read_text(encoding="utf-8"))
    S = d["set_target_data"]
    # time 字段两种形态:[[t0],[t1],...] 或 [t0,t1,...],都归一成一维
    t_rec = np.array([row[0] if isinstance(row, (list, tuple)) else row
                      for row in d["time"]], dtype=np.float64)
    t_rec = t_rec - t_rec[0]

    src = np.zeros((len(S), 9), dtype=np.float64)
    for i, s in enumerate(S):
        h = np.array(s["head"], dtype=np.float64)
        src[i, 0:3] = h[:3, 3]
        src[i, 3:6] = R.from_matrix(h[:3, :3]).as_euler("xyz")
        ant = s["antennas"]
        src[i, 7] = ant[0]
        src[i, 8] = ant[1]
        src[i, 6] = s.get("body_yaw", 0.0)

    # 重采样到目标 fps
    dur = float(t_rec[-1])
    T = max(2, int(round(dur * fps)))
    t_new = np.linspace(0.0, dur, T)
    traj = np.zeros((T, 9), dtype=np.float32)
    for j in range(9):
        traj[:, j] = np.interp(t_new, t_rec, src[:, j])

    # 从中立渐入首帧、末尾渐出回中立
    nr = min(int(ramp_s * fps), T // 2)
    if nr >= 1:
        ramp = (np.linspace(0, 1, nr) ** 2 * (3 - 2 * np.linspace(0, 1, nr))).astype(np.float32)
        traj[:nr] *= ramp[:, None]
        traj[T - nr:] *= ramp[::-1][:, None]
    return traj, d.get("description", "")


def _play(mini, neutral, traj):
    """和 sim/demo_show.py 同款 30Hz 执行层。"""
    import time
    t0 = time.perf_counter()
    for k, a in enumerate(traj):
        m = np.eye(4)
        m[:3, :3] = R.from_euler("xyz", a[3:6]).as_matrix()
        m[:3, 3] = a[:3]
        mini.set_target(head=neutral @ m, antennas=[float(a[7]), float(a[8])],
                        body_yaw=float(a[6]))
        nxt = t0 + (k + 1) * DT
        while time.perf_counter() < nxt:
            pass


def _cmd_list():
    moves = list_moves()
    if not moves:
        print("未在 HF 缓存找到官方动作库。确认已启动过 Reachy Mini Control / sim daemon 触发下载。")
        return
    for name, info in moves.items():
        traj, desc = load_move_9d(info["path"])
        print(f"[{info['kind']:7s}] {name:20s} {len(traj)/FPS:4.1f}s  {desc[:70]}")
    print(f"\n共 {len(moves)} 个动作(用 `play <名字>` 预览)。")


def _cmd_play(names):
    import time
    from reachy_mini import ReachyMini
    trajs = [(n, *load_move_9d(n)) for n in names]
    print(">>> 连接仿真(约 4~5 秒)...", flush=True)
    with ReachyMini(connection_mode="localhost_only", media_backend="no_media") as mini:
        neutral = mini.get_current_head_pose()
        print(">>> 已连接。\n", flush=True)
        for name, traj, desc in trajs:
            print(f"▶ {name}  ({len(traj)}帧/{len(traj)/FPS:.1f}s)  {desc[:60]}", flush=True)
            _play(mini, neutral, traj)
            _play(mini, neutral, np.zeros((int(0.6 * FPS), 9), dtype=np.float32))  # 动作间回中
    print("\n>>> 播放完毕。", flush=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        _cmd_list()
    elif sys.argv[1] == "play":
        names = sys.argv[2:] or ["amazed1"]
        _cmd_play(names)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
