# data_gen/augment.py —— 官方动作库(LIB-01)读取与扰动增广(C2)
"""
把官方录制动作(motion_library 的 action9)读入并做参数化增广,供 compose() 的 `lib:` 路径使用。

接口(C4 依赖,签名不得改):
- load_lib_move(name) -> np.ndarray[T,9]
    从 motion_library 读某动作的 action9,重采样到均匀 50Hz(原生多为 50Hz;4 个动作含重复
    时间戳,重采样消除毛刺)。返回 float32 [T,9],未做任何缩放/限幅。
- augment_lib_move(traj, rng, time_scale=(0.85,1.15), amp_scale=(0.7,1.15)) -> np.ndarray[T',9]
    ① 时间缩放:随机 k∈time_scale,线性重采样到 round(T*k) 步。
    ② 幅度缩放:头部姿态(0-5)/身体(6)/天线(7-8)三组各取独立系数(均来自 amp_scale)。
    ③ 边界连续化:首尾各 0.2s 乘 smoothstep 包络压到 0(官方动作起止未必在中立位,不压会与
       idle 叠加处跳变)。
    ④ 限幅到 limits_real(真机可行域);不改入参。

设计说明见 build_limits_real.py 与 M1.5-C2 报告。
"""
import json
from pathlib import Path

import numpy as np

from data_gen.build_motion_library import load_saved_move
from data_gen.templates import FPS

ROOT = Path(__file__).resolve().parent.parent
BOUNDARY_S = 0.2   # 首尾平滑时长

_LIMITS_REAL = None


def _limits_real():
    global _LIMITS_REAL
    if _LIMITS_REAL is None:
        _LIMITS_REAL = json.loads((ROOT / "sim" / "limits_real.json").read_text(encoding="utf-8"))
    return _LIMITS_REAL


def load_lib_move(name):
    """读官方动作的 action9,重采样到均匀 FPS(50Hz)。返回 float32 [T,9]。"""
    return load_saved_move(name, fps=FPS)["action9"].astype(np.float32)


def _boundary_env(T, ramp_s=BOUNDARY_S):
    """首尾各 ramp_s 秒的 smoothstep 包络(端点=0,中段=1)。"""
    r = min(int(round(ramp_s * FPS)), T // 2)
    env = np.ones(T, dtype=np.float64)
    if r > 0:
        x = np.linspace(0.0, 1.0, r)
        s = x * x * (3 - 2 * x)          # smoothstep 0→1
        env[:r] = s
        env[-r:] = s[::-1]
    env[0] = 0.0
    env[-1] = 0.0
    return env


def augment_lib_move(traj, rng, time_scale=(0.85, 1.15), amp_scale=(0.7, 1.15)):
    a = np.asarray(traj, dtype=np.float64)
    T = len(a)

    # ① 时间缩放:线性重采样到 round(T*k)
    k = float(rng.uniform(*time_scale))
    Tn = max(2, int(round(T * k)))
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, Tn)
    out = np.empty((Tn, 9), dtype=np.float64)
    for j in range(9):
        out[:, j] = np.interp(dst, src, a[:, j])

    # ② 幅度缩放:头部姿态 / 身体 / 天线 三组独立系数
    c_head = float(rng.uniform(*amp_scale))
    c_body = float(rng.uniform(*amp_scale))
    c_ant = float(rng.uniform(*amp_scale))
    coeff = np.array([c_head] * 6 + [c_body] + [c_ant] * 2, dtype=np.float64)
    out *= coeff

    # ③ 边界连续化
    out *= _boundary_env(Tn)[:, None]

    # ④ 限幅到真机可行域
    lo = np.array(_limits_real()["safe_min"], dtype=np.float64)
    hi = np.array(_limits_real()["safe_max"], dtype=np.float64)
    out = np.clip(out, lo, hi)

    return out.astype(np.float32)
