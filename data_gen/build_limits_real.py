# data_gen/build_limits_real.py —— 从官方动作库(LIB-01)提取"真机可行域"限位
"""
扫描 motion_library 全部 104 个官方动作的 action9(重采样到均匀 50Hz),
逐维取 min/max 与最大逐步速度,产出 sim/limits_real.json。

设计依据(M1.5 计划 C2 / 设计文档 §4.1):
- 官方动作录制于真机 → 库包络即"真机可行域"的数据驱动定义,替代 sim/limits.json 的
  仿真虚高限位(比真机大 3~4 倍,见 B1 报告 §4.2)。数据生成(compose)的限幅统一改用本文件。
- safe_min/max:逐维 [min, max] 各向外扩张 range 的 10% 裕量(计划:"加 10% 裕量")。
- max_vel:库内逐维全局最大逐步速度 ×1.1。取全局最大而非分位数,是为了保证任何库动作
  直接插入 compose 后不被"逐步限速"扭曲(库动作已是真机平滑录制);库动作段的连续性由
  augment 的首尾 smoothstep 包络保证,compose 的限速器退化为 NaN/尖峰兜底。
- 速度基于重采样到均匀 50Hz 的 action9 计算:4 个动作(mini-deep-sleep/toc-toc-toc/
  waiting/wake-mini-up)原生含重复时间戳,直接对原生 diff 会得到虚高速度,重采样消除该假象。

用法:python -m data_gen.build_limits_real
"""
import json
from pathlib import Path

import numpy as np

from data_gen.build_motion_library import load_saved_move, DEFAULT_OUT

ROOT = Path(__file__).resolve().parent.parent
FPS = 50
POS_MARGIN = 0.10   # 位置包络向外扩张比例
VEL_MARGIN = 0.10   # 速度上限放大比例


def build(lib_root=DEFAULT_OUT):
    index = json.loads((Path(lib_root) / "index.json").read_text(encoding="utf-8"))
    names = [e["name"] for e in index["moves"]]
    dim_names = index["dim_names"]

    gmin = np.full(9, np.inf)
    gmax = np.full(9, -np.inf)
    gvel = np.zeros(9)
    for name in names:
        a = load_saved_move(name, fps=FPS)["action9"].astype(np.float64)   # 均匀 50Hz
        gmin = np.minimum(gmin, a.min(axis=0))
        gmax = np.maximum(gmax, a.max(axis=0))
        if len(a) > 1:
            gvel = np.maximum(gvel, (np.abs(np.diff(a, axis=0)) * FPS).max(axis=0))

    rng = gmax - gmin
    safe_min = gmin - POS_MARGIN * rng
    safe_max = gmax + POS_MARGIN * rng
    max_vel = gvel * (1.0 + VEL_MARGIN)

    out = {
        "dim_names": dim_names,
        "safe_min": [round(float(v), 4) for v in safe_min],
        "safe_max": [round(float(v), 4) for v in safe_max],
        "max_vel": [round(float(v), 3) for v in max_vel],
        "source": "official_motion_library_envelope",
        "n_moves": len(names),
        "fps": FPS,
        "note": "真机可行域(库包络 min/max ±10% + 速度×1.1);数据生成限幅用本文件,勿用 sim/limits.json(仿真虚高)",
    }
    dst = ROOT / "sim" / "limits_real.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"扫描 {len(names)} 个官方动作 → {dst}")
    for i, d in enumerate(dim_names):
        print(f"  {d:10s} [{out['safe_min'][i]:+.4f}, {out['safe_max'][i]:+.4f}]  vmax={out['max_vel'][i]:.3f}")
    return out


if __name__ == "__main__":
    build()
