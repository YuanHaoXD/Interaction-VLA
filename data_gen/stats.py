# data_gen/stats.py —— 数据集统计聚合(A5)
"""
扫数据集目录的全部 episode,聚合 9 维动作的 min/max/mean/std + 段数/总时长/标签分布,
写数据集根 stats.json。归一化(逐维到 [-1,1])用得上 min/max;训练监控用得上标签分布。

用流式(Welford)累积,10 万段也不吃内存。
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np

DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "body_yaw", "ant_right", "ant_left"]


def aggregate(data_dir: str) -> Dict:
    root = Path(data_dir)
    eps = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("ep_"))

    n_frames = 0
    total_dur = 0.0
    d = 9
    vmin = np.full(d, np.inf, dtype=np.float64)
    vmax = np.full(d, -np.inf, dtype=np.float64)
    vsum = np.zeros(d, dtype=np.float64)
    vsqsum = np.zeros(d, dtype=np.float64)
    label_counter = Counter()
    n_ep = 0

    for ep in eps:
        act_f = ep / "actions.npy"
        meta_f = ep / "meta.json"
        if not act_f.exists():
            continue
        a = np.load(act_f).astype(np.float64)   # [T,9]
        if a.ndim != 2 or a.shape[1] != d:
            continue
        n_ep += 1
        n_frames += len(a)
        vmin = np.minimum(vmin, a.min(0))
        vmax = np.maximum(vmax, a.max(0))
        vsum += a.sum(0)
        vsqsum += (a * a).sum(0)
        if meta_f.exists():
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            total_dur += float(meta.get("duration_s", 0.0))
            for ev in meta.get("template_events", []):
                label_counter[ev.get("label", "?")] += 1

    if n_frames == 0:
        raise RuntimeError(f"{data_dir}: 没有可统计的 actions.npy")

    mean = vsum / n_frames
    var = np.maximum(vsqsum / n_frames - mean * mean, 0.0)
    std = np.sqrt(var)

    stats = {
        "n_episodes": n_ep,
        "n_frames": int(n_frames),
        "total_duration_s": round(total_dur, 1),
        "dim_names": DIM_NAMES,
        "min": vmin.round(5).tolist(),
        "max": vmax.round(5).tolist(),
        "mean": mean.round(5).tolist(),
        "std": std.round(5).tolist(),
        "label_distribution": dict(label_counter),
    }
    return stats


def write_stats(data_dir: str) -> Dict:
    stats = aggregate(data_dir)
    out = Path(data_dir) / "stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def main():
    import argparse
    ap = argparse.ArgumentParser(description="数据集统计聚合(A5)")
    ap.add_argument("--data", required=True, help="数据集目录")
    args = ap.parse_args()
    stats = write_stats(args.data)
    print(f"聚合完成:{stats['n_episodes']} 段 / {stats['n_frames']} 帧 / "
          f"{stats['total_duration_s']}s → {args.data}/stats.json")
    print("标签分布:", stats["label_distribution"])


if __name__ == "__main__":
    main()
