# data_gen/build_motion_library.py
# ─────────────────────────────────────────────────────────────────────────────
# 把官方 104 个录制动作【无损沉淀】成本项目的固定动作模板库(资产,非训练管线)。
#
# 来源:Reachy Mini Control 界面里的表情/舞蹈动作,来自官方两个 HuggingFace 数据集
#   pollen-robotics/reachy-mini-emotions-library  (85 情绪,含 .ogg 语音)
#   pollen-robotics/reachy-mini-dances-library    (19 舞蹈,无音频)
# 启动 daemon 时自动下载到本机 HF 缓存。许可 apache-2.0(需署名保留)。
#
# 【定位/边界】本脚本只做"保存资产":读缓存 → 无损落盘 → 生成可读目录。
#   它【不】导入/修改 templates.py、schema.py、数据管线,也【不】决定这些动作
#   是否/如何纳入训练——那是设计 §4 的"大事",见 docs/decisions/。
#
# 保存原则:无损。每个动作存
#   - 原始 head 4×4 / antennas / body_yaw / time / check_collision(真机执行的真值)
#   - 额外派生我们的 [T,9] 表示(便利视图,与 head 可无损往返)
#   - description(神态/适用情况)、时长、来源、许可 → 写入 index.json / index.md
#   原始 50Hz 不重采样、不加渐入渐出(那是播放器的事,见 sim/recorded_moves.py)。
#
# 用法:
#   PYTHONUTF8=1 python -m data_gen.build_motion_library                 # 落地到默认目录
#   PYTHONUTF8=1 python -m data_gen.build_motion_library <out_dir>       # 指定输出目录
# ─────────────────────────────────────────────────────────────────────────────
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# 9 维顺序: 0x 1y 2z 3roll 4pitch 5yaw 6body_yaw 7ant_右 8ant_左
DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "body_yaw", "ant_right", "ant_left"]
FPS_ORIGINAL = 50  # 两个官方库实测均为 20ms/帧
LICENSE = "apache-2.0"
ATTRIBUTION = "pollen-robotics/reachy-mini-emotions-library, pollen-robotics/reachy-mini-dances-library"

# 默认输出目录:本模块同级的 motion_library/
DEFAULT_OUT = Path(__file__).resolve().parent / "motion_library"

# HuggingFace 缓存里两个官方动作库(datasets 前缀)
_HF = Path.home() / ".cache" / "huggingface" / "hub"
_LIBS = {
    "emotion": "datasets--pollen-robotics--reachy-mini-emotions-library",
    "dance": "datasets--pollen-robotics--reachy-mini-dances-library",
}


def _snapshot_dir(lib_dir_name):
    """定位某数据集当前 snapshot 目录(HF 缓存结构:<repo>/snapshots/<hash>/)。"""
    cands = sorted((_HF / lib_dir_name / "snapshots").glob("*"))
    return cands[-1] if cands else None


def discover_moves():
    """扫描本机 HF 缓存,返回 [{"name","kind","path","audio"(Path|None)}],按名字排序。
    找不到缓存则返回 []。"""
    out = []
    for kind, dirname in _LIBS.items():
        snap = _snapshot_dir(dirname)
        if snap is None:
            continue
        for f in sorted(snap.glob("*.json")):
            ogg = f.with_suffix(".ogg")
            out.append({
                "name": f.stem,
                "kind": kind,
                "path": f,
                "audio": ogg if ogg.exists() else None,
            })
    return sorted(out, key=lambda m: m["name"])


def extract_move(path):
    """把一个官方录制动作 JSON 无损解析成数组字典 + description。
    返回 (arrays: dict, description: str)。
    arrays 含: time[T], head[T,4,4], antennas[T,2], body_yaw[T], check_collision[T], action9[T,9]。
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    S = d["set_target_data"]
    T = len(S)

    # time 字段两种形态:[[t0],...] 或 [t0,...];归一成一维,并平移到从 0 起
    t = np.array([row[0] if isinstance(row, (list, tuple)) else row for row in d["time"]],
                 dtype=np.float64)
    t = t - t[0]

    head = np.zeros((T, 4, 4), dtype=np.float64)
    antennas = np.zeros((T, 2), dtype=np.float64)
    body_yaw = np.zeros(T, dtype=np.float64)
    check_collision = np.zeros(T, dtype=bool)
    for i, s in enumerate(S):
        head[i] = np.array(s["head"], dtype=np.float64)
        antennas[i] = np.array(s["antennas"], dtype=np.float64)[:2]
        body_yaw[i] = float(s.get("body_yaw", 0.0))
        check_collision[i] = bool(s.get("check_collision", False))

    # 派生我们的 9 维便利视图(与 head 无损往返:xyz=平移, rpy=xyz 欧拉)
    action9 = np.zeros((T, 9), dtype=np.float32)
    action9[:, 0:3] = head[:, :3, 3]
    action9[:, 3:6] = R.from_matrix(head[:, :3, :3]).as_euler("xyz")
    action9[:, 6] = body_yaw
    action9[:, 7] = antennas[:, 0]
    action9[:, 8] = antennas[:, 1]

    arrays = {
        "time": t,
        "head": head,
        "antennas": antennas,
        "body_yaw": body_yaw,
        "check_collision": check_collision,
        "action9": action9,
    }
    return arrays, d.get("description", "")


def build_library(out_dir, copy_audio=True):
    """把发现的全部官方动作无损落盘到 out_dir,并生成 index.json / index.md。
    返回 index 字典。out_dir 不存在则创建。"""
    out_dir = Path(out_dir)
    moves = discover_moves()

    entries = []
    for m in moves:
        arrays, desc = extract_move(m["path"])
        rel_npz = f"moves/{m['kind']}s/{m['name']}.npz"  # emotions/ dances/
        npz_path = out_dir / rel_npz
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **arrays)

        rel_audio = None
        if copy_audio and m["audio"] is not None:
            rel_audio = f"audio/{m['name']}.ogg"
            dst = out_dir / rel_audio
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(m["audio"], dst)

        T = arrays["action9"].shape[0]
        a = arrays["action9"]
        entries.append({
            "name": m["name"],
            "kind": m["kind"],
            "description": desc,
            "n_frames": int(T),
            "duration_s": round(float(arrays["time"][-1]), 3),
            "fps": FPS_ORIGINAL,
            "has_audio": rel_audio is not None,
            "npz": rel_npz,
            "audio": rel_audio,
            "range_min": [round(float(x), 4) for x in a.min(axis=0)],
            "range_max": [round(float(x), 4) for x in a.max(axis=0)],
        })

    index = {
        "n_moves": len(entries),
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "source": {kind: repo for kind, repo in _LIBS.items()},
        "fps_original": FPS_ORIGINAL,
        "dim_names": DIM_NAMES,
        "note": "无损保存的官方录制动作资产;是否纳入训练见 docs/decisions/。",
        "moves": entries,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "index.md").write_text(_render_index_md(index), encoding="utf-8")
    return index


def _render_index_md(index):
    """把 index 渲染成人类可读的中文目录(神态/情况一目了然)。"""
    lines = [
        "# 官方录制动作库 · 目录(神态 / 适用情况)",
        "",
        f"> 无损保存的 **{index['n_moves']}** 个官方录制动作。来源 `{index['attribution']}`,"
        f"许可 **{index['license']}**(需署名)。原始 {index['fps_original']}Hz。",
        "> 每个动作的轨迹存于 `moves/<类>/<名>.npz`,情绪动作语音存于 `audio/<名>.ogg`。",
        "> 是否/如何纳入训练 = 设计 §4 的大事,见 `docs/decisions/`。**本目录只是资产保存。**",
        "",
    ]
    for kind, title in [("emotion", "情绪动作 emotions"), ("dance", "舞蹈动作 dances")]:
        rows = [e for e in index["moves"] if e["kind"] == kind]
        if not rows:
            continue
        lines += [f"## {title}({len(rows)} 个)", "",
                  "| 动作名 | 时长 | 音频 | 神态 / 适用情况(description) |",
                  "| --- | --- | --- | --- |"]
        for e in rows:
            audio = "♪" if e["has_audio"] else "—"
            desc = (e["description"] or "").replace("\n", " ").replace("|", "/")
            lines.append(f"| `{e['name']}` | {e['duration_s']:.1f}s | {audio} | {desc} |")
        lines.append("")
    return "\n".join(lines)


def load_saved_move(name, root=DEFAULT_OUT, fps=None):
    """读取已落盘的某个动作,返回 dict:
        {name, kind, description, time[T], head[T,4,4], antennas[T,2],
         body_yaw[T], check_collision[T], action9[T,9]}
    - fps=None:原样返回(原始 50Hz、变步长时间戳)。
    - fps=30 等:对 action9 按原始时间戳线性重采样到等步长 fps,并同步 time
      (head/antennas 等原始真值不重采样,保持无损)。
    """
    root = Path(root)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    entry = next((e for e in index["moves"] if e["name"] == name), None)
    if entry is None:
        raise KeyError(f"未找到动作 {name}(见 {root/'index.json'})")
    d = np.load(root / entry["npz"])
    out = {k: d[k] for k in d.files}
    out.update(name=name, kind=entry["kind"], description=entry["description"])
    if fps is not None:
        t = out["time"]
        dur = float(t[-1])
        T = max(2, int(round(dur * fps)))
        t_new = np.linspace(0.0, dur, T)
        a = out["action9"]
        res = np.zeros((T, 9), dtype=np.float32)
        for j in range(9):
            res[:, j] = np.interp(t_new, t, a[:, j])
        out["action9"] = res
        out["time"] = t_new
    return out


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    moves = discover_moves()
    if not moves:
        print("未在本机 HF 缓存找到官方动作库。请先启动过 Reachy Mini Control / sim daemon 触发下载。")
        return
    print(f">>> 发现 {len(moves)} 个动作,开始无损落盘到:{out}", flush=True)
    index = build_library(out)
    n_audio = sum(1 for e in index["moves"] if e["has_audio"])
    print(f">>> 完成:{index['n_moves']} 个动作(其中 {n_audio} 个带语音)。", flush=True)
    print(f">>> 目录:{out/'index.md'}", flush=True)
    print(f">>> 索引:{out/'index.json'}", flush=True)


if __name__ == "__main__":
    main()
