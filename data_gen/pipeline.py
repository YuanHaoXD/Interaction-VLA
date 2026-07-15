# data_gen/pipeline.py —— 标签→轨迹展开流水线(A5)
"""
把 JoyAI 标注数据(每秒对话时间线 + 5 类离散动作标签)展开为契约 C0 的 episode。

标注样本格式(与 datasets/example.json 同构):
  {
    "video_id": str, "video_duration_s": float,
    "question": [{"content": str, "time": float}, ...],
    "response": [ [ {"content": str, "time": float, "action": <5类之一>}, ... ], ... ]
  }
  response 是"每个 question 对应一组回复"的二维列表(一次提问可有多句回复)。

核心函数 convert_sample(sample, rng, limits) -> (meta, timeline, actions):
  - question 时刻 → user_text 事件 + speech_spans(跨度 = est_dur(content))
  - 带 action 标签的 response → 模板事件 + response 时间线事件(含 est_speech_dur_s、action_label)
  - compose() 生成 30Hz×9 维连续轨迹
  - duration = min(最后事件+5s, 视频长, 120s 上限)
"""

import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from data_gen.templates import compose, FPS
from data_gen.schema import write_episode, validate_episode

VALID_LABELS = {"nod", "shake_head", "tilt_head", "wiggle_antennas", "none"}
DUR_CAP_S = 120.0          # 单段时长上限
TAIL_PAD_S = 5.0           # 最后事件后留白


def est_dur(text: str) -> float:
    """时长代理(设计文档 §4.1/契约 C0):中文 5.5 字/秒、英文 14 字符/秒。
    按 ASCII 占比在两个语速间线性插值,统一给两侧用。"""
    if not text:
        return 0.3
    n = len(text)
    ascii_ratio = sum(c.isascii() for c in text) / n
    rate = 14.0 * ascii_ratio + 5.5 * (1 - ascii_ratio)   # 字符/秒
    return round(max(0.3, n / rate), 2)


def convert_sample(sample: Dict, rng: np.random.Generator, limits: Dict
                   ) -> Tuple[Dict, List[Dict], np.ndarray]:
    """把一个标注样本转成 (meta, timeline, actions)。不写盘。"""
    questions = sample.get("question", [])
    responses = sample.get("response", [])
    video_dur = float(sample.get("video_duration_s", 0.0))

    timeline: List[Dict] = []
    events: List[Dict] = []          # 给 compose 的模板事件
    speech_spans: List[Tuple[float, float]] = []
    last_event_t = 0.0

    # 1) 用户提问 → user_text + speech_span
    for q in questions:
        t = float(q["time"])
        timeline.append({"t": round(t, 2), "type": "user_text", "text": q["content"]})
        speech_spans.append((t, t + est_dur(q["content"])))
        last_event_t = max(last_event_t, t)

    # 2) 回复 → response 时间线事件 + 模板事件(带 action)
    for group in responses:
        for r in group:
            t = float(r["time"])
            label = r.get("action", "none")
            if label not in VALID_LABELS:
                label = "none"
            timeline.append({
                "t": round(t, 2), "type": "response", "text": r["content"],
                "est_speech_dur_s": est_dur(r["content"]), "action_label": label,
            })
            if label != "none":
                events.append({"label": label, "t": t})
            last_event_t = max(last_event_t, t)

    # 3) 时长:min(最后事件+留白, 视频长, 上限);无视频长则退回 最后事件+留白
    cand = last_event_t + TAIL_PAD_S
    duration_s = min(cand, video_dur) if video_dur > 0 else cand
    duration_s = min(duration_s, DUR_CAP_S)
    duration_s = max(duration_s, 1.0)

    timeline.sort(key=lambda e: e["t"])

    # 4) 轨迹
    actions = compose(duration_s, events, speech_spans, rng, limits)

    meta = {
        "episode_id": f"ep_{sample['video_id']}",
        "source": "synthetic_v1_from_annotation",
        "duration_s": round(duration_s, 3),
        "fps_action": FPS,
        "seed": int(rng.integers(0, 2**31 - 1)),
        "template_events": events,
        "origin_video": f"{sample['video_id']}.mp4",
        "schema_version": 1,
    }
    return meta, timeline, actions


# ---------------- 批量驱动(Step 3) ----------------

def _process_one(args) -> Tuple[str, bool, str]:
    """单样本:转换+写盘+校验。返回 (episode_id, ok, reason)。"""
    sample, out_dir, quarantine_dir, seed = args
    ep_id = f"ep_{sample['video_id']}"
    ep_path = Path(out_dir) / ep_id
    # 增量:已存在且校验通过则跳过
    if ep_path.exists() and not validate_episode(ep_path):
        return ep_id, True, "skip(已存在)"
    try:
        rng = np.random.default_rng(seed)
        meta, timeline, actions = convert_sample(sample, rng, _LIMITS_CACHE[out_dir])
        write_episode(ep_path, meta, timeline, actions)
        errs = validate_episode(ep_path)
        if errs:
            _quarantine(ep_path, quarantine_dir)
            return ep_id, False, f"校验失败:{errs[0]}"
        return ep_id, True, "ok"
    except Exception as e:
        return ep_id, False, f"异常:{type(e).__name__}:{e}"


_LIMITS_CACHE: Dict[str, Dict] = {}


def _quarantine(ep_path: Path, quarantine_dir: str) -> None:
    import shutil
    q = Path(quarantine_dir); q.mkdir(parents=True, exist_ok=True)
    if ep_path.exists():
        dst = q / ep_path.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(ep_path), str(dst))


def run_pipeline(annotation_path: str, out_dir: str, limits: Dict,
                 n_max: int = None, n_workers: int = None) -> Dict:
    """批量:扫标注文件 → 逐样本转换/写盘/校验。返回统计 dict。"""
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) else data
    if n_max:
        samples = samples[:n_max]

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    quarantine = str(out.parent / (out.name + "_quarantine"))
    _LIMITS_CACHE[out_dir] = limits

    tasks = [(s, out_dir, quarantine, i) for i, s in enumerate(samples)]
    n_workers = n_workers or min(mp.cpu_count(), 16)

    results = []
    if n_workers > 1 and len(tasks) > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(_process_one, tasks)
    else:
        results = [_process_one(t) for t in tasks]

    ok = sum(1 for _, good, _ in results if good)
    fail = [(eid, r) for eid, good, r in results if not good]
    return {"total": len(results), "ok": ok, "fail": len(fail), "fail_detail": fail[:20]}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="标签→轨迹展开流水线(A5)")
    ap.add_argument("--annotation", required=True, help="标注 json(datasets/example.json 同构)")
    ap.add_argument("--out", required=True, help="输出数据集目录(不入 git)")
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    limits = json.loads((root / "sim" / "limits.json").read_text(encoding="utf-8"))
    stats = run_pipeline(args.annotation, args.out, limits, args.n_max, args.workers)
    print(f"完成:{stats['ok']}/{stats['total']} 通过,{stats['fail']} 失败")
    for eid, reason in stats["fail_detail"]:
        print(f"  ✗ {eid}: {reason}")


if __name__ == "__main__":
    main()
