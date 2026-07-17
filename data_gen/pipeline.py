# data_gen/pipeline.py —— 标签→轨迹展开流水线(A5 / M1.5-C4)
"""
把 JoyAI 标注数据(每秒对话时间线 + 离散动作标签)展开为契约 C0 的 episode。

标注样本格式(M1.5-C4 v2，簇级 demeanor):
  {
    "video_id": str, "video_duration_s": float,
    "question": [{"content": str, "time": float}, ...],
    "response": [ {"content": str, "time": float, "action": <簇名>}, ... ]
      或 chat 格式: [ [ {"content": str, "time": float, "action": <簇名>}, ... ], ... ]
  }
  narration 是 1:N 结构（1 问对 N 答），response 为一维列表。
  chat 是兼容格式（二维列表，仅第一组有效）。

动作标签值域(M1.5-C4 v2):
  - 13 个神态簇名（affirm/explain/attend/think/unsure/joy/surprise/fear/negate/annoy/sad/warm/awkward）
    → 按 cluster_map 加权采样成具体动作 → 生成 lib:<动作名> 事件。
  - 4 类手写模板 nod/shake_head/tilt_head/wiggle_antennas → 走原模板路径(兼容旧 fixture)。
  - none / 未知 → 降级 none,不产事件。

核心函数 convert_sample(sample, rng, limits, cluster_map) -> (meta, timeline, actions):
  - question 时刻 → user_text 事件 + speech_spans(跨度 = est_dur(content))
  - 带 action 簇名的 response → 按渲染概率+下采样规则 → compose 事件 + response 时间线事件
  - compose() 生成 50Hz×9 维连续轨迹；lib: 事件走 augment 增广
  - duration = min(最后事件+5s, 视频长, 120s 上限)

新增(M1.5-C4):
  - 簇名→动作映射（cluster_map.json 加权采样）
  - 渲染概率控制（各簇独立，M2 可调）
  - explain 事件下采样至 ≤70%
  - 长沉默待机注入（分层池，呼吸底座+稀疏点缀+困意升级）
"""

import json
import math
import multiprocessing as mp
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from data_gen.templates import compose, FPS
from data_gen.schema import write_episode, validate_episode

TEMPLATE_LABELS = {"nod", "shake_head", "tilt_head", "wiggle_antennas"}   # 手写正弦模板(旧 4 类)
VALID_LABELS = TEMPLATE_LABELS | {"none"}                                 # 向后兼容别名(模板路径)
CLUSTER_LABELS = {"affirm", "explain", "attend", "think", "unsure", "joy",
                 "surprise", "fear", "negate", "annoy", "sad", "warm", "awkward"}  # 13 神态簇
DUR_CAP_S = 120.0          # 单段时长上限
TAIL_PAD_S = 5.0           # 最后事件后留白
EXPLAIN_DOWNSAMPLE_MAX = 0.70  # explain 事件占比上限（下采样）
IDLE_COOLDOWN_S = 20.0     # 待机注入冷却时间


@lru_cache(maxsize=1)
def load_cluster_map() -> Dict:
    """加载 cluster_map.json（簇→动作映射+渲染概率+待机池）。"""
    path = Path(__file__).parent / "cluster_map.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # 返回空配置（极端环境容错）
        return {
            "clusters": {},
            "render_prob": {},
            "_idle_pool": {"15_30s": [], "30_60s": [], "over_60s": []}
        }


def sample_move_from_cluster(cluster: str, rng: np.random.Generator,
                              cluster_map: Dict = None) -> Optional[str]:
    """按 cluster_map 加权采样一个具体动作名。返回 None 表示该簇无配置。"""
    if cluster_map is None:
        cluster_map = load_cluster_map()
    clusters = cluster_map.get("clusters", {})
    if cluster not in clusters:
        return None
    moves = clusters[cluster].get("moves", [])
    if not moves:
        return None
    # 加权采样
    weights = [m["weight"] for m in moves]
    total = sum(weights)
    if total <= 0:
        return None
    # 归一化
    probs = [w / total for w in weights]
    idx = rng.choice(len(moves), p=probs)
    return moves[idx]["move"]


def should_render_event(cluster: str, rng: np.random.Generator,
                        cluster_map: Dict = None) -> bool:
    """按渲染概率决定是否生成事件动作。"""
    if cluster_map is None:
        cluster_map = load_cluster_map()
    render_prob = cluster_map.get("render_prob", {})
    prob = render_prob.get(cluster, 0.6)  # 默认 60%
    return rng.random() < prob


def inject_idle_events(silence_gaps: List[Tuple[float, float]], events: List[Dict],
                       rng: np.random.Generator, cluster_map: Dict = None) -> None:
    """长沉默待机注入（裁定 §2.1）：分层池，呼吸底座+稀疏点缀+困意升级。
    silence_gaps: 沉默间隙列表 [(start, end), ...]
    events: 就地修改（追加 lib: 事件）
    """
    if cluster_map is None:
        cluster_map = load_cluster_map()
    idle_pool = cluster_map.get("_idle_pool", {})
    pool_15_30 = idle_pool.get("15_30s", [])
    pool_30_60 = idle_pool.get("30_60s", [])
    pool_over_60 = idle_pool.get("over_60s", [])
    boredom_upgrade = idle_pool.get("boredom_upgrade", "boredom2")

    # 记录已注入时间（避免冷却期内重复）
    last_inject_t = -IDLE_COOLDOWN_S * 2

    for gap_start, gap_end in silence_gaps:
        gap_dur = gap_end - gap_start
        if gap_dur < 15.0:  # 短间隙不注入
            continue

        # 选池
        if gap_dur < 30.0:
            pool = pool_15_30
        elif gap_dur < 60.0:
            pool = pool_30_60
        else:
            pool = pool_over_60

        if not pool:
            continue

        # 冷却检查：注入时间必须距离上次注入 ≥20s
        inject_t = gap_start + rng.uniform(0.5, min(5.0, gap_dur - 0.5))
        if inject_t - last_inject_t < IDLE_COOLDOWN_S:
            continue

        # 采样动作
        weights = [m.get("weight", 1.0) for m in pool]
        total = sum(weights)
        if total <= 0:
            continue
        probs = [w / total for w in weights]
        idx = rng.choice(len(pool), p=probs)
        move_name = pool[idx].get("move")

        if move_name:
            events.append({"label": f"lib:{move_name}", "t": inject_t, "type": "idle_injected"})
            last_inject_t = inject_t

            # 困意升级：同间隙内再注入时升级 boredom2
            if gap_dur >= 60.0 and rng.random() < 0.5:
                upgrade_t = inject_t + rng.uniform(10.0, gap_dur - inject_t - gap_start)
                if upgrade_t < gap_end - 5.0:
                    events.append({"label": f"lib:{boredom_upgrade}", "t": upgrade_t,
                                   "type": "idle_injected"})


def detect_silence_gaps(response_times: List[float], total_duration: float,
                        speech_dur_s: List[float]) -> List[Tuple[float, float]]:
    """检测沉默间隙（无 response 的区间）。返回 [(start, end), ...]"""
    gaps = []
    if not response_times:
        gaps.append((0.0, total_duration))
        return gaps

    # 开头间隙
    if response_times[0] > 5.0:
        gaps.append((0.0, response_times[0]))

    # 中间间隙
    for i in range(len(response_times) - 1):
        gap_start = response_times[i] + (speech_dur_s[i] if i < len(speech_dur_s) else 2.0)
        gap_end = response_times[i + 1]
        if gap_end - gap_start >= 5.0:  # 只记录 ≥5s 的间隙
            gaps.append((gap_start, gap_end))

    # 结尾间隙
    last_end = response_times[-1] + (speech_dur_s[-1] if speech_dur_s else 2.0)
    if total_duration - last_end >= 5.0:
        gaps.append((last_end, total_duration))

    return gaps


@lru_cache(maxsize=1)
def lib_emotion_labels() -> frozenset:
    """官方库中 kind=='emotions' 的动作名集合(85 个;C0 修正案的库动作词表)。
    舞蹈类(kind=='dances')不属于对话词表,标注若误给则在 convert_sample 里降级 none。
    找不到 index.json(极端环境)时返回空集,退化为纯模板行为。"""
    idx_path = Path(__file__).parent / "motion_library" / "index.json"
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return frozenset()
    return frozenset(m["name"] for m in idx.get("moves", []) if m.get("kind") == "emotions")


def classify_action(label: str, rng: np.random.Generator,
                    cluster_map: Dict = None) -> Tuple[str, Optional[str]]:
    """把标注 action 归类。返回 (action_label, compose_label)：
      - 模板类 → (label, label) 事件走原模板路径
      - 簇名 → (簇名, "lib:<具体动作>") 事件走库路径，按权重采样
      - 库情绪类（兼容旧 fixture）→ (label, f"lib:{label}") 事件走 augment 增广
      - 其余 → ("none", None) 不产事件
    """
    if label in TEMPLATE_LABELS:
        return label, label
    if label in CLUSTER_LABELS:
        # 簇名 → 采样具体动作
        move_name = sample_move_from_cluster(label, rng, cluster_map)
        if move_name:
            return label, f"lib:{move_name}"
        return label, None  # 簇无配置，不产事件
    if label in lib_emotion_labels():
        # 兼容旧 fixture（直接动作名）
        return label, f"lib:{label}"
    return "none", None


def normalize_response_format(sample: Dict) -> List:
    """标准化 response 格式为二维列表（兼容 chat/narration/旧格式）。
    返回 [[{content, time, action}, ...], ...]
    """
    responses = sample.get("response", [])
    if not responses:
        return []

    # 检测格式
    if isinstance(responses[0], dict):
        # narration 格式：一维列表 [{content, time, action}, ...]
        # 包装成二维列表（仅第一组）
        return [responses]
    elif isinstance(responses[0], list) and len(responses[0]) > 0 and isinstance(responses[0][0], dict):
        # 标准/旧格式：二维列表 [[{...}, ...], ...]
        return responses
    else:
        # 未知格式，尝试包装
        return [responses] if responses else []


def est_dur(text: str) -> float:
    """时长代理(设计文档 §4.1/契约 C0):中文 5.5 字/秒、英文 14 字符/秒。
    按 ASCII 占比在两个语速间线性插值,统一给两侧用。"""
    if not text:
        return 0.3
    n = len(text)
    ascii_ratio = sum(c.isascii() for c in text) / n
    rate = 14.0 * ascii_ratio + 5.5 * (1 - ascii_ratio)   # 字符/秒
    return round(max(0.3, n / rate), 2)


def convert_sample(sample: Dict, rng: np.random.Generator, limits: Dict,
                   cluster_map: Dict = None,
                   explain_downsample_seed: int = None) -> Tuple[Dict, List[Dict], np.ndarray]:
    """把一个标注样本转成 (meta, timeline, actions)。不写盘。

    新增参数:
      cluster_map: 簇→动作映射配置
      explain_downsample_seed: explain 下采样种子（用于 meta 记录）
    """
    # 加载 cluster_map
    if cluster_map is None:
        cluster_map = load_cluster_map()

    questions = sample.get("question", [])
    responses = normalize_response_format(sample)  # 标准化格式
    video_dur = float(sample.get("video_duration_s", 0.0))

    timeline: List[Dict] = []
    events: List[Dict] = []          # 给 compose 的模板事件
    speech_spans: List[Tuple[float, float]] = []
    response_times: List[float] = []
    response_speech_durs: List[float] = []
    last_event_t = 0.0

    # 下采样状态
    explain_count = 0
    total_non_explain = 0

    # 1) 用户提问 → user_text + speech_span
    for q in questions:
        t = float(q["time"])
        timeline.append({"t": round(t, 2), "type": "user_text", "text": q["content"]})
        speech_spans.append((t, t + est_dur(q["content"])))
        last_event_t = max(last_event_t, t)

    # 2) 回复 → response 时间线事件 + compose 事件(带 action;库动作走 lib: 路径)
    for group in responses:
        for r in group:
            t = float(r["time"])
            action_label_raw = r.get("action", "none")

            # 簇名处理
            if action_label_raw in CLUSTER_LABELS:
                cluster = action_label_raw
                # 渲染概率检查
                if not should_render_event(cluster, rng, cluster_map):
                    # 未渲染：只记录时间线，不产 compose 事件
                    timeline.append({
                        "t": round(t, 2), "type": "response", "text": r["content"],
                        "est_speech_dur_s": est_dur(r["content"]), "action_label": cluster,
                        "note": "not_rendered"
                    })
                    response_times.append(t)
                    response_speech_durs.append(est_dur(r["content"]))
                    last_event_t = max(last_event_t, t)
                    continue

                # explain 下采样检查
                if cluster == "explain":
                    explain_count += 1
                    # 预估 explain 占比（简化：按当前计数预估）
                    total_so_far = explain_count + total_non_explain
                    if total_so_far > 0 and explain_count / total_so_far > EXPLAIN_DOWNSAMPLE_MAX:
                        # 超上限，跳过此事件
                        timeline.append({
                            "t": round(t, 2), "type": "response", "text": r["content"],
                            "est_speech_dur_s": est_dur(r["content"]), "action_label": cluster,
                            "note": "downsampled"
                        })
                        response_times.append(t)
                        response_speech_durs.append(est_dur(r["content"]))
                        last_event_t = max(last_event_t, t)
                        continue
                else:
                    total_non_explain += 1

                action_label, compose_label = classify_action(cluster, rng, cluster_map)
            else:
                # 兼容旧格式（直接动作名或模板类）
                action_label, compose_label = classify_action(action_label_raw, rng, cluster_map)

            timeline.append({
                "t": round(t, 2), "type": "response", "text": r["content"],
                "est_speech_dur_s": est_dur(r["content"]), "action_label": action_label,
            })
            if compose_label is not None:
                events.append({"label": compose_label, "t": t})
            response_times.append(t)
            response_speech_durs.append(est_dur(r["content"]))
            last_event_t = max(last_event_t, t)

    # 3) 长沉默待机注入
    silence_gaps = detect_silence_gaps(response_times, last_event_t + TAIL_PAD_S, response_speech_durs)
    inject_idle_events(silence_gaps, events, rng, cluster_map)

    # 4) 时长:min(最后事件+留白, 视频长, 上限);无视频长则退回 最后事件+留白
    cand = last_event_t + TAIL_PAD_S
    duration_s = min(cand, video_dur) if video_dur > 0 else cand
    duration_s = min(duration_s, DUR_CAP_S)
    duration_s = max(duration_s, 1.0)
    duration_s = round(duration_s, 3)

    timeline.sort(key=lambda e: e["t"])

    # 5) 轨迹
    actions = compose(duration_s, events, speech_spans, rng, limits)

    meta = {
        "episode_id": f"ep_{sample.get('video_id', sample.get('video_name', 'unknown'))}",
        "source": "synthetic_v1_from_annotation",
        "duration_s": duration_s,
        "fps_action": FPS,
        "seed": int(rng.integers(0, 2**31 - 1)),
        "template_events": events,
        "origin_video": f"{sample.get('video_id', sample.get('video_name', 'unknown'))}.mp4",
        "schema_version": 1,
        "cluster_map_version": cluster_map.get("_version", "unknown"),
        "explain_downsample_seed": explain_downsample_seed,
    }
    return meta, timeline, actions


# ---------------- 批量驱动(Step 3) ----------------

def _process_one(args) -> Tuple[str, bool, str]:
    """单样本:转换+写盘+校验。返回 (episode_id, ok, reason)。"""
    sample, out_dir, quarantine_dir, seed, cluster_map, explain_seed = args
    ep_id = f"ep_{sample.get('video_id', sample.get('video_name', 'unknown'))}"
    ep_path = Path(out_dir) / ep_id
    # 增量:已存在且校验通过则跳过（捕获异常以防损坏文件）
    if ep_path.exists():
        try:
            if validate_episode(ep_path):
                return ep_id, True, "skip(已存在)"
        except Exception:
            # 损坏文件，删除后重新生成
            import shutil
            shutil.rmtree(ep_path)
    try:
        rng = np.random.default_rng(seed)
        meta, timeline, actions = convert_sample(
            sample, rng, _LIMITS_CACHE[out_dir], cluster_map, explain_seed)
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
                 n_max: int = None, n_workers: int = None,
                 cluster_map: Dict = None) -> Dict:
    """批量:扫标注文件 → 逐样本转换/写盘/校验。返回统计 dict。

    新增参数:
      cluster_map: 簇→动作映射配置（None 则自动加载）
    """
    # 加载 cluster_map
    if cluster_map is None:
        cluster_map = load_cluster_map()

    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) else data
    if n_max:
        samples = samples[:n_max]

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    quarantine = str(out.parent / (out.name + "_quarantine"))
    _LIMITS_CACHE[out_dir] = limits

    # explain 下采样种子（固定以保证可复现）
    import hashlib
    explain_seed = int(hashlib.sha256(annotation_path.encode()).hexdigest()[:8], 16) % (2**31)

    tasks = [(s, out_dir, quarantine, i, cluster_map, explain_seed)
              for i, s in enumerate(samples)]
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
    # C2 起限幅统一改用 limits_real(真机可行域);sim/limits.json 是仿真虚高值,勿用于生成
    limits = json.loads((root / "sim" / "limits_real.json").read_text(encoding="utf-8"))
    stats = run_pipeline(args.annotation, args.out, limits, args.n_max, args.workers)
    print(f"完成:{stats['ok']}/{stats['total']} 通过,{stats['fail']} 失败")
    for eid, reason in stats["fail_detail"]:
        print(f"  ✗ {eid}: {reason}")


if __name__ == "__main__":
    main()
