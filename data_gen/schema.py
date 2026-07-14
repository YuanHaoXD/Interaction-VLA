# data_gen/schema.py —— episode 读写与校验(契约 C0 的代码化)
import json
import numpy as np
from pathlib import Path

REQUIRED_META = ["episode_id","source","duration_s","fps_action","seed",
                 "template_events","origin_video","schema_version"]
EVENT_TYPES = {"user_text","response","silence","delegate"}

def write_episode(path, meta, timeline, actions):
    p = Path(path); p.mkdir(parents=True, exist_ok=True)
    (p/"meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (p/"timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(p/"actions.npy", actions.astype(np.float32))

def validate_episode(path):
    p = Path(path); errs = []
    for f in ("meta.json","timeline.json","actions.npy"):
        if not (p/f).exists(): errs.append(f"缺文件 {f}")
    if errs: return errs
    meta = json.loads((p/"meta.json").read_text(encoding="utf-8"))
    for k in REQUIRED_META:
        if k not in meta: errs.append(f"meta 缺键 {k}")
    a = np.load(p/"actions.npy")
    if a.ndim != 2 or a.shape[1] != 9: errs.append(f"actions 形状 {a.shape} ≠ [T,9]")
    if a.dtype != np.float32: errs.append(f"actions dtype {a.dtype} ≠ float32")
    if np.isnan(a).any(): errs.append("actions 含 NaN")
    if "duration_s" in meta and "fps_action" in meta:
        T = int(round(meta["duration_s"] * meta["fps_action"]))
        if len(a) != T: errs.append(f"帧数 T={len(a)} 与 duration*fps={T} 不符")
    tl = json.loads((p/"timeline.json").read_text(encoding="utf-8"))
    last = -1.0
    for ev in tl:
        if ev.get("type") not in EVENT_TYPES: errs.append(f"非法事件类型 {ev.get('type')}")
        if ev.get("t", 0) < last: errs.append("timeline 未按时间升序")
        last = ev.get("t", last)
        if ev.get("type") == "response" and "est_speech_dur_s" not in ev:
            errs.append(f"response@{ev.get('t')} 缺 est_speech_dur_s")
    return errs
