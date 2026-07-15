# tests/test_pipeline.py —— A5 流水线测试
import json
import pathlib

import numpy as np

from data_gen.pipeline import convert_sample, run_pipeline, est_dur
from data_gen.schema import validate_episode
from data_gen.stats import aggregate

ROOT = pathlib.Path(__file__).parent.parent
LIMITS = json.loads((ROOT / "sim" / "limits.json").read_text(encoding="utf-8"))
ANNO = json.loads((ROOT / "data_gen" / "example_annotation.json").read_text(encoding="utf-8"))
FPS = 30


def _first_sample():
    return ANNO["samples"][0]


def test_convert_sample_timeline_covers_all_events():
    s = _first_sample()
    rng = np.random.default_rng(0)
    meta, timeline, actions = convert_sample(s, rng, LIMITS)

    # 全部 question 变成 user_text
    n_user = sum(1 for e in timeline if e["type"] == "user_text")
    assert n_user == len(s["question"])

    # 全部 response 变成 response 事件,且含 est_speech_dur_s 与 action_label
    n_resp_src = sum(len(g) for g in s["response"])
    resp_events = [e for e in timeline if e["type"] == "response"]
    assert len(resp_events) == n_resp_src
    for e in resp_events:
        assert "est_speech_dur_s" in e and "action_label" in e


def test_convert_sample_action_shape_and_validate(tmp_path):
    s = _first_sample()
    rng = np.random.default_rng(1)
    meta, timeline, actions = convert_sample(s, rng, LIMITS)

    # actions 形状 = duration 秒数 × 30，9 维 float32
    T = int(round(meta["duration_s"] * FPS))
    assert actions.shape == (T, 9)
    assert actions.dtype == np.float32

    # 写盘后过 schema 校验
    from data_gen.schema import write_episode
    ep = tmp_path / meta["episode_id"]
    write_episode(ep, meta, timeline, actions)
    assert validate_episode(ep) == []


def test_duration_capped_and_respects_video_length():
    s = _first_sample()  # video_duration_s = 22.4
    rng = np.random.default_rng(2)
    meta, _, _ = convert_sample(s, rng, LIMITS)
    assert meta["duration_s"] <= 22.4 + 1e-6      # 不超过视频长
    assert meta["duration_s"] <= 120.0            # 上限


def test_est_dur_bilingual():
    # 中文 5.5 字/秒:11 字 ≈ 2.0s；英文 14 字符/秒:28 字符 ≈ 2.0s
    assert 1.5 < est_dur("一二三四五六七八九十一") < 2.5
    assert 1.5 < est_dur("abcdefghijklmnopqrstuvwxyz12") < 2.5


def test_invalid_action_falls_back_to_none():
    s = {"video_id": "bad", "video_duration_s": 6.0,
         "question": [{"content": "hi", "time": 1.0}],
         "response": [[{"content": "yo", "time": 3.0, "action": "backflip"}]]}
    rng = np.random.default_rng(3)
    meta, timeline, _ = convert_sample(s, rng, LIMITS)
    resp = [e for e in timeline if e["type"] == "response"][0]
    assert resp["action_label"] == "none"          # 非法标签降级
    assert meta["template_events"] == []           # none 不进模板事件


def test_run_pipeline_and_stats(tmp_path):
    out = tmp_path / "ds"
    stats = run_pipeline(str(ROOT / "data_gen" / "example_annotation.json"),
                         str(out), LIMITS, n_workers=1)
    assert stats["ok"] == len(ANNO["samples"])
    assert stats["fail"] == 0

    # 增量:重跑应全部 skip 且仍全通过
    stats2 = run_pipeline(str(ROOT / "data_gen" / "example_annotation.json"),
                          str(out), LIMITS, n_workers=1)
    assert stats2["ok"] == stats["ok"] and stats2["fail"] == 0

    # stats 聚合
    agg = aggregate(str(out))
    assert agg["n_episodes"] == len(ANNO["samples"])
    assert len(agg["min"]) == 9 and len(agg["max"]) == 9
