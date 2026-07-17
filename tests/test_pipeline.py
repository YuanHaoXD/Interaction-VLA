# tests/test_pipeline.py —— A5/C4 流水线测试
import json
import pathlib

import numpy as np

from data_gen.pipeline import convert_sample, run_pipeline, est_dur, load_cluster_map
from data_gen.schema import validate_episode
from data_gen.stats import aggregate
from data_gen.templates import FPS

ROOT = pathlib.Path(__file__).parent.parent
# C2 起限幅统一改用 limits_real(真机可行域)
LIMITS = json.loads((ROOT / "sim" / "limits_real.json").read_text(encoding="utf-8"))
ANNO = json.loads((ROOT / "data_gen" / "example_annotation.json").read_text(encoding="utf-8"))
CLUSTER_MAP = load_cluster_map()


def _first_sample():
    return ANNO["samples"][0]


def test_convert_sample_timeline_covers_all_events():
    s = _first_sample()
    rng = np.random.default_rng(0)
    meta, timeline, actions = convert_sample(s, rng, LIMITS, CLUSTER_MAP)

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
    meta, timeline, actions = convert_sample(s, rng, LIMITS, CLUSTER_MAP)

    # actions 形状 = duration 秒数 × FPS(50Hz),9 维 float32
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
    meta, _, _ = convert_sample(s, rng, LIMITS, CLUSTER_MAP)
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
    meta, timeline, _ = convert_sample(s, rng, LIMITS, CLUSTER_MAP)
    resp = [e for e in timeline if e["type"] == "response"][0]
    assert resp["action_label"] == "none"          # 非法标签降级
    assert meta["template_events"] == []           # none 不进模板事件


def test_cluster_mapping_and_sampling():
    """测试簇名→动作映射（M1.5-C4 新增）"""
    rng = np.random.default_rng(42)
    from data_gen.pipeline import sample_move_from_cluster

    # affirm 簇应采样到动作名
    move = sample_move_from_cluster("affirm", rng, CLUSTER_MAP)
    assert move in ["yes1", "understanding1", "understanding2", "proud1",
                    "proud2", "proud3", "success1", "success2"]

    # unknown 簇应返回 None
    assert sample_move_from_cluster("unknown", rng, CLUSTER_MAP) is None


def test_render_probability_control():
    """测试渲染概率控制（M1.5-C4 新增）"""
    from data_gen.pipeline import should_render_event
    rng = np.random.default_rng(43)

    # explain 默认 35% 渲染概率
    explain_render = sum(1 for _ in range(1000) if should_render_event("explain", rng, CLUSTER_MAP))
    assert 0.25 < explain_render / 1000 < 0.45  # 宽容区间

    # joy 默认 85% 渲染概率
    rng = np.random.default_rng(43)
    joy_render = sum(1 for _ in range(1000) if should_render_event("joy", rng, CLUSTER_MAP))
    assert 0.75 < joy_render / 1000 < 0.95


def test_narration_1n_format():
    """测试 narration 1:N 结构（M1.5-C4 关键测试）"""
    # narration 格式：一维 response 列表
    s = {
        "video_name": "narration_test",
        "video_duration_s": 50.0,
        "question": [{"content": "这视频在讲啥?", "time": 0.0}],
        "response": [
            {"content": "画面一开始是主播。", "time": 3.0, "action": "explain"},
            {"content": "注意第三秒的标识。", "time": 8.0, "action": "attend"},
            {"content": "镜头转到车间。", "time": 15.0, "action": "explain"},
            {"content": "这是ASML大楼。", "time": 22.0, "action": "explain"},
            {"content": "光刻设备内部。", "time": 28.0, "action": "explain"}
        ]
    }
    rng = np.random.default_rng(44)
    meta, timeline, actions = convert_sample(s, rng, LIMITS, CLUSTER_MAP)

    # 应处理全部 5 个 response
    resp_events = [e for e in timeline if e["type"] == "response"]
    assert len(resp_events) == 5

    # 每个 response 应有 est_speech_dur_s 和 action_label
    for e in resp_events:
        assert "est_speech_dur_s" in e
        assert "action_label" in e
        # action_label 应为簇名（可能因下采样/渲染概率变化，但仍是簇名或 none）
        assert e["action_label"] in ["explain", "attend", "none"] or "note" in e


def test_run_pipeline_and_stats(tmp_path):
    out = tmp_path / "ds"
    stats = run_pipeline(str(ROOT / "data_gen" / "example_annotation.json"),
                         str(out), LIMITS, n_workers=1, cluster_map=CLUSTER_MAP)
    assert stats["ok"] == len(ANNO["samples"])
    assert stats["fail"] == 0

    # 增量:重跑应全部 skip 且仍全通过
    stats2 = run_pipeline(str(ROOT / "data_gen" / "example_annotation.json"),
                          str(out), LIMITS, n_workers=1, cluster_map=CLUSTER_MAP)
    assert stats2["ok"] == stats["ok"] and stats2["fail"] == 0

    # stats 聚合
    agg = aggregate(str(out))
    assert agg["n_episodes"] == len(ANNO["samples"])
    assert len(agg["min"]) == 9 and len(agg["max"]) == 9
