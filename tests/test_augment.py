# tests/test_augment.py —— C2 官方库增广接入(TDD)
import json
import pathlib

import numpy as np

from data_gen.augment import load_lib_move, augment_lib_move
from data_gen.templates import FPS, compose

ROOT = pathlib.Path(__file__).parent.parent
LIMITS_REAL = json.loads((ROOT / "sim" / "limits_real.json").read_text(encoding="utf-8"))
LO = np.array(LIMITS_REAL["safe_min"])
HI = np.array(LIMITS_REAL["safe_max"])

MOVES = ["amazed1", "curious1", "calming1", "cheerful1", "attentive1"]


def _corr(a, b):
    """把 a 重采样到 len(b) 后,在 9 维展平上算 Pearson 相关。"""
    src = np.linspace(0, 1, len(a))
    dst = np.linspace(0, 1, len(b))
    ar = np.stack([np.interp(dst, src, a[:, j]) for j in range(9)], axis=1)
    return float(np.corrcoef(ar.ravel(), b.ravel())[0, 1])


def test_load_lib_move_shape_and_native_fps():
    for name in MOVES:
        m = load_lib_move(name)
        assert m.ndim == 2 and m.shape[1] == 9
        assert m.dtype == np.float32
        assert len(m) > 2


def test_augment_shape_and_time_scale():
    rng = np.random.default_rng(0)
    orig = load_lib_move("amazed1")
    T = len(orig)
    for _ in range(20):
        aug = augment_lib_move(orig, rng)
        assert aug.ndim == 2 and aug.shape[1] == 9 and aug.dtype == np.float32
        # T' 落在 time_scale=(0.85,1.15) 对应区间内
        assert int(round(T * 0.85)) - 1 <= len(aug) <= int(round(T * 1.15)) + 1


def test_augment_boundaries_near_zero():
    rng = np.random.default_rng(1)
    for name in MOVES:
        orig = load_lib_move(name)
        for _ in range(5):
            aug = augment_lib_move(orig, rng)
            assert np.abs(aug[0]).max() < 0.01
            assert np.abs(aug[-1]).max() < 0.01


def test_augment_within_limits_real():
    rng = np.random.default_rng(2)
    for name in MOVES:
        orig = load_lib_move(name)
        for _ in range(5):
            aug = augment_lib_move(orig, rng)
            assert (aug >= LO - 1e-6).all() and (aug <= HI + 1e-6).all()


def test_augment_correlated_but_different():
    rng = np.random.default_rng(3)
    for name in MOVES:
        orig = load_lib_move(name)
        for _ in range(10):
            aug = augment_lib_move(orig, rng)
            c = _corr(aug, orig)
            assert 0.5 < c < 0.999, f"{name}: corr={c}"


def test_augment_does_not_mutate_input():
    rng = np.random.default_rng(4)
    orig = load_lib_move("curious1")
    snapshot = orig.copy()
    augment_lib_move(orig, rng)
    assert np.array_equal(orig, snapshot)


def test_compose_dispatches_lib_events():
    """compose 对 lib: 事件走库路径:轨迹在 limits_real 内、连续、且库动作确有注入。"""
    rng = np.random.default_rng(7)
    events = [{"label": "lib:curious1", "t": 3.0}, {"label": "nod", "t": 10.0}]
    traj = compose(15.0, events, speech_spans=[(1.0, 3.0)], rng=rng, limits=LIMITS_REAL)
    assert traj.shape == (int(round(15.0 * FPS)), 9) and traj.dtype == np.float32
    # 限幅
    assert (traj >= LO - 1e-6).all() and (traj <= HI + 1e-6).all()
    # 连续:逐步位移不超过 max_vel/FPS(+小裕量)
    mv = np.array(LIMITS_REAL["max_vel"]) / FPS
    step = np.abs(np.diff(traj, axis=0))
    assert (step <= mv + 1e-5).all()
    # 库动作确有注入:curious1 插入区间(约 3s 起)的运动幅度显著高于纯 idle
    seg = traj[int(3.5 * FPS):int(6.0 * FPS)]
    assert np.abs(seg).max() > 0.05
