# tests/test_templates.py
import numpy as np, json, pathlib
from data_gen.templates import render_template, idle_motion, backchannel_nods, compose

RNG = np.random.default_rng(0)
LIMITS = json.loads((pathlib.Path(__file__).parent.parent/"sim"/"limits.json").read_text(encoding="utf-8"))

def test_render_shapes_and_labels():
    for label in ["nod","shake_head","tilt_head","wiggle_antennas","none"]:
        traj, params = render_template(label, 3.0, RNG)
        assert traj.shape == (90, 9) and traj.dtype == np.float32

def test_nod_moves_pitch_only_mainly():
    traj, _ = render_template("nod", 3.0, RNG)
    assert np.abs(traj[:,4]).max() > 0.03          # pitch 显著(阈值留裕量:高频+强衰减的随机组合下首峰可能较小)
    assert np.abs(traj[:,5]).max() < 1e-6          # yaw 不动

def test_trajectory_starts_and_ends_at_zero():
    for label in ["nod","shake_head","tilt_head","wiggle_antennas"]:
        traj, _ = render_template(label, 4.0, RNG)
        assert np.abs(traj[0]).max() < 1e-3 and np.abs(traj[-1]).max() < 0.01

def test_compose_respects_limits_and_continuity():
    events = [{"label":"nod","t":2.0},{"label":"wiggle_antennas","t":5.0}]
    traj = compose(10.0, events, speech_spans=[(1.0,4.0)], rng=RNG, limits=LIMITS)
    assert traj.shape == (300, 9)
    lo, hi = np.array(LIMITS["safe_min"]), np.array(LIMITS["safe_max"])
    assert (traj >= lo - 1e-6).all() and (traj <= hi + 1e-6).all()
    vel = np.abs(np.diff(traj, axis=0)) * 30
    assert (vel <= np.array(LIMITS["max_vel"]) + 1e-6).all()   # 速度限内=连续
