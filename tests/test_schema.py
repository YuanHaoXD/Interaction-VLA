# tests/test_schema.py
import numpy as np
from data_gen.schema import write_episode, validate_episode

META = {"episode_id":"ep_test0001","source":"synthetic_v1","duration_s":2.0,
        "fps_action":30,"seed":1,"template_events":[],"origin_video":"x.mp4","schema_version":1}

def test_roundtrip_valid(tmp_path):
    ep = tmp_path/"ep_test0001"
    write_episode(ep, META, [{"t":1.0,"type":"response","text":"好的",
                              "est_speech_dur_s":0.4,"action_label":"nod"}],
                  np.zeros((60,9), dtype=np.float32))
    assert validate_episode(ep) == []

def test_detects_bad_shape(tmp_path):
    ep = tmp_path/"ep_test0002"
    write_episode(ep, {**META,"episode_id":"ep_test0002"}, [], np.zeros((59,9), dtype=np.float32))
    errs = validate_episode(ep)
    assert any("T" in e or "帧数" in e for e in errs)

def test_detects_nan(tmp_path):
    ep = tmp_path/"ep_test0003"
    a = np.zeros((60,9), dtype=np.float32); a[5,3] = np.nan
    write_episode(ep, {**META,"episode_id":"ep_test0003"}, [], a)
    assert any("NaN" in e for e in validate_episode(ep))
