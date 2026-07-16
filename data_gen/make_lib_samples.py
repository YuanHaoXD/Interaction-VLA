# data_gen/make_lib_samples.py —— 生成 20 段"库动作+模板"混合样例(C2 肉眼验收用)
"""
每段 episode 的 events 混用官方库动作(lib:<名>)与手写模板(nod/tilt_head 等),
用于仿真回放肉眼验收:重点看库动作与 idle 叠加处是否无跳变。

契约 C0 约定:
- timeline 的 response 事件 action_label 存"裸动作名"(如 "curious1");
- compose 的 events(= meta.template_events)存带前缀 "lib:curious1"。
写到 samples_lib/,与 samples/(纯模板 100 段)分开。

用法:python -m data_gen.make_lib_samples
"""
import json
import pathlib

import numpy as np

from data_gen.templates import compose, FPS
from data_gen.schema import write_episode, validate_episode

ROOT = pathlib.Path(__file__).parent.parent
LIMITS = json.loads((ROOT / "sim" / "limits_real.json").read_text(encoding="utf-8"))

# 情绪动作精选(默认排除舞蹈,符合设计文档 §4.1);覆盖聆听/惊讶/安抚/好奇等多种神态
LIB_MOVES = ["curious1", "amazed1", "calming1", "cheerful1", "attentive1",
             "confused1", "come1", "boredom1", "anxiety1", "contempt1",
             "attentive2", "cheerful1", "curious1", "calming1", "amazed1"]
TEMPLATES = ["nod", "shake_head", "tilt_head", "wiggle_antennas"]
UTTER = ["好的,没问题!", "这个我不太同意。", "让我想想…", "哈哈,有意思!", "你说的是那个红色的吗?", "我在听,你继续。"]


def est_dur(text):
    return round(len(text) / 5.5, 2)


def main(n=20):
    rng = np.random.default_rng(20260715)
    ok = 0
    outdir = ROOT / "samples_lib"
    for i in range(n):
        # 事件:1 个库动作 + 1 个模板,时间错开;时长留足(库动作最长约 8s)
        lib_name = LIB_MOVES[i % len(LIB_MOVES)]
        tmpl = TEMPLATES[rng.integers(0, len(TEMPLATES))]
        t_lib = float(round(rng.uniform(3.0, 6.0), 1))
        t_tmpl = float(round(t_lib + rng.uniform(5.0, 8.0), 1))
        dur = round(float(t_tmpl + rng.uniform(5.0, 8.0)), 3)

        timeline, events, spans = [], [], []
        for t, label, txt in [(t_lib, lib_name, UTTER[rng.integers(0, len(UTTER))]),
                              (t_tmpl, tmpl, UTTER[rng.integers(0, len(UTTER))])]:
            q_t = round(max(0.5, t - rng.uniform(1.5, 3.0)), 1)
            timeline.append({"t": q_t, "type": "user_text", "text": "(用户提问)"})
            spans.append((q_t, q_t + 1.5))
            timeline.append({"t": t, "type": "response", "text": txt,
                             "est_speech_dur_s": est_dur(txt), "action_label": label})
        events.append({"label": f"lib:{lib_name}", "t": t_lib})
        events.append({"label": tmpl, "t": t_tmpl})

        traj = compose(dur, events, spans, rng, LIMITS)
        meta = {"episode_id": f"eplib_{i:04d}", "source": "synthetic_v1_lib",
                "duration_s": dur, "fps_action": FPS, "seed": 20260715,
                "template_events": events, "origin_video": "sample_lib", "schema_version": 1}
        ep = outdir / meta["episode_id"]
        write_episode(ep, meta, sorted(timeline, key=lambda e: e["t"]), traj)
        errs = validate_episode(ep)
        assert not errs, f"{ep}: {errs}"
        ok += 1
        print(f"  {meta['episode_id']}: lib:{lib_name}@{t_lib}s + {tmpl}@{t_tmpl}s, dur={dur}s, T={len(traj)}")
    print(f"生成并校验通过 {ok}/{n} 段 → samples_lib/")


if __name__ == "__main__":
    main()
