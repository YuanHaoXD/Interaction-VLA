# data_gen/make_samples.py —— 用模板库+schema 生成 100 段合成样例(无音频/无帧)
import json, pathlib
import numpy as np
from data_gen.templates import compose, FPS
from data_gen.schema import write_episode, validate_episode

ROOT = pathlib.Path(__file__).parent.parent
LIMITS = json.loads((ROOT/"sim"/"limits.json").read_text(encoding="utf-8"))
LABELS = ["nod","shake_head","tilt_head","wiggle_antennas","none"]
UTTER = ["好的,没问题!","这个我不太同意。","让我想想…","哈哈,有意思!","你说的是那个红色的吗?"]

def est_dur(text): return round(len(text) / 5.5, 2)   # 中文时长代理:5.5字/秒

def main(n=100):
    rng = np.random.default_rng(2026)
    ok = 0
    for i in range(n):
        dur = float(rng.uniform(15, 40))
        t_resp = sorted(rng.uniform(3, dur-4, size=rng.integers(1, 4)))
        events, timeline, spans = [], [], []
        for t in t_resp:
            t = float(round(t, 1))
            label = LABELS[rng.integers(0, len(LABELS))]
            text = UTTER[rng.integers(0, len(UTTER))]
            q_t = round(max(0.5, t - rng.uniform(1.5, 3.0)), 1)
            timeline.append({"t": q_t, "type": "user_text", "text": "(用户提问)"})
            spans.append((q_t, q_t + 1.5))
            timeline.append({"t": t, "type": "response", "text": text,
                             "est_speech_dur_s": est_dur(text), "action_label": label})
            if label != "none": events.append({"label": label, "t": t})
        # 轨迹长度须与 meta 存的 duration_s 一致(schema 按 round(dur,3)*fps 校验)。
        # 舍入放在 timeline 构建之后:t_resp/q_t 仍用原始 dur → timeline 与旧版逐字一致。
        dur_r = round(dur, 3)
        traj = compose(dur_r, events, spans, rng, LIMITS)
        meta = {"episode_id": f"ep_{i:08d}", "source": "synthetic_v1",
                "duration_s": dur_r, "fps_action": FPS, "seed": 2026,
                "template_events": events, "origin_video": "sample", "schema_version": 1}
        ep = ROOT/"samples"/meta["episode_id"]
        write_episode(ep, meta, sorted(timeline, key=lambda e: e["t"]), traj)
        errs = validate_episode(ep)
        assert not errs, f"{ep}: {errs}"
        ok += 1
    print(f"生成并校验通过 {ok}/{n} 段 → samples/")

if __name__ == "__main__":
    main()
