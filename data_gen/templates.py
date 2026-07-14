# data_gen/templates.py —— 5 类标签的参数随机化连续轨迹 + idle 微动 + backchannel
import numpy as np

FPS = 30
# 维度索引: 0x 1y 2z 3roll 4pitch 5yaw 6body_yaw 7ant_r 8ant_l

def _T(duration_s): return int(round(duration_s * FPS))

def _envelope(T, attack=0.15, release=0.25):
    """起止平滑包络(smoothstep),保证轨迹从 0 起、回 0 终。"""
    t = np.linspace(0, 1, T)
    def smooth(x): return np.clip(x, 0, 1)**2 * (3 - 2*np.clip(x, 0, 1))
    return smooth(t/attack) * smooth((1-t)/release)

def _sine_burst(T, dim, amp, freq, cycles, decay):
    traj = np.zeros((T, 9), dtype=np.float32)
    t = np.arange(T) / FPS
    dur = min(cycles / freq, T / FPS)
    n = int(dur * FPS)
    if n < 2: return traj
    osc = amp * np.sin(2*np.pi*freq*t[:n]) * np.exp(-decay*t[:n]) * _envelope(n)
    traj[:n, dim] = osc
    return traj

def render_template(label, duration_s, rng, params=None):
    T = _T(duration_s); p = dict(params or {})
    def pick(k, lo, hi):
        if k not in p: p[k] = float(rng.uniform(lo, hi))
        return p[k]
    if label == "nod":
        traj = _sine_burst(T, 4, pick("amp",0.09,0.26), pick("freq",0.8,2.0),
                           pick("cycles",1,3), pick("decay",0.2,0.8))
    elif label == "shake_head":
        traj = _sine_burst(T, 5, pick("amp",0.09,0.35), pick("freq",0.7,1.6),
                           pick("cycles",2,4), pick("decay",0.1,0.5))
    elif label == "tilt_head":
        amp = pick("amp",0.13,0.30) * (1 if rng.random() < 0.5 else -1)
        hold = pick("hold_s",0.5,1.5)
        T_r = int(0.4*FPS); T_h = min(int(hold*FPS), T - 2*T_r)
        traj = np.zeros((T,9), dtype=np.float32)
        if T_h > 0:
            ramp = amp * _envelope(2*T_r, 0.5, 0.5)[:T_r] / max(_envelope(2*T_r,0.5,0.5)[:T_r].max(),1e-9)
            seg = np.concatenate([ramp, np.full(T_h, amp), ramp[::-1]])
            traj[:len(seg), 3] = seg[:T]
    elif label == "wiggle_antennas":
        a = pick("amp",0.3,0.8); f = pick("freq",2.0,4.0); c = pick("cycles",2,6)
        traj = _sine_burst(T, 7, a, f, c, 0.3) + _sine_burst(T, 8, -a, f, c, 0.3)
    elif label == "none":
        traj = np.zeros((T,9), dtype=np.float32)
    else:
        raise ValueError(f"未知标签: {label}")
    return traj.astype(np.float32), p

def idle_motion(duration_s, rng):
    """低幅多正弦漂移 + 类呼吸,贯穿全程的'活着感'。"""
    T = _T(duration_s); t = np.arange(T)/FPS
    traj = np.zeros((T,9), dtype=np.float32)
    traj[:,2] = 0.003 * np.sin(2*np.pi*0.25*t + rng.uniform(0,6.28))          # z 呼吸 ±3mm
    for dim, amp in [(3,0.015),(4,0.015),(5,0.02)]:                            # rpy 慢漂
        for f in (0.07, 0.13, 0.23):
            traj[:,dim] += amp/3 * np.sin(2*np.pi*f*t + rng.uniform(0,6.28))
    env = _envelope(T, attack=0.05, release=0.05)
    return (traj * env[:,None]).astype(np.float32)

def backchannel_nods(duration_s, speech_spans, rng):
    """用户说话区间内,每 2~5 秒随机插入一次小幅点头。"""
    T = _T(duration_s)
    traj = np.zeros((T,9), dtype=np.float32)
    for (s, e) in speech_spans:
        t = s + rng.uniform(0.5, 2.0)
        while t < e - 1.0:
            nod, _ = render_template("nod", 1.5, rng, {"amp": rng.uniform(0.05,0.10)})
            i = int(t*FPS)
            n = min(len(nod), T - i)
            traj[i:i+n] += nod[:n]
            t += rng.uniform(2.0, 5.0)
    return traj

def compose(duration_s, events, speech_spans, rng, limits):
    T = _T(duration_s)
    traj = idle_motion(duration_s, rng) + backchannel_nods(duration_s, speech_spans or [], rng)
    for ev in events:
        seg, _ = render_template(ev["label"], min(4.0, duration_s - ev["t"]), rng, ev.get("params"))
        jitter = rng.uniform(-0.2, 0.5)                                        # 时间锚定抖动
        i = max(0, int((ev["t"] + jitter) * FPS))
        n = min(len(seg), T - i)
        if n > 0: traj[i:i+n] += seg[:n]
    lo = np.array(limits["safe_min"], dtype=np.float32)
    hi = np.array(limits["safe_max"], dtype=np.float32)
    traj = np.clip(traj, lo, hi)
    mv = np.array(limits["max_vel"], dtype=np.float32) / FPS                   # 每步最大位移
    for k in range(1, len(traj)):                                              # 逐步限速=保证连续
        traj[k] = np.clip(traj[k], traj[k-1]-mv, traj[k-1]+mv)
    return traj.astype(np.float32)
