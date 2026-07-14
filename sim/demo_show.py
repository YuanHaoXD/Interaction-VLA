# sim/demo_show.py —— 动作展示秀:依次演示各类动作,给人看
#
# 这是一个"给人演示"用的脚本,不是训练/数据管线的一部分。
# 它手写各种动作(点头/摇头/歪头/摆天线/转身/看四方/呼吸),
# 以 30Hz 连续推给仿真机器人 —— 和真实数据回放走的是同一套执行层。
#
# 用法(先起好 sim daemon,见 human_reading/02 文档):
#   $env:PYTHONUTF8=1
#   E:\...\reachy-mini-demo\.venv\Scripts\python.exe -u sim\demo_show.py          # 跑全套
#   ...\python.exe -u sim\demo_show.py nod                                        # 只跑点头
#   ...\python.exe -u sim\demo_show.py loop                                       # 全套循环(演示时用,Ctrl 关窗停)
#
# 想改成自己的动作:照着下面任一动作函数改幅度/频率即可,它们都返回 [T,9] 数组。
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
import sys, time
import numpy as np
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini

FPS = 30
DT = 1.0 / FPS
# 9 维动作顺序(和 actions.npy 完全一致):
#   索引  0  1  2   3     4      5     6         7        8
#   含义  x  y  z  roll  pitch  yaw  body_yaw  ant_右   ant_左   (米 / 弧度)

def _zeros(dur_s):
    """返回一段"全回中"的 [T,9](机器人保持中立位)。"""
    return np.zeros((int(dur_s * FPS), 9), dtype=np.float32)

def _smoothstep(n):
    """0→1 的平滑渐变(首尾速度为 0,不突跳)。"""
    x = np.linspace(0, 1, n)
    return x * x * (3 - 2 * x)

def _osc(dim, amp, freq, cycles):
    """一个维度上的衰减正弦振荡(点头/摇头/摆天线用)。返回 [T,9]。"""
    n = int(cycles / freq * FPS)
    t = np.arange(n) / FPS
    env = np.sin(np.pi * t / (cycles / freq))          # 0→1→0 钟形包络,保证平滑起止
    a = np.zeros((n, 9), dtype=np.float32)
    a[:, dim] = amp * np.sin(2 * np.pi * freq * t) * env
    return a

def _hold(dim, target, ramp_s=0.6, hold_s=0.8):
    """某维度平滑推到 target、保持、再平滑回 0(歪头/看方向/转身用)。返回 [T,9]。"""
    nr = int(ramp_s * FPS); nh = int(hold_s * FPS)
    up = target * _smoothstep(nr)
    seq = np.concatenate([up, np.full(nh, target, np.float32), up[::-1]])
    a = np.zeros((len(seq), 9), dtype=np.float32)
    a[:, dim] = seq
    return a

def _breathe(dur_s=6.0):
    """低幅多正弦漂移 + 呼吸起伏,展示 idle '活着感'。返回 [T,9]。"""
    n = int(dur_s * FPS); t = np.arange(n) / FPS
    a = np.zeros((n, 9), dtype=np.float32)
    a[:, 2] = 0.004 * np.sin(2 * np.pi * 0.25 * t)                    # z 呼吸
    for dim, amp, f in [(3, 0.02, 0.13), (4, 0.02, 0.11), (5, 0.025, 0.09)]:
        a[:, dim] = amp * np.sin(2 * np.pi * f * t)                   # rpy 慢漂
    return a

# —— 各动作:名字 → [T,9] 轨迹(参数都在 B1 实测安全范围内) ——
def build_shows():
    return [
        ("点头 nod        (pitch 正弦)",      _osc(4, 0.20, 1.2, 3)),
        ("摇头 shake      (yaw 正弦)",        _osc(5, 0.30, 1.1, 3)),
        ("歪头 tilt       (roll 保持)",       _hold(3, 0.25, 0.6, 1.0)),
        ("摆天线 wiggle    (双天线反相)",       _osc(7, 0.5, 3.0, 5) + _osc(8, -0.5, 3.0, 5)),
        ("转身 body_yaw   (身体左右转)",        np.concatenate([_hold(6, 0.4, 0.8, 0.6), _hold(6, -0.4, 0.8, 0.6)])),
        ("看向 look       (右→左→上→下)",      np.concatenate([
            _hold(5, 0.4, 0.5, 0.5), _hold(5, -0.4, 0.5, 0.5),
            _hold(4, -0.3, 0.5, 0.5), _hold(4, 0.3, 0.5, 0.5)])),
        ("呼吸 breathe    (idle 活着感)",      _breathe(6.0)),
    ]

def play(mini, neutral, traj):
    """把 [T,9] 轨迹以 30Hz 推给机器人(和 sim/replay_episode.py 同款执行层)。"""
    t0 = time.perf_counter()
    for k, a in enumerate(traj):
        d = np.eye(4)
        d[:3, :3] = R.from_euler("xyz", a[3:6]).as_matrix()
        d[:3, 3] = a[:3]
        mini.set_target(head=neutral @ d, antennas=[float(a[7]), float(a[8])],
                        body_yaw=float(a[6]))
        nxt = t0 + (k + 1) * DT
        while time.perf_counter() < nxt: pass

def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    print(">>> 连接仿真(首次可能等 10~50 秒)...", flush=True)
    mini = ReachyMini()
    neutral = mini.get_current_head_pose()
    print(">>> 已连接。开始展示。\n", flush=True)
    shows = build_shows()
    loop = (arg == "loop")
    while True:
        for name, traj in shows:
            key = name.split()[0]                                    # 中文名首词,如"点头"
            eng = name.split()[1] if len(name.split()) > 1 else ""   # 英文名,如"nod"
            if arg not in ("all", "loop") and arg not in (key, eng.lower()):
                continue
            print(f"▶ {name}   ({len(traj)}帧 / {len(traj)/FPS:.1f}s)", flush=True)
            play(mini, neutral, traj)
            play(mini, neutral, _zeros(0.8))                         # 动作间回中停顿
        if not loop:
            break
        print("—— 循环一轮结束,重新开始(关闭窗口或 Ctrl+C 停止)——\n", flush=True)
    print("\n>>> 展示完毕。", flush=True)

if __name__ == "__main__":
    main()
