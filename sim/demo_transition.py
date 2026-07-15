# sim/demo_transition.py —— 演示"动作切换"的三种情况,核心看:上一个没做完下一个来了会怎样
#
# 这是本项目最核心的技术点的可视化(设计文档 §2 洞察1):
#   离散硬编码动作的病 = 上一个动作没播完,下一个来了 → 直接突跳(生硬)
#   action chunking 的解 = 重叠窗口时间加权混合(temporal ensembling) → 平滑过渡
#
# 依次演示三种:
#   A. 硬切     : 点头【完整播完】→ 直接接摇头(不回中)。看:动作衔接处有没有小顿挫
#   B. 硬跳打断 : 点头播到一半 → 【瞬间跳】到摇头起点。看:明显的突跳/抽搐(这就是离散token的病)
#   C. 混合打断 : 点头播到一半 → 摇头以【重叠加权混合】接入。看:平滑接管,无突跳(项目的解法)
#
# 用法(daemon 要开着):
#   ...\python.exe -u sim\demo_transition.py           # 跑全部三种
#   ...\python.exe -u sim\demo_transition.py A          # 只看某一种
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
import sys, time
import numpy as np
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini

FPS = 30
DT = 1.0 / FPS
# 维度: 0x 1y 2z 3roll 4pitch(点头) 5yaw(摇头) 6body_yaw 7ant_r 8ant_l

def _bell(n):
    """0→1→0 钟形包络,保证动作平滑起止。"""
    t = np.linspace(0, np.pi, n)
    return np.sin(t)

def nod(dur=2.5, amp=0.6, freq=1.1):
    """点头:pitch(维度4)正弦。返回 [T,9]。"""
    n = int(dur * FPS); t = np.arange(n) / FPS
    a = np.zeros((n, 9), np.float32)
    a[:, 4] = amp * np.sin(2*np.pi*freq*t) * _bell(n)
    return a

def shake(dur=2.5, amp=0.60, freq=1.1):
    """摇头:yaw(维度5)正弦。返回 [T,9]。"""
    n = int(dur * FPS); t = np.arange(n) / FPS
    a = np.zeros((n, 9), np.float32)
    a[:, 5] = amp * np.sin(2*np.pi*freq*t) * _bell(n)
    return a

def play(mini, neutral, traj, tag=""):
    """把 [T,9] 以 30Hz 推给机器人。"""
    if tag: print(f"    ▶ {tag}  ({len(traj)}帧/{len(traj)/FPS:.1f}s)", flush=True)
    t0 = time.perf_counter()
    for k, a in enumerate(traj):
        d = np.eye(4)
        d[:3,:3] = R.from_euler("xyz", a[3:6]).as_matrix(); d[:3,3] = a[:3]
        mini.set_target(head=neutral @ d, antennas=[float(a[7]), float(a[8])], body_yaw=float(a[6]))
        nxt = t0 + (k+1)*DT
        while time.perf_counter() < nxt: pass

# ========== 三种切换方式,各自构造一整条 [T,9] 轨迹,再一次性播放 ==========

def build_A_hardcut():
    """A. 硬切:点头完整 + 摇头完整,首尾相接(不回中)。"""
    return np.concatenate([nod(), shake()], axis=0)

def build_B_hardjump(cut_ratio=0.5):
    """B. 硬跳打断:点头只播前一半,然后【瞬间】切到摇头从头开始。
    模拟'离散token'——新动作直接覆盖旧动作,不管旧的做到哪、当前姿态在哪。"""
    n = nod(); s = shake()
    cut = int(len(n) * cut_ratio)
    return np.concatenate([n[:cut], s], axis=0)      # 点头在最高/最低点被硬生生截断→跳到摇头起点(pitch瞬间归零、yaw突起)

def build_C_blend(cut_ratio=0.5, overlap_s=0.5):
    """C. 混合打断:点头播到一半,摇头以【重叠加权混合】接入(temporal ensembling)。
    重叠区内:旧动作权重从1→0,新动作从0→1,逐帧线性加权。这就是 action chunking 的做法。"""
    n = nod(); s = shake()
    cut = int(len(n) * cut_ratio)
    ov = int(overlap_s * FPS)                        # 重叠帧数
    head = n[:cut-ov] if cut-ov > 0 else n[:0]       # 点头独占段
    # 重叠段:点头的 [cut-ov, cut) 与 摇头的 [0, ov) 加权混合
    w = np.linspace(1, 0, ov)[:, None]               # 旧动作权重 1→0
    old_seg = n[cut-ov:cut] if cut >= ov else n[:ov]
    new_seg = s[:ov]
    m = min(len(old_seg), len(new_seg), ov)
    blend = w[:m] * old_seg[:m] + (1 - w[:m]) * new_seg[:m]
    tail = s[m:]                                      # 摇头剩余段
    return np.concatenate([head, blend, tail], axis=0).astype(np.float32)

def main():
    arg = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    print(">>> 连接仿真(约 4~8 秒)...", flush=True)
    with ReachyMini(connection_mode="localhost_only", media_backend="no_media") as mini:
        neutral = mini.get_current_head_pose()
        print(">>> 已连接。\n", flush=True)

        def rest(sec=1.2):
            play(mini, neutral, np.zeros((int(sec*FPS), 9), np.float32))

        cases = [
            ("A", "硬切:点头【完整】→直接接摇头(不回中)",
             "看动作衔接处——因为两个动作首尾都在0附近,基本顺滑,只是没有停顿", build_A_hardcut()),
            ("B", "硬跳打断:点头播到一半【瞬间跳】到摇头",
             "⚠️看这里!点头在半路(pitch不在0)被硬生生截断,瞬间跳到摇头起点→明显突跳/抽搐。这就是离散token的病", build_B_hardjump()),
            ("C", "混合打断:点头播到一半→摇头【重叠加权混合】接入",
             "✅同样是半路打断,但重叠区平滑接管,无突跳。这就是 action chunking 的解法", build_C_blend()),
        ]
        for key, title, watch, traj in cases:
            if arg != "ALL" and arg != key:
                continue
            print(f"═══ {key}. {title} ═══", flush=True)
            print(f"    提示:{watch}", flush=True)
            rest(0.8)
            play(mini, neutral, traj, tag=title)
            rest(1.5)
            print("", flush=True)
        print(">>> 演示完毕。对比 B(突跳) vs C(平滑)最能体现项目价值。", flush=True)

if __name__ == "__main__":
    main()
