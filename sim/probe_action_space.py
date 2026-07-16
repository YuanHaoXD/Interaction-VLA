# sim/probe_action_space.py —— 动作空间勘察:逐维扫限位 + 50Hz 压测
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
import time, json
import numpy as np
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini

DIMS = ["x","y","z","roll","pitch","yaw","body_yaw","ant_right","ant_left"]
DT = 1.0 / 50.0   # v1.3:30→50Hz

def pose_mat(neutral, x,y,z,roll,pitch,yaw):
    d = np.eye(4); d[:3,:3] = R.from_euler("xyz",[roll,pitch,yaw]).as_matrix(); d[:3,3] = [x,y,z]
    return neutral @ d

def read_state(mini, neutral_inv):
    cur = neutral_inv @ mini.get_current_head_pose()
    xyz = cur[:3,3]; rpy = R.from_matrix(cur[:3,:3]).as_euler("xyz")
    ants = mini.get_present_antenna_joint_positions()  # 顺序假定 [右,左],与 set_target 一致;若观察相反则在报告记录并交换
    return np.array([*xyz, *rpy, 0.0, ants[0], ants[1]])  # body_yaw 读数不可靠,扫描时对第6维特殊处理

def apply(mini, neutral, a):
    mini.set_target(head=pose_mat(neutral, *a[:6]), antennas=[a[7], a[8]], body_yaw=float(a[6]))

def sweep_dim(mini, neutral, neutral_inv, i, step, max_abs, err_tol):
    """逐步加大第 i 维指令,实测跟踪误差,误差超容差即视为触顶。返回 (safe_neg, safe_pos)。"""
    bounds = []
    for sign in (+1, -1):
        v, safe = 0.0, 0.0
        while abs(v) < max_abs:
            v += sign * step
            a = np.zeros(9); a[i] = v
            for _ in range(15): apply(mini, neutral, a); time.sleep(DT)   # 0.5s 逼近
            time.sleep(0.3)
            err = abs(read_state(mini, neutral_inv)[i] - v)
            if err > err_tol: break
            safe = v
        bounds.append(safe)
        a = np.zeros(9)
        for _ in range(30): apply(mini, neutral, a); time.sleep(DT)      # 回中
    return bounds[1], bounds[0]

def rate_test(mini, neutral, seconds=30, freq=1.0, amp=0.17):
    """pitch 正弦 50Hz 流,测实际循环频率与最大周期抖动。"""
    n = int(seconds / DT); tics = []
    t0 = time.perf_counter()
    for k in range(n):
        a = np.zeros(9); a[4] = amp * np.sin(2*np.pi*freq*k*DT)
        apply(mini, neutral, a)
        tics.append(time.perf_counter())
        nxt = t0 + (k+1)*DT
        while time.perf_counter() < nxt: pass
    d = np.diff(tics)
    return {"achieved_hz": float(1/np.mean(d)), "max_jitter_ms": float(np.max(d)*1000 - DT*1000)}

def main():
    mini = ReachyMini()
    neutral = mini.get_current_head_pose()
    neutral_inv = np.linalg.inv(neutral)
    cfg = [  # (step, max_abs, err_tol) 每维扫描参数
        (0.005,0.06,0.004),(0.005,0.06,0.004),(0.005,0.06,0.004),        # xyz (m)
        (0.05,0.8,0.04),(0.05,0.8,0.04),(0.05,1.2,0.04),                 # rpy (rad)
        (0.1,3.2,0.08),(0.1,3.2,0.15),(0.1,3.2,0.15)]                    # body_yaw, ants
    mins, maxs = [], []
    for i,(s,m,e) in enumerate(cfg):
        if i == 6:  # body_yaw 无可靠实测读数:采用保守文档值,肉眼观察仿真中是否达位,A4 负责核实
            lo, hi = -2.8, 2.8
            print(f"{DIMS[i]:9s} 跳过实测,取文档保守值 [{lo:+.3f}, {hi:+.3f}](待 A4 核实)")
        else:
            lo, hi = sweep_dim(mini, neutral, neutral_inv, i, s, m, e)
            print(f"{DIMS[i]:9s} safe range: [{lo:+.3f}, {hi:+.3f}]")
        mins.append(round(lo,4)); maxs.append(round(hi,4))
    rt = rate_test(mini, neutral)
    print("50Hz 压测:", rt)
    out = {"dim_names": DIMS, "safe_min": mins, "safe_max": maxs,
           "max_vel": [0.1,0.1,0.1,2.0,2.0,2.0,2.0,6.0,6.0], **rt}   # max_vel 初值,按报告修订
    with open(os.path.join(os.path.dirname(__file__),"limits.json"),"w",encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("已写出 sim/limits.json")

if __name__ == "__main__":
    main()
