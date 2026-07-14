# _diag_sweep.py —— 复现 B1 的 sweep_dim,每步打印,定位为何 20 分钟不产出
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini

DIMS = ["x","y","z","roll","pitch","yaw","body_yaw","ant_right","ant_left"]
DT = 1.0 / 30.0

def pose_mat(neutral, x,y,z,roll,pitch,yaw):
    d = np.eye(4); d[:3,:3] = R.from_euler("xyz",[roll,pitch,yaw]).as_matrix(); d[:3,3] = [x,y,z]
    return neutral @ d

def read_state(mini, neutral_inv):
    cur = neutral_inv @ mini.get_current_head_pose()
    xyz = cur[:3,3]; rpy = R.from_matrix(cur[:3,:3]).as_euler("xyz")
    ants = mini.get_present_antenna_joint_positions()
    return np.array([*xyz, *rpy, 0.0, ants[0], ants[1]])

def apply(mini, neutral, a):
    mini.set_target(head=pose_mat(neutral, *a[:6]), antennas=[a[7], a[8]], body_yaw=float(a[6]))

print(">>> 连接中(构造可能耗时~50s)...", flush=True)
tc = time.perf_counter()
mini = ReachyMini()
print(f">>> 已连接, 构造耗时 {time.perf_counter()-tc:.1f}s", flush=True)
neutral = mini.get_current_head_pose()
neutral_inv = np.linalg.inv(neutral)

# 只测 pitch(维度4)正向,step=0.05 max_abs=0.8 err_tol=0.04 —— 与主脚本同参
i, step, max_abs, err_tol = 4, 0.05, 0.8, 0.04
print(f">>> 扫描 {DIMS[i]} 正向: step={step} max_abs={max_abs} err_tol={err_tol}", flush=True)
v, safe = 0.0, 0.0
nstep = 0
t_dim = time.perf_counter()
while abs(v) < max_abs:
    v += step
    nstep += 1
    a = np.zeros(9); a[i] = v
    t_apply = time.perf_counter()
    for _ in range(15):
        apply(mini, neutral, a); time.sleep(DT)
    time.sleep(0.3)
    st = read_state(mini, neutral_inv)
    err = abs(st[i] - v)
    print(f"    step{nstep:2d} cmd={v:+.3f} measured={st[i]:+.4f} err={err:.4f} "
          f"{'触顶break' if err>err_tol else 'ok'} [{time.perf_counter()-t_apply:.2f}s]", flush=True)
    if err > err_tol: break
    safe = v
print(f">>> {DIMS[i]} 正向完成: safe={safe:.3f}, {nstep}步, 耗时 {time.perf_counter()-t_dim:.1f}s", flush=True)
# 回中
a = np.zeros(9)
for _ in range(30): apply(mini, neutral, a); time.sleep(DT)
print(">>> 回中完成,诊断结束", flush=True)
