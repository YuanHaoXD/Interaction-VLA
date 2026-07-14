# _diag_sdk_calls.py —— 逐个测 SDK 调用是否阻塞/耗时,定位 B1 卡点
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
import time, sys
import numpy as np
from reachy_mini import ReachyMini

def timed(name, fn):
    t0 = time.perf_counter()
    try:
        r = fn()
        dt = (time.perf_counter() - t0) * 1000
        print(f"[OK]   {name:42s} {dt:8.1f} ms", flush=True)
        return r
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        print(f"[FAIL] {name:42s} {dt:8.1f} ms -> {type(e).__name__}: {e}", flush=True)
        return None

print(">>> 构造 ReachyMini()", flush=True)
mini = timed("ReachyMini()", lambda: ReachyMini())
if mini is None:
    sys.exit("构造失败")

# 列出实例上所有可能的 getter/setter 方法名,便于确认 API 真实存在
methods = [m for m in dir(mini) if not m.startswith("_") and callable(getattr(mini, m))]
print(">>> 可用方法:", methods, flush=True)

pose = timed("get_current_head_pose()", lambda: mini.get_current_head_pose())
if pose is not None:
    print("     head_pose shape:", np.asarray(pose).shape, flush=True)

timed("get_current_joint_positions()", lambda: mini.get_current_joint_positions())

# 天线读数:脚本里用的名字
if hasattr(mini, "get_present_antenna_joint_positions"):
    timed("get_present_antenna_joint_positions()", lambda: mini.get_present_antenna_joint_positions())
else:
    print("[MISSING] get_present_antenna_joint_positions 不存在!", flush=True)

# 测一次 set_target(中立位)
if pose is not None:
    timed("set_target(neutral)x1", lambda: mini.set_target(head=np.asarray(pose), antennas=[0.0,0.0], body_yaw=0.0))

# 连续 30 次 set_target 测吞吐(不 sleep)
if pose is not None:
    p = np.asarray(pose)
    t0 = time.perf_counter()
    for _ in range(30):
        mini.set_target(head=p, antennas=[0.0,0.0], body_yaw=0.0)
    dt = (time.perf_counter()-t0)/30*1000
    print(f"[RATE] set_target x30 平均 {dt:.2f} ms/call -> 理论 {1000/dt:.1f} Hz", flush=True)

# 连续 30 次 get_current_head_pose 测吞吐
if pose is not None:
    t0 = time.perf_counter()
    for _ in range(30):
        mini.get_current_head_pose()
    dt = (time.perf_counter()-t0)/30*1000
    print(f"[RATE] get_current_head_pose x30 平均 {dt:.2f} ms/call -> 理论 {1000/dt:.1f} Hz", flush=True)

print(">>> 诊断完成", flush=True)
