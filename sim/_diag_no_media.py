# _diag_no_media.py —— 测 media_backend=no_media + localhost_only 是否显著加速
# 假设:动作控制不需要音视频,跳过媒体管线初始化可能省下大头时间
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
import time
from reachy_mini import ReachyMini

def try_connect(label, **kwargs):
    t0 = time.perf_counter()
    try:
        mini = ReachyMini(**kwargs)
        dt = time.perf_counter() - t0
        # 顺带验证连上后能否正常控制(取一次姿态)
        try:
            pose = mini.get_current_head_pose()
            ok = "姿态读取OK" if pose is not None else "姿态None"
        except Exception as e:
            ok = f"姿态读取失败:{e}"
        print(f"[OK]   {label:42s} {dt:6.1f}s  ({ok})", flush=True)
        try: mini.__exit__(None, None, None)
        except Exception: pass
        return dt
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"[FAIL] {label:42s} {dt:6.1f}s -> {type(e).__name__}: {e}", flush=True)
    return None

print("测试:媒体后端对连接耗时的影响(动作控制不需要音视频)", flush=True)
print("-" * 66, flush=True)
try_connect("默认(default media)", connection_mode="localhost_only")
try_connect("no_media", connection_mode="localhost_only", media_backend="no_media")
try_connect("no_media (再测一次看稳定性)", connection_mode="localhost_only", media_backend="no_media")
print("-" * 66)
print("若 no_media 明显更快且姿态读取OK → 动作脚本改用它", flush=True)
