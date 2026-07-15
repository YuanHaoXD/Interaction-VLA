# _diag_connect_modes.py —— 对比不同连接参数的构造耗时,验证 localhost_only 是否更快
# 依据官方源码 ReachyMini.__init__: host 默认 reachy-mini.local, connection_mode 默认 auto
#   auto = 先试 localhost(timeout=5s) 失败再试网络域名(慢) → 这是之前连接慢的元凶
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
import time, inspect
from reachy_mini import ReachyMini

# 先打印真实签名,确认参数名(以本机安装的 1.8.0 为准,可能和 main 源码略有出入)
try:
    sig = inspect.signature(ReachyMini.__init__)
    print("本机 ReachyMini.__init__ 签名:")
    print("  ", sig)
except Exception as e:
    print("取签名失败:", e)
print("-" * 60, flush=True)

def try_connect(label, **kwargs):
    t0 = time.perf_counter()
    try:
        mini = ReachyMini(**kwargs)
        dt = time.perf_counter() - t0
        print(f"[OK]   {label:38s} {dt:6.1f}s", flush=True)
        # 干净释放(用 __exit__ 或显式方法)
        try:
            if hasattr(mini, "__exit__"): mini.__exit__(None, None, None)
        except Exception: pass
        return dt
    except TypeError as e:
        print(f"[跳过] {label:38s} 参数不被支持: {e}", flush=True)
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"[FAIL] {label:38s} {dt:6.1f}s -> {type(e).__name__}: {e}", flush=True)
    return None

# 逐个测(每次都是一次全新构造)
try_connect("裸 ReachyMini() [默认auto,我之前用的]")
try_connect("localhost_only=True", localhost_only=True)
try_connect("connection_mode=localhost_only", connection_mode="localhost_only")
try_connect("host=localhost", host="localhost")
try_connect("use_sim=True", use_sim=True)

print("-" * 60)
print("结论:哪种最快就在正式脚本里用哪种", flush=True)
