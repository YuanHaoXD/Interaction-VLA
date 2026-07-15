# sim/replay_episode.py —— 把 episode 的动作轨迹以 30Hz 回放到 MuJoCo 仿真
# 用法: python sim/replay_episode.py <episode_dir>
import os, sys, time, json
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini

def main(ep_dir):
    p = Path(ep_dir)
    actions = np.load(p/"actions.npy")
    meta = json.loads((p/"meta.json").read_text(encoding="utf-8"))
    # 官方最佳实践:localhost_only(仿真在本机)+ no_media(动作不需音视频,连接快~40%)+ with(干净释放)
    with ReachyMini(connection_mode="localhost_only", media_backend="no_media") as mini:
        neutral = mini.get_current_head_pose()
        print(f"回放 {meta['episode_id']}: {len(actions)} 帧 @30Hz")
        t0 = time.perf_counter()
        for k, a in enumerate(actions):
            d = np.eye(4)
            d[:3,:3] = R.from_euler("xyz", a[3:6]).as_matrix(); d[:3,3] = a[:3]
            mini.set_target(head=neutral @ d, antennas=[float(a[7]), float(a[8])],
                            body_yaw=float(a[6]))
            nxt = t0 + (k+1)/30
            while time.perf_counter() < nxt: pass
        print(f"完成,实际用时 {time.perf_counter()-t0:.1f}s (标称 {len(actions)/30:.1f}s)")

if __name__ == "__main__":
    main(sys.argv[1])
