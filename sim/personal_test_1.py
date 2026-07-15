from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time
import numpy as np
# 连接到localhost上运行的仿真
with ReachyMini() as mini:
    print("已连接到仿真！")

    # 向上看并倾斜头部
    print("移动头部...")
    mini.goto_target(
        head=create_head_pose(z=200, roll=10, mm=True, degrees=True),
        duration=3
    )
    time.sleep(2)
    # 摆动天线
    print("摆动天线...")
    mini.goto_target(antennas=[0.6, -0.6], duration=0.3)
    mini.goto_target(antennas=[-0.6, 0.6], duration=0.3)
    time.sleep(2)
    print("Back")
    # 重置到休息位置
    mini.goto_target(
        head=create_head_pose(),
        antennas=[0, 0],
        duration=1.0
    )