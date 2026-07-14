# M1-A4：SDK/电机适配研究笔记

**日期**：2026-07-14
**环境**：8×昇腾 910B 64G（无仿真环境，无真机）
**研究方法**：官方文档 + 本机 B1 实测数据

---

## 1. SDK 安装情况

**服务器安装尝试**：❌ 失败

```bash
pip install reachy-mini[mujoco]
```

**错误**：系统依赖缺失（cairo），服务器为 headless 环境

**结论**：
- 服务器无需安装 reachy-mini（无 MuJoCo 仿真、无真机）
- SDK 适配主要依赖本机测试数据
- A4 纯调研性质，无需服务器运行 SDK

## 2. `set_target` 到电机的完整链路

**官方文档（reachymini.net/zh/developers.html）**：

```python
from reachy_mini import ReachyMini
robot = ReachyMini()

# 高层接口（封装后）
robot.head.look_at(x=10, y=0, z=30)
robot.head.rotate(pitch=15, roll=0, yaw=45)

# 底层接口（设计文档 §3.2）
robot.set_target(head=4×4矩阵, antennas=[右,左], body_yaw=φ)
```

**控制链路推断**：
1. Python SDK → REST/WebSocket → reachy-mini-daemon
2. daemon 内插值行为 → 电机控制器
3. 控制频率上限：30Hz（本机 B1 实测 achieved_hz: 30.000）

**注意**：
- 设计文档要求"SDK 源码为准"，但服务器无法安装 SDK
- 链路细节需本地 Fable 5 确认或参考 reachy-mini-demo 仓库

## 3. 控制频率上限与超频行为

**本机 B1 实测数据（sim/limits.json）**：
- `achieved_hz`: 30.000
- `max_jitter_ms`: 4.6ms

**结论**：
- 30Hz 稳定可达，抖动 < 5ms
- 超频行为未知（服务器无法测试），需本地确认

## 4. 9 维各自硬限位

**本机 B1 实测（sim/limits.json）**：

| 维度 | safe_min | safe_max | max_vel | 单位 |
|------|----------|----------|---------|------|
| x    | -0.05    | 0.03     | 0.1     | m    |
| y    | -0.05    | 0.05     | 0.1     | m    |
| z    | -0.055   | 0.02     | 0.1     | m    |
| roll | -0.8     | 0.8      | 2.0     | rad  |
| pitch| -0.75    | 0.7      | 2.0     | rad  |
| yaw  | -1.2     | 0.9      | 2.0     | rad  |
| body_yaw | -2.8| 2.8      | 2.0     | rad  |
| ant_right | -3.2| 3.2      | 6.0     | rad  |
| ant_left  | -3.2| 3.2      | 6.0     | rad  |

**对照设计文档 §3.2**：
- 头部运动范围小（±30° 量级）← 实测约 ±0.7~0.8 rad ≈ ±40~45°，稍有超出但仍远离万向锁
- 天线范围 ±3.2 rad ≈ ±180°，全范围可用

## 5. 电机型号/减速比/断力矩能力

**官方文档未提供**。

**推断**：
- Reachy Mini 为开源桌面人形机器人
- 电机一般为标准舵机（本机 B1 脚本中有 `disable_motors()` 引用，说明支持断力矩）

**需确认**（M3 示教相关）：
- 具体电机型号
- 是否可整机断力矩手掰
- 减速比

## 6. Headless MuJoCo 服务器可用性

**服务器安装尝试**：❌ 失败（同 SDK，系统依赖缺失）

**本机测试**（B1-B4）：
- MuJoCo 3.3.0 可用（`E:\Github\ReachyMni_Project\reachy-mini-demo\.venv`）
- 环境变量：`MUJOCO_GL=egl` 或 `osmesa`（headless 渲染）

**结论**：
- 服务器不适合运行 MuJoCo（无 GPU、系统依赖缺失）
- 后续自动化评测建议在本机运行

## 7. Stewart 平台 body_yaw 补偿语义

**设计文档 §3.2 注记**：
> "SDK 的 Stewart 逆解自动补偿 body_yaw"

**推断**：
- Reachy Mini 底座为 Stewart 平台结构
- `set_target(head=矩阵, body_yaw=φ)` 时，SDK 内部自动处理 Stewart 运动学
- 动作空间中的 `body_yaw` 独立于头部 rpy

**需确认**（本地或 SDK 源码）：
- 补偿的具体实现
- body_yaw 与 head yaw 的叠加关系

---

## 总结

| 问题 | 状态 | 来源 |
|------|------|------|
| set_target 链路 | ⚠️ 部分 | 官方文档 + 推断 |
| 频率上限 | ✅ 30Hz | 本机 B1 实测 |
| 硬限位 | ✅ 已测 | 本机 B1 limits.json |
| 电机参数 | ❌ 待查 | 需本地/SDK源码 |
| headless MuJoCo | ❌ 不可用 | 服务器环境限制 |
| Stewart 补偿 | ⚠️ 部分 | 设计文档 + 推断 |

**后续工作**：
- M3 真机部署前，需本地确认电机参数、断力矩手掰可行性
- 服务器端无需 SDK，直接使用本机提供的 limits.json

---

## 附件

- 本机 B1 实测数据：`sim/limits.json`
- 官方开发者文档：https://reachymini.net/zh/developers.html
