# M1-A4：SDK/电机适配研究笔记

**日期**：2026-07-14 首版（仅官方文档，SDK 装不上）；2026-07-15 **补 SDK 源码实读复核**
**环境**：8×昇腾 910B2 服务器（headless，无 GPU 显示、无真机）
**方法升级**：首版只能靠官方文档 + 推断；本次**已在服务器装上 `reachy_mini==1.9.0` 并直读源码**，
把"推断/待查"改为源码实据。安装绕坑见 §0。

> ⚠️ 首版的 §2–§7 多为"推断/需确认"。**以本文 §A（源码实读）为准**；下方旧节保留 B1 实测表。

---

## 0. 服务器安装 reachy_mini 的绕坑（可复现）

`pip install reachy-mini` 直接失败：拉 `PyGObject` → `pycairo` → 需系统 `cairo`（无 root，装不了）。
但 `PyGObject/cairo` 只服务 **音频/GStreamer + 人脸跟踪 vision**，与 `set_target`/运动学无关。绕法：

```bash
pip install reachy_mini --no-deps
pip install reachy-mini-rust-kinematics reachy_mini_motor_controller   # 运动学+电机控制(核心)
pip install aiohttp asgiref fastapi huggingface-hub libusb_package log-throttling \
            platformdirs psutil pyserial pyusb pyyaml questionary rustypot toml \
            "uvicorn[standard]" zeroconf python-multipart                # 纯 python 依赖,跳过 PyGObject
```
效果：**运动学/协议/电机控制器源码可读、纯运算可跑**；`from reachy_mini import ReachyMini`
顶层导入仍会因 `vision/face_tracking.py: import gi` 中断（不影响读源码与做 IK/FK 计算）。
> 若将来真要在服务器起 daemon，需系统装 `libcairo2` + `gobject-introspection`（要 root）。

---

# §A 源码实读结论（reachy_mini 1.9.0，六问逐条）

### ① `set_target` 到电机的完整链路
`ReachyMini.set_target(head=4×4, antennas=[右,左], body_yaw=φ)`
（`reachy_mini/reachy_mini.py:521`）→ 组 `SetFullTargetCmd`（head 展平成 16 元 list）
→ `client.send_command()` → **WebSocket / WebRTC DataChannel** → reachy-mini-daemon → 后端电机控制器。
- **一次 set_target = 一个 tick**（源码原话："one set_target per tick"）。是**流式命令**，
  daemon **不回 per-command ack**（fire-and-forget，WebRTC SCTP 有序可靠）。
- **有内置安全**：daemon 侧 IK 用 `inverse_kinematics_safe`（见 ⑥），自动把 body/head yaw 限进机械限位。
- 另有**上传-回放**通道（`UploadMove*` + `PlayUploadedMoveCmd`）：把整段 move 先传到 daemon，
  由机器人本地 `Backend.play_move` 跑内循环——**无线链路下比逐帧 set_target 抖动小**。
  → 对我们：仿真/有线可逐帧 set_target；**真机 demo 若无线，考虑改用上传-回放通道**。

### ② 控制频率上限与超频行为
- 状态发布固定 **50 Hz**（`JointStateMsg`/`HeadPoseMsg`/`ImuDataMsg` 源码注释 "published at 50 Hz"）。
- 上传-回放的 `play_frequency` **默认 100 Hz，硬上限 `le=200.0`**（`io/protocol.py:656`）。
  → **30 Hz 目标远在能力内**；块回放可开到 100 Hz。
- 逐帧流式无显式频率字段，按 tick 到达即执行；超发不会崩，但受 50 Hz 状态回读与网络 RTT 约束。
- **修订认知**：首版说"频率上限 30Hz"是把 B1 的**实测达成率**误当成上限；实际 SDK 支持到 200 Hz，
  30 Hz 只是我们选的控制率。

### ③ 9 维硬限位（源码机械限位 vs B1 实测）
- 电机侧是 **9 元数组**，顺序 = `[body_yaw, stewart×6, antennas×2]`
  （`reachy_mini_motor_controller/__init__.pyi:76`："positions = body_yaw, stewart, antennas"）。
  ⚠️ **注意**：电机数组是 `body_yaw + 6 个 Stewart 支腿关节 + 2 天线`，与我们**任务空间** 9 维
  `[x,y,z,rpy,body_yaw,antR,antL]` **不是同一组 9 维**——中间隔着 Stewart IK。我们喂任务空间，
  SDK 负责解到 6 支腿。
- 机械限位（源码硬值）：**`max_body_yaw = 160°`、head 相对 body 的 `max_relative_yaw = 65°`**。
- 头部平移/rpy 的精细盒仍以 B1 实测 `sim/limits.json` 为准（旧表见下），二者不冲突：
  B1 表是我们收窄的**安全工作盒**，源码是**机械极限**。

### ④ 电机型号/减速比/断力矩手掰（关系 M3 示教）
- **断力矩可行**：`disable_motors(ids=None)` / `enable_motors(ids=None)`（`reachy_mini.py:974/985`）
  → 发 `SetTorqueCmd(on=False/True, ids=...)`，可整机或按 id 关力矩。**M3 手掰示教路径成立**。
  ⚠️ 源码警告：`enable_motors()` 会把所有 target 钉到当前姿态，故示教录完重新使能后要**先 enable 再 set_target**。
- 电机可按 id 读 PID、读位置（`get_motor_name_id()` 给名→id 映射，9 个电机）。
- **型号/减速比**：源码是 rustypot 驱动（Feetech/Dynamixel 类总线舵机），具体型号未在 py 层写死，
  在 rust 扩展里；**非 M1 阻塞项**，M3 真机前本机对照即可。

### ⑤ Headless MuJoCo 服务器可用性（关系自动化评测在哪跑）——**已实测**
在服务器（`pip install mujoco==3.10.0`）实测：

| 用法 | 结果 | 说明 |
| --- | --- | --- |
| **纯物理 `mj_step`（无渲染）** | ✅ **可用** | 100 步仿真正常，qpos 正常更新 |
| 离屏渲染 `MUJOCO_GL=egl` | ❌ 失败 | GLContext 无法创建（headless 无渲染设备） |
| 离屏渲染 `MUJOCO_GL=osmesa` | ❌ 失败 | 无 osmesa GL 后端（`glGetError` NoneType） |

- 系统有 `libEGL.so.1/libEGL_mesa.so.0`，但节点无可用 render device，EGL 上下文建不起来。
- **结论**：**轨迹级校验/物理压测可在服务器 headless 跑**（不需要 GL）；
  **但 VLM 裁判要的"渲染成视频"（M2/M3）在本服务器当前起不来**——
  需运维补 GPU EGL 或 `osmesa` 软渲染库，否则**渲染类评测放本机**（本机有 GUI MuJoCo）。
  → 这是一个要早点告诉运维的点，写入"给本地的话"。

### ⑥ Stewart 平台 body_yaw 补偿的准确语义——**源码实据**
`AnalyticalKinematics.ik()`（`kinematics/analytical_kinematics.py:65`）：
- `automatic_body_yaw=True`（默认）时走 `inverse_kinematics_safe(pose, body_yaw,
  max_relative_yaw=65°, max_body_yaw=160°)`：**在保证头姿态的前提下自动调 body_yaw**，
  使 body 绝对偏航 ≤160°、head-相对-body 偏航 ≤65° 都不越机械限。
- `automatic_body_yaw=False` 时 `reachy_joints = [body_yaw] + stewart_joints`：**不动 body_yaw，直接透传**。
- ik 输出 7 维 = `[body_yaw] + 6 Stewart 支腿`；加 2 天线才是电机 9 维。
- ik 内部对 pose 有 `head_z_offset` 平移补偿（源码 `_pose[:3,3][2] += head_z_offset`）。
- **对我们（§3.2 "Stewart 逆解自动补偿 body_yaw"）**：证实无误。数据侧我们的 9 维含独立 `body_yaw`，
  执行时若开 `automatic_body_yaw` SDK 会再微调它——**建议训练/回放统一固定一种模式**（自动 or 手动），
  避免"我们给的 body_yaw"与"SDK 改过的 body_yaw"在 FK 对不上。这是个**待本地确认的一致性点**（见文末）。

---

# §B 旧节（首版，B1 实测表保留，其余以 §A 为准）

## 4'. 9 维各自硬限位（本机 B1 实测 `sim/limits.json`，仍有效）

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

> 注：B1 表 `body_yaw ±2.8 rad ≈ ±160°` 与源码 `max_body_yaw=160°` **吻合**，互为印证。

---

## 六问结论总表（复核版）

| 问题 | 结论 | 来源 |
|------|------|------|
| ① set_target 链路 | ✅ WS/WebRTC 流式 fire-and-forget，一 tick 一命令；另有上传-回放通道 | 源码 |
| ② 频率上限 | ✅ 状态 50Hz；回放 play_frequency 默认 100 上限 200Hz；30Hz 富余 | 源码 |
| ③ 9 维硬限位 | ✅ 机械 body_yaw 160°/rel 65°；电机 9 维=[body_yaw,stewart×6,ant×2]≠任务 9 维 | 源码+B1 |
| ④ 电机/断力矩 | ✅ disable/enable_motors 可整机断力矩→M3 示教成立；型号在 rust 层，非阻塞 | 源码 |
| ⑤ headless MuJoCo | ⚠️ 物理 mj_step 可用；**离屏渲染 EGL/OSMesa 起不来**→渲染类评测需运维补库或放本机 | 实测 |
| ⑥ Stewart body_yaw 补偿 | ✅ automatic_body_yaw 用 inverse_kinematics_safe 自动调偏航保限位；建议固定一种模式 | 源码 |

## 给本地的话（待决/需协调）
1. **渲染评测落点**（⑤）：本服务器离屏渲染起不来，M2/M3 的 MuJoCo→视频→VLM 裁判要么运维补 EGL/osmesa，
   要么在本机渲染。**请本地定：渲染评测放哪台机。**
2. **body_yaw 一致性**（⑥）：数据里的 `body_yaw` 与 SDK `automatic_body_yaw` 自动微调可能不一致。
   建议训练与回放统一 `automatic_body_yaw=False`（透传我们给的 body_yaw），否则 FK 对不上。
   **这涉及动作表示的执行语义，属"需确认"，先记在此不擅改 schema。**

## 附件
- SDK 源码：`.../site-packages/reachy_mini/`（1.9.0）；电机控制器 `.../reachy_mini_motor_controller/__init__.pyi`
- B1 实测：`sim/limits.json`；官方文档：https://reachymini.net/zh/developers.html
