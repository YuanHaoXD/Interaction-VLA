# 从官方 AGENTS.md 学到的:连接最佳实践 + 我之前理解的修正

> 读了官方 [pollen-robotics/reachy_mini AGENTS.md](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md) + SDK 源码后的收获与实测。
> 面向:想理解"为什么这样连仿真"的人,以及以后接手改脚本的人。
> 结论都在本机(reachy-mini 1.8.0)实测验证过。

---

## 一、最重要的收获:连接参数用官方推荐的,更快更稳

### 之前的写法(能用,但不最优)

```python
mini = ReachyMini()          # 裸构造,全吃默认值
```

### 现在的写法(官方最佳实践)

```python
with ReachyMini(connection_mode="localhost_only", media_backend="no_media") as mini:
    neutral = mini.get_current_head_pose()
    ...  # 30Hz set_target 控制
# 退出 with 块(正常结束 或 Ctrl+C/关窗被打断)都会自动干净释放连接
```

三个改动,每个都有实测依据:

| 改动 | 为什么 | 实测效果 |
|---|---|---|
| `connection_mode="localhost_only"` | 仿真 daemon 就在本机,不用探测网络。默认 `"auto"` 会先试 localhost 再回退网络域名 `reachy-mini.local` | 略快(~7s→~7s,省探测那一下),更重要是**杜绝极端情况下去解析 reachy-mini.local 造成的超长卡顿** |
| `media_backend="no_media"` | **动作控制根本不需要音视频**。默认会初始化整条音视频管线 | **连接快约 40%:7.4s → 4.5s**,且**避开所有音视频坑**(`No Audio USB device`、媒体管线卡死等) |
| `with ... as mini:` 上下文管理器 | 官方 AGENTS.md 首推的连接模式。保证 `__exit__` 释放连接 | **修复"强杀导致连接管线坏"的坑**:即使脚本被 Ctrl+C/关窗打断,连接也干净释放,不会污染 daemon |

### 实测数据(本机,同一 daemon)

```
连接模式对比:
  裸 ReachyMini() [默认auto]          8.0s
  connection_mode=localhost_only      6.8s
  host=localhost                      7.4s
  use_sim=True                        7.7s

媒体后端对比(都用 localhost_only):
  默认 media                          7.4s   姿态读取OK
  no_media                            4.8s   姿态读取OK   ← 快40%
  no_media (复测)                     4.5s   姿态读取OK
```

**结论:动作脚本一律用 `connection_mode="localhost_only", media_backend="no_media"` + `with`。** 已应用到 `sim/demo_show.py`、`sim/replay_episode.py`。

---

## 二、修正我之前的一个误判(诚实记录)

**我之前以为**"连接慢 10~52 秒是因为 auto 模式回退到网络域名探测"。**实测证明这个判断不完全对**:五种连接模式都在 7-8 秒,差异很小。真正的大头是**媒体管线初始化**(no_media 能省 40%),不是连接模式。

之前观察到的"10~52 秒甚至更久"的巨大波动,主因是:① 媒体管线初始化(现在 no_media 跳过)、② daemon 冷热状态、③ 那次 GUI 冲突(见文档 02)。加了 no_media + localhost_only 后,连接稳定在 4~5 秒,波动也小多了。

> 教训:猜想要用实验验证。我一开始的假设方向对(连接参数有优化空间)但归因错了(以为是网络探测,其实是媒体管线)。做了对比实验才找到真正的加速点。

---

## 三、官方 SDK 的权威签名(1.8.0,本机 inspect 确认)

```python
ReachyMini.__init__(
    robot_name="reachy_mini",
    host="reachy-mini.local",          # 默认是网络域名!仿真务必配 localhost_only 覆盖
    port=8000,
    connection_mode="auto",            # "auto" | "localhost_only" | "network"
    spawn_daemon=False,
    use_sim=False,
    timeout=5.0,                       # 单次连接尝试超时
    automatic_body_yaw=True,
    log_level="INFO",
    media_backend="default",           # "default" | "no_media" | LOCAL | WEBRTC
    localhost_only=None,               # 已废弃,改用 connection_mode
)
```

**动作控制常用方法(源码确认):**

```python
mini.set_target(head=4x4矩阵, antennas=[右,左]弧度, body_yaw=弧度)   # 30Hz实时控制,绕过插值
mini.goto_target(head=..., antennas=..., duration=0.5, method="minjerk", body_yaw=0.0)  # 带插值的慢动作
mini.get_current_head_pose() -> 4x4矩阵
mini.get_current_joint_positions() -> (head关节[7], 天线[2])  # 弧度
mini.get_present_antenna_joint_positions() -> [右,左]  # 弧度
```

- **`set_target` 是我们做连续动作的正确接口**(每帧算好姿态直接推,不要插值)。
- `goto_target` 是"给个目标+时长,自己缓慢过去"——硬编码版小艺的 nod() 用的就是它,这也是它生硬的原因(见文档 01)。

> ⚠️ 源码里一个坑注记:`enable_motors()` 会把所有目标钉到当前姿态再上力矩,所以 `set_target(X); enable_motors()` 不会驱动到 X。要 **先 enable_motors 再 set_target**。仿真里一般不涉及,真机注意。

---

## 四、官方文档还提到、但我们暂时用不上的东西(备查)

- **App 框架**:官方推荐用 `reachy-mini-app-assistant` 脚手架建 app(Python app 或 JS/Web app),**不要手动建 app 文件夹**。我们做研究/数据管线,不走 app 那套,直接用 SDK 即可。
- **JS/Web app**:浏览器通过 WebRTC 连 daemon,有 `setHeadRpyDeg`/`setAntennasDeg`/`setBodyYawDeg` 等接口。我们全用 Python,不涉及。
- **真机/无线连接**:`host="reachy-mini.local"` 或机器人 IP,`connection_mode="network"`。真机部署(M3)再用。
- 官方进一步的文档:`docs/source/SDK/python-sdk.md`(API 概览)、`docs/source/troubleshooting.md`(连接问题)、`docs/source/platforms/`(各平台启动)。

---

## 五、三句话速记

1. **动作脚本连接一律用** `with ReachyMini(connection_mode="localhost_only", media_backend="no_media") as mini:` —— 快40%、避音视频坑、被打断也干净释放。
2. **no_media 是关键加速点**(不是连接模式)——动作控制不需音视频,跳过它省一半时间。
3. **set_target 做连续控制,goto_target 是慢插值**;真机注意先 enable_motors 再 set_target。
