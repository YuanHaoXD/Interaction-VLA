# 自己动手:开仿真、跑动作演示,以及看懂 reachy-mini 和 MuJoCo

> 写给想**自己上手**跑仿真、给别人做动作演示的人。
> 全部命令都在本机(Windows 11)真实验证过。照着做即可。
> 配套:`01-模板动作vs模型动作-以及B1到B4在做什么.md`(先读那篇理解"为什么")。

---

## 零、三个概念,先建立正确的心智模型

### ① reachy-mini 代码 = 两半:服务端(本体) + 客户端(你的脚本)

```
┌─────────────────────────────┐         ┌──────────────────────────┐
│  reachy-mini-daemon --sim   │◀───────▶│  你的 Python 脚本         │
│  (服务端 = 虚拟机器人本体)   │  指令/  │  用 ReachyMini SDK 当客户端│
│                             │  状态   │                          │
│  · 内部跑 MuJoCo 物理引擎    │         │  mini.set_target(...)    │
│  · 算重力/关节/碰撞         │         │  mini.get_current_...()  │
│  · 渲染那个 3D 窗口         │         │                          │
│  · 在 localhost:8000 开服务 │         │  (B1-B4、demo_show 都是它)│
└─────────────────────────────┘         └──────────────────────────┘
```

**关键理解:你写的所有脚本永远是"客户端"。** 它们不直接控制电机、不直接算物理,只是通过 SDK 向 daemon**发指令**(set_target)和**读状态**(get_current_head_pose)。daemon 才是"机器人本体"——在仿真里它是虚拟的(跑 MuJoCo),在真机上它是真的(驱动 USB 电机)。**同一份客户端脚本,仿真和真机通用**——这就是仿真的价值。

- `--sim` = 虚拟本体(MuJoCo 物理 + 3D 窗口),我们现在全用它。
- 真机时把 `--sim` 去掉,daemon 连真电机,你的脚本一行不用改。

### ② MuJoCo:你几乎不直接碰它

MuJoCo 是一个**物理引擎**(Google DeepMind 出品),负责"给定关节指令,算出机器人在重力/碰撞下真实会怎么动,并渲染出来"。

**但你不需要学 MuJoCo、不需要写 MuJoCo 代码。** 它整个藏在 `reachy-mini-daemon` 内部。你唯一和它打交道的方式,就是:
- 通过 SDK 发 `set_target` → daemon 内部让 MuJoCo 步进一帧 → 你在 3D 窗口看到结果。
- 那个能用鼠标拖拽旋转视角的 3D 窗口,就是 MuJoCo 的 viewer。

换句话说:**MuJoCo 对你是"透明"的**。你只管发 9 维动作,物理和渲染 daemon+MuJoCo 全包了。(唯一可能需要碰 MuJoCo 的场景:改场景里的桌子/苹果等道具,那是编辑 `reachy-mini-demo` 里的 `.xml` 场景文件,M1 用不到。)

### ③ 动作 = 每 1/30 秒的 9 个数字

所有动作,无论点头还是复杂序列,本质都是一张表:每 1/30 秒一行,每行 9 个数。

```
索引:  0    1    2     3      4       5      6          7        8
含义:  x    y    z    roll   pitch   yaw   body_yaw   ant_右   ant_左
单位:  米   米   米   弧度   弧度    弧度   弧度       弧度     弧度
       └── 头部平移 ──┘ └──── 头部转动 ────┘ 身体转   └─ 两根天线 ─┘
```

- **pitch**(俯仰)正弦振荡 = 点头
- **yaw**(偏航)正弦振荡 = 摇头
- **roll**(侧倾)保持一个角度 = 歪头
- **两根天线**反相振荡 = 摆天线卖萌
- **body_yaw** = 身体原地左右转

记住这张表,你就能"读懂"任何一段动作数据,也能自己编动作。

---

## 一、完整流程:从零开到看见机器人动

### 前提(一次性,已具备)

- 仿真环境 venv:`E:\Github\ReachyMni_Project\reachy-mini-demo\.venv`(Python 3.12 + reachy-mini 1.8.0 + mujoco 3.3.0)。
- 本文所有 `python.exe` 都指这个 venv 的绝对路径。

### 步骤(PowerShell,按顺序)

**⚠️ 第 0 步——先确认没有别的 Python 程序在跑!** 下面第 1 步会杀掉本机所有 python 进程。如果你正开着 Jupyter、别的脚本、或 IDE 在调试 Python,先存好关掉,否则会被误杀。(reachy-mini-daemon 在 Windows 上也显示为 python.exe。)

```powershell
# 1) 杀掉所有残留的 python 和 daemon(确保干净启动)
Get-Process python,reachy-mini-daemon -EA SilentlyContinue | Stop-Process -Force

# 2) 起仿真 daemon(这个窗口会一直占用,不要关它;3D 窗口也会弹出)
E:\Github\ReachyMni_Project\reachy-mini-demo\.venv\Scripts\reachy-mini-daemon.exe --sim --scene minimal

# 3) 等就绪:浏览器打开 http://localhost:8000/docs,能打开就说明 REST 就绪
#    ⭐ 然后【再静置约 25 秒】——让内部媒体管线彻底稳定。这一步别省(冷 daemon 直连会卡)。

# 4) 【另开一个 PowerShell 窗口】跑动作脚本
$env:PYTHONUTF8 = 1
cd E:\Github\ReachyMni_Project\interaction-vla
E:\Github\ReachyMni_Project\reachy-mini-demo\.venv\Scripts\python.exe -u sim\demo_show.py all
```

**预期:** 脚本先打印"连接仿真(首次可能等 10~50 秒)",连上后 3D 窗口里机器人依次做:点头 → 摇头 → 歪头 → 摆天线 → 转身 → 看四方 → 呼吸,共约 26 秒。

> **关于"连接等 10~50 秒":** 这是本机特性,`ReachyMini()` 首次连接握手就是慢,而且时间不稳定(实测 10~52 秒)。**耐心等,别以为卡死了**。判断死活:看 3D 窗口机器人有没有动、或脚本有没有继续打印。连上之后所有动作都是流畅实时的。

### 演示用法(给别人看时)

```powershell
# 全套循环播放(一直演到你关窗口 / Ctrl+C),适合摆着当展台
...\python.exe -u sim\demo_show.py loop

# 只演单个动作
...\python.exe -u sim\demo_show.py nod        # 点头
...\python.exe -u sim\demo_show.py 摇头        # 中文名也行
...\python.exe -u sim\demo_show.py wiggle      # 摆天线
```

### 也可以回放真实的合成数据(B4 造的 100 段)

```powershell
...\python.exe -u sim\replay_episode.py samples\ep_00000003
```

`demo_show.py`(手编动作,适合演示)和 `replay_episode.py`(播放数据文件,适合看训练数据)**走的是同一套执行层**——都是"每 1/30 秒 set_target 一次"。

---

## 二、怎么改成自己的动作(调幅度、编新动作)

`sim/demo_show.py` 就是为"好改"设计的。每个动作是一个函数调用,返回 `[T,9]` 数组。

### 调幅度:改一个数字

打开 `sim/demo_show.py`,找到 `build_shows()` 函数,里面是这样的:

```python
("点头 nod", _osc(4, 0.20, 1.2, 3)),   # _osc(维度, 幅度, 频率, 周期数)
#                 │    │     │   └─ 点几下
#                 │    │     └───── 多快(Hz)
#                 │    └─────────── 多大幅度(弧度)← 想更明显就调大这个
#                 └──────────────── 维度4=pitch=点头
```

- **想点头更明显** → 把 `0.20` 改成 `0.30`(注意 B1 实测 pitch 安全上限约 0.70,别超)。
- **想点快点** → 把 `1.2`(Hz)改成 `1.8`。
- **想多点几下** → 把 `3` 改成 `5`。

摇头同理(维度 5=yaw),歪头看 `_hold(3, 0.25, ...)`(维度 3=roll,`0.25` 是保持的角度)。

### 各维度安全幅度参考(来自 B1 实测 `sim/limits.json`)

| 动作 | 维度 | 安全范围(弧度) | 换算 | 演示建议值 |
|---|---|---|---|---|
| 点头 pitch | 4 | -0.75 ~ +0.70 | ±40° | 0.20~0.35 |
| 摇头 yaw | 5 | -1.20 ~ +0.90 | -69°~+52° | 0.25~0.40 |
| 歪头 roll | 3 | ±0.80 | ±46° | 0.20~0.30 |
| 天线 | 7,8 | ±3.2(仿真) | 很大 | 0.4~0.8 |
| 转身 body_yaw | 6 | ±2.8 | ±160° | 0.3~0.5 |

> ⚠️ 这些是**仿真**安全值,比真机大很多。真机上点头别超 ±15°(0.26 弧度),详见 `../docs/experiments/M1-B1-动作空间勘察报告.md` §4.2。演示若在仿真里,用上表没问题。

### 编一个全新动作(比如"疑惑地缓慢歪头 + 摆一下天线")

在 `build_shows()` 的列表里加一行,把几个基础动作**相加**(numpy 数组直接 `+`,同长才行)或**拼接**(`np.concatenate` 前后接):

```python
("疑惑", _hold(3, 0.3, 1.0, 1.5) + _osc(7, 0.4, 2.0, 2)[:_hold(3,0.3,1.0,1.5).shape[0]]),
#        歪头保持 1.5 秒          同时右天线摆两下(长度对齐)
```

或更简单,先歪头**再**摆天线(拼接):
```python
("疑惑2", np.concatenate([_hold(3, 0.3, 0.6, 1.0), _osc(7,0.5,3,3)+_osc(8,-0.5,3,3)])),
```

改完存盘,重跑 `python -u sim\demo_show.py 疑惑` 即可。**不用重启 daemon**(daemon 是本体,一直开着;你只是重跑客户端脚本)。

---

## 三、遇到问题怎么办(我踩过的坑,给你排雷)

### 坑 1:脚本一直停在"连接仿真...",好久不动

- **多半是正常的**——本机连接握手就是慢(10~52 秒)。先耐心等 1 分钟。
- 判断死活:3D 窗口机器人有没有动?没动但脚本没退出 = 还在连,继续等。

### 坑 2:等了 2 分钟还连不上(连接卡死)

我真遇到过。**根因**:上一个脚本被中途强杀(比如前台命令超时、Ctrl+C 打断在连接握手中途),会把 daemon 的连接管线搞进坏状态,导致后续新连接全卡住。

**恢复办法——重启 daemon:**
```powershell
# 1) 全杀
Get-Process python,reachy-mini-daemon -EA SilentlyContinue | Stop-Process -Force
# 2) 重新起 daemon(回到步骤 2),等就绪 + 静置 25 秒
# 3) 再跑脚本
```
干净重启后立刻就能连上(我验证过)。

**预防**:别在脚本"连接中"的时候强杀它;让它自己跑完或正常退出。

### 坑 3:中文/emoji 报 UnicodeEncodeError

跑任何脚本前先设 `$env:PYTHONUTF8 = 1`。

### 坑 4:别开两个 daemon、别频繁杀重启

一次只开一个 daemon。频繁杀重启会把音视频管线搞乱(仿真里我们不用音视频,影响小,但连接也可能受累)。正常用法:daemon 开着不动,反复重跑客户端脚本即可。

### 坑 5:代理劫持 localhost

脚本里已经设了 `os.environ["NO_PROXY"]="localhost,127.0.0.1"`。如果你自己写新脚本,记得在 `import reachy_mini` **之前**加这句,否则本机代理会拦截和 daemon 的通信。

---

## 四、和"训练出模型后"的关系(呼应 01 文档)

你现在手编的 `demo_show.py`,和将来模型输出,**执行层完全一样**:

```
现在:  你手编的动作函数  → [T,9] → set_target@30Hz → MuJoCo 窗口
将来:  训练好的动作专家  → [T,9] → set_target@30Hz → MuJoCo 窗口 / 真机
                                    └──── 你现在就在用的这套 ────┘
```

所以你现在把仿真玩熟、把"发 9 维动作"的手感建立起来,等模型训出来,你已经完全知道它是怎么"动起来"的了——无非是把"手编动作"换成"模型生成动作",管道一模一样。

---

## 五、三句话速记

1. **两半结构**:`daemon --sim`(本体,内含 MuJoCo)+ 你的脚本(客户端,用 SDK 发 `set_target`)。你不直接碰 MuJoCo。
2. **流程**:杀残留 → 起 daemon → 等就绪+静置 25 秒 → 另开窗口跑 `python -u sim\demo_show.py all`。连接慢(10~50s)是正常的,耐心等。
3. **改动作**:`demo_show.py` 的 `build_shows()` 里改数字调幅度/频率,加行编新动作,重跑脚本即可(不用重启 daemon)。
