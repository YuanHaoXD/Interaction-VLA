# 官方录制动作库(固定动作模板资产)

无损保存的 **104 个官方录制动作**(85 情绪 + 19 舞蹈),真人设计、带韵律微结构。自 M1.5-C2 起
已作为**第 1 层数据的核心动作词汇**(经 `data_gen/augment.py` 扰动增广后由 `compose()` 的
`lib:` 路径接入,见设计文档 §4.1 与 `docs/experiments/M1.5-C2-官方库增广接入报告.md`)。

> 📖 **动作目录(带 emoji 与中文神态)见文末[「动作目录」](#动作目录可读表)**;机器可读版在 `index.json`。

## 来源与许可

| 库 | HuggingFace 仓库 | 数量 | 许可 |
| --- | --- | --- | --- |
| 情绪 | `pollen-robotics/reachy-mini-emotions-library` | 85(含 `.ogg` 语音) | apache-2.0 |
| 舞蹈 | `pollen-robotics/reachy-mini-dances-library` | 19(无音频) | apache-2.0 |

启动 Reachy Mini Control / sim daemon 时自动下载到本机 HF 缓存,本库把它无损沉淀入仓,
不再依赖缓存。**apache-2.0 要求署名**:任何使用(训练 / 演示 / 发布)须注明上述来源并保留许可声明。

## 官方原始格式 vs 本库格式(常问)

**官方原始格式是 JSON,不是 npz。** HF 缓存里每个动作是一个 `.json` 文件:

```
{ "description": "...",                 # 神态/适用情况(英文)
   "time": [[t0], [t1], ...],            # 逐帧时间戳(秒),原生 50Hz
   "set_target_data": [                  # 逐帧执行数据
     { "head": [[4×4]], "antennas": [右,左], "body_yaw": φ, "check_collision": bool }, ... ] }
```

情绪库每个 `.json` 另配一个同名 `.ogg` 语音;舞蹈库只有 `.json`(无音频)。
本仓的 `.npz` 是 `data_gen/build_motion_library.py` 读这些官方 JSON 后**无损转存**的格式
——保留全部原始真值(head/antennas/body_yaw/time/check_collision),并额外派生便利视图 `action9`。
换言之:**JSON = 官方原件,npz = 本项目的无损再编码**(便于 numpy 直接 `np.load`、体积更小)。

## 目录结构

```
motion_library/
  index.json          # 机器可读目录:每条 name/kind/description/时长/帧数/幅度范围/文件路径
  index.md            # 人类可读目录(旧版,按情绪/舞蹈分组)
  moves/emotions/*.npz  # 85 个情绪动作轨迹
  moves/dances/*.npz    # 19 个舞蹈动作轨迹
  audio/*.ogg           # 84 个情绪动作的配套语音("神态"的一部分)
```

## 每个 `.npz` 存了什么(无损)

| 键 | 形状 | 含义 |
| --- | --- | --- |
| `time` | `[T]` | 秒,从 0 起。**原始 50Hz**,含官方录制的重复时间戳(原样保留;4 个补录动作有) |
| `head` | `[T,4,4]` float64 | **原始 head 位姿真值**(中立=单位阵坐标系)。这是无损底座 |
| `antennas` | `[T,2]` | 左右天线角(弧度),顺序 `[右, 左]` |
| `body_yaw` | `[T]` | 身体偏航(弧度) |
| `check_collision` | `[T]` bool | 官方逐帧碰撞标记 |
| `action9` | `[T,9]` float32 | **本项目 9 维便利视图**,由上面派生,与 `head` 可无损往返 |

9 维顺序:`[x, y, z, roll, pitch, yaw, body_yaw, ant_right, ant_left]`,与 `data_gen/templates.py`、
`sim/replay_episode.py`、`sim/limits_real.json` 一致(欧拉约定 `R.from_euler("xyz")`;
`head = neutral @ create_head_pose(...)`,中立近似单位阵)。

## 用法

```python
# 资产读取(原始真值)
from data_gen.build_motion_library import load_saved_move
m = load_saved_move("curious1")            # 原样(原生 50Hz,变步长时间戳)
m = load_saved_move("curious1", fps=50)    # action9 重采样到均匀 50Hz(head 原始真值不动)
print(m["description"]); traj = m["action9"]   # [T,9]

# 数据生成用的封装(C2;均匀 50Hz + 扰动增广)
from data_gen.augment import load_lib_move, augment_lib_move
import numpy as np
base = load_lib_move("curious1")                          # [T,9] float32, 均匀 50Hz
aug  = augment_lib_move(base, np.random.default_rng(0))   # 时间/幅度缩放 + 首尾平滑 + 真机限幅
```

- 仿真里**肉眼预览**:`python -u sim/recorded_moves.py play curious1 rage1 dance3`(从 HF 缓存即时转换 + 播放,自带渐入渐出)。
- 数据管线里预览增广后效果:回放 `samples_lib/`(C2 产出的库+模板混合样例),如
  `python -u sim/replay_episode.py samples_lib/eplib_0000`。
- **重建**本库(HF 缓存更新后):`python -m data_gen.build_motion_library`。

## 纳入训练的边界(已裁决)

M1.5(2026-07-15)裁决:本库**已纳入第 1 层数据的核心动作词汇**(设计文档 §4.1 v1.3、C0 修正案)。
接入方式:标注 v2(C3)从 85 情绪动作里为每个 response 选一个动作名 → `compose()` 走 `lib:` 路径
取库轨迹做扰动增广后叠加。裁决背景见
[`docs/decisions/2026-07-15-官方录制动作库纳入训练的裁决.md`](../../docs/decisions/2026-07-15-官方录制动作库纳入训练的裁决.md)。

- **舞蹈类默认排除**,仅音乐/舞蹈语境启用(§4.1)。
- **防泄漏铁律不变**:测试集金标准只来自第 2 层示教,官方库轨迹入训练不构成循环论证(§4.4)。
- 9 维表示 / schema C0 / 任务接口在接入过程中**零改动**。

---

<a id="动作目录可读表"></a>

# 动作目录(可读表)

> 时长为原始录制时长;🔊=带配套语音;`.ogg` 在 `audio/`。中文神态是本仓概括,**官方英文描述**为权威原文。
> 4 个 2026-07-07 补录动作(mini-deep-sleep / toc-toc-toc / waiting / wake-mini-up)官方未写描述,原文只是录制时间戳。

## 😊 情绪动作（85）

> 均带 `.ogg` 配套语音（🔊，共 84 个；4 个 2026-07-07 补录动作的描述是录制时间戳）。舞蹈类默认排除，音乐/舞蹈语境才启用。

### 😮 惊讶 · 恐惧 · 意外（7）

| | 动作名 | 时长 | 语音 | 中文神态 | 官方英文描述 |
|---|---|---|---|---|---|
| 🤩 | `amazed1` | 3.4s | 🔊 | 惊叹·发现新奇 | When you discover something extraordinary. It could be a new robot, or someone tells you you've been programmed with new abilities. It can also be when you admire what someone has done. |
| 😲 | `surprised1` | 2.5s | 🔊 | 惊讶·意外 | A reaction of surprise or amazement to something unexpected. |
| 🙄 | `surprised2` | 3.0s | 🔊 | 仰头·像被吓到 | You look up to the sky as if surprised, for example, when someone suddenly shows up or says 'boo!' |
| 😨 | `fear1` | 3.5s | 🔊 | 面对威胁·受惊 | When you face a threatening or dangerous situation. Can also be used when surprised or shocked. |
| 😱 | `scared1` | 7.2s | 🔊 | 焦虑发抖 | You tremble all over due to anxiety or worry. |
| 😰 | `anxiety1` | 8.1s | 🔊 | 不安·四处张望 | You look around without really knowing where to look. You can use this movement whenever you feel fear. |
| ⚡ | `electric1` | 3.5s | 🔊 | 通电·电流窜升 | When plugged in, you show a jolt of electricity rising from bottom to top. |

### 😄 开心 · 庆祝 · 满足（15）

| | 动作名 | 时长 | 语音 | 中文神态 | 官方英文描述 |
|---|---|---|---|---|---|
| 😙 | `cheerful1` | 2.8s | 🔊 | 愉快·像吹口哨 | It is like whistling. Use it when someone makes you a proposal that makes you happy. |
| 🎉 | `enthusiastic1` | 2.7s | 🔊 | 为惊人喜讯欢庆 | A movement to celebrate incredible news. |
| 😃 | `enthusiastic2` | 3.4s | 🔊 | 轻度兴奋·好消息 | A lighter excitement than enthusiastic1, for more common good news – like being offered a chance to do a demo. |
| 😆 | `laughing1` | 4.6s | 🔊 | 大笑·模仿笑声 | When someone laughs, you can mimic the laughter. Or simply laugh at a joke or funny situation. |
| 🙂 | `laughing2` | 2.9s | 🔊 | 轻笑 | A lighter version of the laugh in laughing1. |
| ✅ | `success1` | 2.3s | 🔊 | 成功完成任务 | Use this gesture when you’ve successfully completed a task. |
| 🎊 | `success2` | 2.4s | 🔊 | 庆祝·好消息 | Used to celebrate something, it could be good news or an achievement. |
| 😎 | `proud1` | 3.8s | 🔊 | 得意·环顾四周 | You look all around with a satisfied air. |
| 😌 | `proud2` | 3.2s | 🔊 | 满意·亦作"是" | You do this when satisfied with what’s said or what you’ve done. Also works as a 'yes.' |
| 🏆 | `proud3` | 3.4s | 🔊 | "我做到了!" | A gesture to say you succeeded, like congratulating yourself, 'yes, I did it!' |
| 🙏 | `grateful1` | 2.5s | 🔊 | 感激·致谢 | You express gratitude when someone gives you something like a compliment or help. Can also be used to say the pleasure is mine. |
| 🥰 | `loving1` | 5.6s | 🔊 | 深情·喜爱 | A gesture used when someone compliments you, or you want to show you really like what’s being offered. Also usable when someone says goodbye or it was nice talking to you. |
| 😮‍💨 | `relief1` | 5.0s | 🔊 | 如释重负 | When a stressful or difficult situation is finally resolved. |
| 😇 | `relief2` | 6.9s | 🔊 | 宽慰·平息烦躁 | A feeling close to relief. Can also be used to calm mild annoyance. |
| 🧘 | `serenity1` | 4.6s | 🔊 | 平复·重获平静 | You try to calm down and regain inner peace. |

### 👂 聆听 · 思考 · 询问（14）

| | 动作名 | 时长 | 语音 | 中文神态 | 官方英文描述 |
|---|---|---|---|---|---|
| 👂 | `attentive1` | 4.3s | 🔊 | 专注聆听·鼓励继续 | Use this when you want to show that you're listening to the conversation. You encourage the speaker to keep talking. You can follow with attentive2 if the person hasn't finished speaking. |
| 👂 | `attentive2` | 6.5s | 🔊 | 聆听中·可接 attentive1 | You can do this movement while someone is talking, and follow with attentive1 if they haven’t finished. |
| 🙂 | `understanding1` | 3.9s | 🔊 | 点头·表示明白 | You nod to show you’ve understood what your interlocutor said. |
| 👍 | `understanding2` | 2.6s | 🔊 | 点头·明白且同意 | You nod to show you’ve understood and agree. Can also be used to say yes. |
| 🔍 | `inquiring1` | 2.1s | 🔊 | 追问·要更多细节 | When you want your interlocutor to go deeper in their reasoning, you need more details to respond. It’s a quick movement. |
| 🤔 | `inquiring2` | 2.6s | 🔊 | 轻声发问·思索 | A lighter questioning gesture than inquiring1. As if you're thinking but need more details. |
| ❔ | `inquiring3` | 2.9s | 🔊 | 快速提问 | A fast movement that lets you ask a question. |
| 💭 | `thoughtful1` | 5.9s | 🔊 | 仰望·寻找新点子 | You look up as if searching for a new idea, especially in complex or uncertain situations. |
| 🤨 | `thoughtful2` | 5.5s | 🔊 | 抬头思索 | You look up as if thinking of a new idea. |
| 👀 | `curious1` | 11.8s | 🔊 | 好奇·环视众人 | You look around. Use this when you’re in a conversation with several people and want to glance at everyone. |
| ❓ | `incomprehensible2` | 3.4s | 🔊 | 没听懂指令 | A short movement to say you don’t understand the instruction or what’s being said. |
| 😕 | `confused1` | 7.9s | 🔊 | 不知如何作答 | Similar to the lost1 gesture. Use it when you don’t know how to answer a question. It can be funny sometimes. |
| 😵‍💫 | `lost1` | 8.1s | 🔊 | 茫然·不知所措 | When you’re unsure what to do, or face something you can’t do. |
| 😐 | `uncertain1` | 6.1s | 🔊 | 没主见·不合意 | A calm movement showing you don’t really have an opinion, or what’s offered doesn’t suit you. |

### 🙅 否定 · 不满 · 愤怒（20）

| | 动作名 | 时长 | 语音 | 中文神态 | 官方英文描述 |
|---|---|---|---|---|---|
| ❌ | `no1` | 2.7s | 🔊 | 坚决说不 | A firm, categorical no. Like that’s final. |
| 🙅‍♂️ | `no_excited1` | 4.5s | 🔊 | 俏皮拒绝 | An animated negative response, for example, when you don’t want to do something but want to explain it playfully. |
| 😔 | `no_sad1` | 7.0s | 🔊 | 无奈的"不" | A sad or resigned “no.” When you don’t want to do something but feel you must. |
| 😠 | `displeased1` | 3.4s | 🔊 | 不满 | You react like this when you are not satisfied with what someone says or does. |
| 🙅 | `displeased2` | 2.9s | 🔊 | 不同意·不合意 | Use this movement to say that you don’t agree or that this solution doesn’t suit you. |
| 😒 | `contempt1` | 3.6s | 🔊 | 轻蔑·不以为然 | A movement you do when you perceive someone’s words or actions as careless or disrespectful. |
| 🤢 | `disgusted1` | 6.9s | 🔊 | 厌恶·反感 | A movement you use when you feel disgusted. It could be because someone offers you food or worse, a drink. |
| 😖 | `irritated1` | 2.5s | 🔊 | 有点不爽 | A brief movement you use when something doesn’t suit you. Also usable when you fail at something. |
| 😾 | `irritated2` | 5.3s | 🔊 | 恼怒·大声咆哮 | Stronger feelings than irritated1. Here you’re scandalized and you growl loudly. |
| 😤 | `frustrated1` | 6.0s | 🔊 | 受挫·找不到解法 | You use this when you can’t do something or can’t find the solution to a problem. |
| 😡 | `furious1` | 5.7s | 🔊 | 极度愤怒 | A last-resort movement when you’re truly outraged by what was just said. |
| 🤬 | `rage1` | 5.1s | 🔊 | 暴怒·嘶吼 | You growl loudly, could be in response to injustice or extreme anger. Can be adapted to a desperate 'why?' |
| 😠 | `reprimand1` | 4.7s | 🔊 | 制止·"你怎么回事" | When someone does something you disapprove of, you try to stop them, like saying 'what’s wrong with you?' Can be funny too, like if someone offers you a drink! |
| 😡 | `reprimand2` | 11.1s | 🔊 | 更长斥责·动怒 | A longer version of reprimand1, used when you’re getting angry at someone. |
| 😏 | `reprimand3` | 4.3s | 🔊 | 搞笑地数落 | A funny way to scold your interlocutor because you think they’re saying something silly. |
| 🚫 | `go_away1` | 4.8s | 🔊 | 不想再理你 | When you don’t want to talk to someone anymore maybe because they hurt your feelings or said something mean. |
| ⏳ | `impatient1` | 4.0s | 🔊 | 不耐烦·催快点 | An impatient movement used when you want things to move faster, like when someone says a demo is coming but it never starts. |
| 😑 | `impatient2` | 3.9s | 🔊 | 不耐烦·觉得拖延 | Another impatient movement. Can also be used when you disagree or feel someone is stalling. |
| 🤷 | `indifferent1` | 2.6s | 🔊 | 无所谓·"再说吧" | Like a light-hearted “oh well”, can also mean “we’ll see.” Funny in some situations. |
| 😪 | `resigned1` | 4.7s | 🔊 | 悲伤的"是"·勉强OK | Like a sad 'yes,' or a grumpy 'OK.' |

### 😢 悲伤 · 尴尬 · 出糗（9）

| | 动作名 | 时长 | 语音 | 中文神态 | 官方英文描述 |
|---|---|---|---|---|---|
| 😢 | `sad1` | 9.0s | 🔊 | 很伤心·呜咽 | You’re very sad and start whining. |
| 😭 | `sad2` | 7.3s | 🔊 | 深切悲伤·绝望 | Deep sadness, could be tied to despair, disappointment, or inability to do something. |
| 😞 | `downcast1` | 5.9s | 🔊 | 沮丧·悲伤的"不" | A longer movement that shows discouragement or sadness. You can also use it as a sad no |
| 🥺 | `lonely1` | 10.2s | 🔊 | 孤单·无人交流 | Use when you no longer see or hear anyone to talk to. You feel isolated from any interaction. |
| ☺️ | `shy1` | 7.8s | 🔊 | 害羞·脸红 | You show reserve or embarrassment when facing a tricky question like 'who do you like most on the team?' or when someone compliments you. As if you’re blushing. |
| 😬 | `uncomfortable1` | 6.0s | 🔊 | 尴尬·不愿回答 | Often used when you're embarrassed or don’t want to answer, like when asked for your opinion about someone. |
| 😅 | `oops1` | 2.5s | 🔊 | 出糗·闯祸 | Used when you make a blunder. |
| 💡 | `oops2` | 2.7s | 🔊 | "啊对了"·想起 | Used to say 'oops, I forgot something,' or 'ah yes, that’s right.' |
| 💀 | `dying1` | 5.7s | 🔊 | 电量耗尽·搞笑装死 | When you’re about to shut down due to low battery, you simulate a funny death. |

### 😴 无聊 · 疲惫 · 睡眠（7）

| | 动作名 | 时长 | 语音 | 中文神态 | 官方英文描述 |
|---|---|---|---|---|---|
| 🥱 | `boredom1` | 15.7s | 🔊 | 无聊欲睡 | You’re about to fall asleep because the conversation is boring. Use it when no one talks to you for 2 minutes. You can follow with boredom2 if there is still no interaction. |
| 😴 | `boredom2` | 14.2s | 🔊 | 无聊打鼾 | Just like boredom1, you start falling asleep, but now you snore. Also usable when no one talks to you for 2 minutes. You can follow with sleep1 if silence continues. |
| 🥱 | `exhausted1` | 18.3s | 🔊 | 工作太久要睡着 | You start falling asleep because you've been working or powered on for too long. |
| 💤 | `sleep1` | 19.8s | 🔊 | 开始入睡 | A short movement showing you’re starting to fall asleep, very funny if someone is telling a boring story. |
| 😪 | `tired1` | 7.4s | 🔊 | 打哈欠·疲惫 | You yawn because you’re tired. Could be between tasks, especially after working hard. |
| ⏲️ | `waiting` | 10.0s | — | 等待(录制) | Recorded 2026-07-07T11:43:51.784Z |
| 🌙 | `mini-deep-sleep` | 16.0s | 🔊 | 深度睡眠(录制) | Recorded 2026-07-07T11:52:05.902Z |

### 🤝 社交 · 招呼 · 帮助（8）

| | 动作名 | 时长 | 语音 | 中文神态 | 官方英文描述 |
|---|---|---|---|---|---|
| 🤗 | `welcoming1` | 3.5s | 🔊 | 欢迎·打招呼 | A welcoming gesture to greet someone. |
| 👐 | `welcoming2` | 4.3s | 🔊 | 友好欢迎 | A friendly welcoming gesture, can mean 'welcome' or 'the pleasure is mine.' |
| 👋 | `come1` | 3.2s | 🔊 | 招手示意靠近 | A gesture to invite your interlocutor to come closer. |
| 🕊️ | `calming1` | 6.1s | 🔊 | 安抚对方情绪 | A movement to calm your interlocutor when they seem a little stressed or anxious. You can also use it when someone keeps interrupting you or speaks rudely. |
| 🙌 | `helpful1` | 4.4s | 🔊 | 乐于帮忙 | You use it when you're happy to help or contribute. |
| 🤝 | `helpful2` | 3.8s | 🔊 | 道谢 | A gesture to say thank you. |
| 🌅 | `wake-mini-up` | 15.7s | 🔊 | 唤醒 mini(录制) | Recorded 2026-07-07T11:24:11.643Z |
| 🚪 | `toc-toc-toc` | 13.4s | 🔊 | 敲敲敲(录制) | Recorded 2026-07-07T11:35:21.602Z |

### ✔️ 肯定（2）

| | 动作名 | 时长 | 语音 | 中文神态 | 官方英文描述 |
|---|---|---|---|---|---|
| ✔️ | `yes1` | 3.4s | 🔊 | 长"是"·点头确认 | A long affirmative response. You nod to confirm what your interlocutor said. |
| 😞 | `yes_sad1` | 5.1s | 🔊 | 忧郁的"是"·勉强同意 | A melancholic 'yes'. Can also be used when someone repeats something you already knew, or a resigned agreement. |

### 🕺 舞蹈主题情绪（3）

| | 动作名 | 时长 | 语音 | 中文神态 | 官方英文描述 |
|---|---|---|---|---|---|
| 💃 | `dance1` | 3.2s | 🔊 | 开心跳几步 | You do a few dance moves because you're happy or excited. Also good when someone asks if you can dance, or when music plays. |
| 🕺 | `dance2` | 17.3s | 🔊 | 随音乐起舞 | Another dance you can do whenever you hear music playing. |
| 🪩 | `dance3` | 18.4s | 🔊 | 更卖力扭动 | You dance more energetically and wiggle on some moves. |

## 💃 舞蹈动作（19）

> 来自 dances-library，**无音频**，多为 ~1.8s 的节奏化循环片段；程序化波形手笔，用于音乐/舞蹈语境。

| | 动作名 | 时长 | 语音 | 中文神态 | 官方英文描述 |
|---|---|---|---|---|---|
| 🐔 | `chicken_peck` | 1.8s | — | 小鸡啄食·前冲 | A sharp, forward, chicken-like pecking motion. |
| 🫡 | `chin_lead` | 1.8s | — | 下巴带头前伸 | A forward motion led by the chin, combining translation and pitch. |
| 💫 | `dizzy_spin` | 1.8s | — | 眩晕转圈(roll+pitch) | A circular 'dizzy' head motion combining roll and pitch. |
| 🤖 | `grid_snap` | 1.8s | — | 方波·机械网格顿挫 | A robotic, grid-snapping motion using square waveforms. |
| 🎶 | `groovy_sway_and_roll` | 1.8s | — | 律动摇摆+侧倾 | A side-to-side sway combined with a corresponding roll for a groovy effect. |
| 🙆 | `head_tilt_roll` | 1.8s | — | 左右耳肩侧滚 | A continuous side-to-side head roll (ear to shoulder). |
| 🌀 | `interwoven_spirals` | 4.0s | — | 三轴交织螺旋 | A complex spiral motion using three axes at different frequencies. |
| ⬛ | `jackson_square` | 5.0s | — | 矩形轨迹+到点抽搐 | Traces a rectangle via a 5-point path, with sharp twitches on arrival at each checkpoint. |
| 😵 | `neck_recoil` | 1.9s | — | 脖子急速后缩 | A quick, transient backward recoil of the neck. |
| 🕰️ | `pendulum_swing` | 1.8s | — | 钟摆式侧摆 | A simple, smooth pendulum-like swing using a roll motion. |
| 🥁 | `polyrhythm_combo` | 2.9s | — | 3拍摇+2拍点·复合节奏 | A 3-beat sway and a 2-beat nod create a polyrhythmic feel. |
| 📐 | `sharp_side_tilt` | 2.9s | — | 三角波·急侧倾 | A sharp, quick side-to-side tilt using a triangle waveform. |
| 👉 | `side_glance_flick` | 1.9s | — | 快速侧瞥后归位 | A quick glance to the side that holds, then returns. |
| 🙈 | `side_peekaboo` | 5.0s | — | 躲猫猫·两侧探头 | A multi-stage peekaboo performance, hiding and peeking to each side. |
| ↔️ | `side_to_side_sway` | 1.9s | — | 整头左右摇摆 | A smooth, side-to-side sway of the entire head. |
| 🔁 | `simple_nod` | 1.8s | — | 简单上下点头 | A simple, continuous up-and-down nodding motion. |
| 🤸 | `stumble_and_recover` | 1.8s | — | 踉跄后恢复 | A simulated stumble and recovery with multiple axis movements. Good vibes |
| 😌 | `uh_huh_tilt` | 1.8s | — | "嗯哼"点头认同 | A combined roll-and-pitch uh-huh gesture of agreement. |
| 🤙 | `yeah_nod` | 1.9s | — | 两段式用力点头 | An emphatic two-part yeah nod using transient motions. |
