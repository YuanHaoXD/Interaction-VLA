# 官方录制动作库(固定动作模板资产)

无损保存的 **104 个官方录制动作**(85 情绪 + 19 舞蹈),真人设计、带韵律微结构,是本项目
的固定动作模板资产 / 第 1 层数据的候选真人轨迹源。

## 来源与许可

| 库 | HuggingFace 仓库 | 数量 | 许可 |
| --- | --- | --- | --- |
| 情绪 | `pollen-robotics/reachy-mini-emotions-library` | 85(含 `.ogg` 语音) | apache-2.0 |
| 舞蹈 | `pollen-robotics/reachy-mini-dances-library` | 19(无音频) | apache-2.0 |

启动 Reachy Mini Control / sim daemon 时自动下载到本机 HF 缓存,本库把它无损沉淀入仓,
不再依赖缓存。**apache-2.0 要求署名**:任何使用(训练 / 演示 / 发布)须注明上述来源并保留许可声明。

## 目录结构

```
motion_library/
  index.json          # 机器可读目录:每条 name/kind/description/时长/帧数/幅度范围/文件路径
  index.md            # 人类可读目录:按情绪/舞蹈分组,含每条"神态 / 适用情况"
  moves/emotions/*.npz  # 85 个情绪动作轨迹
  moves/dances/*.npz    # 19 个舞蹈动作轨迹
  audio/*.ogg           # 84 个情绪动作的配套语音("神态"的一部分)
```

## 每个 `.npz` 存了什么(无损)

| 键 | 形状 | 含义 |
| --- | --- | --- |
| `time` | `[T]` | 秒,从 0 起。**原始 50Hz**,含官方录制的重复时间戳(原样保留) |
| `head` | `[T,4,4]` float64 | **原始 head 位姿真值**(中立=单位阵坐标系)。这是无损底座 |
| `antennas` | `[T,2]` | 左右天线角(弧度),顺序 `[右, 左]` |
| `body_yaw` | `[T]` | 身体偏航(弧度) |
| `check_collision` | `[T]` bool | 官方逐帧碰撞标记 |
| `action9` | `[T,9]` float32 | **本项目 9 维便利视图**,由上面派生,与 `head` 可无损往返 |

9 维顺序:`[x, y, z, roll, pitch, yaw, body_yaw, ant_right, ant_left]`,与 `sim/limits.json`、
`data_gen/templates.py`、`sim/replay_episode.py` 一致(欧拉约定 `R.from_euler("xyz")`;
`head = neutral @ create_head_pose(...)`,中立近似单位阵)。

## 用法

```python
from data_gen.build_motion_library import load_saved_move

m = load_saved_move("curious1")            # 原样 50Hz
m = load_saved_move("curious1", fps=30)    # action9 重采样到 30Hz(head 原始真值不动)
print(m["description"])                     # "You look around. Use this when..."
traj = m["action9"]                         # [T,9]
```

- 想在仿真里**肉眼预览**这些动作:用 `sim/recorded_moves.py`(从 HF 缓存即时转换 + 30Hz 播放,
  自带渐入渐出),例如 `python -u sim/recorded_moves.py play curious1 rage1 dance3`。
- 想**重建**本库(HF 缓存更新后):`python -m data_gen.build_motion_library`。

## 边界(重要)

**本库只是"资产保存"。** 它是否 / 如何纳入训练模板库,是设计文档 §4 数据设计的"大事"
(涉及算第几层、§4.4 测试集防泄漏铁律、标签体系、署名),尚**未裁决**——详见
[`docs/decisions/2026-07-15-官方录制动作库纳入训练的裁决.md`](../../docs/decisions/2026-07-15-官方录制动作库纳入训练的裁决.md)。

在裁决前:本库**不被** `data_gen/templates.py`、`schema.py` 或任何数据生成管线导入,
9 维表示 / schema / 任务接口零改动。
