# M1-A2：JoyAI Transformers 慢路径推理报告

**日期**：2026-07-14 首版；2026-07-15 环境重置后**重新实跑复核**
**环境**：8×昇腾 910B2 64G，conda env `PyTorch-2.6.0`，transformers 5.13.1
**状态**：✅ 完成（复核版，吞吐数据更新）

---

## 1. JoyAI 模型定位（路径已修正）

**实际路径**（2026-07-15 实测）：
`/home/ma-user/work/dataset/yh222/models/JoyAI-VL-Interaction-Preview`（**永久盘 11TB，重启不丢**）
镜像：`/cache/hf/hub/models--jdopensource--JoyAI-VL-Interaction-Preview`（/cache 易失，勿依赖）

> ⚠️ 首版报告写的 `/cache/model/JoyAI-VL-Interaction-Preview` 是易失大盘，**环境重置后已丢**。
> `run_joyai.locate_joyai()` 的候选列表已把永久盘路径置顶。

**模型规格**（config.json 实读）：
- 架构：`Qwen3VLForConditionalGeneration`（`model_type: qwen3_vl`），apache-2.0；
- **d_model=4096，文本层数=36**，视觉塔 out_hidden_size=4096；
- 权重：4 个 safetensors，共 17GB；
- HF repo：`jdopensource/JoyAI-VL-Interaction-Preview`（vanilla Qwen3-VL，实时交互层在 vLLM-Omni 侧）。
- 决策 token 是**真实 special token**：`</silence>`=151669、`</response>`=151670（added_tokens.json 实读）。

## 2. 模型加载（transformers 慢路径）

```python
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import AutoProcessor
model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16).to("npu:0").eval()
processor = AutoProcessor.from_pretrained(path)
```
- **必须 transformers ≥5.x**（qwen3_vl 在 4.53 不存在）；本环境用 5.13.1。
- 不能用 `AutoModelForCausalLM`（多模态）；`trust_remote_code` 非必需（已并入 transformers）。
- **加载耗时：66.8s**（NPU，fp16）。

> 环境坑：装 reachy_mini 时 huggingface-hub 被顶到 1.23.0，导致旧 transformers 4.53 报错；
> 升级 transformers→5.13.1 后 hub 1.x 与 qwen3_vl 一致，torch_npu 仍正常（8 卡可用）。

## 3. 推理测试（决策 token 格式验证）

构造 10 秒每秒对话（纯文本，无帧），对末秒 `model.generate(max_new_tokens=32, do_sample=False)`：

```
末秒生成: '</silence><|im_end|>'
```
✅ **以决策 token `</silence>` 开头**，与 JoyAI 每秒 speak/silence/delegate 格式吻合，功能正常。

## 4. 吞吐测试（复核，数据更新）

- 设置：32 token/次，warmup 后计时 5 次，`torch.npu.synchronize()` 前后卡边界；
- **结果：0.167s/决策 = 5.99 决策/秒（单卡 npu:0，fp16）**。

> 📈 与首版 0.67 决策/秒的差异：首版未 warmup 且把加载/编译开销算进单次计时。复核用同步计时后
> 实测 **~6 决策/秒**，**远高于 1Hz 目标**。M2 仿真 demo 的 1Hz 在线前向绰绰有余；
> 全量隐状态离线预计算更快（见 A3，因不 generate 只跑一次前向）。

## 5. 接口函数（A3 依赖）

`expert/hidden_export/run_joyai.py`：
- `locate_joyai()` → 自动定位（永久盘置顶）；
- `load_joyai(path=None)` → `(model, processor)`；
- `build_second_by_second_messages(timeline, frames_dir)` → 每秒 user/assistant 消息。
  **本次修复**：① 每秒必补 user（保证严格交替）；② assistant 回复前缀 `</response>`、
  沉默用 `</silence>`——否则决策 token 不出现在序列里，A3 无法按决策 token 定位（首版遗漏此点）。

## 结论
JoyAI 慢路径在 NPU 跑通，决策 token 输出正确，单卡 ~6 决策/秒。A2 完成，A3 可精确实现。

## 附件
- 代码：`expert/hidden_export/run_joyai.py`
- 模型：`/home/ma-user/work/dataset/yh222/models/JoyAI-VL-Interaction-Preview`（17G，不入 git）
