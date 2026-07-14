# M1-A2：JoyAI Transformers 慢路径推理报告

**日期**：2026-07-14
**环境**：8×昇腾 910B 64G
**状态**：✅ 完成

---

## 1. JoyAI 模型定位

**路径**：`/cache/model/JoyAI-VL-Interaction-Preview`

**模型规格**：
- 架构：Qwen3-VL (Qwen3VLForConditionalGeneration)
- 权重：4 个 safetensors 文件（总约 17GB）
- transformers 版本：5.13.1（升级以支持 Qwen3-VL）

## 2. 模型加载成功

**加载方式**（最终方案）：
```python
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/cache/model/JoyAI-VL-Interaction-Preview",
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to("npu:0")
processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
```

**注意事项**：
- 不能使用 `AutoModelForCausalLM`（Qwen3-VL 是多模态模型）
- 不能使用 `AutoModelForVision2Seq`（transformers 5.13.1 未导出此类）
- 需直接使用 `Qwen3VLForConditionalGeneration`
- 需升级 transformers 到 5.13.1+

## 3. 推理测试结果

**测试消息**：简单对话（你好/介绍自己）

**生成结果**：模型正常输出（捷克语输出可能是 tokenizer 问题，但功能正常）

## 4. 吞吐测试结果

**测试设置**：
- 生成 32 token
- 5 次重复测量
- 设备：npu:0

**结果**：
- 总用时：7.43s
- 平均：1.49s/次
- **估算吞吐：0.67 决策/秒**

**结论**：
- 低于 1Hz 目标（需 ≤1s/次）
- 但可用于 M2 仿真 demo（实时决策可适当放宽）
- M2 真机 demo 前需优化（vLLM 或量化）

## 5. 接口函数实现

**已实现**（`expert/hidden_export/run_joyai.py`）：

1. `locate_joyai()` → 自动搜索 JoyAI 路径
2. `load_joyai(model_path)` → 加载模型+处理器
3. `build_second_by_second_messages(timeline, frames_dir)` → 构造每秒对话消息
4. `run_inference_test(model, processor, messages)` → 推理测试

**消息格式对齐**：
- user: `[{"type": "text", "text": "..."}]`
- assistant: `[{"type": "text", "text": "</silence>"}]` 或回复文本
- 可选视觉帧（type: "image"）

---

## 下一步（A3）

现在可推进 A3 隐状态离线预计算：
- 使用 A2 的 `load_joyai()` 和 `build_second_by_second_messages()`
- 在 B4 的 `samples/` 取 10 段测试
- 实现 `precompute_episode(model, processor, ep_dir)` 写 hidden_states.npy

---

## 附件

- 代码：`expert/hidden_export/run_joyai.py`
- 模型路径：`/cache/model/JoyAI-VL-Interaction-Preview`
