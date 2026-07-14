# M1-A2：JoyAI Transformers 慢路径推理报告

**日期**：2026-07-14
**环境**：8×昇腾 910B 64G
**状态**：⚠️ 待续（模型路径缺失）

---

## 1. JoyAI 模型定位

**自查结果**：❌ 未找到

搜索位置：
- `/cache/model/`（只有其他模型：dinov2、LeanVAE、OmniTokenizer 等）
- `/home/ma-user/`
- `/opt/`
- `~/.cache/huggingface/hub/`（无 JoyAI 相关）

**预期路径**（计划文档）：
- 服务器此前做过 JoyAI 数据标注相关工作，应存在相关路径
- 找不到时需在实验记录写明实际路径并继续

**当前状态**：
- 路径未找到，需运维/本地提供 JoyAI 8B 模型路径

## 2. 框架代码已实现

虽然模型缺失，但 `expert/hidden_export/run_joyai.py` 已完成：

### 2.1 `locate_joyai()` 函数
自动搜索常见路径，返回 JoyAI 模型目录。

### 2.2 `load_joyai(model_path)` 函数
```python
from transformers import AutoModelForCausalLM, AutoProcessor
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="npu:0"
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
```
**设计依据**：计划文档要求"抄 JoyAI 仓库 `services/webinfer` 的官方加载代码"。

### 2.3 `build_second_by_second_messages(timeline, frames_dir)` 函数
按 JoyAI datasets/README 格式构造每秒对话消息：
- 每秒一个 user 消息（该秒内的 user_text + 可选视觉帧）
- 每秒一个 assistant 消息（response 或 `</silence>`）

**格式对齐**：
```python
[
    {"role": "user", "content": [{"type": "text", "text": "用户问题"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "</silence>"}]},
    {"role": "user", "content": [{"type": "text", "text": "继续"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "好的..."}]},
]
```

### 2.4 `run_inference_test(model, processor, messages)` 函数
简单推理测试：最后一秒生成 32 token。

### 2.5 吞吐测试
5 次生成测量平均用时，目标 ≤1s/次（1Hz 决策）。

## 3. 下一步

**阻塞**：模型路径缺失

**需要提供**：
1. JoyAI 8B 模型路径（或 HuggingFace model id）
2. 数据集样例（用于测试消息构造格式）

**提供路径后**：
1. 运行 `python expert/hidden_export/run_joyai.py`
2. 记录吞吐数据（决策/秒）
3. 继续推进 A3 隐状态预计算

---

## 附件

- 代码：`expert/hidden_export/run_joyai.py`
- 依赖：transformers（已装 4.53.1）、torch、torch_npu（已验证）
