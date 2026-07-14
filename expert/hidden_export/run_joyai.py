# expert/hidden_export/run_joyai.py —— JoyAI transformers 慢路径推理跑通
"""
依赖：JoyAI 8B 模型权重与代码

服务器路径自查结果（2026-07-14）：
- 未在 /cache、/home、/opt 找到 JoyAI 仓库或权重
- 需运维/本地提供路径后继续

计划加载方式（抄官方 webinfer 代码）：
```python
from transformers import AutoModelForCausalLM, AutoProcessor
model = AutoModelForCausalLM.from_pretrained(
    <joyai_path>,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="npu:0"
)
processor = AutoProcessor.from_pretrained(<joyai_path>, trust_remote_code=True)
```
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from typing import List, Dict, Optional
from pathlib import Path

def locate_joyai() -> Optional[str]:
    """定位服务器上 JoyAI 模型路径"""
    candidates = [
        "/cache/model/JoyAI",
        "/home/ma-user/JoyAI",
        "/opt/JoyAI",
        "~/.cache/huggingface/hub/models--JoyAI",
    ]
    for cand in candidates:
        p = Path(cand).expanduser()
        if p.exists():
            return str(p)
    return None


def load_joyai(model_path: Optional[str] = None):
    """
    加载 JoyAI 8B 模型与处理器

    Args:
        model_path: JoyAI 模型路径（为 None 时自动搜索）

    Returns:
        (model, processor) 元组
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    path = model_path or locate_joyai()
    if path is None:
        raise RuntimeError(
            "JoyAI 模型路径未找到。请手动指定 model_path 或检查以下位置：\n"
            "  - /cache/model/JoyAI\n"
            "  - /home/ma-user/JoyAI\n"
            "  - ~/.cache/huggingface/hub/models--*JoyAI*"
        )

    print(f"加载 JoyAI 模型: {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="npu:0"
    )
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    print(f"✓ 模型加载完成，设备: {model.device}")
    return model, processor


def build_second_by_second_messages(timeline: List[Dict], frames_dir: Optional[str] = None) -> List[Dict]:
    """
    按 JoyAI datasets/README 格式构造每秒对话消息

    Args:
        timeline: 事件列表，每项含 {t, type, text, ...}
        frames_dir: 视觉帧目录（每秒一帧，命名 frame_%06d.jpg）

    Returns:
        消息列表，每秒一个 {"role": "user"/"assistant", "content": [...]}
    """
    import math

    messages = []
    duration_s = max((ev["t"] for ev in timeline), default=0) + 5
    n_seconds = math.ceil(duration_s)

    # 按 type 分组事件
    user_events = [e for e in timeline if e["type"] == "user_text"]
    response_events = [e for e in timeline if e["type"] == "response"]

    for sec in range(n_seconds):
        # 用户输入（该秒内的 user_text）
        user_texts = [e["text"] for e in user_events if sec <= e["t"] < sec + 1]

        # 视觉帧
        if frames_dir:
            frame_path = f"{frames_dir}/frame_{sec:06d}.jpg"
            frame_content = {"type": "image", "image": frame_path}
        else:
            # 无帧时用纯文本模式
            frame_content = None

        # 构造 user 消息
        content = []
        if frame_content:
            content.append(frame_content)
        if user_texts:
            content.extend([{"type": "text", "text": t} for t in user_texts])

        if content:
            messages.append({"role": "user", "content": content})

        # assistant 响应（该秒内的 response 或 silence）
        resp = [e for e in response_events if sec <= e["t"] < sec + 1]
        if resp:
            # 有回复
            texts = [e["text"] for e in resp]
            content = [{"type": "text", "text": "\n".join(texts)}]
            messages.append({"role": "assistant", "content": content})
        else:
            # 沉默
            messages.append({"role": "assistant", "content": [{"type": "text", "text": "</silence>"}]})

    return messages


def run_inference_test(model, processor, messages: List[Dict]):
    """简单推理测试"""
    import torch

    print("\n=== 推理测试 ===")
    print(f"消息数: {len(messages)}")

    # 简单测最后一秒生成
    prompt = processor.apply_chat_template(messages[-2:], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt").to(model.device)

    print("生成中...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False
        )

    generated = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"生成结果: {generated[:200]}...")


def main():
    """主函数：定位、加载、测试"""
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"NPU 可用: {torch.npu.is_available()}")

    # 1. 定位模型
    model_path = locate_joyai()
    if model_path:
        print(f"✓ 找到 JoyAI 路径: {model_path}")
    else:
        print("⚠ 未找到 JoyAI 模型路径")
        print("   请在服务器部署 JoyAI 模型或手动指定路径")
        print("   预期路径：/cache/model/JoyAI 或 ~/.cache/huggingface/hub/models--*JoyAI*")
        return

    # 2. 加载模型
    try:
        model, processor = load_joyai(model_path)
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return

    # 3. 构造测试消息
    timeline = [
        {"t": 1.0, "type": "user_text", "text": "你好"},
        {"t": 3.0, "type": "response", "text": "你好！有什么我可以帮助你的吗？", "est_speech_dur_s": 2.5},
        {"t": 5.0, "type": "user_text", "text": "介绍一下你自己"},
        {"t": 7.0, "type": "silence"},
    ]
    messages = build_second_by_second_messages(timeline)

    # 4. 推理测试
    try:
        run_inference_test(model, processor, messages)

        # 5. 吞吐测量
        print("\n=== 吞吐测试 ===")
        import time
        n = 5
        t0 = time.time()
        for _ in range(n):
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        elapsed = time.time() - t0
        print(f"{n} 次生成用时 {elapsed:.2f}s，平均 {elapsed/n:.2f}s/次")
        print(f"估算吞吐: {1/(elapsed/n):.2f} 决策/秒（1Hz 目标需 ≤1s/次）")

    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
