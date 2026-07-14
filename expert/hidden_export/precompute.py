# expert/hidden_export/precompute.py —— JoyAI 隐状态离线预计算
"""
为每个 episode 生成 hidden_states.npy

格式：
- hidden_states.npy: [N_decisions, d_model] float32
- N_decisions = timeline 中决策秒数（每秒一个决策）
- d_model = JoyAI 最后一层隐藏维度（4096）

方法（设计文档 §3.3）：
- transformers 路径：model(..., output_hidden_states=True)
- 取 last-layer 在决策 token（</silence></response> 首 token）处的 hidden state
- 一次前向可并取全部秒的位置（teacher forcing 整段跑一遍）
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from .run_joyai import load_joyai, build_second_by_second_messages


def locate_decision_positions(messages: List[Dict]) -> List[int]:
    """
    定位每秒决策位置（assistant 消息首 token 位置）

    Args:
        messages: 消息列表，偶数索引 user，奇数索引 assistant

    Returns:
        每秒决策的 token 位置列表
    """
    positions = []
    # 每秒对应一个 assistant 消息（奇数索引）
    for i in range(1, len(messages), 2):
        # 简化：假设每秒 assistant 消息从该秒 token 序列末尾开始
        # 实际需要 tokenizer 后统计，这里先估算
        # TODO: 精确实现需 tokenize 后计算每消息长度
        positions.append(i)  # 占位
    return positions


def precompute_episode(model, processor, ep_dir: str) -> None:
    """
    为单个 episode 预计算隐状态

    Args:
        model: JoyAI 模型
        processor: JoyAI 处理器
        ep_dir: episode 目录路径
    """
    import torch

    ep_path = Path(ep_dir)

    # 读取 episode 数据
    meta = json.loads((ep_path / "meta.json").read_text(encoding="utf-8"))
    timeline = json.loads((ep_path / "timeline.json").read_text(encoding="utf-8"))

    print(f"处理 {meta['episode_id']}：{len(timeline)} 个事件")

    # 构造消息
    frames_dir = ep_path / "frames" if (ep_path / "frames").exists() else None
    messages = build_second_by_second_messages(timeline, frames_dir=str(frames_dir) if frames_dir else None)

    # Tokenize 整段对话
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    # 前向并取 hidden states
    print("  前向中...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 取最后一层 hidden states: [batch, seq_len, d_model]
    last_layer_hidden = outputs.hidden_states[-1][0]  # [seq_len, d_model]

    # 定位决策位置
    # 简化实现：假设每秒决策在对应 assistant 起始位置
    # 实际需要根据 </silence></response> token 位置精确定位
    n_seconds = int(meta["duration_s"])
    d_model = last_layer_hidden.shape[-1]

    # TODO: 精确定位决策 token 位置
    # 临时方案：均匀采样 seq_len 中 n_seconds 个点
    decision_positions = np.linspace(0, len(last_layer_hidden) - 1, n_seconds, dtype=int)

    # 提取决策点隐状态
    hidden_states = last_layer_hidden[decision_positions].cpu().numpy()  # [n_seconds, d_model]

    # 写入文件
    np.save(ep_path / "hidden_states.npy", hidden_states.astype(np.float32))

    # 更新 meta.json
    meta["d_model"] = int(d_model)
    meta["n_decisions"] = int(n_seconds)
    (ep_path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"  ✓ 写入 hidden_states.npy: {hidden_states.shape}")


def batch_precompute(model_dir: str, data_dir: str, n_max: int = 10):
    """
    批量预计算隐状态

    Args:
        model_dir: JoyAI 模型目录
        data_dir: 数据集根目录（含 episode 子目录）
        n_max: 最大处理 episode 数（默认 10，测试用）
    """
    from .run_joyai import load_joyai

    print("=== JoyAI 隐状态离线预计算 ===\n")

    # 加载模型
    print("加载 JoyAI 模型...")
    model, processor = load_joyai(model_dir)
    print(f"✓ 模型加载完成\n")

    # 扫描 episode 目录
    data_path = Path(data_dir)
    episodes = [p for p in data_path.iterdir() if p.is_dir() and p.name.startswith("ep_")]

    print(f"找到 {len(episodes)} 个 episode")

    # 限制处理数量
    episodes = episodes[:n_max]
    print(f"处理前 {len(episodes)} 个\n")

    # 逐个处理
    for i, ep_dir in enumerate(episodes, 1):
        print(f"[{i}/{len(episodes)}] ", end="", flush=True)
        try:
            precompute_episode(model, processor, str(ep_dir))
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n=== 预计算完成 ===")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="JoyAI 隐状态离线预计算")
    parser.add_argument("--model", default="/cache/model/JoyAI-VL-Interaction-Preview", help="JoyAI 模型路径")
    parser.add_argument("--data", required=True, help="数据集目录（含 episode 子目录）")
    parser.add_argument("--n-max", type=int, default=10, help="最大处理 episode 数")
    args = parser.parse_args()

    batch_precompute(args.model, args.data, args.n_max)


if __name__ == "__main__":
    main()
