# expert/hidden_export/precompute.py —— JoyAI 隐状态离线预计算(精确版)
"""
为每个 episode 生成 hidden_states.npy

格式：
- hidden_states.npy: [N_decisions, d_model] float32
- N_decisions = timeline 中决策秒数（每秒一个决策：</silence> 或 </response>）
- d_model = JoyAI 最后一层隐藏维度（4096）

方法（设计文档 §3.3，M1-A3 精确实现）：
- transformers 慢路径：teacher forcing 整段跑一遍 `model(..., output_hidden_states=True)`
- 决策 token 是**单 token**：`</silence>`=151669、`</response>`=151670
- 在整段 token 序列里按这两个 id 定位每秒决策位置，取 last-layer 在该位置的 hidden state
- 一次前向并取全部秒 → 不需要逐秒 generate

⚠️ 与首版(2026-07-14 简化版)的区别：首版用 np.linspace 均匀采样占位,并非真正的决策
token 位置;本版按决策 token id 精确定位,是设计文档要求的"判定通过"标准实现。
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict

from .run_joyai import load_joyai, build_second_by_second_messages

# 决策 token id（来自 JoyAI added_tokens.json，已实测）
SILENCE_ID = 151669   # </silence>
RESPONSE_ID = 151670  # </response>
DECISION_IDS = (SILENCE_ID, RESPONSE_ID)


def locate_decision_positions(input_ids: np.ndarray) -> List[int]:
    """
    在整段 token 序列里精确定位每个决策 token 的位置。

    Args:
        input_ids: teacher-forcing 整段的 token id 序列 [seq_len]

    Returns:
        决策 token 位置列表(升序),每个对应一秒的 </silence> 或 </response>
    """
    a = np.asarray(input_ids)
    mask = np.isin(a, DECISION_IDS)
    return np.where(mask)[0].tolist()


def precompute_episode(model, processor, ep_dir: str, verbose: bool = True) -> int:
    """
    为单个 episode 预计算隐状态,写 <ep_dir>/hidden_states.npy 并回写 meta.d_model。

    Returns:
        N_decisions(写入的决策数),便于调用方统计。
    """
    import torch

    ep_path = Path(ep_dir)
    meta = json.loads((ep_path / "meta.json").read_text(encoding="utf-8"))
    timeline = json.loads((ep_path / "timeline.json").read_text(encoding="utf-8"))

    # 构造每秒 user/assistant 消息(带决策 token 前缀)
    frames_dir = ep_path / "frames" if (ep_path / "frames").exists() else None
    messages = build_second_by_second_messages(
        timeline, frames_dir=str(frames_dir) if frames_dir else None
    )

    # teacher forcing:整段一次前向(不 generate)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0].detach().cpu().numpy()

    # 精确定位决策 token
    positions = locate_decision_positions(input_ids)
    if not positions:
        raise RuntimeError(
            f"{meta['episode_id']}: 序列中未找到任何决策 token(</silence>/</response>)。"
            "检查 build_second_by_second_messages 是否带了决策 token 前缀。"
        )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # last-layer hidden: [seq_len, d_model]
    last_layer = outputs.hidden_states[-1][0]
    d_model = last_layer.shape[-1]

    # 取决策 token 位置的隐状态 → [N_decisions, d_model]
    idx = torch.tensor(positions, device=last_layer.device)
    hidden = last_layer.index_select(0, idx).float().cpu().numpy()

    assert not np.isnan(hidden).any(), f"{meta['episode_id']}: 隐状态含 NaN"

    np.save(ep_path / "hidden_states.npy", hidden.astype(np.float32))
    meta["d_model"] = int(d_model)
    meta["n_decisions"] = int(len(positions))
    (ep_path / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    if verbose:
        n_sil = int((input_ids[positions] == SILENCE_ID).sum())
        n_resp = int((input_ids[positions] == RESPONSE_ID).sum())
        print(f"  ✓ {meta['episode_id']}: hidden {hidden.shape} "
              f"(沉默 {n_sil} / 回复 {n_resp}), seq_len={len(input_ids)}")
    return len(positions)


def batch_precompute(model_dir: str, data_dir: str, n_max: int = 10):
    """批量预计算隐状态(顺序,单卡)。全量多卡并行见报告 §吞吐。"""
    import time

    print("=== JoyAI 隐状态离线预计算(精确版) ===\n")
    print("加载 JoyAI 模型...")
    model, processor = load_joyai(model_dir)
    print("✓ 模型加载完成\n")

    data_path = Path(data_dir)
    episodes = sorted(p for p in data_path.iterdir()
                      if p.is_dir() and p.name.startswith("ep_"))[:n_max]
    print(f"处理 {len(episodes)} 个 episode\n")

    ok, total_dec, t0 = 0, 0, time.time()
    for i, ep_dir in enumerate(episodes, 1):
        print(f"[{i}/{len(episodes)}] ", end="", flush=True)
        try:
            total_dec += precompute_episode(model, processor, str(ep_dir))
            ok += 1
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()

    dt = time.time() - t0
    print(f"\n=== 完成 {ok}/{len(episodes)} 段, {total_dec} 决策, "
          f"耗时 {dt:.1f}s ({dt/max(ok,1):.2f}s/段, {ok/dt*60:.1f}段/分·卡) ===")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="JoyAI 隐状态离线预计算(精确版)")
    parser.add_argument("--model", default=None, help="JoyAI 路径(None 则自动定位)")
    parser.add_argument("--data", required=True, help="数据集目录(含 episode 子目录)")
    parser.add_argument("--n-max", type=int, default=10, help="最大处理 episode 数")
    args = parser.parse_args()
    batch_precompute(args.model, args.data, args.n_max)


if __name__ == "__main__":
    main()
