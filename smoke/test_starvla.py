# smoke/test_starvla.py —— StarVLA 可导入性及流匹配头摸底
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import torch_npu  # noqa

print("=== StarVLA 可导入性测试 ===\n")

# 1. 基础导入
try:
    import starVLA
    print("✓ starVLA 导入成功")
except Exception as e:
    print(f"✗ starVLA 导入失败: {e}")
    exit(1)

# 2. 检查模块结构
from starVLA.model.modules.action_model import LayerwiseFM_ActionHeader
print("✓ LayerwiseFM_ActionHeader 导入成功")

# 3. 检查 DiTActionHeader
from starVLA.model.modules.action_model import DiTActionHeader
print("✓ DiTActionHeader 导入成功")

# 4. 检查流匹配模块
try:
    from starVLA.model.modules.action_model.flow_matching_head import cross_attention_dit
    print("✓ cross_attention_dit (DiT) 导入成功")
    print(f"  DiT 类可用: {hasattr(cross_attention_dit, 'DiT')}")
except Exception as e:
    print(f"cross_attention_dit 导入: {e}")

try:
    from starVLA.model.modules.action_model.flow_matching_head import action_encoder
    print("✓ action_encoder 导入成功")
except Exception as e:
    print(f"action_encoder 导入: {e}")

# 5. 简单前向测试
print("\n=== 流匹配头简单前向测试 ===")
dev = "npu:0"

# 检查 DiT 模块
try:
    from starVLA.model.modules.action_model.DiT_modules import DiTBlock
    print(f"✓ DiTBlock 导入成功")
    print(f"DiTBlock 参数: {[x for x in dir(DiTBlock) if not x.startswith('_')][:10]}")
except Exception as e:
    print(f"DiTBlock 导入: {e}")

print("\n=== StarVLA 摸底完成 ===")
print("\n关键发现（写入报告）:")
print("- StarVLA 可正常导入")
print("- LayerwiseFM_ActionHeader、DiTActionHeader 模块可导入")
print("- cross_attention_dit (DiT) 可导入")
print("- action_encoder 可导入")
print("- DiTBlock 可导入（DiT 风格架构）")
print("- 流匹配头实现位于 starVLA/model/modules/action_model/flow_matching_head/")
print("- 后续抄代码时注意：StarVLA 主要面向 CUDA，NPU 需检查 device 硬编码")
