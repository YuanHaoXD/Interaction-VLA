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
print("\n=== 流匹配头真前向测试(NPU) ===")
dev = "npu:0"

# 直接实例化 DiT 流匹配头,映射到本项目条件流:
#   输出 9 维动作 / 45 步动作块 / 条件=语义隐状态序列(投影到 inner_dim)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

dit = DiT(num_attention_heads=8, attention_head_dim=64, output_dim=9,
          num_layers=6, cross_attention_dim=512, compute_dtype=torch.float32).to(dev)
B, T, S, D = 4, 45, 8, 512
x = torch.randn(B, T, D, device=dev)            # 噪声动作块 token
cond = torch.randn(B, S, D, device=dev)         # 语义条件(隐状态序列)
tstep = torch.randint(0, 1000, (B,), device=dev)
with torch.no_grad():
    out = dit(x, cond, timestep=tstep)
o = out[0] if isinstance(out, (tuple, list)) else out
assert tuple(o.shape) == (B, T, 9), o.shape
assert not torch.isnan(o).any()
print(f"✓ DiT 前向 OK, 输出 {tuple(o.shape)} (期望 (4,45,9)), 无 NaN")

print("\n=== StarVLA 摸底完成 ===")
print("\n关键发现（写入报告）:")
print("- StarVLA 可正常导入(pip install -e . --no-deps + 补 diffusers/peft>=0.17)")
print("- LayerwiseFM_ActionHeader、DiTActionHeader、cross_attention_dit、action_encoder 均可导入")
print("- DiT 流匹配头可在 NPU 上真前向,输出可配成 (B,45,9) 直接对齐我们的动作块")
print("- gaussian_diffusion 类 DiT_modules 无 __init__ 导出,需从 models.py 直接导入(非 DiTBlock)")
print("- 唯一 CUDA 硬编码:spike_action_model_multitimestep.py:446 torch.device('cuda:0');")
print("  抄流匹配头主线(cross_attention_dit/LayerwiseFM)不碰它")
