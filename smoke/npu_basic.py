# smoke/npu_basic.py —— torch_npu 基础与融合注意力可用性
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import torch_npu  # noqa

dev = "npu:0"
print(f"PyTorch: {torch.__version__}")
print(f"torch_npu: {torch_npu.__version__}")
print(f"NPU 可用: {torch.npu.is_available()}")
print(f"NPU 数量: {torch.npu.device_count()}")
print(f"当前设备: {torch.npu.get_device_name(0)}")

# 基础张量操作
x = torch.randn(4, 8, 128, 64, device=dev, dtype=torch.float16)
print(f"\n输入张量: {x.shape}, {x.dtype}, {x.device}")

# 尝试 npu_fusion_attention
try:
    out = torch_npu.npu_fusion_attention(x, x, x, head_num=8, input_layout="BNSD")[0]
    print("npu_fusion_attention: OK", out.shape)
except Exception as e:
    print("npu_fusion_attention: FAIL ->", e, "(回退朴素 attention,记录性能差距)")

# 朴素 SDPA（备选）
y = torch.nn.functional.scaled_dot_product_attention(x, x, x)
print("sdpa: OK", y.shape)

# 简单前向+反传测试
a = torch.randn(10, 10, device=dev, requires_grad=True)
b = torch.randn(10, 10, device=dev)
c = a @ b
loss = c.sum()
loss.backward()
print(f"\n简单矩阵乘法反传: OK, grad shape: {a.grad.shape}")

print("\n=== NPU 基础冒烟测试通过 ===")
