# smoke/npu_dit.py —— 一个 6 层 DiT 风格块在 NPU 上的前向+反传+优化器 step
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import time
import torch
import torch.nn as nn
import torch_npu  # noqa

dev = "npu:0"
print(f"使用设备: {torch.npu.get_device_name(0)}")


class Block(nn.Module):
    """简化的 DiT 风格块：自注意力 + MLP + AdaLN 条件注入"""

    def __init__(self, d=384, h=6):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d)
        )
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.ada = nn.Linear(d, 2 * d)  # AdaLN 风格条件注入

    def forward(self, x, c):
        # 条件调制
        g, b = self.ada(c).chunk(2, -1)
        h = self.n1(x) * (1 + g[:, None]) + b[:, None]

        # 自注意力
        x = x + self.attn(h, h, h, need_weights=False)[0]

        # MLP
        return x + self.mlp(self.n2(x))


print("\n初始化 6 层 DiT 风格网络...")
net = nn.ModuleList([Block() for _ in range(6)]).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4)

x = torch.randn(8, 45, 384, device=dev)  # 45 步动作块 token
c = torch.randn(8, 384, device=dev)  # 条件向量

print(f"输入形状: x={x.shape}, c={c.shape}")

# 预热
print("\n预热中...")
for blk in net:
    x = blk(x, c)
loss = x.pow(2).mean()
loss.backward()
opt.step()
opt.zero_grad()

# 正式测试
print("\n开始 20 步训练测试...")
t0 = time.time()
losses = []
for i in range(20):
    h = torch.randn(8, 45, 384, device=dev)
    c = torch.randn(8, 384, device=dev)
    for blk in net:
        h = blk(h, c)
    loss = h.pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    losses.append(loss.item())
elapsed = time.time() - t0

print(f"\n=== DiT 冒烟测试结果 ===")
print(f"20 步完成, 用时 {elapsed:.2f}s")
print(f"平均每步 {elapsed/20*1000:.1f} ms")
print(f"最终 loss: {losses[-1]:.4f}")
print(f"loss 趋势: {losses[0]:.4f} -> {losses[-1]:.4f}")
print("\n=== DiT 冒烟测试通过 ===")
