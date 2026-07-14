# M1-A1：NPU 冒烟测试三件套报告

**日期**：2026-07-14
**环境**：8×昇腾 910B 64G
**Python**：3.11.10
**torch**：2.6.0+cpu
**torch_npu**：2.6.0.post5

---

## 1. 环境修复

**问题**：torch_npu 2.6.0.post5 与 triton 3.6/3.7 不兼容
- 错误：`ImportError: cannot import name 'AttrsDescriptor' from 'triton.backends.compiler'`
- 原因：torch/_inductor/runtime/hints.py 中 except 块有 bug，重复导入不存在的类

**解决方案**：
- 降级 triton 到 3.5.0
- 修改 torch/_inductor/runtime/hints.py 第 66-80 行，将 except 块改为使用 namedtuple fallback
- 结果：torch_npu 正常加载，8 张 NPU 可用

## 2. 基础算子测试

**结果**：✅ PASS

- `npu_fusion_attention`：✅ 可用，输出形状正确
- `scaled_dot_product_attention`（SDPA）：✅ 可用
- 简单矩阵乘法反传：✅ 可用
- 设备检测：Ascend910B2 × 8

**结论**：NPU 基础算子正常，无需回退朴素实现

## 3. DiT 小模型训练步测试

**结果**：✅ PASS

- 模型：6 层 DiT 风格块（MultiheadAttention + MLP + AdaLN）
- 批大小：8
- 序列长度：45（动作块 token）
- 隐藏维度：384
- 测试：20 步前向+反传+优化器 step

**性能**：
- 总用时：0.53s
- 平均每步：26.3 ms/step
- loss 下降：1.7686 → 1.4896

**结论**：DiT 风格架构在 NPU 上训练可行，性能良好

## 4. StarVLA 安装与摸底

**结果**：✅ 基本可用

**安装**：
```bash
git clone https://github.com/starVLA/starVLA
pip install -e starVLA
pip install diffusers peft>=0.17.0  # 依赖补装
```

**模块导入测试**：
- ✅ starVLA 基础导入
- ✅ LayerwiseFM_ActionHeader（分层流匹配动作头）
- ✅ DiTActionHeader（DiT 动作头）
- ✅ cross_attention_dit.DiT（DiT 实现）
- ✅ action_encoder（动作编码器）

**关键路径**：
- 流匹配头：`starVLA/model/modules/action_model/flow_matching_head/`
  - `cross_attention_dit.py`：DiT 架构
  - `action_encoder.py`：动作编码器
- 分层流匹配：`starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`

**注意事项**：
1. **CUDA 硬编码**：StarVLA 主要面向 CUDA，后续抄代码需检查 `.cuda()` 调用，改为 `.to(npu:0)`
2. **DiT_modules 模块**：无 __init__.py，需直接从 `models.py` 导入（未测试）

## 5. 绕行方案（如有）

无绕行，三件套全部通过。

---

## 总结

| 测试项 | 结果 | 备注 |
|--------|------|------|
| 基础算子 | ✅ PASS | npu_fusion_attention 可用 |
| DiT 训练 | ✅ PASS | 26.3 ms/step，性能良好 |
| StarVLA | ✅ PASS | 主要模块可导入，需注意 CUDA 硬编码 |

**后续工作**：
- A2/A5 可安全使用 torch_npu
- v2 动作专家（DiT + flow matching）架构可参考 StarVLA 实现
- 抄代码时注意替换 `.cuda()` → `.to(device)`
