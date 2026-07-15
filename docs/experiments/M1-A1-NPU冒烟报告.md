# M1-A1：NPU 冒烟测试三件套报告

**日期**：2026-07-14 首版；2026-07-15 环境重置后**重新实跑复核**（本文为复核版）
**环境**：华为云 ModelArts Notebook（cn-north-9），8×昇腾 910B2 64G，Python 3.11.10，conda env `PyTorch-2.6.0`
**torch**：2.6.0+cpu（eager，走 torch_npu 后端）
**torch_npu**：2.6.0.post5
**复现脚本**：`smoke/setup_npu_env.sh`（一键把环境修好，见 §1）

---

## 1. 环境修复（可复现，已脚本化）

**问题**：`import torch_npu` 失败 →
`ImportError: cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'`，
最终 `RuntimeError: Failed to load the backend extension: torch_npu`。

**根因**：torch 2.6.0 的 `torch/_inductor/runtime/hints.py` 在 import 期尝试从 triton 导入
`AttrsDescriptor`；本环境 triton 为 3.6.0（降到 3.5.0 也一样），该符号在两个候选路径
（`triton.backends.compiler`、`triton.compiler.compiler`）都已被删除，内层 `except` 二次
import 再次抛错，冒泡使 torch_npu 后端自动加载整体失败 → 8 张卡全部不可用。

**修复**（两步，`smoke/setup_npu_env.sh` 自动完成）：
1. `pip install triton==3.5.0`（对齐 torch 2.6.0 inductor 预期版本）；
2. 打补丁 `hints.py`：把内层 `except ImportError` 分支从"再次 import 不存在的符号"改为
   **namedtuple 回退**（与该文件"triton 不可用"分支同构）。只影响 inductor 编译路径，
   eager/NPU 执行不受影响。

⚠️ **注意**：补丁落在 conda 的 site-packages 里，**环境重置会被还原**。重置后重跑
`bash smoke/setup_npu_env.sh` 即可（幂等：已打过补丁会跳过）。

修复后：`torch.npu.is_available()=True`，`device_count()=8`，`get_device_name(0)=Ascend910B2`。

## 2. 基础算子冒烟（`smoke/npu_basic.py`）

**结果**：✅ PASS

| 项 | 结果 |
| --- | --- |
| `npu_fusion_attention`（BNSD, head=8） | ✅ OK，输出 `[4,8,128,64]` |
| `scaled_dot_product_attention`（SDPA 备选） | ✅ OK |
| 矩阵乘法前向+反传 | ✅ OK |

**结论**：融合注意力可用，**无需**回退朴素 attention。

## 3. DiT 小模型训练步冒烟（`smoke/npu_dit.py`）

**结果**：✅ PASS

- 6 层 DiT 风格块（MultiheadAttention + MLP + AdaLN），batch=8，seq=45（动作块步数），d=384；
- 20 步 前向+反传+AdamW step；
- **性能：0.53s / 20 步 = 26.5 ms/step**（与首版 26.3 ms 一致）；
- loss 1.7493 → 1.4355，正常下降。

**结论**：DiT 风格架构在 NPU 上训练可行，性能良好；v2 动作专家（DiT+流匹配）无算子障碍。

## 4. StarVLA 安装与摸底（`smoke/test_starvla.py`）

**结果**：✅ PASS（比首版更进一步：跑通**真前向**，非仅 import）

**安装方式**（避免拖入 deepspeed/pytorch3d/flash-attn 等重 CUDA 依赖）：
```bash
git clone --depth 1 https://github.com/starVLA/starVLA  # → tools/starVLA(永久盘)
pip install -e tools/starVLA --no-deps
pip install diffusers "peft>=0.17.0" einops timm omegaconf
```
> `diffusers` 强制 `peft>=0.17.0`（本环境原装 0.14.0 → 报错，必须升级）。

**导入**：`starVLA` / `LayerwiseFM_ActionHeader` / `DiTActionHeader` /
`flow_matching_head.cross_attention_dit.DiT` / `action_encoder` 全部 ✅。

**真前向（新增，NPU 上跑通）**：直接实例化 `DiT` 流匹配头，配成本项目条件流——
`output_dim=9`（9 维动作）、`num_layers=6`、`cross_attention_dim=512`（语义隐状态条件）：
```
输入: 噪声块 x=[4,45,512], 条件 cond=[4,8,512], timestep=[4]
输出: [4,45,9]  ✅ 恰好对齐我们的动作块 (B,45,9), 无 NaN, 参数量 ~23M
```
**这直接验证了 v2 的可行性**：StarVLA 的 DiT 头改 `output_dim=9` 即可产出我们的动作块。

**抄代码时的注意点（写给 v2）**：
1. **CUDA 硬编码仅一处**：`action_model/spike_action_model_multitimestep.py:446`
   `torch.device("cuda:0" ...)`；主线流匹配头（`cross_attention_dit` / `LayerwiseFM_ActionHeader`）
   **不含** `.cuda()` 硬编码，抄它们不受影响。
2. `DiT_modules`（gaussian_diffusion 一支）无 `__init__` 导出，需从其 `models.py` 直接导入；
   首版报告写的 `DiTBlock` 导入路径有误，已在脚本中修正。
3. `TimestepEncoder` 有一处 `self.config.compute_dtype` 的 eval 期 BUG（源码 TODO 标注），
   实例化时显式传 `compute_dtype=` 规避。

## 5. 三件套结论

| 测试项 | 结果 | 关键数字 / 备注 |
| --- | --- | --- |
| 基础算子 | ✅ PASS | npu_fusion_attention 可用 |
| DiT 训练步 | ✅ PASS | 26.5 ms/step，loss 正常下降 |
| StarVLA | ✅ PASS | DiT 流匹配头 NPU 真前向 → (4,45,9)，23M 参数 |

**后续**：A2/A5 可安全用 torch_npu；v2 动作专家可直接借 StarVLA 的 `cross_attention_dit.DiT`
（改 `output_dim=9`）作为流匹配解码器骨架，抄代码只需避开那一处 `cuda:0` 硬编码。

## 附：踩坑与耗时

- 主要耗时在环境修复（triton 版本 + hints.py 根因定位）与 StarVLA 依赖裁剪（diffusers→peft 版本链）。
- 首版报告的 triton 降级+patch **未持久化**（环境重置被还原），本次已脚本化为 `setup_npu_env.sh`，
  杜绝重复踩坑。
