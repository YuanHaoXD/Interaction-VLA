#!/usr/bin/env bash
# smoke/setup_npu_env.sh —— 复现 NPU 冒烟环境(M1-A1)
#
# 背景:本机 ModelArts Notebook 的 conda 环境 PyTorch-2.6.0 里,
#   torch 2.6.0 的 _inductor 会在 import 时尝试从 triton 导入 AttrsDescriptor,
#   而 triton 3.5/3.6 均已删除该符号 → torch_npu 自动加载失败,整个 NPU 不可用。
# 环境重置(/、/tmp 重启即丢;site-packages 被还原)后此问题会复现,故脚本化。
#
# 用法: bash smoke/setup_npu_env.sh   (在 conda env PyTorch-2.6.0 下)
set -e
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"

echo "[1/4] 固定 triton==3.5.0(与 torch 2.6.0 inductor 期望一致)"
pip install "triton==3.5.0" $MIRROR

echo "[2/4] 打补丁:torch/_inductor/runtime/hints.py 的 AttrsDescriptor 回退"
python - <<'PY'
import re, pathlib, torch, collections
f = pathlib.Path(torch.__file__).parent / "_inductor" / "runtime" / "hints.py"
s = f.read_text()
marker = "# NPU 适配补丁(M1-A1)"
if marker in s:
    print("  已打过补丁,跳过"); raise SystemExit
# 把内层 `except ImportError: from triton.compiler.compiler import AttrsDescriptor ...` 整块
# 替换为 namedtuple 回退(与 triton 不可用分支同构)
bad = """    except ImportError:
        from triton.compiler.compiler import AttrsDescriptor

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
        ):
            # Prepare the arguments for AttrsDescriptor
            kwargs = {
                "divisible_by_16": divisible_by_16,
                "equal_to_1": equal_to_1,
            }

            # Instantiate AttrsDescriptor with the prepared arguments
            return AttrsDescriptor(**kwargs)"""
good = """    except ImportError:
        # NPU 适配补丁(M1-A1):triton 3.5/3.6 均不再导出 AttrsDescriptor,
        # 原代码在此二次 import 会再次 ImportError 并使 torch_npu 加载失败。
        # 回退为 namedtuple(与 triton 不可用分支同构),不影响 eager/NPU 路径。
        AttrsDescriptorWrapper = collections.namedtuple(  # type: ignore[no-redef, name-match]
            "AttrsDescriptor",
            ["divisible_by_16", "equal_to_1"],
            defaults=[(), ()],
        )"""
assert bad in s, "hints.py 结构与预期不符,请手动检查(可能 torch 版本变了)"
f.write_text(s.replace(bad, good))
print("  补丁完成:", f)
PY

echo "[3/4] 安装 StarVLA(零件库,--no-deps)+ 冒烟所需轻量依赖"
STARVLA_DIR="/home/ma-user/work/dataset/yh222/tools/starVLA"
if [ ! -d "$STARVLA_DIR" ]; then
  mkdir -p "$(dirname "$STARVLA_DIR")"
  git clone --depth 1 https://github.com/starVLA/starVLA "$STARVLA_DIR"
fi
pip install -e "$STARVLA_DIR" --no-deps $MIRROR
pip install "diffusers" "peft>=0.17.0" einops timm omegaconf $MIRROR

echo "[4/4] 验证 torch_npu 可用"
python -c "import torch, torch_npu; assert torch.npu.is_available(); print('NPU OK, 卡数=', torch.npu.device_count())"
echo "=== 环境就绪:可跑 smoke/npu_basic.py / npu_dit.py / test_starvla.py ==="
