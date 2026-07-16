# JoyAI-VL-Interaction 动作标注

给 HF 数据集 `jdopensource/JoyAI-VL-Interaction` 的每条 `response[].content` 打一个动作标签
（`nod` / `shake_head` / `wiggle_antennas` / `tilt_head` / `none`），用"一步"平台（OpenAI 兼容接口）
调 Gemini 做**纯文本语义判断**。产出带 `action` 字段的干净数据，供后续 `</act_xxx>` marker 转换 +
resize embedding(+4) SFT 使用。

> 完整背景、设计决策、后续训练步骤见 [`../docs/数据集标注任务.md`](../docs/数据集标注任务.md)。
> 本目录是该交接文档的**代码落地**。

## 目录

```
action-annotation/
├── annotate_actions_api.py     # 标注主脚本
├── local_api_logger/           # API 调用日志库（每次调用都记录，硬性要求）
├── JoyAI-VL-Interaction/       # HF 数据集克隆（git-lfs，见下）—— 已 gitignore
├── annotated/                  # 标注输出（镜像输入目录结构）—— 已 gitignore
├── api_logs/                   # API 调用日志（自动生成）—— 已 gitignore
├── .env.example                # 复制为 .env 填凭证（脚本会自动加载 .env）
└── requirements.txt            # openai + ijson（流式读大文件）
```

## 环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

Copy-Item .env.example .env         # 然后编辑 .env 填 BASE_URL / API_KEY
```

脚本启动时会**自动加载同目录的 `.env`**（纯标准库，不覆盖已有环境变量）。所以配置只需写进 `.env` 即可，
无需手动 `$env:...`。命令行/shell 里显式设的环境变量优先级更高（方便临时覆盖）。

## 数据布局（重要）

HF 克隆下来是**按 task_type 分子目录**的（不是文档假设的 `raw_data/` 平铺）：

```
JoyAI-VL-Interaction/
├── background/background.json
├── chat/chat_shard_01..06.json
├── narration/narration.json
└── event_grounding/event_grounding.json
```

脚本的 `--task_types` 过滤是按**路径片段**匹配的，所以这套嵌套布局天然适配，无需搬动文件。
直接把 `--input` 指向克隆目录即可，输出会镜像成 `annotated/background/background.json` 等。

> ⚠️ 用 git-lfs 克隆时，未下完的 json 是 ~134 字节的 **LFS 指针占位**。必须等
> `git lfs pull` / clone 全部完成（文件恢复成几百 MB）再跑，否则 `json.loads` 会报错。
> 检查：`ls -lh JoyAI-VL-Interaction/*/*.json`，都应是几十~几百 MB 而非 134B。

## 跑法

```powershell
# 1) 先小测 2000 条，验证凭证/模型/标签分布/调用健康度
python annotate_actions_api.py --input .\JoyAI-VL-Interaction --sample_limit 2000

# 2) 分文件跑（推荐，降内存压力）
python annotate_actions_api.py --input .\JoyAI-VL-Interaction --task_types background
python annotate_actions_api.py --input .\JoyAI-VL-Interaction --task_types chat
python annotate_actions_api.py --input .\JoyAI-VL-Interaction --task_types narration
python annotate_actions_api.py --input .\JoyAI-VL-Interaction --task_types event_grounding

# 3) 查看 API 调用统计（成本/token/成功率）
python -c "from local_api_logger import print_stats_summary; print_stats_summary()"
```

**流式处理（内存 O(CHUNK_SAMPLES)）**：用 `ijson` 流式读输入，标注结果先按「一行一个 sample」追加进
`<输出>.json.jsonl` 工作文件，整文件跑完再流式合成下游要的 `.json` 数组、删掉工作文件。所以
`background.json` / `narration.json` 这种 ~580MB 大文件**不再整文件 `json.loads`、也不再整文件重写
checkpoint**，几 GB 内存 / O(n²) I/O 的问题都没了。已实测。

**断点续跑**：直接重跑同一命令即可。
- 输出 `.json` 已存在 = 该文件上轮已跑完 → 跳过。
- 只剩工作文件 `.json.jsonl` = 上次跑到一半 → 数出已完成的 sample、跳过它们、只补剩下的（崩溃留下的
  半截尾行会被自动丢弃重标）。中途重启不丢已标数据；`--no_resume` 强制重标。**实测续跑结果与一次跑完逐字节一致。**

> 支持"多问题多回复"样本（`question[i] ↔ response[i]` 按位置配对）、`response` 的 flat 与 nested
> （`[[{...}]]`）两种布局、以及同一 sample 内混入 `</delegation>`（直接标 `none`、不调 API）。均已实测。

## 关键参数（环境变量，均有默认）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_NAME` | `gemini-2.5-flash` | 标注模型 |
| `MAX_WORKERS` | `20` | 并发；频繁 429 就降到 10 |
| `MAX_TOKENS` | `512` | 输出上限；给漏关思考的 ~10% 留余量（见「关思考」） |
| `THINK_OFF` | `1` | 默认关思考（yibuapi 上实测有效、省约 7×）；设 `0` 开思考 |
| `THINKING_BUDGET` | `0` | 关思考时的 thinking_budget 值 |
| `REASONING_EFFORT` | 空 | yibuapi 上**无效**，留空；仅换平台时可能用到 |
| `CHUNK_SAMPLES` | `500` | 每读入并标注多少 sample 落一次盘（内存 / checkpoint 粒度） |

> 连通性与关思考已在 `https://yibuapi.com/v1` + `gemini-2.5-flash` 上实测通过（5/5 标注正确、
> 关思考经 SDK 9/10 生效）。`.venv/` 已建好并装了 `openai`，直接用 `.\.venv\Scripts\python.exe` 跑即可。

## ⚠️ 思考模型的坑（务必看小测的"LLM 调用健康度"）

Gemini 2.5 系列是**思考模型**。若思考 token 吃光 `MAX_TOKENS`，会返回**空 content**
（`finish_reason == "length"`）。原始交接脚本会把空返回**无声兜底成 `none`**，且标签分布看起来
一切正常——这会造成**大面积静默误标**。

本版做了三件事防这个坑：
1. 检测空/截断返回 → **自动加大预算（×4）重试一次**；
2. 重试后仍失败的单独计为 `empty`，结尾"LLM 调用健康度"里**醒目列出**；
3. 若 `empty + error` 占比 > 2%，打印告警并给对策（调大 `MAX_TOKENS` 或设 `REASONING_EFFORT=none`）。

**小测后先看这块**：`ok` 应占绝大多数；`empty` 高就说明思考 token 在坑你，全量前必须先解决
（否则 950 万次里一大片会被误标 `none`）。

## 分阶段与验收（对齐设计文档 §4.5/4.6）

服务器端 vLLM 版的官方流程，API 版沿用：

| 阶段 | 规模 | 目的 | 通关标准 |
|---|---|---|---|
| **S1 校准** | 1-2 千条 | 人工抽检、迭代 prompt | 分布合理（none 占多数）、抽检准确率 ≥ 85% |
| **S2 验证** | 5-10 万 | 验证分布稳定性 | 分布与 S1 一致、抽检稳定 |
| **S3 全量** | 全部（~1000 万 response） | 出最终数据资产 | 断点续跑跑完、终检抽检 |

**人工抽检**：每个动作类 + none 各抽 50-100 条，S1 必做。

预期标签分布（服务器 vLLM 版 S1 实测参考）：

```
none            ~87%   （设计文档下限：应 ≥50%）
wiggle_antennas ~10%
nod             ~2%
shake_head      ~0.5%
tilt_head       ~0.1%
```

`none` 不占多数、或**任一动作类 > 25%**（报警阈值），说明 prompt 或调用有问题，需排查。

## 关思考（已实测确定方案，默认开启）

源头 vLLM 版（`JoyAI-VL-Interaction/datasets/annotate_actions.py`）用 Qwen3-32B、temp=0、
`enable_thinking=False`——关思考是设计刚需。API 版在 yibuapi 网关上**实测**了各种关法：

| 方法 | 是否生效 |
|---|---|
| OpenAI `reasoning_effort: none/low/minimal` | ❌ 网关忽略，照样思考 |
| body 顶层 `google: {thinking_config:...}` | ❌ 无效 |
| **body 顶层字面 `extra_body: {google:{thinking_config:{thinking_budget:0}}}`** | ✅ ~90% 生效（reasoning_tok=0） |
| Gemini 原生端点 `x-goog-api-key` 头 + `thinkingConfig.thinkingBudget=0` | ✅ 生效（另一条调用路径） |

**结论 / 已落地**：脚本默认 `THINK_OFF=1`，经 openai SDK 的 `extra_body` 参数产出上面那个字段。
实测对比（gemini-2.5-flash，temp=0）：

```
不关思考： 每条 ~250 reasoning token，输出 ~243 token，3-6s
关思考：   ~90% 调用 reasoning=0、输出 ~6-14 token，2-3s；剩 ~10% 网关漏关仍思考
```

- 判断质量**等价**（实测每次"错"都是 token 截断兜底成 none，不是标错）。
- 约 **省 7× 输出 token**、更快，且与 vLLM 原版一致。
- `MAX_TOKENS=512` 是给那漏关的 ~10%（最多 ~350 token）留的余量，让它们一次跑完、不触发重试；
  输出按实际计费，设大不额外花钱。**不要**因为"关了思考"就把它降到 32——漏关的那 10% 会被截断。

## 后续阶段的下游文件（本次不做，仅登记，来自设计文档 §5）

- 格式转换：`JoyAI-VL-Interaction/datasets/convert_data.py` → `action` 透传为 `</act_xxx>` marker
- adapter 解析：`JoyAI-VL-Interaction/services/webinfer/live_adapter.py:203-226`
- reachy 对接：`reachy-mini-demo/voice/realtime.py` / `voice/d01_realtime_chat.py`（按 action 调动作函数）
- 4 个动作定义源头：`reachy-mini-demo/voice/config.py:242-267`

## 相对交接文档的改动

代码整体忠实于 `docs/数据集标注任务.md` 第 6/7 节，仅做以下**针对性修正**（详见各文件注释）：
- 默认模型 `gemini-2.5-pro` → `gemini-2.5-flash`；
- 空/截断检测 + 加大预算重试 + 健康度告警（防静默误标 `none`）；
- `MAX_TOKENS` / `REASONING_EFFORT` 环境变量化；
- 修 `local_api_logger.set_log_dir` 对 `log_completion` 静默失效的 latent bug（改为原地改 `log_dir`）；
- 补齐平台专有的可重试错误串（对齐 `JoyAI-VL-Interaction/api_call/api_test.py`）：429 分支加
  `负载`/`饱和`，5xx 分支加 `unexpected end`/`bad_response_body`/`upstream_error`——漏了它们会
  把本该重试的错误当成不可重试 → 静默标 `none`；
- **流式处理**（`ijson` 读 + JSONL 工作文件增量写 + 收尾合成 `.json` 数组）：替换原版整文件
  `json.loads` + 每 200 条整文件重写 checkpoint——那套在 ~580MB 大文件上会吃几 GB 内存、且 O(n²) I/O，
  百万级样本不可行。现在内存 O(CHUNK_SAMPLES)、追加式落盘、按 sample 崩溃续跑；顺带根治了原始 vLLM 版
  重跑覆盖已标结果的 bug（现在续跑只追加、从不整文件重写，实测与一次跑完逐字节一致）；
- 加 `.env` 自动加载（纯标准库，脚本同目录 `.env`，不覆盖已有环境变量）；
- 单文件 `--input` 修正（指到单个 `.json` 时输出路径不再出错）。
