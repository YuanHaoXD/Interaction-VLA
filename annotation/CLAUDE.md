# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 本目录延续父仓库的中文优先约定:文档、注释、commit 均用中文。

## 这个目录是什么

`action-annotation/` 是 `docs/数据集标注任务.md` 交接文档的**代码落地**:给 HF 数据集
`jdopensource/JoyAI-VL-Interaction` 的每条 `response[].content` 打一个动作单标签
(`nod` / `shake_head` / `wiggle_antennas` / `tilt_head` / `none`),用"一步"平台(yibuapi,
OpenAI 兼容网关)调 Gemini 做**纯文本语义判断**。产出带 `action` 字段的干净数据,供下游
`</act_xxx>` marker 转换 + resize embedding(+4) SFT。

⚠️ 定位(2026-07 用户澄清):**不是"被 VLA 方案取代的一次性 toy"**。这套 5 类动作的
**分类标准 + 判定口径**("Yep/口语确认→nod"、delegation/caption→`none`、不确定偏 `none`)是后续
VLA **连续动作标注要遵循的语义锚**——现在定"哪条 response 配哪种动作",VLA 阶段再把离散标签细化成
连续轨迹,两阶段必须共用同一套动作词表与触发口径。所以:①这些口径是要长期引用的地基,别当废稿;
②全量应用**同一 model/prompt/config** 一致跑完才有前瞻价值。当前上层设计见
`../interaction-vla/docs/2026-07-14-交互VLA设计文档_zh.md`(父 `MEMORY.md` 里"toy 前置"应理解为
"动作标注地基",非"废弃")。

> 本目录**不是** git 仓库(`git rev-parse` 会失败)。父容器根目录也不是 git 仓库;唯一的
> git 仓库是 `../reachy-mini-demo/`。这里的产物靠 `.gitignore` + 数据目录排除来管理。

## 环境与运行

**必须用本目录 `.venv` 的 python**(装了 `openai` + `ijson`):

```powershell
.\.venv\Scripts\python.exe annotate_actions_api.py --input .\JoyAI-VL-Interaction --sample_limit 2000
```

- 配置只需写进 `.env`(从 `.env.example` 复制)。脚本启动**自动加载同目录 `.env`**(纯标准库,
  不覆盖已有环境变量;shell 里显式 `$env:` 优先)。无需手动 export。
- 依赖装法(清华镜像):`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
- 无 pytest / lint 框架。验证靠"小测 2000 条 → 看健康度与分布"这一人工流程。

**典型全量流程**(README §跑法):先 `--sample_limit 2000` 小测校准,再按 `--task_types`
分文件跑(`background` / `chat` / `narration` / `event_grounding`,降内存),最后
`.\.venv\Scripts\python.exe -c "from local_api_logger import print_stats_summary; print_stats_summary()"`
看成本/token/成功率。

## 数据布局(易踩坑)

HF 克隆是**按 task_type 分子目录**的(不是文档假设的 `raw_data/` 平铺):
`JoyAI-VL-Interaction/{background,chat,narration,event_grounding}/*.json`,单文件可达 ~580MB。
`--task_types` 是按**路径片段**匹配(`f.parts`),所以嵌套布局天然适配,`--input` 直接指克隆目录、
输出镜像成 `annotated/background/background.json`。

- ⚠️ git-lfs 未拉完的 json 是 ~134 字节的 **LFS 指针占位**,`json.loads` 会报错。跑前查
  `ls -lh JoyAI-VL-Interaction/*/*.json` 都应是几十~几百 MB。
- 样本结构:`question[i] ↔ response[i]` **按位置配对**(不依赖 `time` 字段)。`response` 有 flat 和
  nested(`[[{...}]]`)两种布局;`iter_response_dicts` 统一处理。标注即在每个 response dict 原地加
  `action` 字段,其余字段透传。

## 代码架构(big picture)

单主脚本 `annotate_actions_api.py`,核心是**流式 + 崩溃安全续跑**的设计,替换了原始 vLLM 版整文件
`json.loads` + 每 200 条整文件重写 checkpoint(那套在 580MB 文件上吃几 GB 内存、O(n²) I/O):

- **读**:`ijson.items(fin, "item")` 流式读,内存 O(`CHUNK_SAMPLES`)。
- **写**:每标完一个 sample,以"一行一个 compact JSON"**追加**进 `<out>.json.jsonl` 工作文件
  (`_resume_from_jsonl` / `flush_chunk`)。整文件跑完再 `_finalize_to_json_array` 流式合成下游要的
  `.json` 数组、删工作文件。
- **续跑**:重跑同命令即可。`.json` 已存在=已跑完→跳过;只剩 `.json.jsonl`=跑到一半→数已完成行、
  截断崩溃留下的半截尾行、只补剩下的。`--no_resume` 强制重标。**续跑只追加、从不整文件重写**,
  实测与一次跑完逐字节一致。
- **并发**:`ThreadPoolExecutor(max_workers)`,共享一个全局 `OpenAI` client(线程安全,`get_client` 加锁)。
- **配对/兜底**:`</delegation>` 样本直接标 `none` 不调 API;输入自带 `action` 的直接留用。

**local_api_logger/**:轻量本地 LLM 调用日志库(JSONL,零配置)。`annotate` 每次调用都
`log_completion(...)`(硬性要求),日志落到 `../api_logs/`(经 `set_log_dir` 设置)。公开入口:
`log_completion` / `set_log_dir` / `print_stats_summary`。

## 两个必须理解的陷阱(改代码前务必读)

1. **思考模型静默误标 `none`**:Gemini 2.5 是思考模型。若思考 token 吃光 `MAX_TOKENS`,返回**空 content**
   (`finish_reason == "length"`),原始交接脚本会**无声兜底成 `none`**,且标签分布看起来正常→大面积静默误标。
   本版对策(`label_one_task` + 结尾"LLM 调用健康度"):检测空/截断→自动 ×4 预算重试一次→仍失败计为
   `empty` 并醒目告警。**小测后先看这块**:`ok` 应占绝大多数;`empty+error > 2%` 会打印告警。

2. **关思考只有一种有效写法**(yibuapi 网关实测):唯一生效的是请求体顶层放**字面 `extra_body` 字段**
   `{"extra_body":{"google":{"thinking_config":{"thinking_budget":0}}}}`。经 openai SDK 实现方式是把它
   整体作为 SDK 的 `extra_body` 参数(SDK 会摊平进 body 根→根出现 `extra_body` 键,见 `call_api`)。
   **OpenAI 的 `reasoning_effort`、顶层 `google` 键在本网关上均无效**。默认 `THINK_OFF=1` 省约 7× 输出 token。
   `MAX_TOKENS=512` 是给网关漏关思考的 ~10% 调用留的余量——**不要**因"关了思考"就把它降到 32。

## 关键参数(环境变量,均有默认,详见 README 表)

`MODEL_NAME`(默认 `gemini-2.5-flash`)、`MAX_WORKERS`(20,频繁 429 降到 10)、`MAX_TOKENS`(512)、
`THINK_OFF`(1)、`THINKING_BUDGET`(0)、`CHUNK_SAMPLES`(500,落盘/checkpoint 粒度)。
`REASONING_EFFORT` 在 yibuapi 上**无效**,留空,仅换平台时可能用到。

## 验收标准(对齐设计文档 §4.5/4.6)

S1 校准(1-2 千)→ S2 验证(5-10 万)→ S3 全量(~1000 万 response)。预期分布:`none` ~87%(应占多数)、
`wiggle_antennas` ~10%、`nod` ~2%、`shake_head` ~0.5%、`tilt_head` ~0.1%。**任一动作类 > 25% 即报警**,
说明 prompt 或调用有问题。人工抽检:每个动作类 + none 各抽 50-100 条,S1 必做,抽检准确率 ≥ 85%。

## 下游(本目录不做,仅登记,来自设计文档 §5)

格式转换 `JoyAI-VL-Interaction/datasets/convert_data.py`(`action`→`</act_xxx>` marker)→ adapter 解析
`services/webinfer/live_adapter.py` → reachy 对接 `../reachy-mini-demo/voice/realtime.py`;4 个动作定义源头
`../reachy-mini-demo/voice/config.py:242-267`。
