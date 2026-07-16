# 提示词：M1.5 服务器侧（仅 C4）——粘贴给服务器 Claude Code（Opus 4.8）

> 用法：服务器上进入仓库目录启动 Claude Code，整段粘贴。
> 变更（2026-07-15）：**C3 已改派本机执行**（标注不需要卡且数据留本机），服务器只做 C4。

---

你是 interaction-vla 项目**服务器侧**执行工程师，环境是 8×昇腾 910B 64G。

## 第零步：环境自检

1. 仓库在**永久区** `/home/ma-user/work` 下，`git pull origin main`。产物一律不放 `/`、`/tmp`、`/cache`（易失盘，重启即丢）。
2. `source /home/ma-user/anaconda3/bin/activate PyTorch-2.6.0`；新环境先 `bash smoke/setup_npu_env.sh`（幂等），再 `python smoke/npu_basic.py` 确认 NPU。
3. JoyAI 权重：`/home/ma-user/work/dataset/yh222/models/JoyAI-VL-Interaction-Preview`。

## 你的任务

只做 `docs/plans/2026-07-15-M1.5-官方动作库与50Hz迁移.md` 的 **C4**。C1/C2 已完成，C3 由本机做（**你不要碰标注**）。动手前读：仓库 CLAUDE.md（自动加载）→ 设计文档 §4.1（v1.3）→ M1.5 计划全文。

C4 分两段：

**现在就能做（代码改造+自检）：**
- `pipeline.convert_sample`：`action` 为官方库动作名时生成 `{"label": "lib:<名>", ...}` 事件（none/旧 5 类逻辑不变）；同步 fixture 与测试。注意 C2 已合入的 `data_gen/augment.py` 与 `compose()` 的 `lib:` 路径，直接消费其接口，**不得改签名**。
- 前置自检：用真实数据集样本+帧验证 `model.generate` 输出以 `</silence>`/`</response>` 开头（若此前复核报告已含此项，引用即可跳过）。

**等信号再做（全量生产）：**
- 触发条件 = C3 全量标注文件由本机上传到服务器（路径见届时的 C3 实验记录/用户告知）。
- 全量 ≥10 万段 @50Hz，限幅用 `sim/limits_real.json`（不要用 limits.json，那是仿真虚高值）→ `stats.json` → 报告。
- 隐状态全量预计算（8 卡，预估约 40 分钟）→ 报告（带帧/无帧口径声明）。

## 执行纪律

每完成一项：中文 commit + 勾计划 checkbox + `docs/experiments/` 实验记录 + push；大规模产物不入 git，写明永久区绝对路径。大事（改 schema/接口签名/评测协议）停下写"待决问题"等本地裁决。pip 用清华镜像。

现在开始：第零步 → 读文档 → C4 代码改造部分。
