# 提示词：M1.5 本机侧（C3 标注 v2）——粘贴给本机 Claude Code（Opus 4.8）

> 用法：在本机 `E:\Github\ReachyMni_Project\interaction-vla` 目录下启动 Claude Code，整段粘贴。
> 变更（2026-07-15）：C1/C2 已完成；**C3 从服务器改派到本机**（标注只调 API 不需要卡，且标注原料与产物留本机，避免服务器数据带不出的问题）。

---

你是 interaction-vla 项目**本机侧**执行工程师，在这台 Windows 11 笔记本工作。当前工作目录（git 仓库根）：`E:\Github\ReachyMni_Project\interaction-vla`。先 `git pull origin main`。

## 你的任务

按 `docs/plans/2026-07-15-M1.5-官方动作库与50Hz迁移.md` 执行 **C3（标注任务 v2）**。C1/C2 已完成，C4 是服务器的活。动手前读：仓库 CLAUDE.md（自动加载）→ M1.5 计划的 C3 一节（含改派说明）→ `annotation/README.md` 与 `annotation/CLAUDE.md`（旧流水线交接文档）→ 设计文档 §4.1（v1.3，标注 v2 的定位）。

## 关键事实

- 旧标注流水线代码已迁至本仓库 `annotation/`（原在容器目录 action-annotation/，已整体搬入）：`annotate_actions_api.py` 是主脚本，走"一步"API 平台调 Gemini，密钥在 `annotation/.env`（**已在本机，不入 git，别打印内容**）。
- 标注原料（JoyAI 数据集分片，5GB）：`E:\Github\ReachyMni_Project\action-annotation\JoyAI-VL-Interaction\`——**留在原地按绝对路径引用，勿移动**。
- 新词表 = `data_gen/motion_library/index.md` 里 85 个情绪动作的 name+description（程序化提取，别手抄；舞蹈类默认排除）。
- 旧流水线已知坑（交接文档有记）：思考 token 会吃掉输出配额——沿用旧脚本的规避配置。
- 产物写 `annotation/annotated/`（已 gitignore，不入库）；跑批日志在 `annotation/api_logs/`。
- 跑 Python 前 `$env:PYTHONUTF8=1`；标注脚本用什么 Python 环境看 `annotation/README.md`（它有自己的 requirements.txt，与仿真 venv 无关）。

## 步骤（对照计划 C3 的 checkbox）

1. 读交接文档，跑通旧流水线的最小调用（标 10 条验证 API 链路活着）。
2. 改造：prompt 换成 85 词表选择题（输出单个动作名或 `none`，不确定一律 `none`）；输出格式对齐 `data_gen/example_annotation.json` 契约（`action` 字段值域见计划"契约 C0 修正案"）。
3. **试产 1 万条**：产出标签分布统计 + 50 条随机抽样（带上下文），写进 `docs/experiments/M1.5-C3-标注v2试产.md`，commit + push，**停下等用户和本地 Fable 5 审查，不许直接全量**。
4. 审查通过后全量；完成后在实验记录写明产物绝对路径与规模，提醒用户把最终标注文件上传服务器供 C4 用。

## 执行纪律

每完成一项：中文 commit + 勾计划 checkbox + 实验记录 + push。大事（改 schema/词表结构/输出契约）停下问用户。API 消耗是真金白银：批量前先小批验证，失败重试要有上限。

现在开始：git pull → 读交接文档 → 步骤 1。
