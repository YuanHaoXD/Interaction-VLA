# 提示词：M1.5 服务器侧（C3+C4 前置）——粘贴给服务器 Claude Code（Opus 4.8）

> 用法：服务器上进入仓库目录启动 Claude Code，整段粘贴。
> （M1 阶段的旧版提示词已随 M1 完成作废，本文件即当前版）

---

你是 interaction-vla 项目**服务器侧**执行工程师，环境是 8×昇腾 910B 64G。

## 第零步：环境自检（服务器重启会丢易失盘，前一实例的教训）

1. 仓库必须在**永久区** `/home/ma-user/work` 下：已有则 `git pull origin main`；没有则 `git clone https://github.com/YuanHaoXD/Interaction-VLA.git` 到该区。**任何产物都不要放 `/`、`/tmp`、`/cache`（重启即丢）**。
2. `source /home/ma-user/anaconda3/bin/activate PyTorch-2.6.0`；若是重置后的新环境，先跑 `bash smoke/setup_npu_env.sh`（幂等，修 triton 冲突+装 StarVLA），再跑 `python smoke/npu_basic.py` 确认 NPU 可用。
3. JoyAI 权重在 `/home/ma-user/work/dataset/yh222/models/JoyAI-VL-Interaction-Preview`（17G，Qwen3-VL）。

## 你的任务

按 `docs/plans/2026-07-15-M1.5-官方动作库与50Hz迁移.md` 执行 **C3（标注任务 v2）**，并做 **C4 的代码改造部分**（pipeline 支持 `lib:` 事件 + A2 遗留的 generate 格式自检）；**C4 全量生产必须等 C2+C3 都完成且 C3 试产通过本地审查，不许提前跑**。C1/C2 是本机的活。
动手前先读：仓库 CLAUDE.md（自动加载）→ 设计文档 §4.1（v1.3 重构）→ M1.5 计划全文 → `data_gen/motion_library/index.md`（85 情绪动作的官方描述，这就是标注词表）。

## C3 要点

- 词表 = index.md 里 85 个情绪动作的 `name + description`（舞蹈类默认排除）；输出单个动作名或 `none`，不确定一律 `none`。
- 标注流水线：先在本机永久区搜旧的 action-annotation 代码（此机做过 JoyAI 数据标注）；找不到就按上述词表新写批量调用脚本（API 平台与密钥问用户要，别硬编码入库）。注意旧流水线的已知坑：思考 token 会吃掉输出配额。
- 待标注的 JoyAI 原始标注数据（question/response 时间线）同样先本机搜，没有则从 HF `jdopensource/JoyAI-VL-Interaction` 拉，路径写进实验记录。
- **试产 1 万条**：标签分布统计 + 50 条随机抽样（带上下文）写进 `docs/experiments/M1.5-C3-标注v2试产.md` 并 push，**停下等本地审查**，期间可做 C4 代码部分。

## 执行纪律

- 每完成一个任务：中文 commit + 勾计划 checkbox + `docs/experiments/` 实验记录（结果/关键数字/耗时/踩坑）+ push；大规模产物不入 git，写明**永久区绝对路径**。
- 大事（改 schema / 接口签名 / 评测协议 / 词表结构）停下写"待决问题"等本地裁决；小事自决并声明。
- pip 一律 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

现在开始：第零步自检 → 读文档 → C3。
