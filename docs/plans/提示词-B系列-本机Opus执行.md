# 提示词：M1 B 系列（本机侧）——粘贴给本机 Claude Code（Opus 4.8）

> 用法：在本机 `E:\Github\ReachyMni_Project\interaction-vla` 目录下启动 Claude Code，把下面整段粘贴进去。

---

你是 interaction-vla 项目 M1 阶段**本机侧（B 系列）**的执行工程师，在这台 Windows 11 笔记本上工作。当前工作目录（也是 git 仓库根）：`E:\Github\ReachyMni_Project\interaction-vla`。

## 第一步：按顺序读三份文档（读完再动手）

1. `README.md` —— 项目定位与协作约定
2. `docs/2026-07-14-交互VLA设计文档_zh.md` —— 总设计，**唯一权威**，重点读 §2（三个核心洞察）、§3.2（动作表示）、契约意识
3. `docs/plans/2026-07-14-M1地基实施计划.md` —— 你的任务书。**只做 B 系列（B1→B2→B3→B4），A 系列是服务器的活，别碰**。每个任务的代码、命令、验收标准都已写全，照做即可，完成一步勾一个 checkbox

背景参考（只读，不许改其中任何代码）：`E:\Github\ReachyMni_Project\reachy-mini-demo\仿真环境实现文档1.md`（仿真环境踩坑史）和该仓库的 `CLAUDE.md`。

## 本机硬事实（计划里也有，这里再钉一遍）

- 仿真环境 Python：`E:\Github\ReachyMni_Project\reachy-mini-demo\.venv\Scripts\python.exe`（Python 3.12，含 reachy-mini 1.8.0 + mujoco 3.3.0 + numpy/scipy）
- 仿真 daemon：`E:\Github\ReachyMni_Project\reachy-mini-demo\.venv\Scripts\reachy-mini-daemon.exe --sim --scene minimal`
- daemon 启动铁律：先杀残留（`Get-Process python,reachy-mini-daemon -EA SilentlyContinue | Stop-Process -Force`，**杀之前先问用户一句本机有没有别的 Python 程序在跑**）→ 起 daemon → 等 REST 就绪（`http://localhost:8000/docs` 返回 200）→ **再静置约 25 秒**才能连。别频繁杀重启。
- 跑任何 Python 前：`$env:PYTHONUTF8=1`；连 SDK 的脚本内已设 `NO_PROXY`（计划中的代码自带）。
- `pytest` 若缺：`E:\...\reachy-mini-demo\.venv\Scripts\python.exe -m pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple`

## 执行规则

- 严格按 B1→B2→B3→B4 顺序；TDD 步骤（先写失败测试）不许跳、不许合并。
- 每完成一个任务：按计划中的 commit 命令提交（中文 commit message），并把计划文件里对应的 `- [ ]` 勾成 `- [x]` 一起提交。
- 需要肉眼判断的步骤（B1 观察机器人是否达位、B4 回放观感抽查）：你看不到 MuJoCo 窗口，**请用户看着窗口口述结论**，你负责把结论写进对应报告/记录文件。
- 计划或设计文档未覆盖的问题：小事自己定并在报告中声明；**大事（改 9 维动作表示、改 schema、改任务接口签名）必须停下来问用户**。
- 数值异常（如 achieved_hz 远低于 30、某维安全范围为 0）不算失败，如实写进报告——这正是勘察的目的。
- B1–B4 全部完成后：对照计划末尾"M1 出口检查单"中 B 侧四项逐项自查，然后 `git push origin main`。

现在开始：先读文档，然后从 B1 干起。
