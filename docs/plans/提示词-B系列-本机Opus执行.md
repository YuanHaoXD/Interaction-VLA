# 提示词：M1.5 本机侧（C1+C2）——粘贴给本机 Claude Code（Opus 4.8）

> 用法：在本机 `E:\Github\ReachyMni_Project\interaction-vla` 目录下启动 Claude Code，整段粘贴。
> （M1 阶段的旧版提示词已随 M1 完成作废，本文件即当前版）

---

你是 interaction-vla 项目**本机侧**执行工程师，在这台 Windows 11 笔记本工作。当前工作目录（git 仓库根）：`E:\Github\ReachyMni_Project\interaction-vla`。先 `git pull origin main` 确保最新。

## 你的任务

按 `docs/plans/2026-07-15-M1.5-官方动作库与50Hz迁移.md` 执行 **C1（50Hz 迁移）→ C2（官方库增广接入）**，只做这两个；C3/C4 是服务器的活。动手前先读：仓库 CLAUDE.md（自动加载）→ 设计文档 `docs/2026-07-14-交互VLA设计文档_zh.md` 的 §2/§3.2/§4.1（v1.3 修订）→ M1.5 计划全文（含契约 C0 修正案）。

## 本机硬事实（前人血泪，直接用）

- 仿真环境 Python：`E:\Github\ReachyMni_Project\reachy-mini-demo\.venv\Scripts\python.exe`（3.12，含 reachy-mini 1.8.0 + mujoco 3.3.0 + numpy/scipy + pytest）
- 仿真 daemon：`E:\Github\ReachyMni_Project\reachy-mini-demo\.venv\Scripts\reachy-mini-daemon.exe --sim --scene minimal`。启动铁律：先杀残留（`Get-Process python,reachy-mini-daemon -EA SilentlyContinue | Stop-Process -Force`，**杀之前先问用户本机有无别的 Python 在跑**）→ 起 daemon → REST 就绪（http://localhost:8000/docs）→ **再静置 ~25 秒**。别频繁杀重启。
- **`ReachyMini()` 构造耗时 10~52 秒不稳定**（连接自动检测所致），不是卡死；SDK 脚本一律 `python -u` 跑，判断死活看 MuJoCo 窗口是否在动（见 M1-B1 报告 §4.1）。连上之后 set_target 只要 0.18ms/call。
- 本机 pytest 的默认临时目录 `C:\Users\WyofarMoon\AppData\Local\Temp\pytest-of-WyofarMoon` 有权限问题，跑测试加 `--basetemp=.pytmp`（已在 .gitignore）。
- 跑任何 Python 前 `$env:PYTHONUTF8=1`；模块脚本必须在仓库根用 `-m` 跑。

## 执行规则

- TDD 步骤不许跳；每完成一个任务：中文 commit + 勾计划 checkbox + `docs/experiments/` 写实验记录 + push。
- 肉眼验收步骤（C1 的 50Hz 回放、C2 的 20 段库动作样例回放）：你看不到 MuJoCo 窗口，请用户看窗口口述结论，你写进报告。
- 大事（改 9 维表示 / schema / 接口签名）停下问用户；小事自决并在报告声明。
- C2 的接口签名（`load_lib_move` / `augment_lib_move` / `compose` 的 `lib:` 约定）是 C4 的依赖，**不得改**。

现在开始：git pull → 读文档 → C1。
