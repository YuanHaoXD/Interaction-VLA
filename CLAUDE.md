# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言与协作约定

- **所有产出用中文**：文档、代码注释、commit message、实验报告一律中文。
- **你的角色**：服务器侧（8×昇腾 910B 64G NPU）实施/实验 AI（Opus 4.8）。本机侧（Fable 5，有 MuJoCo 仿真+真机）负责设计/审查、仿真测试、数据模板、真机工作。协作链路：本机（设计）⇄ GitHub ⇄ 服务器（实施）。
- **权威文档**：`docs/2026-07-14-交互VLA设计文档_zh.md` 是唯一权威总设计，与任何其它文档冲突时以它为准。任务书按阶段看 `docs/plans/`：M1 已基本完成，**当前阶段是 `2026-07-15-M1.5-官方动作库与50Hz迁移.md`**（含契约 C0 修正案：fps 50Hz、动作词汇扩为官方库 85 情绪动作）。
- **决策边界**：设计未覆盖时——小事自决并在实验报告中声明；**大事**（改动作表示 / 数据 schema / 接口签名 / 评测协议）必须停下，在 `docs/experiments/` 写"待决问题"报告等本地裁决，同时继续做不受阻塞的任务。
- **pip 一律用清华镜像**：`pip install <pkg> -i https://pypi.tuna.tsinghua.edu.cn/simple`。

## 每完成一个任务的纪律

1. 中文 commit；2. 把计划文件里对应 `- [ ]` 勾成 `- [x]`；3. 在 `docs/experiments/` 写一篇中文实验记录（结果、关键数字、耗时、踩坑、绕行方案）；4. `git push origin main`。
大规模产物（数据集 episodes、隐状态缓存、模型权重）**不入 git**（.gitignore 已配），存服务器本地盘，实验记录里写明**绝对路径**。

## 常用命令

```bash
# 测试（必须在仓库根运行；conft.py 空文件的作用是把根目录加进 sys.path，data_gen 才可导入）
pytest tests/ -v
pytest tests/test_templates.py -v          # 单文件
pytest tests/test_schema.py::test_detects_nan -v   # 单个用例

# 生成合成样例数据（必须用 -m 从仓库根运行，否则 data_gen 导入失败）
python -m data_gen.make_samples

# 仿真回放一段 episode（需先起 reachy_mini sim daemon；本机侧环境，服务器 headless 可用性见 A4 笔记）
python sim/replay_episode.py samples/ep_00000000

# NPU 冒烟三件套（服务器侧）
python smoke/npu_basic.py        # torch_npu 基础算子 + npu_fusion_attention
python smoke/npu_dit.py          # DiT 风格块前向+反传+optimizer step
python smoke/test_starvla.py     # StarVLA DiT 流匹配头 NPU 真前向

# JoyAI 慢路径推理 / 隐状态离线预计算（服务器侧，权重在永久盘）
python -m expert.hidden_export.run_joyai
python -m expert.hidden_export.precompute --data samples --n-max 10

# 标签→轨迹展开流水线（A5，需真标注数据；小样用 fixture）
python -m data_gen.pipeline --annotation data_gen/example_annotation.json --out /cache/tmp/ds --workers 2
python -m data_gen.stats --data /cache/tmp/ds
```

## 服务器环境（关键，2026-07-15 实测）

- **conda env**：`source /home/ma-user/anaconda3/bin/activate PyTorch-2.6.0`（torch 2.6.0+cpu / torch_npu 2.6.0.post5 / transformers 5.13.1 / 8×Ascend 910B2）。
- **环境重置会丢东西**：`/`、`/tmp`、`/cache` 是易失盘（重启即丢）；**唯一永久区是 `/home/ma-user/work`（11TB）**。torch_npu 的 triton 补丁落在 site-packages，重置后要重跑 `bash smoke/setup_npu_env.sh`（幂等，修 triton/torch AttrsDescriptor 冲突 + 装 StarVLA）。
- **JoyAI 权重**：`/home/ma-user/work/dataset/yh222/models/JoyAI-VL-Interaction-Preview`（17G，Qwen3-VL，d_model=4096，36 层；`</silence>`=151669、`</response>`=151670）。旧报告写的 `/cache/model/...` 是易失路径，已失效。
- **实测数字**：JoyAI 单卡 ~6 决策/秒；隐状态精确预计算 310 段/分·卡（10 万段 8 卡约 40 分钟）。
- **pip 用清华镜像**：`-i https://pypi.tuna.tsinghua.edu.cn/simple`。
- **git 身份**：`user.name=YuanHao`、`user.email=3228984401@qq.com`（本地已配，匹配历史提交）。

## 架构大图（慢脑 + 快脑）

核心思想：在冻结的 JoyAI-VL-Interaction 8B（**慢脑，1Hz 决策**）上外挂一个可训练的连续动作专家（**快脑，50–200M，50Hz**），让机器人在说话/聆听/沉默全程流畅输出拟人动作。

```
用户文本 + 摄像头帧(≈1fps) ─▶ JoyAI 8B(慢脑,1Hz,冻结) ─▶ 文本回复
                                     │ 决策token隐状态(训练期离线预计算)
                                     ▼
                            动作专家(快脑,50Hz) ─▶ 动作块 ─▶ 安全层 ─▶ set_target
```

三个必须理解的核心洞察（设计文档 §2，驱动全部设计）：
1. **节奏差桥接 = action chunking**：慢脑 1Hz、控制需 50Hz（v1.3 起，原 30Hz，迁移见 M1.5 计划 C1）。每个决策步输出未来 1.5s 动作块（75步×9维），块间重叠区 temporal ensembling 混合 → 天然连续无跳变。
2. **沉默也是数据**：JoyAI 训练格式是每秒一对 user/assistant（`</silence>` 或 `</response> 文本`）。沉默秒也带动作监督（聆听点头、思考歪头）——这是回合制 VLA 结构上做不到的，是故事核心新颖点。
3. **无音频原则**：动作专家**初期不接任何音频通道**（JoyAI 本身没有语音模态）。目标收紧为「图像/文本 → 动作」。发声区间用**时长代理**（字数÷语速：中文 5.5 字/秒、英文 14 字符/秒）摊开语义条件。音频通道是 future work（条件接口已预留扩展位，勿擅自并入）。

### 动作表示（"字母表"，改动即大事）
9 维任务空间向量 @50Hz（v1.3 起）：`[x, y, z, roll, pitch, yaw, body_yaw, ant_R, ant_L]`（头部平移/姿态 + 身体旋转 + 双天线）。维度索引在 `data_gen/templates.py` 顶部注释；限位：数据生成用 `sim/limits_real.json`（官方库包络，C2 产出），`sim/limits.json` 是仿真勘察值（比真机大 3–4 倍，勿用于生成）。

### 数据契约 C0（schema v1，改动即大事）
每个 episode 是一个目录：`meta.json` + `timeline.json`（时间升序事件列表，`type ∈ {user_text, response, silence, delegate}`）+ `actions.npy`（float32 `[T,9]`，`T = round(duration_s*fps_action)`，fps 规范值 50）。可选：`hidden_states.npy`（`[N_decisions, d_model]`）、`frames/`、`audio_user.wav`（预留）。`response` 事件必须带 `est_speech_dur_s`。schema 的代码化在 `data_gen/schema.py`（`write_episode` / `validate_episode`）——任何数据生产/消费都应过 `validate_episode`。

## 代码地图

- `data_gen/` —— 数据构造。`templates.py`（5 类标签 nod/shake_head/tilt_head/wiggle_antennas/none → 参数随机化连续轨迹，含 idle 微动与 backchannel 点头；`compose()` 是总装：叠加事件轨迹后做限幅+逐步限速保证连续）；`schema.py`（C0 契约代码化）；`make_samples.py`（生成 100 段合成样例）。
- `expert/hidden_export/` —— 慢脑侧。`run_joyai.py`（JoyAI transformers 慢路径加载/推理，加载方式抄官方 `services/webinfer`，**别自己发明**）；`precompute.py`（隐状态离线预计算，**精确版**：按决策 token id `</silence>`=151669/`</response>`=151670 定位，0.19s/段·卡，见 A3 报告 2026-07-15 修订）。
- `sim/` —— 仿真与勘察（多为本机侧环境）。`replay_episode.py`（按 meta 的 `fps_action` 回放到 MuJoCo，50Hz；用官方最佳实践 `connection_mode="localhost_only"` + `media_backend="no_media"`）；`probe_action_space.py`（限位/频率勘察，产出 `limits.json`）；`_diag_*.py` 是连接模式排障脚本。
- `tests/` —— pytest（TDD：先写失败测试再实现）。
- `smoke/` —— NPU 冒烟测试（算子/DiT/StarVLA 兼容性）。
- `docs/` —— `2026-07-14-交互VLA设计文档_zh.md`（权威设计）、`plans/`（分阶段任务书 + 服务器/本机执行提示词）、`experiments/`（每任务实验报告）。
- `human_reading/` —— 面向人类的通俗解释文档（非执行依据）。

## 关键上下文与陷阱

- **A3（隐状态离线预计算）是全项目头号判定点**：若受阻，Plan B 是「回复文本 → 小型文本编码器」（设计文档 §3.3，接口不变），代价是丢失视觉上下文与沉默秒语义。精确版已跑通、判定真正通过（2026-07-15），语义通道锁定隐状态方案。
- **8B 微调链路在本环境从未跑通**（设计文档 §7 风险 3），M1/M2 不需要，别顺手去试——那是 M3 的 v3 预研项（候选 LLaMA-Factory / ms-swift）。
- **StarVLA 当零件库不当整机**：借流匹配动作头、训练 loop、动作归一化工具；数据管线自写。
- 训练课程：v1 回归版（8月）→ v2 DiT 流匹配（9月，论文主结果，因动作分布多峰，回归会平均成"木头动作"）→ v3 联合微调（10月，可选）。
- JoyAI 权重/代码/数据集路径需在服务器自行查找（此机做过 JoyAI 数据标注），找到后写进实验记录。默认候选路径见 `run_joyai.py:locate_joyai`。
