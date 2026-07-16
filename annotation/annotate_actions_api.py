#!/usr/bin/env python3
"""给 JoyAI 数据集的 response 打动作标签（API 版本）。

用一步平台 API 调用 Gemini 模型，替代本地 vLLM。

用法:
  # 小规模测试
  python annotate_actions_api.py --sample_limit 2000

  # 处理指定文件
  python annotate_actions_api.py --task_types background

  # 全量
  python annotate_actions_api.py

输出写到 --output 目录，镜像 input 目录结构，每个 response dict 新增 "action" 字段。
支持断点续跑: 已有 action 字段的 response 自动跳过。

Q-R 配对: question[i] <-> response[i]，按位置一一对应，不依赖 time 字段。

── 相对原始交接文档做的修正（2026-07，落地时）──
1. 默认模型改为 gemini-2.5-flash（原为 gemini-2.5-pro）：情感分类用 pro 又慢又贵，
   flash 质量足够、便宜快一个量级。可用 MODEL_NAME 环境变量覆盖。
   （M1.5-C3 起标签集由旧 5 类升级为官方库 85 情绪动作 + none，见下方「动作词表 v2」。）
2. 防"静默误标 none"：Gemini 2.5 系列是思考模型，若 max_tokens 被思考 token 吃光，会返回
   空 content（finish_reason == "length"），原脚本会把它无声兜底成 "none"，且统计看不出异常。
   本版检测空/截断返回 → 自动加大预算重试一次 → 仍失败的单独计为 "empty" 并在结尾醒目告警。
   小测（--sample_limit 2000）时看这个计数就能判断思考 token 有没有在坑你。
3. 默认关思考（THINK_OFF=1）：Gemini 2.5 在 yibuapi 网关上默认思考，每次烧 ~250 reasoning
   token。实测唯一有效的关法是顶层 "extra_body":{"google":{"thinking_config":{"thinking_budget":0}}}
   （reasoning_effort、顶层 google 键均无效）。关后 ~90% 调用只 ~10 token、省约 7×，判断质量
   等价，且与 vLLM 原版 enable_thinking=False 一致。剩 ~10% 网关漏关的由 MAX_TOKENS 余量兜住。
4. 补齐平台专有可重试错误串（对齐 api_call/api_test.py）：429 加 负载/饱和，5xx 加
   unexpected end / bad_response_body / upstream_error。
5. 流式处理（ijson 读 + JSONL 工作文件增量写 + 收尾合成 .json 数组）：原版整文件 json.loads
   ~580MB 会吃几 GB 内存，且每 200 条重写整份输出 = O(n²) I/O，百万级样本不可行。现在内存
   O(CHUNK_SAMPLES)、追加式落盘、按 sample 崩溃续跑；输出仍是下游 convert_data.py 要的 .json 数组。
   （这也顺带修了原始 vLLM 版重跑覆盖已标结果的 bug——现在续跑只追加、从不整文件重写。）
"""
import argparse
import collections
import json
import os
import re
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from local_api_logger import log_completion, set_log_dir


# ── .env 自动加载（纯标准库，无依赖）──
def _load_dotenv(path=None):
    """把脚本同目录的 .env 读进 os.environ。已存在的环境变量不覆盖（命令行/shell 优先）。

    仅支持最简格式：每行 KEY=VALUE，# 开头为注释，值两侧的成对引号会被去掉。
    """
    p = Path(path) if path else (Path(__file__).resolve().parent / ".env")
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key and key not in os.environ:  # 不覆盖已有环境变量
            os.environ[key] = val


_load_dotenv()  # 必须在下面读取配置常量之前执行

# ── 配置 ──
INPUT_DEFAULT = "./raw_data"
OUTPUT_DEFAULT = "./annotated"
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "20"))

# 输出上限。关思考后 ~90% 的调用只用 ~6-14 token；剩 ~10% 网关漏关思考的会用到 ~350，
# 512 的余量能让它们一次跑完、不触发截断重试。输出按实际计费，设大不额外花钱。
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))

# 关思考（实测决定，见 README「关思考」一节）。
# yibuapi 网关上，唯一有效的关思考方式是在请求体顶层放一个字面 "extra_body" 字段：
#   {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}}
# 经 openai SDK 要产出这个字段，需把它整体作为 SDK 的 extra_body 参数传入（SDK 会把外层
# extra_body 摊平进 body 根，于是 body 根就多了一个 "extra_body" 键）。实测 ~90% 生效
# （reasoning_tok=0），剩 ~10% 网关抖动仍会思考，由上面的 MAX_TOKENS 余量兜住。
# 注意：顶层 "google" 键、以及 OpenAI 的 reasoning_effort 参数在本网关上都【无效】。
THINK_OFF = os.environ.get("THINK_OFF", "1").strip() not in ("", "0", "false", "False")
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "0"))
# 保留但默认不发；本网关忽略它，仅为换平台时的可移植性留口子。
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "").strip()

# 流式处理：每次读入并标注这么多 sample 就落一次盘。内存 O(CHUNK_SAMPLES)，
# 避免对 ~580MB 大文件整文件 json.loads / 整文件重写 checkpoint。
CHUNK_SAMPLES = int(os.environ.get("CHUNK_SAMPLES", "500"))

ACTIONS = ("nod", "shake_head", "wiggle_antennas", "tilt_head", "none")  # 旧 5 类(下方 v2 覆盖)

# ── 动作词表 v2(官方情绪动作库,M1.5-C3 起)──
# 标签集由旧 5 类升级为官方动作库的 85 个情绪动作 + none。词表【程序化提取】自
# ../data_gen/motion_library/index.json(kind=="emotion"),不手抄;舞蹈库(19 个 kind=="dance")
# 默认排除;情绪库里的 dance1/2/3 保留,但仅在音乐/舞蹈语境启用(见 prompt 规则 5)。
MOTION_LIBRARY_INDEX = (
    Path(__file__).resolve().parent.parent / "data_gen" / "motion_library" / "index.json"
)

# 4 个补录动作官方只留了录制时间戳、无语义描述;按名义补最简英文注释
# (本机 C3 自决,已在实验记录声明;不改动作本身,只为让 prompt 词条可读)。
_DESC_OVERRIDE = {
    "mini-deep-sleep": "Falls into a deep sleep / powered-down rest.",
    "toc-toc-toc": "A knock-knock gesture, playfully knocking to get attention.",
    "waiting": "Idly waiting for something to happen, with nothing to do.",
    "wake-mini-up": "Waking up and booting back to life.",
}


def build_action_catalog(index_path=MOTION_LIBRARY_INDEX):
    """从官方库 index.json 程序化提取 85 个情绪动作的 (name, description)。
    返回 (names: tuple, catalog_lines: list[str])。"""
    idx = json.loads(Path(index_path).read_text(encoding="utf-8"))
    rows = []
    for m in idx["moves"]:
        if m.get("kind") != "emotion":       # 舞蹈库默认排除
            continue
        name = m["name"]
        desc = _DESC_OVERRIDE.get(name) or (m.get("description") or "").replace("\n", " ").strip()
        rows.append((name, desc))
    rows.sort(key=lambda r: r[0])
    names = tuple(r[0] for r in rows)
    lines = [f"- {n:16} : {d}" for n, d in rows]
    return names, lines


_EMOTION_NAMES, _CATALOG_LINES = build_action_catalog()
_CATALOG_TEXT = "\n".join(_CATALOG_LINES)

# 标注模式(M1.5-C3):
#   demeanor(默认,用户 2026-07-16 定):判断"人说这句话时自然会有的神态",【无 none】,每条必选一个动作;
#   emotion(旧 v2 可回退):判断"回答本身表达的情绪",保留 none。
LABEL_MODE = os.environ.get("LABEL_MODE", "demeanor").strip().lower()
INCLUDE_NONE = (LABEL_MODE == "emotion")

ACTIONS = _EMOTION_NAMES + (("none",) if INCLUDE_NONE else ())

# 解析/兜底失败时的哨兵(demeanor 模式下不允许 none;失败用可见哨兵,不静默塞情绪)
_FALLBACK = "none" if INCLUDE_NONE else "__unparsed__"
# delegation(功能性委派 token,非自然话语)的处理:emotion 保持 none;demeanor 用可见哨兵、不调 API
_DELEGATION_LABEL = "none" if INCLUDE_NONE else "__delegation__"

PROMPT_SYSTEM_EMOTION = (
    "You label which emotional body-language animation a small desktop robot should play "
    "while it says a given reply utterance. Reply utterances may be English or Chinese.\n\n"
    "Choose EXACTLY ONE action name from the catalog below, or \"none\".\n"
    "Each catalog line is `name : when to use it`.\n\n"
    "=== ACTION CATALOG (85 emotions) ===\n"
    f"{_CATALOG_TEXT}\n"
    "=== END CATALOG ===\n\n"
    "Rules:\n"
    "1. Output STRICTLY a JSON object: {\"action\": \"<name>\"}. The name MUST be EXACTLY one "
    "catalog name, or \"none\". No explanation, no prose.\n"
    "2. DEFAULT TO \"none\". Most replies are plain / neutral / factual and must be \"none\". "
    "Pick a named emotion ONLY when that specific emotion or intent is CLEARLY expressed in the reply text.\n"
    "3. When unsure, or when several actions fit only weakly, choose \"none\" (never guess).\n"
    "4. Functional / delegation sentences (e.g. handing off to a background model) are \"none\".\n"
    "5. dance1 / dance2 / dance3 ONLY when the reply is explicitly about music or dancing; "
    "otherwise never pick them."
)

PROMPT_SYSTEM_DEMEANOR = (
    "A small desktop robot is about to SAY the reply line given below (English or Chinese).\n"
    "Your job: decide the facial expression / demeanor (神态) that a person would NATURALLY have "
    "while speaking THIS exact line. Even a plain, factual or descriptive line is spoken with SOME "
    "demeanor — so you must ALWAYS pick one. Then choose the single best-matching animation.\n\n"
    "Each catalog line is `name : the mood / situation it fits`.\n\n"
    "=== ANIMATION CATALOG (85) ===\n"
    f"{_CATALOG_TEXT}\n"
    "=== END CATALOG ===\n\n"
    "Rules:\n"
    "1. Output STRICTLY a JSON object: {\"action\": \"<name>\"}. The name MUST be EXACTLY one "
    "catalog name. No explanation, no prose.\n"
    "2. ALWAYS choose one animation. There is NO \"neutral\" / \"none\" option. Judge how the line "
    "is naturally delivered (its tone, intent and content) and pick the demeanor that fits best.\n"
    "3. For a calm, plain or matter-of-fact line, prefer gentle everyday demeanors "
    "(e.g. attentive1, understanding1, thoughtful1, welcoming1, curious1) rather than intense ones "
    "(rage1, surprised1, enthusiastic1...). Reserve intense demeanors for lines that truly warrant them.\n"
    "4. If several fit, choose the most natural / most likely one for a friendly desktop robot.\n"
    "5. dance1 / dance2 / dance3 ONLY when the line is explicitly about music or dancing."
)

PROMPT_SYSTEM = PROMPT_SYSTEM_EMOTION if INCLUDE_NONE else PROMPT_SYSTEM_DEMEANOR

PROMPT_USER_TMPL = (
    "Context (what was asked, for disambiguation only):\n{context}\n\n"
    "Reply line to label:\n"
    "\"\"\"\n"
    "{reply}\n"
    "\"\"\"\n\n"
    "Respond with only the JSON: {{\"action\": \"<name>\"}}"
)


# ── OpenAI client (线程安全，全局共享) ──
_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(
                api_key=os.environ.get("API_KEY"),
                base_url=os.environ.get("BASE_URL"),
            )
        return _client


def build_messages(reply, context):
    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": PROMPT_USER_TMPL.format(
            context=context or "(none)", reply=reply
        )},
    ]


def _extract_output_and_finish(response):
    """从 OpenAI 响应对象取 (content_text, finish_reason)。任一缺失返回 None。"""
    try:
        choice = response.choices[0]
        content = choice.message.content
        finish = getattr(choice, "finish_reason", None)
        return content, finish
    except Exception:
        return None, None


def call_api(messages, max_tokens=None, reasoning_effort=None):
    """调用一步 API，自动重试。返回 (output_text, response_dict, finish_reason)。

    不可重试的错误返回 ("ERROR: ...", None, None)。
    """
    client = get_client()
    mt = MAX_TOKENS if max_tokens is None else max_tokens
    effort = REASONING_EFFORT if reasoning_effort is None else reasoning_effort

    create_kwargs = dict(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=mt,
        timeout=300,
    )
    if THINK_OFF:
        # 见配置区注释：yibuapi 只认顶层字面 "extra_body" 字段来关思考。
        # SDK 会把这里的外层 extra_body 摊平进 body 根 → body 根出现 "extra_body" 键。
        create_kwargs["extra_body"] = {
            "extra_body": {"google": {"thinking_config": {"thinking_budget": THINKING_BUDGET}}}
        }
    if effort:
        # 本网关忽略；仅在显式配置时发送，供换平台可移植（部分代理不认识会 400）。
        create_kwargs["reasoning_effort"] = effort

    wait = 15
    while True:
        try:
            response = client.chat.completions.create(**create_kwargs)
            output, finish = _extract_output_and_finish(response)

            # 记录 API 日志
            try:
                req_data = {"model": MODEL_NAME, "messages": messages}
                log_completion(
                    model=MODEL_NAME,
                    request_data=req_data,
                    response_data=response.model_dump(),
                    user="action_annotator",
                )
            except Exception:
                pass

            return output, response.model_dump(), finish

        except Exception as e:
            err_str = str(e)

            # 429 限流（含"一步/云雾"平台返回的中文过载串：负载/饱和）
            if "429" in err_str or "rate" in err_str.lower() \
               or "负载" in err_str or "饱和" in err_str:
                print(f"[429] 限流，等待 {wait}s...", file=sys.stderr)
                time.sleep(wait)
                wait = min(wait * 2, 120)
                continue

            # 超时
            if "timeout" in err_str.lower() or "timed out" in err_str.lower():
                print(f"[TIMEOUT] 超时，等待 {wait}s...", file=sys.stderr)
                time.sleep(wait)
                wait = min(wait * 2, 120)
                continue

            # 5xx 服务端错误（含平台把上游异常包成的 JSON 截断 / 网关串）
            if any(code in err_str for code in ["500", "502", "503", "504"]) \
               or "unexpected end" in err_str.lower() \
               or "bad_response_body" in err_str.lower() \
               or "upstream_error" in err_str.lower():
                print(f"[5xx] 服务端错误，等待 {wait}s...", file=sys.stderr)
                time.sleep(wait)
                wait = min(wait * 2, 120)
                continue

            # 连接错误
            if "connection" in err_str.lower() or "connect" in err_str.lower():
                print(f"[CONN] 连接错误，等待 {wait}s...", file=sys.stderr)
                time.sleep(wait)
                wait = min(wait * 2, 120)
                continue

            # 不可重试的错误
            return f"ERROR: {e}", None, None


# ── response 遍历 ──
def iter_response_dicts(resp):
    """Yield (outer_idx, dict) for each content-dict inside a response field.

    outer_idx 用于按位置配对 question[idx] <-> response[idx]。
    同时处理 flat 和 nested 布局。
    """
    if not isinstance(resp, list):
        return
    for idx, item in enumerate(resp):
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict):
                    yield idx, sub
        elif isinstance(item, dict):
            yield idx, item


def get_question_by_index(sample, idx):
    """Return sample['question'][idx]'s content, paired purely by position.

    Returns '' if question is missing/malformed or idx is out of range.
    """
    qs = sample.get("question") or []
    if not isinstance(qs, list) or not (0 <= idx < len(qs)):
        return ""
    q = qs[idx]
    if isinstance(q, dict) and q.get("content"):
        return str(q["content"])[:500]
    return ""


def is_delegation(content):
    return "</delegation>" in (content or "")


_ACTIONS_SET = set(ACTIONS)
_ACTIONS_BY_LEN = sorted(ACTIONS, key=len, reverse=True)  # 最长优先,防短名是长名子串(如 sad1 ⊂ no_sad1)


def parse_action(text):
    """把 LLM 输出解析成 ACTIONS 之一;失败兜底 _FALLBACK(emotion 模式=none,demeanor 模式=可见哨兵)。"""
    if not text:
        return _FALLBACK
    text = text.strip()
    # 1) JSON object(主路径)
    m = re.search(r'\{[^}]*\}', text)
    if m:
        try:
            d = json.loads(m.group())
            a = str(d.get("action", "")).strip()
            if a in _ACTIONS_SET:
                return a
            if a.lower() in _ACTIONS_SET:    # 官方名本就是小写,容错大小写
                return a.lower()
        except Exception:
            pass
    # 2) 关键字兜底:最长名优先,避免短名误命中
    low = text.lower()
    for a in _ACTIONS_BY_LEN:
        if a != "none" and a in low:
            return a
    return _FALLBACK


def collect_json_files(path):
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(p.rglob("*.json"))


# ── 流式输出 / 断点续跑（JSONL 工作文件）──
# 标注时把每个已完成 sample 以「一行一个 compact JSON」追加进 <out>.jsonl 工作文件
# （崩溃安全、可按行续跑）；整文件跑完再流式合成下游要的 .json 数组，然后删掉工作文件。
def _resume_from_jsonl(work_path):
    """数工作文件里已完成的 sample 行数，并截断掉崩溃留下的半截尾行。返回已完成数。"""
    if not work_path.exists():
        return 0
    n = 0
    good_bytes = 0
    with open(work_path, "rb") as f:
        for raw in f:
            try:
                json.loads(raw)          # 完整的一行 = 一个已标 sample
            except Exception:
                break                    # 半截尾行（写到一半崩了），丢弃
            n += 1
            good_bytes += len(raw)
    with open(work_path, "r+b") as f:    # 截断残缺尾巴，保证可安全追加
        f.truncate(good_bytes)
    return n


def _finalize_to_json_array(work_path, out_path):
    """把 JSONL 工作文件流式合成为下游要的 .json 数组（O(1) 内存），成功后删掉工作文件。"""
    with open(work_path, "rb") as fin, open(out_path, "w", encoding="utf-8") as fout:
        fout.write("[")
        first = True
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            fout.write(("\n" if first else ",\n") + line.decode("utf-8"))
            first = False
        fout.write("\n]" if not first else "]")
    work_path.unlink()


class Stats:
    """跨文件累计的计数器。"""
    def __init__(self):
        self.processed = 0            # 本 run 新标注的 sample 数
        self.skipped_done = 0         # 续跑跳过的已完成 sample 数
        self.skipped_delegation = 0
        self.llm_calls = 0
        self.len_mismatch = 0
        self.mismatch_examples = []
        self.labels = collections.Counter()
        self.status = collections.Counter()


def process_file_streaming(fp, rel, out_path, args, stats):
    """流式标注单个文件：ijson 读 + JSONL 增量写 + 收尾合成 .json 数组。

    内存 O(CHUNK_SAMPLES)，追加式落盘（无整文件重写 checkpoint），崩溃/重启按 sample 续跑。
    """
    import ijson

    work_path = Path(str(out_path) + ".jsonl")

    # 最终 .json 已存在 = 上一轮已整文件跑完，跳过（--no_resume 例外）
    if out_path.exists() and not args.no_resume:
        print(f"[annotate] {rel} 已完成(.json 存在)，跳过", flush=True)
        return
    if args.no_resume and work_path.exists():
        work_path.unlink()

    n_done = 0 if args.no_resume else _resume_from_jsonl(work_path)
    stats.skipped_done += n_done
    if n_done:
        print(f"[annotate] 断点续跑: 已完成 {n_done} 个 sample，跳过", flush=True)

    remaining = None
    if args.sample_limit:
        # 续跑时已完成的 n_done 也占 sample_limit 额度，否则会超标。
        remaining = args.sample_limit - stats.processed - n_done
        if remaining <= 0:
            # 额度已用尽：把已有工作文件收尾成 .json 再返回（别把 n_done 丢在 jsonl 里）
            if work_path.exists():
                _finalize_to_json_array(work_path, out_path)
                stats.processed += n_done
            return

    pool = ThreadPoolExecutor(max_workers=args.max_workers)
    fout = open(work_path, "ab")
    buffer = []
    new_count = 0

    def flush_chunk():
        nonlocal buffer
        if not buffer:
            return
        tasks = []
        for s in buffer:
            qs = s.get("question")
            resp = s.get("response", [])
            if isinstance(qs, list) and isinstance(resp, list) \
               and len(qs) != len(resp):
                stats.len_mismatch += 1
                if len(stats.mismatch_examples) < 5:
                    stats.mismatch_examples.append(
                        (s.get("video_name", ""), len(qs), len(resp)))
            for idx, d in iter_response_dicts(resp):
                content = d.get("content", "")
                if "action" in d:                    # 输入自带(极少)→ 直接留用
                    stats.labels[d["action"]] += 1
                    continue
                if is_delegation(content):
                    d["action"] = _DELEGATION_LABEL
                    stats.skipped_delegation += 1
                    stats.labels[_DELEGATION_LABEL] += 1
                    continue
                ctx = get_question_by_index(s, idx)
                tasks.append((d, content, ctx, 0, idx))
        futures = {pool.submit(label_one_task, t): t for t in tasks}
        for fut in as_completed(futures):
            task, action, status = fut.result()
            task[0]["action"] = action
            stats.labels[action] += 1
            stats.status[status] += 1
            stats.llm_calls += 1
        # 整块标完再一次性追加到工作文件（每行一个 compact sample）
        for s in buffer:
            fout.write(json.dumps(s, ensure_ascii=False).encode("utf-8") + b"\n")
        fout.flush()
        buffer = []

    skipped = 0
    try:
        with open(fp, "rb") as fin:
            for s in ijson.items(fin, "item"):
                if skipped < n_done:                 # 跳过上一轮已完成的前 n_done 个
                    skipped += 1
                    continue
                if remaining is not None and new_count >= remaining:
                    break
                if args.per_file_limit and (n_done + new_count) >= args.per_file_limit:
                    break
                buffer.append(s)
                new_count += 1
                if len(buffer) >= CHUNK_SAMPLES:
                    flush_chunk()
                    print(f"[annotate]   已处理 {n_done + new_count} sample "
                          f"(LLM {stats.llm_calls})...", flush=True)
        flush_chunk()
    finally:
        pool.shutdown()
        fout.close()

    _finalize_to_json_array(work_path, out_path)
    stats.processed += new_count
    print(f"[annotate] 已保存 -> {out_path}  (本文件共 {n_done + new_count} sample)",
          flush=True)


def label_one_task(task):
    """标注一个 task，返回 (task, action, status)。

    status: ok       正常拿到非空、未截断的输出
            recovered 首次空/截断，加大预算重试后成功
            empty    重试后仍空/截断（思考 token 吃光预算，已兜底为 none）
            error    API 不可重试错误（已兜底为 none）
    """
    target_dict, content, ctx, sample_idx, resp_idx = task
    messages = build_messages(content, ctx)

    output, _resp, finish = call_api(messages)

    if output and output.startswith("ERROR:"):
        return task, "none", "error"

    status = "ok"
    truncated = (not output or not output.strip()) or (finish == "length")
    if truncated:
        # 疑似思考 token 吃光了 max_tokens：加大预算重试一次（仅调大 max_tokens，
        # 不加平台专有参数，保证对未知代理安全）。
        output2, _resp2, finish2 = call_api(messages, max_tokens=MAX_TOKENS * 4)
        if output2 and not output2.startswith("ERROR:"):
            output, finish = output2, finish2
        still_bad = (not output or not output.strip()) or (finish == "length")
        status = "empty" if still_bad else "recovered"

    action = parse_action(output)
    return task, action, status


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=INPUT_DEFAULT,
                    help="输入目录/文件")
    ap.add_argument("--output", default=OUTPUT_DEFAULT,
                    help="输出目录")
    ap.add_argument("--task_types", nargs="*", default=None,
                    help="只处理这些 task_type 子目录")
    ap.add_argument("--sample_limit", type=int, default=0,
                    help="限制总 sample 数(0=不限)")
    ap.add_argument("--per_file_limit", type=int, default=0,
                    help="每个文件最多处理 N 条(0=不限)")
    ap.add_argument("--max_workers", type=int, default=MAX_WORKERS,
                    help=f"并发数(默认 {MAX_WORKERS})")
    ap.add_argument("--no_resume", action="store_true",
                    help="忽略已有输出，强制重标")
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    # 设置 API 日志目录
    set_log_dir(str(out_root.parent / "api_logs"))

    print(f"[annotate] 模型={MODEL_NAME}  模式={LABEL_MODE}(none={'有' if INCLUDE_NONE else '无'})  "
          f"max_tokens={MAX_TOKENS}  "
          f"关思考={'是(budget=%d)' % THINKING_BUDGET if THINK_OFF else '否'}  "
          f"reasoning_effort={REASONING_EFFORT or '(未发送)'}  并发={args.max_workers}",
          flush=True)

    files = collect_json_files(in_root)
    if args.task_types:
        wanted = set(args.task_types)
        files = [f for f in files if any(p in wanted for p in f.parts)]
    print(f"[annotate] 输入 {len(files)} 个 JSON 文件", flush=True)
    for f in files:
        print(f"          - {f.relative_to(in_root)}", flush=True)

    stats = Stats()

    for fp in files:
        if args.sample_limit and stats.processed >= args.sample_limit:
            print("[annotate] 已达 sample_limit，停止", flush=True)
            break
        # --input 指到单个文件时，relative_to(自身) 会得到 "."，输出路径就错了；
        # 此时用文件名做 rel，输出为 <output>/<文件名>。
        rel = Path(fp.name) if in_root.is_file() else fp.relative_to(in_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[annotate] === {rel} ===", flush=True)
        process_file_streaming(fp, rel, out_path, args, stats)

    # 分布统计
    total_processed = stats.processed
    total_skipped_done = stats.skipped_done
    total_skipped_delegation = stats.skipped_delegation
    total_llm_calls = stats.llm_calls
    total_len_mismatch = stats.len_mismatch
    mismatch_examples = stats.mismatch_examples
    label_counter = stats.labels
    status_counter = stats.status
    print("\n=== 标注完成 ===", flush=True)
    print(f"处理 samples: {total_processed}", flush=True)
    print(f"跳过(续跑已完成 sample): {total_skipped_done}", flush=True)
    print(f"跳过(delegation→none): {total_skipped_delegation}", flush=True)
    print(f"LLM 调用数:   {total_llm_calls}", flush=True)
    print(f"question/response 长度不一致的 sample 数: {total_len_mismatch}",
          flush=True)
    if mismatch_examples:
        print("  例如(最多列5个):", flush=True)
        for vn, qn, rn in mismatch_examples:
            print(f"    video_name={vn}  len(question)={qn}  "
                  f"len(response)={rn}", flush=True)

    # LLM 调用健康度（识别"思考 token 吃光预算→静默误标 none"）
    print("\n=== LLM 调用健康度 ===", flush=True)
    for st in ("ok", "recovered", "empty", "error"):
        print(f"  {st:10} {status_counter.get(st, 0):>8}", flush=True)
    n_empty = status_counter.get("empty", 0)
    n_error = status_counter.get("error", 0)
    n_recovered = status_counter.get("recovered", 0)
    if total_llm_calls and (n_empty + n_error) / total_llm_calls > 0.02:
        print(f"\n  ⚠️  空/截断({n_empty}) + 错误({n_error}) 占比偏高！"
              f"这些已被兜底成 none，可能是**静默误标**。", flush=True)
        print(f"     若多为 empty：思考 token 很可能吃光了 max_tokens。对策：调大 "
              f"MAX_TOKENS，或（若平台支持）设 REASONING_EFFORT=none 关思考。", flush=True)
    elif n_recovered:
        print(f"\n  提示：{n_recovered} 条首次空/截断、加大预算重试后救回。"
              f"若占比高，建议直接调大 MAX_TOKENS 省一次往返。", flush=True)

    print("\n=== 标签分布 ===", flush=True)
    tot = sum(label_counter.values()) or 1
    # 先单列非情绪桶(none / 委派 / 解析失败哨兵);demeanor 模式下这些应极少
    shown = set()
    for special in ("none", _DELEGATION_LABEL, _FALLBACK):
        c = label_counter.get(special, 0)
        if c and special not in shown:
            print(f"  {special:18} {c:>8}  ({c/tot*100:5.1f}%)", flush=True)
            shown.add(special)
    nonzero = [(a, c) for a, c in label_counter.most_common() if a not in shown and c > 0]
    n_emo = len([a for a, _ in nonzero if a in _EMOTION_NAMES])
    print(f"  -- 命中的情绪动作 {n_emo}/{len(_EMOTION_NAMES)} 类(按计数降序)--", flush=True)
    for a, c in nonzero:
        print(f"  {a:18} {c:>8}  ({c/tot*100:5.1f}%)", flush=True)


if __name__ == "__main__":
    main()
