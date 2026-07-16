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

# 批处理(Fable5 裁定 §5.2 / §6 成本账):每次调用打包 N 条 response,把簇准则表(~700 token)
# 摊薄到每条 ~120 token 输入。N=20 是裁定给的初值。失败按单条重试,上限 MAX_ITEM_RETRY。
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "20"))
MAX_ITEM_RETRY = int(os.environ.get("MAX_ITEM_RETRY", "3"))

# ── 神态簇词表 v3(Fable5 裁定 2026-07-16:簇级 demeanor)──
# 关键裁定:【不做 85 选 1】。understanding1/understanding2/yes1 的差别在【轨迹形态】而非
# 【语句内容】——同一句"对,他完成了"配这三个都成立,让 LLM 强行 argmax 是在造噪声标签。
# 文本可判定的是"这句话在肯定/讲解/思考",所以 LLM 只判 13 个【神态簇】;簇内具体动作由
# 数据生成侧(data_gen/cluster_map.json)按权重随机采样。副产物:catalog 从 ~2400 token 降到
# ~700,且近义动作靠采样天然全覆盖(与沉默段多峰处理哲学一致)。
# 裁定原文:docs/decisions/2026-07-16-C3标注裁定-簇级demeanor与候选裁剪.md §2
#
# 每簇 = (簇名, 中文名, 触发条件, 反例)。逐行进 prompt。
CLUSTERS = [
    ("affirm",   "肯定认同", "The line's core is agreement/confirmation: yes, right, exactly, I agree.",
     "Neutral retelling of what is on screen -> explain"),
    ("explain",  "讲解陈述", "Neutral explaining: stating facts, reporting numbers/durations, describing what is seen, no clear emotional swing.",
     "Has 'look/notice/let's see' guidance -> attend; has 'seems/maybe' speculation -> think"),
    ("attend",   "引导关注", "Directing attention to something, opening/introducing a topic, encouraging the other to keep talking.",
     "Purely neutral statement -> explain"),
    ("think",    "思考推测", "Pondering/inferring/self-questioning: 'maybe, seems like, let me think, why is that'.",
     "Cannot answer at all -> unsure; neutral scenery description -> explain"),
    ("unsure",   "困惑不解", "Did not understand / cannot answer / not enough information.",
     "A speculation with a direction -> think"),
    ("joy",      "开心兴奋", "Good news, a spectacular moment, praise, a success achieved.",
     "Neutral 'it is done' -> explain; mild satisfaction -> affirm"),
    ("surprise", "惊讶",     "Unexpected turn, contrast, 'actually/never thought'.",
     "Surprised AND happy -> joy; frightened -> fear"),
    ("fear",     "紧张担忧", "Danger approaching, worried for someone, nervous/uneasy.",
     "Sadness about a bad outcome that already happened -> sad"),
    ("negate",   "否定拒绝", "The line's core is negation: no, it isn't, that won't work, I disagree — calm tone.",
     "Rebuttal with anger -> annoy"),
    ("annoy",    "不满恼火", "Being offended, criticizing/blaming, strong dissatisfaction.",
     "Calm disagreement -> negate"),
    ("sad",      "悲伤低落", "Sad, regretful, parting, empathizing with someone's pain.",
     "Nervous on someone's behalf -> fear"),
    ("warm",     "温暖安抚", "Greeting, thanking, comforting, welcoming, expressing fondness.",
     "Plain happiness -> joy"),
    ("awkward",  "尴尬歉意", "Own mistake, apologizing, embarrassed.",
     "Criticizing someone else's mistake -> annoy"),
]

CLUSTER_NAMES = tuple(c[0] for c in CLUSTERS)
ACTIONS = CLUSTER_NAMES          # annotation 产物的 action 值域 = 13 簇名(A5 契约修正,裁定 §4)

# 无 none(demeanor 哲学:人说话必带神态,必选一个)。解析失败/委派用可见哨兵,不静默塞标签。
_FALLBACK = "__unparsed__"
_DELEGATION_LABEL = "__delegation__"

_CLUSTER_TABLE = "\n".join(
    f"- {name:9} | USE WHEN: {trig}\n            NOT: {anti}"
    for name, _zh, trig, anti in CLUSTERS
)

PROMPT_SYSTEM = (
    "A small desktop robot is about to SAY each reply line given below (English or Chinese).\n"
    "For EACH line decide: what demeanor (神态) would a person naturally have while speaking "
    "THIS exact line? Even a plain, factual or descriptive line is spoken with SOME demeanor — "
    "so you must ALWAYS pick one. Choose the single best-fitting DEMEANOR CLUSTER.\n\n"
    "=== DEMEANOR CLUSTERS (choose exactly one name per line) ===\n"
    f"{_CLUSTER_TABLE}\n"
    "=== END CLUSTERS ===\n\n"
    "Rules:\n"
    "1. Use ONLY the cluster names exactly as spelled above. There is NO 'neutral'/'none' option.\n"
    "2. Judge how the line is naturally delivered (its function, tone and intent) — not merely how "
    "exciting the described content is. A calm narrator describing a fight is still `explain`, not `joy`.\n"
    "3. `explain` is only for lines that are PURELY neutral report. Before settling on `explain`, "
    "actively check the other clusters — a describing line that points something out is `attend`, "
    "one that speculates is `think`, one that confirms is `affirm`, one that marvels is `surprise`/`joy`. "
    "Use `explain` only when none of the others genuinely fit.\n"
    "4. If two clusters fit, use the NOT-hints above to disambiguate."
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


def _item_text(n, reply, context):
    return (f"--- ITEM {n} ---\n"
            f"Context (what was asked, for disambiguation only): {context or '(none)'}\n"
            f"Line: \"\"\"{reply}\"\"\"")


def build_batch_messages(items):
    """items: [(task, reply, context)] → 编号列表进、JSON 数组出(裁定 §5.2 批处理)。"""
    body = "\n\n".join(_item_text(n, r, c) for n, (_t, r, c) in enumerate(items, 1))
    user = (
        f"Label EACH of the {len(items)} items below.\n\n{body}\n\n"
        f"Respond with ONLY a JSON array of exactly {len(items)} objects, in the same order:\n"
        '[{"i": 1, "action": "<cluster>"}, ...]\n'
        "No prose, no markdown fence."
    )
    return [{"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user}]


def build_single_messages(reply, context):
    """单条重试用(批内某条解析失败时回退到单条调用)。"""
    user = (f"{_item_text(1, reply, context)}\n\n"
            'Respond with ONLY the JSON: {"i": 1, "action": "<cluster>"}')
    return [{"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user}]


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
_ACTIONS_BY_LEN = sorted(ACTIONS, key=len, reverse=True)   # 最长优先,防短簇名是长簇名子串


def _norm_cluster(a):
    """把一个候选字符串规整成合法簇名;不合法返回 None。"""
    if a is None:
        return None
    a = str(a).strip().strip('"\'').lower()
    return a if a in _ACTIONS_SET else None


def parse_action(text):
    """单条输出 → 簇名;失败返回 _FALLBACK(可见哨兵,不静默塞标签)。"""
    if not text:
        return _FALLBACK
    text = text.strip()
    # 1) JSON(主路径):{"i":1,"action":"explain"} 或裸 {"action":...}
    m = re.search(r'\{[^}]*\}', text)
    if m:
        try:
            got = _norm_cluster(json.loads(m.group()).get("action"))
            if got:
                return got
        except Exception:
            pass
    # 2) 裸簇名(prompt 规则1 要求只输出簇名)
    got = _norm_cluster(text)
    if got:
        return got
    # 3) 关键字兜底:最长优先
    low = text.lower()
    for a in _ACTIONS_BY_LEN:
        if a in low:
            return a
    return _FALLBACK


def parse_batch(text, n):
    """批输出 → 长度 n 的簇名列表;某位解析不出则该位为 None(交由单条重试)。"""
    out = [None] * n
    if not text:
        return out
    m = re.search(r'\[.*\]', text, re.S)          # 抓 JSON 数组
    if not m:
        return out
    try:
        arr = json.loads(m.group())
    except Exception:
        return out
    if not isinstance(arr, list):
        return out
    for pos, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        # 优先按 i 字段归位(模型可能乱序);i 缺失/越界则按数组位置
        idx = item.get("i")
        try:
            idx = int(idx) - 1
        except Exception:
            idx = pos
        if not (0 <= idx < n):
            idx = pos if pos < n else None
        if idx is None:
            continue
        got = _norm_cluster(item.get("action"))
        if got:
            out[idx] = got
    return out
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
                tasks.append((d, content, ctx))
        # 批处理(裁定 §5.2):每 BATCH_SIZE 条打一次包,摊薄 system prompt 的簇准则表
        batches = [tasks[i:i + BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)]
        futures = [pool.submit(label_batch, b) for b in batches]
        for fut in as_completed(futures):
            for d, action, status in fut.result():
                d["action"] = action
                stats.labels[action] += 1
                stats.status[status] += 1
            stats.llm_calls += 1                    # 计"批调用数"(单条重试另计)
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


def _call_with_budget_retry(messages, budget_mult=4):
    """调一次 API;空/截断则加大预算重试一次。返回 (output, status_hint)。

    status_hint: ok / recovered / empty / error
    """
    output, _resp, finish = call_api(messages)
    if output and output.startswith("ERROR:"):
        return None, "error"
    truncated = (not output or not output.strip()) or (finish == "length")
    if not truncated:
        return output, "ok"
    # 疑似思考 token 吃光了 max_tokens:加大预算重试一次(仅调大 max_tokens,对未知代理安全)
    output2, _r2, finish2 = call_api(messages, max_tokens=MAX_TOKENS * budget_mult)
    if output2 and not output2.startswith("ERROR:"):
        still_bad = (not output2.strip()) or (finish2 == "length")
        return (None, "empty") if still_bad else (output2, "recovered")
    return None, "empty"


def label_batch(batch):
    """标注一批(裁定 §5.2)。batch: [(d, reply, ctx)] → [(d, action, status)]。

    流程:整批一次调用 → 逐条校验 → 失败的单条重试(上限 MAX_ITEM_RETRY) → 仍失败记哨兵。
    status: ok / recovered / retried_ok(批内失败但单条救回) / empty / error
    """
    n = len(batch)
    output, hint = _call_with_budget_retry(build_batch_messages(batch))
    labels = parse_batch(output, n) if output else [None] * n

    results = []
    for pos, (d, reply, ctx) in enumerate(batch):
        action, status = labels[pos], (hint if labels[pos] is None else hint)
        if action is not None:
            results.append((d, action, "ok" if hint == "ok" else hint))
            continue
        # 批内该条没解析出来 → 单条重试(有上限,失败不无限重试)
        got, st = None, hint
        for _ in range(MAX_ITEM_RETRY):
            out1, st1 = _call_with_budget_retry(build_single_messages(reply, ctx))
            if out1:
                cand = parse_action(out1)
                if cand != _FALLBACK:
                    got, st = cand, "retried_ok"
                    break
            st = st1
        results.append((d, got if got else _FALLBACK,
                        st if got else ("empty" if st in ("ok", "empty") else st)))
    return results


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

    print(f"[annotate] 模型={MODEL_NAME}  簇级demeanor({len(CLUSTER_NAMES)}簇,无none)  "
          f"批大小={BATCH_SIZE}  max_tokens={MAX_TOKENS}  "
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
              f"这些已记为 {_FALLBACK} 哨兵（不静默塞标签，可在产物里筛出重标）。", flush=True)
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
    n_hit = len([a for a, _ in nonzero if a in CLUSTER_NAMES])
    print(f"  -- 命中的神态簇 {n_hit}/{len(CLUSTER_NAMES)}(按计数降序)--", flush=True)
    for a, c in nonzero:
        flag = "  ⚠️>40% 塌陷" if (a in CLUSTER_NAMES and c / tot > 0.40) else ""
        print(f"  {a:18} {c:>8}  ({c/tot*100:5.1f}%){flag}", flush=True)
    # 裁定 §5.3 的塌陷阈值:单簇 >40% 即回报再裁
    worst = max(((a, c) for a, c in nonzero if a in CLUSTER_NAMES), default=None,
                key=lambda x: x[1])
    if worst and worst[1] / tot > 0.40:
        print(f"\n  ⚠️  单簇 {worst[0]} 占 {worst[1]/tot*100:.1f}% > 40%(裁定 §5.3 塌陷阈值)"
              f"——需回报 Fable5 再裁准则/再裁簇。", flush=True)


if __name__ == "__main__":
    main()
