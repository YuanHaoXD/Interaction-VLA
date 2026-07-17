# annotation/_analyze_trial.py —— 试产结果分析(诊断脚本,下划线前缀)
# 用法: .\.venv\Scripts\python.exe _analyze_trial.py [产物目录]
# 默认产物目录 annotated_trial10k_cluster/;产出:
#   1) 总体 + 分 task_type 的簇分布
#   2) 成本/token 实测
#   3) 50 条分层抽样 → 写入 _trial_samples50.md(供实验报告引用)
import collections
import glob
import json
import os
import random
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "annotated_trial10k_cluster"
CLUSTERS = ["affirm", "explain", "attend", "think", "unsure", "joy",
            "surprise", "fear", "negate", "annoy", "sad", "warm", "awkward"]
PIN, POUT = 2.0, 5.0          # $/1M token(网关价)


def load(p):
    """兼容 .json 数组与半成品 .jsonl。"""
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        out = []
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out


def iter_resp(resp):
    if not isinstance(resp, list):
        return
    for it in resp:
        if isinstance(it, list):
            for s in it:
                if isinstance(s, dict):
                    yield s
        elif isinstance(it, dict):
            yield it


def main():
    if not os.path.isdir(OUT):
        print(f"找不到产物目录 {OUT}")
        return

    rows = []                                   # (task_type, cluster, question, reply)
    per_type = collections.defaultdict(collections.Counter)
    for f in glob.glob(f"{OUT}/**/*.json", recursive=True) + \
             glob.glob(f"{OUT}/**/*.jsonl", recursive=True):
        if "api_logs" in f.replace("\\", "/"):
            continue
        tt = f.replace("\\", "/").split(OUT.strip("./") + "/")[-1].split("/")[0]
        for s in load(f):
            qs = s.get("question") or []
            for idx, it in enumerate(s.get("response", [])):
                items = it if isinstance(it, list) else [it]
                for r in items:
                    if not isinstance(r, dict):
                        continue
                    a = r.get("action", "?")
                    per_type[tt][a] += 1
                    q = qs[idx]["content"] if idx < len(qs) and isinstance(qs[idx], dict) else ""
                    rows.append((tt, a, str(q), str(r.get("content", ""))))

    total = collections.Counter()
    for c in per_type.values():
        total.update(c)
    tot = sum(total.values()) or 1

    print("=" * 64)
    print(f"总标注 response: {tot}")
    print("=" * 64)
    print("\n### 总体簇分布")
    for a, c in total.most_common():
        flag = "  <== >40%" if (a in CLUSTERS and c / tot > 0.40) else ""
        print(f"  {a:16}{c:>7}  ({c/tot*100:5.1f}%){flag}")
    miss = [c for c in CLUSTERS if total.get(c, 0) == 0]
    print(f"  [未命中的簇] {miss if miss else '无(13簇全覆盖)'}")

    print("\n### 分 task_type")
    for tt, c in per_type.items():
        t = sum(c.values()) or 1
        top = ", ".join(f"{a} {n/t*100:.1f}%" for a, n in c.most_common(4))
        print(f"  [{tt:16}] {t:>6} 条 | {top}")

    # ---- 成本 ----
    pin = pout = ncall = 0
    for f in glob.glob(f"{OUT}/api_logs/**/*.jsonl", recursive=True):
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            r = d.get("response")
            if isinstance(r, str):
                try:
                    r = json.loads(r.replace("'", '"'))
                except Exception:
                    r = None
            u = (r or {}).get("usage") if isinstance(r, dict) else None
            if u:
                pin += u.get("prompt_tokens", 0)
                pout += u.get("completion_tokens", 0)
                ncall += 1
    if ncall:
        cost = pin / 1e6 * PIN + pout / 1e6 * POUT
        print(f"\n### 成本实测")
        print(f"  批调用 {ncall} 次 | 输入 {pin:,} tok | 输出 {pout:,} tok")
        print(f"  本次花费 ${cost:.2f}  |  每条 response ${cost/tot:.8f} ({pin/tot:.1f} tok 输入)")
        print(f"  外推首期 50 万条: ${cost/tot*500_000:,.0f}   (裁定预算上限 $200)")
        print(f"  外推 100 万条(§7.3 加量): ${cost/tot*1_000_000:,.0f}")

    # ---- 50 条分层抽样 ----
    random.seed(2026)
    nonexp = [r for r in rows if r[1] not in ("explain", "__delegation__", "__unparsed__")]
    exp = [r for r in rows if r[1] == "explain"]
    pick = random.sample(nonexp, min(30, len(nonexp))) + random.sample(exp, min(20, len(exp)))
    random.shuffle(pick)
    lines = ["| # | 类型 | 簇 | 提问(截) | 回复(截) |", "|---|---|---|---|---|"]
    for i, (tt, a, q, c) in enumerate(pick, 1):
        qq = q[:55].replace("\n", " ").replace("|", "/")
        cc = c[:85].replace("\n", " ").replace("|", "/")
        lines.append(f"| {i} | {tt} | `{a}` | {qq} | {cc} |")
    with open("_trial_samples50.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n### 50 条分层抽样(非explain {min(30,len(nonexp))} + explain {min(20,len(exp))})")
    print(f"  已写入 annotation/_trial_samples50.md")
    print("\n  前 8 条预览:")
    for l in lines[2:10]:
        print("   ", l)


if __name__ == "__main__":
    main()
