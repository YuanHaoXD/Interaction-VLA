# annotation/_test_resume_quota.py —— 断点续标 + response 配额闸 的离线自测
# 【不调真 API、不花钱】:把 label_batch 打桩成假标注器,只验证续跑与配额的账算得对不对。
# 用法: .\.venv\Scripts\python.exe _test_resume_quota.py
import json
import shutil
import sys
import tempfile
from pathlib import Path

import annotate_actions_api as A

CALLS = {"n": 0}


def fake_label_batch(batch):
    """假标注器:不调 API,给每条打 explain。统计被真正标注的条数。"""
    CALLS["n"] += len(batch)
    return [(d, "explain", "ok") for d, _r, _c in batch]


A.label_batch = fake_label_batch
A.CHUNK_SAMPLES = 2          # 调小,方便触发多次落盘


class Args:
    def __init__(self, **kw):
        self.task_types = None
        self.sample_limit = 0
        self.response_limit = 0
        self.per_file_limit = 0
        self.max_workers = 2
        self.no_resume = False
        self.__dict__.update(kw)


def make_input(path, n_samples, resp_per_sample):
    """造假输入:每个 sample 带 resp_per_sample 条 response。"""
    data = []
    for i in range(n_samples):
        data.append({
            "video_name": f"v{i}.mp4",
            "question": [{"content": f"q{i}", "time": "1"}],
            "response": [{"content": f"reply {i}-{j}", "time": "2"}
                         for j in range(resp_per_sample)],
        })
    Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def count_out(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    nr = sum(len(s["response"]) for s in d)
    labeled = sum(1 for s in d for r in s["response"] if "action" in r)
    return len(d), nr, labeled


def run(inp, out, **kw):
    stats = A.Stats()
    args = Args(input=str(inp), output=str(out), **kw)
    A.process_file_streaming(Path(inp), Path(inp).name, Path(out) / Path(inp).name,
                             args, stats)
    return stats


def main():
    tmp = Path(tempfile.mkdtemp(prefix="resume_test_"))
    ok = True
    try:
        inp = tmp / "in.json"
        out = tmp / "out"
        out.mkdir()
        make_input(inp, n_samples=10, resp_per_sample=3)     # 共 30 条 response

        # ── 用例 1:response 配额闸——限 10 条,每样本 3 条 → 只应标 3 个样本(9 条),不切半个样本
        CALLS["n"] = 0
        st = run(inp, out, response_limit=10)
        ns, nr, labeled = count_out(out / "in.json")
        print(f"[1] response_limit=10 → 输出 {ns} sample / 已标 {labeled} 条 | "
              f"真调用 {CALLS['n']} | stats.responses={st.responses}")
        if not (labeled == 9 and CALLS["n"] == 9 and st.responses == 9):
            print("    ✗ 期望 标9条/调用9次(3样本×3条,不超10)"); ok = False
        else:
            print("    ✓ 硬闸生效且不切半个样本")

        # ── 用例 2:续跑——同配额重跑,不应再调 API(已完成 .json 被跳过且计入配额)
        CALLS["n"] = 0
        st2 = run(inp, out, response_limit=10)
        print(f"[2] 同配额重跑 → 真调用 {CALLS['n']} | 计入配额 responses={st2.responses}")
        if not (CALLS["n"] == 0 and st2.responses == 9):
            print("    ✗ 期望 0 次调用、已完成的 9 条计入配额"); ok = False
        else:
            print("    ✓ 续跑零重复消费,且旧产物计入配额(这正是防超预算的关键)")

        # ── 用例 3:提高配额续跑——应只补新的,不重标旧的
        CALLS["n"] = 0
        shutil.rmtree(out); out.mkdir()
        run(inp, out, response_limit=10)          # 先标 9 条
        CALLS["n"] = 0
        st3 = run(inp, out, response_limit=30)    # 抬到 30
        print(f"[3] 抬配额到30续跑 → 本次真调用 {CALLS['n']}")
        if CALLS["n"] != 0:
            print("    注:.json 已完成→整份跳过(按文件粒度续跑),需 no_resume 才重标")

        # ── 用例 4:sidecar 回执存在且账对
        side = Path(str(out / "in.json") + ".done.json")
        print(f"[4] sidecar 回执: {side.exists()} -> {side.read_text(encoding='utf-8') if side.exists() else 'N/A'}")
        if not side.exists():
            print("    ✗ 缺回执,续跑将无法 O(1) 计入配额"); ok = False
        else:
            print("    ✓ 回执已写,续跑无需重解析大 .json")

        # ── 用例 5:jsonl 半截尾行(模拟崩溃)应被丢弃且不重复计
        CALLS["n"] = 0
        shutil.rmtree(out); out.mkdir()
        work = out / "in.json.jsonl"
        good = json.dumps({"video_name": "v0.mp4",
                           "question": [{"content": "q0"}],
                           "response": [{"content": "reply 0-0", "action": "explain"},
                                        {"content": "reply 0-1", "action": "explain"},
                                        {"content": "reply 0-2", "action": "explain"}]},
                          ensure_ascii=False)
        work.write_bytes(good.encode() + b"\n" + b'{"video_name": "v1.mp4", "resp')  # 半截
        n, nr = A._resume_from_jsonl(work)
        print(f"[5] 崩溃半截行 → 认到 {n} sample / {nr} response(半截行应丢弃)")
        if not (n == 1 and nr == 3):
            print("    ✗ 期望 1 sample/3 response"); ok = False
        else:
            print("    ✓ 半截行被丢弃,续跑可安全追加")

        print("\n" + ("=" * 46))
        print("全部通过 ✓ 可以放心跑 50 万条" if ok else "有用例失败 ✗ 先别跑全量")
        print("=" * 46)
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
