import asyncio
import json
import os
import time

from dotenv import load_dotenv
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator as _RetEval
from agent.main_agent import AgentV1, AgentV2, AgentV3

load_dotenv()

# ── Cấu hình ──────────────────────────────────────────────────────────────────
MAX_CASES = 10  # Giới hạn số test cases mỗi lần chạy (None = chạy hết)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ── Giữ nguyên tên class từ template, fill thật vào ──────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class ExpertEvaluator:
    def __init__(self):
        self._eval = _RetEval()

    async def score(self, case, resp):
        ground_truth_ids = case.get("ground_truth_ids", [])
        if case.get("follow_up_question") and case.get("follow_up_ground_truth_ids"):
            ground_truth_ids = case.get("follow_up_ground_truth_ids", ground_truth_ids)
        retrieved_ids = resp.get("retrieved_ids", [])
        hit_rate = self._eval.calculate_hit_rate(ground_truth_ids, retrieved_ids)
        mrr = self._eval.calculate_mrr(ground_truth_ids, retrieved_ids)
        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "faithfulness": 0.0,
            "relevancy": 0.0,
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
            },
        }

class MultiModelJudge:
    """Wrapper giữ tên gốc từ template, delegate sang LLMJudge thật."""

    def __init__(self):
        self._judge = LLMJudge(openai_api_key=OPENAI_API_KEY)

    async def evaluate_multi_judge(self, q, a, gt):
        return await self._judge.evaluate_multi_judge(q, a, gt)


# ── Giữ nguyên signature gốc, thêm agent param ───────────────────────────────

async def run_benchmark_with_results(agent_version: str, agent=None):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    import random
    random.seed(42)
    random.shuffle(dataset)

    if MAX_CASES is not None:
        dataset = dataset[:MAX_CASES]
        print(f"  ⚡ Giới hạn {MAX_CASES} cases (đổi MAX_CASES trong main.py để chạy hết)")

    if agent is None:
        agent = AgentV1()

    runner = BenchmarkRunner(agent, ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)

    for r in results:
        r["agent_version"] = agent_version

    total = len(results)

    def _extract_hit_rate(row: dict) -> float:
        ragas = row.get("ragas", {})
        if "hit_rate" in ragas:
            return float(ragas.get("hit_rate", 0.0))
        return float(ragas.get("retrieval", {}).get("hit_rate", 0.0))

    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(_extract_hit_rate(r) for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
        }
    }
    return results, summary

async def run_benchmark(version, agent=None):
    _, summary = await run_benchmark_with_results(version, agent)
    return summary


def _save_regression_cases(v1: list, v2: list, v3: list):
    """Write cases where V1 beats V2 or V3, side by side, to reports/regression_cases.txt."""
    lines = []
    flagged = 0

    for r1, r2, r3 in zip(v1, v2, v3):
        s1 = r1["judge"]["final_score"]
        s2 = r2["judge"]["final_score"]
        s3 = r3["judge"]["final_score"]
        v1_beats_v2 = s1 > s2
        v1_beats_v3 = s1 > s3
        if not v1_beats_v2 and not v1_beats_v3:
            continue

        flagged += 1
        sep = "=" * 80
        lines.append(sep)
        label = []
        if v1_beats_v2:
            label.append(f"V1({s1:.2f}) > V2({s2:.2f})")
        if v1_beats_v3:
            label.append(f"V1({s1:.2f}) > V3({s3:.2f})")
        lines.append(f"CASE #{flagged}  |  {' , '.join(label)}")
        lines.append(f"Type: {r1.get('test_case', '')[:120]}")
        lines.append("")

        q = r1.get("eval_question") or r1.get("test_case", "")
        lines.append(f"QUESTION: {q}")
        lines.append("")

        lines.append(f"{'V1-Base':^26} | {'V2-Rewrite':^26} | {'V3-Clarify':^26}")
        lines.append(f"{'score: ' + str(s1):^26} | {'score: ' + str(s2):^26} | {'score: ' + str(s3):^26}")
        lines.append("-" * 80)

        # Print answers wrapped at ~78 chars per column — simple line-by-line approach
        def wrap(text, width=76):
            words, cur, result = text.split(), "", []
            for w in words:
                if len(cur) + len(w) + 1 > width:
                    result.append(cur)
                    cur = w
                else:
                    cur = (cur + " " + w).strip()
            if cur:
                result.append(cur)
            return result or [""]

        a1 = wrap(r1.get("agent_response", ""))
        a2 = wrap(r2.get("agent_response", ""))
        a3 = wrap(r3.get("agent_response", ""))
        rows = max(len(a1), len(a2), len(a3))
        a1 += [""] * (rows - len(a1))
        a2 += [""] * (rows - len(a2))
        a3 += [""] * (rows - len(a3))
        for l1, l2, l3 in zip(a1, a2, a3):
            lines.append(f"{l1:<26} | {l2:<26} | {l3:<26}")

        lines.append("")
        # Per-criterion scores
        pc1 = r1["judge"].get("per_criterion", {})
        pc2 = r2["judge"].get("per_criterion", {})
        pc3 = r3["judge"].get("per_criterion", {})
        for crit in pc1:
            lines.append(f"  {crit:<16} V1={pc1.get(crit,'?'):.1f}  V2={pc2.get(crit,'?'):.1f}  V3={pc3.get(crit,'?'):.1f}")
        lines.append("")

    if flagged == 0:
        lines.append("No regression cases found (V1 did not beat V2 or V3 on any case).")

    out = "\n".join(lines)
    path = "reports/regression_cases.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"REGRESSION REPORT — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Cases where V1-Base outscores V2 or V3: {flagged}\n\n")
        f.write(out)
    print(f"  📄 Regression cases saved → {path}  ({flagged} cases)")


async def main():
    (v1_results, v1_summary), (v2_results, v2_summary), (v3_results, v3_summary) = \
        await asyncio.gather(
            run_benchmark_with_results("Agent_V1_Base",    AgentV1()),
            run_benchmark_with_results("Agent_V2_Rewrite", AgentV2()),
            run_benchmark_with_results("Agent_V3_Clarify", AgentV3()),
        )

    if not v1_summary or not v3_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    for label, s in [("V1-Base", v1_summary), ("V2-Rewrite", v2_summary), ("V3-Clarify", v3_summary)]:
        if s:
            print(f"  {label}: score={s['metrics']['avg_score']:.2f}  agreement={s['metrics']['agreement_rate']:.2f}")

    delta = v3_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"\nDelta V3 vs V1: {'+' if delta >= 0 else ''}{delta:.2f}")

    os.makedirs("reports", exist_ok=True)
    comparison = {
        "summaries": {
            "v1": v1_summary,
            "v2": v2_summary,
            "v3": v3_summary,
            "V1-Base":    v1_summary,
            "V2-Rewrite": v2_summary,
            "V3-Clarify": v3_summary,
        },
        "delta_v3_vs_v1": round(delta, 3),
    }
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "v1": v1_results,
                "v2": v2_results,
                "v3": v3_results,
                "V1-Base": v1_results,
                "V2-Rewrite": v2_results,
                "V3-Clarify": v3_results,
            },
            f, ensure_ascii=False, indent=2,
        )

    _save_regression_cases(v1_results, v2_results, v3_results)

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")


if __name__ == "__main__":
    asyncio.run(main())
